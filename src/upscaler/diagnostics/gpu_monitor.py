"""GPU monitoring via pynvml — polls temperature, utilization, VRAM, and power."""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuSnapshot:
    """Single point-in-time GPU reading."""

    timestamp: float
    temperature_c: int
    utilization_pct: int
    vram_used_mb: int
    vram_total_mb: int
    power_draw_w: float
    fan_speed_pct: int | None = None


class GpuMonitor:
    """Daemon thread that polls GPU metrics via pynvml.

    Usage::

        with GpuMonitor(device_index=0) as mon:
            # ... long-running work ...
            snap = mon.latest()
            print(f"GPU temp: {snap.temperature_c}°C")
    """

    def __init__(
        self,
        device_index: int = 0,
        poll_interval: float = 2.0,
        history_limit: int = 10_000,
    ) -> None:
        self._device_index = device_index
        self._poll_interval = poll_interval
        self._history: deque[GpuSnapshot] = deque(maxlen=history_limit)
        self._latest: GpuSnapshot | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Initialise pynvml and start the polling thread."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception:
            logger.warning("pynvml unavailable — GPU monitoring disabled")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="gpu-monitor"
        )
        self._thread.start()
        logger.debug("GPU monitor started (poll every %.1fs)", self._poll_interval)

    def stop(self) -> None:
        """Signal the thread to stop, join, and shut down pynvml."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval * 2)
            self._thread = None
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass

    def __enter__(self) -> GpuMonitor:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- accessors ---------------------------------------------------------

    def latest(self) -> GpuSnapshot | None:
        """Return the most recent snapshot (thread-safe)."""
        with self._lock:
            return self._latest

    def history(self) -> list[GpuSnapshot]:
        """Return a copy of all collected snapshots."""
        with self._lock:
            return list(self._history)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return min/max/avg for numeric metrics."""
        snaps = self.history()
        if not snaps:
            return {}
        result: dict[str, dict[str, float]] = {}
        for key in ("temperature_c", "utilization_pct", "vram_used_mb", "power_draw_w"):
            values = [getattr(s, key) for s in snaps]
            result[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }
        return result

    # -- CSV export --------------------------------------------------------

    def export_csv(self, path: Path) -> None:
        """Write collected metrics history to a CSV file."""
        snaps = self.history()
        if not snaps:
            return
        field_names = [f.name for f in fields(GpuSnapshot) if f.name != "fan_speed_pct"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            for snap in snaps:
                row = {k: v for k, v in asdict(snap).items() if k in field_names}
                writer.writerow(row)
        logger.info("GPU metrics written to %s (%d samples)", path, len(snaps))

    # -- internal ----------------------------------------------------------

    def _poll_loop(self) -> None:
        """Thread target: poll pynvml at the configured interval."""
        while not self._stop_event.is_set():
            try:
                snap = self._read_snapshot()
                with self._lock:
                    self._latest = snap
                    self._history.append(snap)
            except Exception:
                logger.debug("GPU poll failed", exc_info=True)
            self._stop_event.wait(timeout=self._poll_interval)

    def _read_snapshot(self) -> GpuSnapshot:
        """Read current GPU state via pynvml."""
        import pynvml

        temp = pynvml.nvmlDeviceGetTemperature(
            self._handle, pynvml.NVML_TEMPERATURE_GPU
        )
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W

        import contextlib

        fan: int | None = None
        with contextlib.suppress(pynvml.NVMLError):
            fan = pynvml.nvmlDeviceGetFanSpeed(self._handle)

        return GpuSnapshot(
            timestamp=time.monotonic(),
            temperature_c=temp,
            utilization_pct=util.gpu,
            vram_used_mb=mem.used // (1024 * 1024),
            vram_total_mb=mem.total // (1024 * 1024),
            power_draw_w=round(power, 1),
            fan_speed_pct=fan,
        )
