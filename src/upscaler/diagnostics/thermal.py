"""Thermal policy — evaluate GPU temperature and pause if overheating."""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass

from upscaler.diagnostics.gpu_monitor import GpuMonitor, GpuSnapshot


class ThermalAction(enum.Enum):
    """Result of evaluating a GPU snapshot against the thermal policy."""

    OK = "ok"
    WARN = "warn"
    PAUSE = "pause"


@dataclass
class ThermalPolicy:
    """Temperature thresholds for GPU thermal management.

    Must satisfy: cooldown_target_c < warn_temp_c < critical_temp_c
    """

    warn_temp_c: int = 85
    critical_temp_c: int = 90
    cooldown_target_c: int = 80
    cooldown_poll_s: float = 5.0

    def __post_init__(self) -> None:
        if not (self.cooldown_target_c < self.warn_temp_c < self.critical_temp_c):
            raise ValueError(
                f"Temperature ordering violated: cooldown({self.cooldown_target_c}) "
                f"< warn({self.warn_temp_c}) < critical({self.critical_temp_c})"
            )


def evaluate(snapshot: GpuSnapshot, policy: ThermalPolicy) -> ThermalAction:
    """Evaluate a snapshot against the thermal policy."""
    if snapshot.temperature_c >= policy.critical_temp_c:
        return ThermalAction.PAUSE
    if snapshot.temperature_c >= policy.warn_temp_c:
        return ThermalAction.WARN
    return ThermalAction.OK


def wait_for_cooldown(
    monitor: GpuMonitor,
    policy: ThermalPolicy,
    log: logging.Logger | None = None,
) -> None:
    """Block until GPU temperature drops below the cooldown target.

    Polls the monitor at ``policy.cooldown_poll_s`` intervals and logs
    each check.  Returns as soon as a reading is at or below the target.
    """
    log = log or logging.getLogger(__name__)
    log.warning("Waiting for GPU to cool below %d\u00b0C...", policy.cooldown_target_c)
    while True:
        time.sleep(policy.cooldown_poll_s)
        snap = monitor.latest()
        if snap is None:
            log.warning("No GPU reading available — resuming")
            return
        log.info(
            "GPU temp: %d\u00b0C (target: %d\u00b0C)",
            snap.temperature_c,
            policy.cooldown_target_c,
        )
        if snap.temperature_c <= policy.cooldown_target_c:
            log.info(
                "GPU cooled to %d\u00b0C — resuming processing", snap.temperature_c
            )
            return
