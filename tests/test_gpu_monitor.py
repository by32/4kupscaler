"""Tests for GPU monitoring module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from upscaler.diagnostics.gpu_monitor import GpuMonitor, GpuSnapshot


class TestGpuSnapshot:
    def test_creation(self):
        snap = GpuSnapshot(
            timestamp=1.0,
            temperature_c=72,
            utilization_pct=85,
            vram_used_mb=8192,
            vram_total_mb=10240,
            power_draw_w=250.0,
            fan_speed_pct=60,
        )
        assert snap.temperature_c == 72
        assert snap.vram_used_mb == 8192
        assert snap.power_draw_w == 250.0
        assert snap.fan_speed_pct == 60

    def test_fan_speed_optional(self):
        snap = GpuSnapshot(
            timestamp=1.0,
            temperature_c=70,
            utilization_pct=50,
            vram_used_mb=4096,
            vram_total_mb=10240,
            power_draw_w=200.0,
        )
        assert snap.fan_speed_pct is None

    def test_frozen(self):
        snap = GpuSnapshot(
            timestamp=1.0,
            temperature_c=70,
            utilization_pct=50,
            vram_used_mb=4096,
            vram_total_mb=10240,
            power_draw_w=200.0,
        )
        with pytest.raises(AttributeError):
            snap.temperature_c = 80


class TestGpuMonitor:
    def _mock_pynvml(self):
        """Create a mock pynvml module."""
        mock_nvml = MagicMock()
        mock_nvml.NVML_TEMPERATURE_GPU = 0
        mock_nvml.NVMLError = type("NVMLError", (Exception,), {})

        # Mock utilization rates
        util = MagicMock()
        util.gpu = 85
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = util

        # Mock memory info
        mem = MagicMock()
        mem.used = 8192 * 1024 * 1024
        mem.total = 10240 * 1024 * 1024
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem

        mock_nvml.nvmlDeviceGetTemperature.return_value = 72
        mock_nvml.nvmlDeviceGetPowerUsage.return_value = 250000  # mW
        mock_nvml.nvmlDeviceGetFanSpeed.return_value = 60

        return mock_nvml

    def test_start_stop_with_mock(self):
        mock_nvml = self._mock_pynvml()
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            monitor = GpuMonitor(poll_interval=0.1)
            monitor.start()
            time.sleep(0.3)
            monitor.stop()

            snap = monitor.latest()
            assert snap is not None
            assert snap.temperature_c == 72
            assert snap.utilization_pct == 85

    def test_context_manager(self):
        mock_nvml = self._mock_pynvml()
        with (
            patch.dict("sys.modules", {"pynvml": mock_nvml}),
            GpuMonitor(poll_interval=0.1) as mon,
        ):
            time.sleep(0.3)
            assert mon.latest() is not None

    def test_history_bounded(self):
        mock_nvml = self._mock_pynvml()
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            monitor = GpuMonitor(poll_interval=0.05, history_limit=5)
            monitor.start()
            time.sleep(0.5)
            monitor.stop()

            hist = monitor.history()
            assert len(hist) <= 5

    def test_summary(self):
        mock_nvml = self._mock_pynvml()
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            monitor = GpuMonitor(poll_interval=0.1)
            monitor.start()
            time.sleep(0.3)
            monitor.stop()

            summary = monitor.summary()
            assert "temperature_c" in summary
            assert summary["temperature_c"]["min"] == 72
            assert summary["temperature_c"]["max"] == 72

    def test_export_csv(self, tmp_path):
        mock_nvml = self._mock_pynvml()
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            monitor = GpuMonitor(poll_interval=0.1)
            monitor.start()
            time.sleep(0.3)
            monitor.stop()

            csv_path = tmp_path / "metrics.csv"
            monitor.export_csv(csv_path)
            assert csv_path.exists()
            content = csv_path.read_text()
            assert "temperature_c" in content
            assert "72" in content

    def test_latest_none_before_start(self):
        monitor = GpuMonitor()
        assert monitor.latest() is None

    def test_graceful_when_pynvml_unavailable(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GpuMonitor()
            monitor.start()  # Should not raise
            monitor.stop()
            assert monitor.latest() is None
