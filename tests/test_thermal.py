"""Tests for thermal policy evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from upscaler.diagnostics.gpu_monitor import GpuSnapshot
from upscaler.diagnostics.thermal import (
    ThermalAction,
    ThermalPolicy,
    evaluate,
    wait_for_cooldown,
)


def _snap(temp: int) -> GpuSnapshot:
    """Create a snapshot with the given temperature."""
    return GpuSnapshot(
        timestamp=1.0,
        temperature_c=temp,
        utilization_pct=80,
        vram_used_mb=8000,
        vram_total_mb=10240,
        power_draw_w=250.0,
    )


class TestThermalPolicy:
    def test_default_values(self):
        policy = ThermalPolicy()
        assert policy.warn_temp_c == 85
        assert policy.critical_temp_c == 90
        assert policy.cooldown_target_c == 80

    def test_custom_values(self):
        policy = ThermalPolicy(warn_temp_c=80, critical_temp_c=85, cooldown_target_c=75)
        assert policy.warn_temp_c == 80

    def test_invalid_ordering_raises(self):
        with pytest.raises(ValueError, match="ordering violated"):
            ThermalPolicy(warn_temp_c=90, critical_temp_c=85, cooldown_target_c=80)

    def test_equal_values_raises(self):
        with pytest.raises(ValueError, match="ordering violated"):
            ThermalPolicy(warn_temp_c=85, critical_temp_c=85, cooldown_target_c=80)

    def test_cooldown_above_warn_raises(self):
        with pytest.raises(ValueError, match="ordering violated"):
            ThermalPolicy(warn_temp_c=80, critical_temp_c=90, cooldown_target_c=85)


class TestEvaluate:
    def test_ok_below_warn(self):
        assert evaluate(_snap(84), ThermalPolicy()) == ThermalAction.OK

    def test_ok_well_below(self):
        assert evaluate(_snap(60), ThermalPolicy()) == ThermalAction.OK

    def test_warn_at_threshold(self):
        assert evaluate(_snap(85), ThermalPolicy()) == ThermalAction.WARN

    def test_warn_between_thresholds(self):
        assert evaluate(_snap(87), ThermalPolicy()) == ThermalAction.WARN

    def test_warn_just_below_critical(self):
        assert evaluate(_snap(89), ThermalPolicy()) == ThermalAction.WARN

    def test_pause_at_critical(self):
        assert evaluate(_snap(90), ThermalPolicy()) == ThermalAction.PAUSE

    def test_pause_above_critical(self):
        assert evaluate(_snap(95), ThermalPolicy()) == ThermalAction.PAUSE

    def test_custom_thresholds(self):
        policy = ThermalPolicy(warn_temp_c=70, critical_temp_c=80, cooldown_target_c=60)
        assert evaluate(_snap(65), policy) == ThermalAction.OK
        assert evaluate(_snap(75), policy) == ThermalAction.WARN
        assert evaluate(_snap(85), policy) == ThermalAction.PAUSE


class TestWaitForCooldown:
    def test_returns_when_cooled(self):
        monitor = MagicMock()
        # First read: hot, second read: cool
        monitor.latest.side_effect = [_snap(92), _snap(78)]

        policy = ThermalPolicy(cooldown_poll_s=0.01)
        with patch("upscaler.diagnostics.thermal.time.sleep"):
            wait_for_cooldown(monitor, policy)

        assert monitor.latest.call_count == 2

    def test_returns_when_no_reading(self):
        monitor = MagicMock()
        monitor.latest.return_value = None

        policy = ThermalPolicy(cooldown_poll_s=0.01)
        with patch("upscaler.diagnostics.thermal.time.sleep"):
            wait_for_cooldown(monitor, policy)
