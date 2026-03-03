"""Tests for hardware presets."""

import pytest

from upscaler.core.presets import PRESETS, get_preset


class TestGetPreset:
    def test_rtx3080(self) -> None:
        preset = get_preset("rtx3080")
        assert preset["model"] == "3b-fp8"
        assert preset["batch_size"] == 1
        assert preset["blocks_to_swap"] == 20
        assert preset["preserve_vram"] is True
        assert preset["segment_size"] == 5

    def test_rtx3090(self) -> None:
        preset = get_preset("rtx3090")
        assert preset["segment_size"] == 13

    def test_rtx4090(self) -> None:
        preset = get_preset("rtx4090")
        assert preset["model"] == "7b-fp8"
        assert preset["batch_size"] == 5
        assert preset["segment_size"] == 21

    def test_returns_copy(self) -> None:
        """Mutating the returned dict shouldn't affect the original."""
        preset = get_preset("rtx3080")
        preset["model"] = "changed"
        assert PRESETS["rtx3080"]["model"] == "3b-fp8"

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("rtx9999")
