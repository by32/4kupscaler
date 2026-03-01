"""Tests for model registry and resolution."""

import pytest

from upscaler.core.models import (
    MODEL_REGISTRY,
    get_max_blocks,
    get_model_variant,
    resolve_model_filename,
)


class TestResolveModelFilename:
    def test_friendly_names(self) -> None:
        for name, expected in MODEL_REGISTRY.items():
            assert resolve_model_filename(name) == expected

    def test_raw_safetensors_passthrough(self) -> None:
        raw = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        assert resolve_model_filename(raw) == raw

    def test_raw_gguf_passthrough(self) -> None:
        raw = "seedvr2_ema_3b-Q4_K_M.gguf"
        assert resolve_model_filename(raw) == raw

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_filename("nonexistent")


class TestGetModelVariant:
    def test_3b_models(self) -> None:
        assert get_model_variant("3b-fp8") == "3b"
        assert get_model_variant("3b-fp16") == "3b"

    def test_7b_models(self) -> None:
        assert get_model_variant("7b-fp8") == "7b"
        assert get_model_variant("7b-fp16") == "7b"
        assert get_model_variant("7b-sharp-fp8") == "7b"
        assert get_model_variant("7b-sharp-fp16") == "7b"


class TestGetMaxBlocks:
    def test_3b_max_32(self) -> None:
        assert get_max_blocks("3b-fp8") == 32
        assert get_max_blocks("3b-fp16") == 32

    def test_7b_max_36(self) -> None:
        assert get_max_blocks("7b-fp8") == 36
        assert get_max_blocks("7b-fp16") == 36
