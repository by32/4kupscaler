"""Tests for the upscale engine (GPU-free via mocking)."""

from pathlib import Path

from upscaler.config.schema import UpscaleConfig
from upscaler.core.engine import UpscaleEngine


class TestUpscaleEngineInit:
    def test_creates_with_config(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video)
        engine = UpscaleEngine(config)
        assert engine.config is config
        assert engine._runner is None

    def test_default_model_is_3b_fp8(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video)
        engine = UpscaleEngine(config)
        assert engine.config.model == "3b-fp8"

    def test_config_preserved(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(
            input=video,
            model="7b-fp8",
            batch_size=5,
            resolution=720,
        )
        engine = UpscaleEngine(config)
        assert engine.config.model == "7b-fp8"
        assert engine.config.batch_size == 5
        assert engine.config.resolution == 720


class TestModelResolutionInEngine:
    """Verify the engine resolves model names correctly."""

    def test_resolves_friendly_name(self, tmp_path: Path) -> None:
        from upscaler.core.models import resolve_model_filename

        assert (
            resolve_model_filename("3b-fp8") == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        )

    def test_resolves_7b_variant(self, tmp_path: Path) -> None:
        from upscaler.core.models import get_model_variant

        assert get_model_variant("7b-sharp-fp16") == "7b"
