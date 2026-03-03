"""Tests for the upscale engine (GPU-free via mocking)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestSegmentedRouting:
    """Verify the engine routes to segmented vs single-pass correctly."""

    def test_no_segment_size_uses_single_pass(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video)
        engine = UpscaleEngine(config)
        assert config.segment_size is None
        # Without segment_size, _process should call _process_single_pass
        with patch.object(engine, "_process_single_pass") as mock_sp:
            mock_sp.return_value = video
            engine._process(skip_frames=0, max_frames=5)
            mock_sp.assert_called_once()

    def test_segment_size_with_max_frames_uses_single_pass(
        self, tmp_path: Path
    ) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video, segment_size=5)
        engine = UpscaleEngine(config)
        # When max_frames is set, should use single pass (e.g. preview mode)
        with patch.object(engine, "_process_single_pass") as mock_sp:
            mock_sp.return_value = video
            engine._process(skip_frames=0, max_frames=3)
            mock_sp.assert_called_once()

    def test_segment_size_routes_to_segmented(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video, segment_size=5)
        engine = UpscaleEngine(config)
        # Mock get_video_meta to return enough frames to trigger segmentation
        mock_meta = MagicMock(frame_count=20)
        with (
            patch("upscaler.core.engine.get_video_meta", return_value=mock_meta),
            patch.object(engine, "_process_segmented") as mock_seg,
        ):
            mock_seg.return_value = video
            engine._process(skip_frames=0, max_frames=None)
            mock_seg.assert_called_once()

    def test_segment_size_larger_than_video_uses_single_pass(
        self, tmp_path: Path
    ) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video, segment_size=21)
        engine = UpscaleEngine(config)
        mock_meta = MagicMock(frame_count=10)
        with (
            patch("upscaler.core.engine.get_video_meta", return_value=mock_meta),
            patch.object(engine, "_process_single_pass") as mock_sp,
        ):
            mock_sp.return_value = video
            engine._process(skip_frames=0, max_frames=None)
            mock_sp.assert_called_once()


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
