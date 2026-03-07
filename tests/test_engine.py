"""Tests for the upscale engine (GPU-free via mocking)."""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from upscaler.config.schema import UpscaleConfig, VAETilingConfig
from upscaler.core.engine import UpscaleEngine, _format_duration


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


class TestFormatDuration:
    """Verify human-readable duration formatting."""

    def test_seconds_only(self) -> None:
        assert _format_duration(0) == "0s"
        assert _format_duration(45) == "45s"
        assert _format_duration(59.9) == "59s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_duration(60) == "1m00s"
        assert _format_duration(90) == "1m30s"
        assert _format_duration(3599) == "59m59s"

    def test_hours_and_minutes(self) -> None:
        assert _format_duration(3600) == "1h00m"
        assert _format_duration(7380) == "2h03m"
        assert _format_duration(36000) == "10h00m"


class TestInterPhaseVramClearing:
    """Verify empty_cache is called between inference phases."""

    def test_clear_vram_called_between_phases(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video)
        engine = UpscaleEngine(config)

        # Set up cached runner state
        engine._runner = MagicMock()
        engine._base_ctx = {"dit_device": "cpu", "compute_dtype": "float32"}
        engine._cache_context = MagicMock()
        engine._text_embeds = MagicMock()
        engine._debug = MagicMock()

        # Mock result tensor that avoids needing real torch dtypes
        mock_result = MagicMock()
        mock_result.is_cuda = False
        mock_result.dtype = "float32"

        # Mock torch module for the final dtype check
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float8_e4m3fn = "float8_e4m3fn"
        mock_torch.float8_e5m2 = "float8_e5m2"
        mock_torch.float32 = "float32"

        def make_ctx_returner(result_on_last=False):
            def phase_fn(*args, **kwargs):
                ctx = kwargs.get("ctx", args[1] if len(args) > 1 else {})
                if result_on_last:
                    ctx["final_video"] = mock_result
                return ctx

            return phase_fn

        import sys

        # Mock vendored modules that import torch/torchvision at module level
        mock_phases = MagicMock()
        mock_phases.encode_all_batches = MagicMock(side_effect=make_ctx_returner())
        mock_phases.upscale_all_batches = MagicMock(side_effect=make_ctx_returner())
        mock_phases.decode_all_batches = MagicMock(side_effect=make_ctx_returner())
        mock_phases.postprocess_all_batches = MagicMock(
            side_effect=make_ctx_returner(result_on_last=True)
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": mock_torch,
                    "upscaler._vendor.seedvr2.src.core.generation_phases": mock_phases,
                },
            ),
            patch.object(UpscaleEngine, "_clear_vram") as mock_clear,
        ):
            engine._run_inference_cached(frames=MagicMock())
            # Should be called twice: after encode→upscale, after upscale→decode
            assert mock_clear.call_count == 2


class TestSegmentTiming:
    """Verify segment timing is logged."""

    def test_segment_completion_logged(self, tmp_path: Path, caplog) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video, segment_size=5)
        engine = UpscaleEngine(config)

        mock_meta = MagicMock(frame_count=12, fps=24.0)
        mock_frames = MagicMock()
        mock_result = MagicMock()
        mock_result.shape = (5, 1080, 1920, 3)

        # Mock time.monotonic to return predictable values
        # Calls: run_t0, seg1_start, seg1_end, seg2_start, seg2_end, total_end
        time_values = iter([0.0, 100.0, 210.0, 210.0, 325.0, 325.0])

        mock_writer = MagicMock()

        with (
            patch("upscaler.core.engine.get_video_meta", return_value=mock_meta),
            patch("upscaler.core.engine.ensure_models", return_value=tmp_path),
            patch.object(engine, "_ensure_models_loaded"),
            patch(
                "upscaler.core.engine.read_video_segment",
                return_value=(mock_frames, mock_meta),
            ),
            patch("upscaler.core.engine.pad_to_4n1", return_value=(mock_frames, 0)),
            patch(
                "upscaler.core.engine.compute_segments",
                return_value=[(0, 5), (5, 10)],
            ),
            patch.object(engine, "_run_inference_cached", return_value=mock_result),
            patch("upscaler.core.engine.time") as mock_time,
            patch(
                "upscaler.core.engine.StreamingVideoWriter",
                return_value=mock_writer,
            ),
            caplog.at_level(logging.INFO, logger="upscaler.core.engine"),
        ):
            mock_time.monotonic = MagicMock(side_effect=time_values)
            engine._process_segmented(skip_frames=0)

        # Check that segment completion messages were logged
        completion_msgs = [r for r in caplog.records if "complete" in r.message]
        assert len(completion_msgs) == 2
        assert "avg" in completion_msgs[0].message
        assert "ETA" in completion_msgs[0].message


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


class TestTilingParamsPassedToRunner:
    """Verify tiling parameters are forwarded to prepare_runner."""

    def test_tiling_params_passed_to_prepare_runner(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(
            input=video,
            resolution=2160,
            vae_tiling=VAETilingConfig(
                encode_tiled=True,
                decode_tiled=True,
                encode_tile_size=512,
                encode_tile_overlap=160,
                decode_tile_size=512,
                decode_tile_overlap=160,
            ),
        )
        engine = UpscaleEngine(config)

        # Mock all vendored imports
        mock_prepare_runner = MagicMock(return_value=(MagicMock(), MagicMock()))
        mock_gen_utils = MagicMock()
        mock_gen_utils.prepare_runner = mock_prepare_runner
        mock_gen_utils.setup_generation_context = MagicMock(
            return_value={"dit_device": "cpu", "compute_dtype": "float32"}
        )
        mock_gen_utils.load_text_embeddings = MagicMock()

        mock_constants = MagicMock()
        mock_constants.get_script_directory = MagicMock(return_value="/tmp")

        mock_debug_mod = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "upscaler._vendor.seedvr2.src.core.generation_utils": mock_gen_utils,
                "upscaler._vendor.seedvr2.src.utils.constants": mock_constants,
                "upscaler._vendor.seedvr2.src.utils.debug": mock_debug_mod,
            },
        ):
            engine._ensure_models_loaded(tmp_path, "model.safetensors")

        mock_prepare_runner.assert_called_once()
        kwargs = mock_prepare_runner.call_args.kwargs
        assert kwargs["encode_tiled"] is True
        assert kwargs["decode_tiled"] is True
        assert kwargs["encode_tile_size"] == (512, 512)
        assert kwargs["encode_tile_overlap"] == (160, 160)
        assert kwargs["decode_tile_size"] == (512, 512)
        assert kwargs["decode_tile_overlap"] == (160, 160)
