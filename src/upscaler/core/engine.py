"""Upscale orchestrator — ties model loading, video I/O, and inference together.

The UpscaleEngine is the central module that:
1. Resolves and downloads model weights
2. Reads input video into tensors
3. Enforces frame constraints (4n+1 padding)
4. Configures the SeedVR2 inference pipeline
5. Runs the 4-phase generation pipeline (encode -> upscale -> decode -> postprocess)
6. Writes output video or PNGs

Supports two modes:
- Single-pass: loads all frames, processes, writes (legacy behavior)
- Segmented: processes in small chunks, streaming output to disk (low memory)
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from pathlib import Path

from upscaler.config.schema import UpscaleConfig
from upscaler.core.models import VAE_FILENAME, ensure_models, resolve_model_filename
from upscaler.core.video_io import (
    StreamingPngWriter,
    StreamingVideoWriter,
    compute_segments,
    get_video_meta,
    pad_to_4n1,
    read_video,
    read_video_segment,
    write_frames_as_png,
    write_video,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]
"""Callback signature: (current_step, total_steps, phase_name)"""


class UpscaleEngine:
    """Orchestrates the full upscale pipeline.

    Usage:
        config = UpscaleConfig(input=Path("video.mp4"))
        engine = UpscaleEngine(config)
        output_path = engine.run()
    """

    def __init__(self, config: UpscaleConfig) -> None:
        self.config = config
        self._runner = None
        self._ctx = None

    def run(self, progress_callback: ProgressCallback | None = None) -> Path:
        """Run the full upscale pipeline.

        Args:
            progress_callback: Optional callback for progress reporting.

        Returns:
            Path to the output file/directory.
        """
        return self._process(
            skip_frames=self.config.skip_first_frames,
            max_frames=self.config.max_frames,
            progress_callback=progress_callback,
        )

    def preview(
        self,
        n_frames: int = 5,
        start_at: int = 0,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Run a quick preview on a subset of frames.

        Args:
            n_frames: Number of frames to process.
            start_at: Frame to start from.
            progress_callback: Optional callback for progress reporting.

        Returns:
            Path to preview output directory (PNGs).
        """
        # Force PNG output for previews
        original_format = self.config.output_format
        self.config.output_format = "png"
        try:
            return self._process(
                skip_frames=start_at,
                max_frames=n_frames,
                progress_callback=progress_callback,
            )
        finally:
            self.config.output_format = original_format

    def _process(
        self,
        skip_frames: int = 0,
        max_frames: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Internal processing pipeline — routes to single-pass or segmented."""
        cfg = self.config

        # Determine if segmented processing applies
        if cfg.segment_size is not None and max_frames is None:
            meta = get_video_meta(cfg.input)
            effective_frames = meta.frame_count - skip_frames
            if effective_frames > cfg.segment_size:
                return self._process_segmented(
                    skip_frames=skip_frames,
                    progress_callback=progress_callback,
                )

        return self._process_single_pass(
            skip_frames=skip_frames,
            max_frames=max_frames,
            progress_callback=progress_callback,
        )

    def _process_single_pass(
        self,
        skip_frames: int = 0,
        max_frames: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Original single-pass pipeline: read all -> infer all -> write all."""
        cfg = self.config
        filename = resolve_model_filename(cfg.model)

        # Step 1: Ensure model weights are downloaded
        logger.info("Step 1/5: Ensuring model weights...")
        model_dir = ensure_models(cfg.model, cfg.model_dir)

        # Step 2: Read input video
        logger.info("Step 2/5: Reading input video...")
        frames, meta = read_video(
            cfg.input, skip_frames=skip_frames, max_frames=max_frames
        )

        # Step 3: Pad frames to satisfy 4n+1 constraint
        frames, pad_count = pad_to_4n1(frames)

        # Step 4: Run inference
        logger.info("Step 3/5: Running SeedVR2 inference...")
        result = self._run_inference(
            frames=frames,
            model_dir=model_dir,
            filename=filename,
            progress_callback=progress_callback,
        )

        # Remove padding frames from result
        if pad_count > 0:
            result = result[:-pad_count]

        # Step 5: Write output
        logger.info("Step 4/5: Writing output...")
        output_path = cfg.output
        assert output_path is not None  # set_default_output validator guarantees this

        if cfg.output_format == "png":
            write_frames_as_png(result, output_path)
        else:
            write_video(result, output_path, fps=meta.fps)

        logger.info("Done! Output: %s", output_path)
        return output_path

    def _process_segmented(
        self,
        skip_frames: int = 0,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Segmented pipeline: process video in chunks, streaming output to disk."""
        cfg = self.config
        segment_size = cfg.segment_size
        assert segment_size is not None
        filename = resolve_model_filename(cfg.model)

        # Step 1: Ensure model weights
        logger.info("Step 1: Ensuring model weights...")
        model_dir = ensure_models(cfg.model, cfg.model_dir)

        # Step 2: Get video metadata and compute segments
        meta = get_video_meta(cfg.input)
        effective_frames = meta.frame_count - skip_frames
        segments = compute_segments(effective_frames, segment_size)
        logger.info(
            "Segmented processing: %d frames in %d segments (size=%d)",
            effective_frames,
            len(segments),
            segment_size,
        )

        # Step 3: Load models once for all segments
        logger.info("Step 2: Loading models...")
        self._ensure_models_loaded(model_dir, filename)

        output_path = cfg.output
        assert output_path is not None

        # Step 4: Open streaming writer and process each segment
        # Compute output resolution — we need the upscaled dimensions.
        # The short side is cfg.resolution, compute the long side proportionally.
        scale = cfg.resolution / min(meta.width, meta.height)
        out_h = round(meta.height * scale)
        out_w = round(meta.width * scale)
        # Ensure even dimensions for video codec
        out_h = out_h + (out_h % 2)
        out_w = out_w + (out_w % 2)

        if cfg.output_format == "png":
            writer_ctx = StreamingPngWriter(output_path)
        else:
            writer_ctx = StreamingVideoWriter(
                output_path, fps=meta.fps, width=out_w, height=out_h
            )

        with writer_ctx as writer:
            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                abs_start = skip_frames + seg_start
                abs_end = skip_frames + seg_end
                logger.info(
                    "Segment %d/%d: frames [%d:%d]",
                    seg_idx + 1,
                    len(segments),
                    abs_start,
                    abs_end,
                )

                # Read segment frames
                frames, _ = read_video_segment(cfg.input, abs_start, abs_end)

                # Pad to 4n+1
                frames, pad_count = pad_to_4n1(frames)

                # Run inference with cached models
                result = self._run_inference_cached(
                    frames=frames,
                    progress_callback=progress_callback,
                )

                # Remove padding frames
                if pad_count > 0:
                    result = result[:-pad_count]

                # Stream to writer
                writer.write_tensor(result)

                # Free segment memory
                del frames, result
                gc.collect()

                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        logger.info("Done! Output: %s", output_path)
        return output_path

    def _ensure_models_loaded(self, model_dir: Path, filename: str) -> None:
        """Load models once, caching runner and text embeddings for reuse."""
        if self._runner is not None:
            return

        from upscaler._vendor.seedvr2.src.core.generation_utils import (
            load_text_embeddings,
            prepare_runner,
            setup_generation_context,
        )
        from upscaler._vendor.seedvr2.src.utils.constants import get_script_directory
        from upscaler._vendor.seedvr2.src.utils.debug import Debug

        cfg = self.config
        debug = Debug(enabled=logger.isEnabledFor(logging.DEBUG))

        device = f"cuda:{cfg.cuda_device}"
        ctx = setup_generation_context(
            dit_device=device,
            vae_device=device,
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            debug=debug,
        )

        block_swap_cfg = {
            "blocks_to_swap": cfg.block_swap.blocks_to_swap,
            "swap_io_components": cfg.block_swap.offload_io_components,
        }

        runner, cache_context = prepare_runner(
            dit_model=filename,
            vae_model=VAE_FILENAME,
            model_dir=str(model_dir),
            debug=debug,
            ctx=ctx,
            block_swap_config=block_swap_cfg,
        )

        script_dir = get_script_directory()
        text_embeds = load_text_embeddings(
            script_dir, ctx["dit_device"], ctx["compute_dtype"], debug
        )

        self._runner = runner
        self._cache_context = cache_context
        self._base_ctx = ctx
        self._text_embeds = text_embeds
        self._debug = debug

    def _run_inference_cached(
        self,
        frames: object,
        progress_callback: ProgressCallback | None = None,
    ) -> object:
        """Run inference using cached models — creates a fresh ctx per segment."""
        from upscaler._vendor.seedvr2.src.core.generation_phases import (
            decode_all_batches,
            encode_all_batches,
            postprocess_all_batches,
            upscale_all_batches,
        )

        cfg = self.config
        runner = self._runner

        # Fresh context per segment, reusing the base settings
        ctx = dict(self._base_ctx)
        ctx["cache_context"] = self._cache_context
        ctx["text_embeds"] = self._text_embeds

        # Phase 1: Encode
        if progress_callback:
            progress_callback(1, 4, "Encoding")
        ctx = encode_all_batches(
            runner,
            ctx=ctx,
            images=frames,
            debug=self._debug,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            resolution=cfg.resolution,
        )

        # Phase 2: Upscale (cache_model=True to keep DiT for next segment)
        if progress_callback:
            progress_callback(2, 4, "Upscaling")
        ctx = upscale_all_batches(
            runner,
            ctx=ctx,
            debug=self._debug,
            seed=cfg.seed,
            cache_model=True,
        )

        # Phase 3: Decode (cache_model=True to keep VAE for next segment)
        if progress_callback:
            progress_callback(3, 4, "Decoding")
        ctx = decode_all_batches(
            runner,
            ctx=ctx,
            debug=self._debug,
            cache_model=True,
        )

        # Phase 4: Post-process
        if progress_callback:
            progress_callback(4, 4, "Post-processing")
        ctx = postprocess_all_batches(
            ctx=ctx,
            debug=self._debug,
            batch_size=cfg.batch_size,
        )

        import torch

        result = ctx["final_video"]
        if result.is_cuda:
            result = result.cpu()
        if result.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
            result = result.to(torch.float32)

        return result

    def _run_inference(
        self,
        frames: object,  # torch.Tensor, typed loosely to avoid import at module level
        model_dir: Path,
        filename: str,
        progress_callback: ProgressCallback | None = None,
    ) -> object:
        """Configure and run the SeedVR2 generation pipeline.

        This is the GPU-dependent step. All vendored code calls happen here.
        """
        from upscaler._vendor.seedvr2.src.core.generation_phases import (
            decode_all_batches,
            encode_all_batches,
            postprocess_all_batches,
            upscale_all_batches,
        )
        from upscaler._vendor.seedvr2.src.core.generation_utils import (
            prepare_runner,
            setup_generation_context,
        )
        from upscaler._vendor.seedvr2.src.utils.debug import Debug

        cfg = self.config
        debug = Debug(enabled=logger.isEnabledFor(logging.DEBUG))

        # Setup generation context
        device = f"cuda:{cfg.cuda_device}"
        ctx = setup_generation_context(
            dit_device=device,
            vae_device=device,
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            debug=debug,
        )

        # Build block swap config
        block_swap_cfg = {
            "blocks_to_swap": cfg.block_swap.blocks_to_swap,
            "swap_io_components": cfg.block_swap.offload_io_components,
        }

        # Prepare runner (loads models)
        runner, cache_context = prepare_runner(
            dit_model=filename,
            vae_model=VAE_FILENAME,
            model_dir=str(model_dir),
            debug=debug,
            ctx=ctx,
            block_swap_config=block_swap_cfg,
        )
        ctx["cache_context"] = cache_context

        # Load text embeddings
        from upscaler._vendor.seedvr2.src.core.generation_utils import (
            load_text_embeddings,
        )
        from upscaler._vendor.seedvr2.src.utils.constants import get_script_directory

        script_dir = get_script_directory()
        ctx["text_embeds"] = load_text_embeddings(
            script_dir, ctx["dit_device"], ctx["compute_dtype"], debug
        )

        # Phase 1: Encode
        if progress_callback:
            progress_callback(1, 4, "Encoding")
        ctx = encode_all_batches(
            runner,
            ctx=ctx,
            images=frames,
            debug=debug,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            resolution=cfg.resolution,
        )

        # Phase 2: Upscale
        if progress_callback:
            progress_callback(2, 4, "Upscaling")
        ctx = upscale_all_batches(
            runner,
            ctx=ctx,
            debug=debug,
            seed=cfg.seed,
        )

        # Phase 3: Decode
        if progress_callback:
            progress_callback(3, 4, "Decoding")
        ctx = decode_all_batches(
            runner,
            ctx=ctx,
            debug=debug,
        )

        # Phase 4: Post-process
        if progress_callback:
            progress_callback(4, 4, "Post-processing")
        ctx = postprocess_all_batches(
            ctx=ctx,
            debug=debug,
            batch_size=cfg.batch_size,
        )

        import torch

        result = ctx["final_video"]
        if result.is_cuda:
            result = result.cpu()
        if result.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
            result = result.to(torch.float32)

        return result
