"""Upscale subcommand — single video upscaling."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from upscaler.cli.common import print_error, resolve_config, validate_input_path
from upscaler.core.engine import UpscaleEngine
from upscaler.progress.reporter import ProgressReporter


def upscale(
    input_video: Annotated[str, typer.Argument(help="Path to input video file.")],
    output: Annotated[
        str | None, typer.Option("-o", "--output", help="Output file path.")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("-m", "--model", help="Model variant (e.g. 3b-fp8, 7b-fp8)."),
    ] = None,
    resolution: Annotated[
        int | None,
        typer.Option("-r", "--resolution", help="Target short-side resolution."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Frames per batch (must follow 4n+1 rule)."),
    ] = None,
    blocks_to_swap: Annotated[
        int | None,
        typer.Option("--blocks-to-swap", help="Transformer blocks offloaded to CPU."),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to TOML config file."),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option("--preset", help="Hardware preset (rtx3080, rtx3090, rtx4090)."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    skip_frames: Annotated[
        int | None,
        typer.Option("--skip-frames", help="Frames to skip from start."),
    ] = None,
    max_frames: Annotated[
        int | None,
        typer.Option("--max-frames", help="Maximum frames to process."),
    ] = None,
    output_format: Annotated[
        str | None,
        typer.Option("--output-format", help="Output format: video or png."),
    ] = None,
    segment_size: Annotated[
        int | None,
        typer.Option(
            "--segment-size",
            help="Frames per segment for streaming processing (must follow 4n+1 rule).",
        ),
    ] = None,
    vae_tiling: Annotated[
        bool | None,
        typer.Option(
            "--vae-tiling",
            help="Enable VAE tiling (auto-enabled for resolutions > 1080).",
        ),
    ] = None,
    gpu_monitor: Annotated[
        bool | None,
        typer.Option(
            "--gpu-monitor/--no-gpu-monitor",
            help="Enable GPU temperature monitoring during processing.",
        ),
    ] = None,
    checkpoint: Annotated[
        bool,
        typer.Option(
            "--checkpoint/--no-checkpoint",
            help="Enable checkpointing for resumable segmented runs.",
        ),
    ] = False,
    segment_range: Annotated[
        list[int] | None,
        typer.Option(
            "--segment-range",
            help="Only process segments [START, END). Used by fleet jobs.",
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="Resume a previously interrupted job from its work directory.",
        ),
    ] = None,
) -> None:
    """Upscale a single video to 4K resolution."""
    try:
        input_path = validate_input_path(Path(input_video))
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    cli_overrides: dict = {}
    if model is not None:
        cli_overrides["model"] = model
    if resolution is not None:
        cli_overrides["resolution"] = resolution
    if batch_size is not None:
        cli_overrides["batch_size"] = batch_size
    if blocks_to_swap is not None:
        cli_overrides["block_swap"] = {"blocks_to_swap": blocks_to_swap}
    if seed is not None:
        cli_overrides["seed"] = seed
    if skip_frames is not None:
        cli_overrides["skip_first_frames"] = skip_frames
    if max_frames is not None:
        cli_overrides["max_frames"] = max_frames
    if output_format is not None:
        cli_overrides["output_format"] = output_format
    if segment_size is not None:
        cli_overrides["segment_size"] = segment_size
    if vae_tiling is not None and vae_tiling:
        cli_overrides["vae_tiling"] = {"encode_tiled": True, "decode_tiled": True}
    if gpu_monitor is not None:
        cli_overrides["gpu_monitor"] = {"enabled": gpu_monitor}
    if segment_range is not None:
        if len(segment_range) != 2:
            print_error("--segment-range requires exactly 2 values: START END")
            raise typer.Exit(code=1)
        cli_overrides["segment_range"] = tuple(segment_range)

    try:
        cfg = resolve_config(
            input_path=input_path,
            output=Path(output) if output else None,
            preset=preset,
            config_path=Path(config) if config else None,
            **cli_overrides,
        )
    except (ValueError, ValidationError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    # GPU monitoring setup
    monitor = None
    if cfg.gpu_monitor.enabled:
        try:
            from upscaler.diagnostics.gpu_monitor import GpuMonitor

            monitor = GpuMonitor(
                device_index=int(cfg.cuda_device),
                poll_interval=cfg.gpu_monitor.poll_interval,
            )
            monitor.start()
        except Exception:
            pass  # pynvml unavailable — monitor stays None

    # Checkpoint / resume setup
    manifest = None
    if resume is not None:
        from upscaler.core.checkpoint import JobManifest

        manifest = JobManifest.load(Path(resume) / "manifest.json")
    elif checkpoint and cfg.segment_size is not None:
        from upscaler.core.checkpoint import JobManifest
        from upscaler.core.video_io import compute_segments, get_video_meta

        meta = get_video_meta(cfg.input)
        effective = meta.frame_count - cfg.skip_first_frames
        segments = compute_segments(effective, cfg.segment_size)
        work_dir = (
            cfg.output.parent / f".{cfg.output.stem}_segments"
            if cfg.output
            else Path(f".{cfg.input.stem}_segments")
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest = JobManifest.create(
            manifest_path=work_dir / "manifest.json",
            input_file=cfg.input,
            config=cfg,
            segments=segments,
        )

    try:
        reporter = ProgressReporter(gpu_monitor=monitor)
        engine = UpscaleEngine(cfg, gpu_monitor=monitor)
        result_path = engine.run(
            progress_callback=reporter.callback,
            reporter=reporter,
            manifest=manifest,
        )
        reporter.complete(result_path)
    finally:
        if monitor is not None:
            if cfg.gpu_monitor.log_metrics and cfg.output:
                csv_path = cfg.output.with_suffix(".gpu_metrics.csv")
                monitor.export_csv(csv_path)
            monitor.stop()
