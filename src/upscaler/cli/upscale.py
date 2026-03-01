"""Upscale subcommand — single video upscaling."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from upscaler.cli.common import print_error, resolve_config, validate_input_path
from upscaler.core.engine import UpscaleEngine
from upscaler.progress.reporter import ProgressReporter

app = typer.Typer()


@app.callback(invoke_without_command=True)
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

    reporter = ProgressReporter()
    engine = UpscaleEngine(cfg)
    result_path = engine.run(progress_callback=reporter.callback)
    reporter.complete(result_path)
