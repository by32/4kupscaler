"""Preview subcommand — quick quality check on a few frames."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from upscaler.cli.common import print_error, resolve_config, validate_input_path
from upscaler.core.engine import UpscaleEngine
from upscaler.progress.reporter import ProgressReporter


def preview(
    input_video: Annotated[str, typer.Argument(help="Path to input video file.")],
    n_frames: Annotated[
        int,
        typer.Option("-n", "--frames", help="Number of frames to preview."),
    ] = 5,
    start_at: Annotated[
        int,
        typer.Option("--start-at", help="Frame to start preview from."),
    ] = 0,
    model: Annotated[
        str | None,
        typer.Option("-m", "--model", help="Model variant (e.g. 3b-fp8, 7b-fp8)."),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to TOML config file."),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option("--preset", help="Hardware preset (rtx3080, rtx3090, rtx4090)."),
    ] = None,
) -> None:
    """Preview upscale quality on a small number of frames."""
    try:
        input_path = validate_input_path(Path(input_video))
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    cli_overrides: dict = {}
    if model is not None:
        cli_overrides["model"] = model

    try:
        cfg = resolve_config(
            input_path=input_path,
            preset=preset,
            config_path=Path(config) if config else None,
            **cli_overrides,
        )
    except (ValueError, ValidationError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    reporter = ProgressReporter()
    engine = UpscaleEngine(cfg)
    result_path = engine.preview(
        n_frames=n_frames,
        start_at=start_at,
        progress_callback=reporter.callback,
    )
    reporter.complete(result_path)
