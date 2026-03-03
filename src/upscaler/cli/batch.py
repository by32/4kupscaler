"""Batch subcommand — process a directory of videos."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from upscaler.cli.common import print_error, resolve_config
from upscaler.core.engine import UpscaleEngine
from upscaler.core.video_io import SUPPORTED_EXTENSIONS
from upscaler.progress.reporter import ProgressReporter


def _find_videos(input_dir: Path, pattern: str) -> list[Path]:
    """Glob for video files matching the pattern with supported extensions."""
    candidates = sorted(input_dir.glob(pattern))
    return [f for f in candidates if f.suffix.lower() in SUPPORTED_EXTENSIONS]


def batch(
    input_dir: Annotated[
        str, typer.Argument(help="Directory containing input videos.")
    ],
    output_dir: Annotated[
        str | None,
        typer.Option("-o", "--output-dir", help="Output directory."),
    ] = None,
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="Glob pattern for video files."),
    ] = "*",
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing", help="Skip already-upscaled files."),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option("-m", "--model", help="Model variant."),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to TOML config file."),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option("--preset", help="Hardware preset."),
    ] = None,
    segment_size: Annotated[
        int | None,
        typer.Option(
            "--segment-size",
            help="Frames per segment for streaming processing (must follow 4n+1 rule).",
        ),
    ] = None,
) -> None:
    """Batch upscale all videos in a directory."""
    console = Console()
    dir_path = Path(input_dir)
    if not dir_path.is_dir():
        print_error(f"Not a directory: {dir_path}")
        raise typer.Exit(code=1)

    videos = _find_videos(dir_path, pattern)
    if not videos:
        print_error(f"No video files found in {dir_path} matching '{pattern}'")
        raise typer.Exit(code=1)

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    cli_overrides: dict = {}
    if model is not None:
        cli_overrides["model"] = model
    if segment_size is not None:
        cli_overrides["segment_size"] = segment_size

    console.print(f"Found [bold]{len(videos)}[/bold] video(s) to process.")

    for i, video_path in enumerate(videos, 1):
        output_path = None
        if out_dir:
            output_path = out_dir / f"{video_path.stem}_upscaled.mp4"

        if skip_existing and output_path and output_path.exists():
            console.print(f"  [{i}/{len(videos)}] Skipping (exists): {video_path.name}")
            continue

        console.print(f"  [{i}/{len(videos)}] Processing: {video_path.name}")

        try:
            cfg = resolve_config(
                input_path=video_path,
                output=output_path,
                preset=preset,
                config_path=Path(config) if config else None,
                **cli_overrides,
            )
        except (ValueError, ValidationError) as exc:
            print_error(f"{video_path.name}: {exc}")
            continue

        reporter = ProgressReporter(console=console)
        engine = UpscaleEngine(cfg)
        result = engine.run(progress_callback=reporter.callback)
        reporter.complete(result)

    console.print("[bold green]Batch processing complete.[/bold green]")
