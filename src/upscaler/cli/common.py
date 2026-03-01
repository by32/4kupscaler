"""Shared CLI options, validators, and callbacks."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from upscaler.config.loader import merge_config
from upscaler.config.schema import UpscaleConfig

SUPPORTED_FORMATS = {".mp4", ".mpg", ".mpeg", ".avi", ".mkv", ".mov"}

_console = Console(stderr=True)


def validate_batch_size(value: int) -> int:
    """Validate that batch_size follows the 4n+1 rule."""
    if (value - 1) % 4 != 0:
        valid = [4 * n + 1 for n in range(6)]
        msg = f"batch_size must follow 4n+1 rule (e.g., {valid}), got {value}"
        raise ValueError(msg)
    return value


def validate_input_path(path: Path) -> Path:
    """Check that the input file exists and has a supported extension."""
    if not path.exists():
        msg = f"Input file not found: {path}"
        raise FileNotFoundError(msg)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        msg = f"Unsupported format '{path.suffix}'. Supported: {supported}"
        raise ValueError(msg)
    return path


def resolve_config(
    input_path: Path,
    output: Path | None = None,
    preset: str | None = None,
    config_path: Path | None = None,
    **cli_overrides: object,
) -> UpscaleConfig:
    """Merge all config sources and return a validated UpscaleConfig."""
    merged = merge_config(
        preset=preset,
        config_path=config_path,
        cli_overrides=cli_overrides,
    )
    merged["input"] = input_path
    if output is not None:
        merged["output"] = output
    return UpscaleConfig(**merged)


def print_error(message: str) -> None:
    """Print a formatted error message to stderr."""
    _console.print(f"[bold red]Error:[/bold red] {message}")
