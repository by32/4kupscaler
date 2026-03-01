"""Config subcommand — init, show, and validate configuration."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from upscaler.cli.common import print_error
from upscaler.config.loader import load_toml, merge_config
from upscaler.config.schema import UpscaleConfig

app = typer.Typer()

_EXAMPLE_TOML = Path(__file__).resolve().parents[3] / "configs" / "example.toml"


@app.command()
def init() -> None:
    """Create an example config file in the current directory."""
    dest = Path("upscaler.toml")
    if dest.exists():
        print_error(f"{dest} already exists. Remove it first to re-initialize.")
        raise typer.Exit(code=1)

    if not _EXAMPLE_TOML.exists():
        print_error(f"Example template not found at {_EXAMPLE_TOML}")
        raise typer.Exit(code=1)

    shutil.copy2(_EXAMPLE_TOML, dest)
    console = Console()
    console.print(f"[green]Created[/green] {dest}")


@app.command()
def show(
    config: str | None = typer.Option(None, "--config", help="TOML config path."),
    preset: str | None = typer.Option(None, "--preset", help="Hardware preset."),
) -> None:
    """Print the active merged configuration."""
    console = Console()
    try:
        merged = merge_config(
            preset=preset,
            config_path=Path(config) if config else None,
        )
    except (ValueError, OSError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    table = Table(title="Active Configuration", show_lines=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    for key, value in sorted(merged.items()):
        if isinstance(value, dict):
            for sub_key, sub_val in sorted(value.items()):
                table.add_row(f"{key}.{sub_key}", str(sub_val))
        else:
            table.add_row(key, str(value))

    console.print(table)


@app.command()
def validate(
    path: str = typer.Argument(..., help="Path to TOML config file."),
) -> None:
    """Validate a TOML configuration file."""
    console = Console()
    toml_path = Path(path)

    if not toml_path.exists():
        print_error(f"File not found: {toml_path}")
        raise typer.Exit(code=1)

    try:
        load_toml(toml_path)
    except Exception as exc:
        print_error(f"Failed to parse TOML: {exc}")
        raise typer.Exit(code=1) from None

    # Need a dummy input to validate the full config
    merged = merge_config(config_path=toml_path)
    merged["input"] = Path("dummy.mp4")

    try:
        UpscaleConfig(**merged)
    except ValidationError as exc:
        console.print("[bold red]Validation failed:[/bold red]")
        console.print(str(exc))
        raise typer.Exit(code=1) from None

    # Show the parsed TOML for confirmation
    with open(toml_path) as f:
        toml_text = f.read()
    console.print(Syntax(toml_text, "toml", theme="monokai"))
    console.print("[bold green]Configuration is valid.[/bold green]")
