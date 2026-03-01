"""Config subcommand — init, show, and validate configuration."""

import typer

app = typer.Typer()


@app.command()
def init() -> None:
    """Create an example config file in the current directory."""
    typer.echo("Config init not yet implemented.")
    raise typer.Exit(code=1)


@app.command()
def show() -> None:
    """Print the active merged configuration."""
    typer.echo("Config show not yet implemented.")
    raise typer.Exit(code=1)


@app.command()
def validate(
    path: str = typer.Argument(..., help="Path to TOML config file."),
) -> None:
    """Validate a TOML configuration file."""
    typer.echo(f"Validating: {path}")
    raise typer.Exit(code=1)
