"""Batch subcommand — process a directory of videos."""

import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing input videos."),
) -> None:
    """Batch upscale all videos in a directory."""
    typer.echo(f"Batch processing: {input_dir}")
    raise typer.Exit(code=1)  # Not yet implemented
