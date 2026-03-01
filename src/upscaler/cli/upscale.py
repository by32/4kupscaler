"""Upscale subcommand — single video upscaling."""

import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def upscale(
    input_video: str = typer.Argument(..., help="Path to input video file."),
) -> None:
    """Upscale a single video to 4K resolution."""
    typer.echo(f"Upscaling: {input_video}")
    raise typer.Exit(code=1)  # Not yet implemented
