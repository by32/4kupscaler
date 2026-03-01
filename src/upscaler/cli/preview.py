"""Preview subcommand — quick quality check on a few frames."""

import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def preview(
    input_video: str = typer.Argument(..., help="Path to input video file."),
) -> None:
    """Preview upscale quality on a small number of frames."""
    typer.echo(f"Previewing: {input_video}")
    raise typer.Exit(code=1)  # Not yet implemented
