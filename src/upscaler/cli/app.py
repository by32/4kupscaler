"""Main Typer application with subcommand registration."""

import typer

from upscaler.cli import batch, config_cmd, preview, upscale

app = typer.Typer(
    name="upscaler",
    help="4K video upscaler powered by SeedVR2.",
    no_args_is_help=True,
)

app.add_typer(upscale.app, name="upscale", help="Upscale a single video to 4K.")
app.add_typer(preview.app, name="preview", help="Preview upscale on a few frames.")
app.add_typer(batch.app, name="batch", help="Batch upscale a directory of videos.")
app.add_typer(config_cmd.app, name="config", help="Manage configuration.")
