"""Main Typer application with subcommand registration."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from upscaler import __version__
from upscaler.cli import batch, config_cmd, preview, upscale

app = typer.Typer(
    name="upscaler",
    help="4K video upscaler powered by SeedVR2.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"upscaler {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            help="Increase verbosity (-v for info, -vv for debug).",
            count=True,
        ),
    ] = 0,
) -> None:
    """4K video upscaler powered by SeedVR2."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


app.add_typer(upscale.app, name="upscale", help="Upscale a single video to 4K.")
app.add_typer(preview.app, name="preview", help="Preview upscale on a few frames.")
app.add_typer(batch.app, name="batch", help="Batch upscale a directory of videos.")
app.add_typer(config_cmd.app, name="config", help="Manage configuration.")
