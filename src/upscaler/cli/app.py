"""Main Typer application with subcommand registration."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from upscaler import __version__
from upscaler.cli import cloud, config_cmd
from upscaler.cli.batch import batch
from upscaler.cli.preview import preview
from upscaler.cli.upscale import upscale

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


app.command(name="upscale", help="Upscale a single video to 4K.")(upscale)
app.command(name="preview", help="Preview upscale on a few frames.")(preview)
app.command(name="batch", help="Batch upscale a directory of videos.")(batch)
app.add_typer(config_cmd.app, name="config", help="Manage configuration.")
app.add_typer(cloud.app, name="cloud", help="Run jobs on vast.ai cloud GPUs.")
