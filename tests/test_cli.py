"""Tests for CLI argument parsing and subcommand routing."""

from typer.testing import CliRunner

from upscaler.cli.app import app

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_upscale_help() -> None:
    result = runner.invoke(app, ["upscale", "--help"])
    assert result.exit_code == 0
    assert "video" in result.output.lower()


def test_preview_help() -> None:
    result = runner.invoke(app, ["preview", "--help"])
    assert result.exit_code == 0


def test_batch_help() -> None:
    result = runner.invoke(app, ["batch", "--help"])
    assert result.exit_code == 0


def test_config_help() -> None:
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "show" in result.output
    assert "validate" in result.output
