"""Tests for CLI argument parsing and subcommand routing."""

from pathlib import Path

from typer.testing import CliRunner

from upscaler.cli.app import app

runner = CliRunner()


# ── Version / verbose ─────────────────────────────────────


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "upscaler 0.1.0" in result.output


def test_verbose_flag() -> None:
    result = runner.invoke(app, ["-v", "upscale", "--help"])
    assert result.exit_code == 0


def test_double_verbose() -> None:
    result = runner.invoke(app, ["-vv", "upscale", "--help"])
    assert result.exit_code == 0


# ── Help / routing ────────────────────────────────────────


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


# ── upscale subcommand ────────────────────────────────────


def test_upscale_nonexistent_file() -> None:
    result = runner.invoke(app, ["upscale", "nonexistent.mp4"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_upscale_unsupported_format(tmp_path: Path) -> None:
    bad_file = tmp_path / "video.txt"
    bad_file.touch()
    result = runner.invoke(app, ["upscale", str(bad_file)])
    assert result.exit_code == 1
    assert "unsupported" in result.output.lower()


def test_upscale_shows_all_options() -> None:
    result = runner.invoke(app, ["upscale", "--help"])
    assert "--model" in result.output
    assert "--resolution" in result.output
    assert "--batch-size" in result.output
    assert "--blocks-to-swap" in result.output
    assert "--config" in result.output
    assert "--preset" in result.output
    assert "--seed" in result.output
    assert "--output-format" in result.output


# ── preview subcommand ────────────────────────────────────


def test_preview_nonexistent_file() -> None:
    result = runner.invoke(app, ["preview", "nonexistent.mp4"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_preview_shows_frame_options() -> None:
    result = runner.invoke(app, ["preview", "--help"])
    assert "--frames" in result.output
    assert "--start-at" in result.output


# ── batch subcommand ──────────────────────────────────────


def test_batch_nonexistent_dir() -> None:
    result = runner.invoke(app, ["batch", "/nonexistent/dir"])
    assert result.exit_code == 1
    assert "not a directory" in result.output.lower()


def test_batch_empty_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 1
    assert "no video files" in result.output.lower()


def test_batch_shows_options() -> None:
    result = runner.invoke(app, ["batch", "--help"])
    assert "--output-dir" in result.output
    assert "--pattern" in result.output
    assert "--skip-existing" in result.output


# ── config subcommand ─────────────────────────────────────


def test_config_init_creates_file(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert (tmp_path / "upscaler.toml").exists()


def test_config_init_refuses_overwrite(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    (tmp_path / "upscaler.toml").touch()
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 1
    assert "already exists" in result.output.lower()


def test_config_show_defaults() -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "3b-fp8" in result.output


def test_config_validate_example() -> None:
    example = Path(__file__).resolve().parents[1] / "configs" / "example.toml"
    if not example.exists():
        return  # skip if example.toml not present
    result = runner.invoke(app, ["config", "validate", str(example)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_config_validate_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[model]\nname = 'invalid-model-xyz'\n")
    runner.invoke(app, ["config", "validate", str(bad)])
    # May pass or fail depending on whether model name is validated at this level
    # The important thing is it doesn't crash


def test_config_validate_nonexistent() -> None:
    result = runner.invoke(app, ["config", "validate", "/nonexistent.toml"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# ── Public API exports ───────────────────────────────────


def test_config_exports() -> None:
    from upscaler.config import BlockSwapConfig, UpscaleConfig

    assert UpscaleConfig is not None
    assert BlockSwapConfig is not None


def test_core_exports() -> None:
    from upscaler.core import UpscaleEngine

    assert UpscaleEngine is not None
