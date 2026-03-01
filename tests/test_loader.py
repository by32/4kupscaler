"""Tests for TOML loading and config merging."""

from pathlib import Path

import pytest

from upscaler.config.defaults import get_defaults
from upscaler.config.loader import load_toml, merge_config

# ── get_defaults ──────────────────────────────────────────


class TestGetDefaults:
    def test_returns_dict(self) -> None:
        d = get_defaults()
        assert isinstance(d, dict)

    def test_default_model(self) -> None:
        assert get_defaults()["model"] == "3b-fp8"

    def test_default_block_swap_nested(self) -> None:
        d = get_defaults()
        assert isinstance(d["block_swap"], dict)
        assert d["block_swap"]["blocks_to_swap"] == 20

    def test_returns_fresh_copy(self) -> None:
        a = get_defaults()
        b = get_defaults()
        a["model"] = "changed"
        assert b["model"] == "3b-fp8"


# ── load_toml ─────────────────────────────────────────────


class TestLoadToml:
    def test_loads_example_toml(self) -> None:
        example = Path(__file__).resolve().parents[1] / "configs" / "example.toml"
        result = load_toml(example)
        assert result["model"] == "3b-fp8"
        assert result["resolution"] == 1072
        assert result["batch_size"] == 1

    def test_maps_model_name_field(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[model]\nname = "7b-fp8"\n')
        result = load_toml(toml_file)
        assert result["model"] == "7b-fp8"

    def test_maps_model_dir_field(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[model]\ndir = "/custom/models"\n')
        result = load_toml(toml_file)
        assert result["model_dir"] == "/custom/models"

    def test_maps_output_format(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[output]\nformat = "png"\n')
        result = load_toml(toml_file)
        assert result["output_format"] == "png"

    def test_maps_block_swap(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            "[performance.block_swap]\nblocks_to_swap = 10\nuse_non_blocking = false\n"
        )
        result = load_toml(toml_file)
        assert result["block_swap"]["blocks_to_swap"] == 10
        assert result["block_swap"]["use_non_blocking"] is False

    def test_maps_processing_section(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[processing]\nskip_first_frames = 5\nmax_frames = 50\n")
        result = load_toml(toml_file)
        assert result["skip_first_frames"] == 5
        assert result["max_frames"] == 50

    def test_invalid_toml_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("this is not valid toml [[[")
        with pytest.raises(ValueError):
            load_toml(toml_file)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_toml(Path("/nonexistent/config.toml"))


# ── merge_config ──────────────────────────────────────────


class TestMergeConfig:
    def test_defaults_only(self) -> None:
        result = merge_config()
        assert result["model"] == "3b-fp8"
        assert result["batch_size"] == 1

    def test_preset_overrides_defaults(self) -> None:
        result = merge_config(preset="rtx4090")
        assert result["model"] == "7b-fp8"
        assert result["batch_size"] == 5
        assert result["block_swap"]["blocks_to_swap"] == 8

    def test_toml_overrides_preset(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[model]\nname = "3b-fp16"\n')
        result = merge_config(preset="rtx4090", config_path=toml_file)
        # TOML overrides preset model
        assert result["model"] == "3b-fp16"
        # Preset batch_size is preserved (TOML didn't set it)
        assert result["batch_size"] == 5

    def test_cli_overrides_all(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[output]\nresolution = 720\n")
        result = merge_config(
            preset="rtx3080",
            config_path=toml_file,
            cli_overrides={"resolution": 540},
        )
        assert result["resolution"] == 540

    def test_cli_none_values_ignored(self) -> None:
        result = merge_config(cli_overrides={"model": None, "seed": 42})
        assert result["model"] == "3b-fp8"  # default, not None
        assert result["seed"] == 42

    def test_block_swap_merge_across_layers(self, tmp_path: Path) -> None:
        """block_swap dicts should merge, not replace."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[performance.block_swap]\nblocks_to_swap = 15\n")
        result = merge_config(config_path=toml_file)
        # TOML overrides blocks_to_swap
        assert result["block_swap"]["blocks_to_swap"] == 15
        # Default use_non_blocking is preserved
        assert result["block_swap"]["use_non_blocking"] is True

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            merge_config(preset="unknown_gpu")
