"""TOML config loading and merging with CLI args."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from upscaler.config.defaults import get_defaults
from upscaler.core.presets import get_preset

# Maps TOML section keys to flat UpscaleConfig field names.
_TOML_FIELD_MAP: dict[str, str] = {
    "name": "model",
    "dir": "model_dir",
    "format": "output_format",
}


def load_toml(path: Path) -> dict:
    """Load a TOML config file and return it as a flat config dict.

    Maps TOML sections to flat UpscaleConfig fields:
      [model].name       -> model
      [model].dir        -> model_dir
      [output].resolution -> resolution
      [output].format    -> output_format
      [output].seed      -> seed
      [performance].*    -> direct mapping
      [performance.block_swap].* -> block_swap dict
      [processing].*     -> direct mapping
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    result: dict = {}

    for section in ("model", "output", "processing"):
        for key, value in raw.get(section, {}).items():
            field = _TOML_FIELD_MAP.get(key, key)
            result[field] = value

    perf = raw.get("performance", {})
    for key, value in perf.items():
        if key == "block_swap":
            result["block_swap"] = dict(value)
        elif key == "vae_tiling":
            result["vae_tiling"] = dict(value)
        else:
            field = _TOML_FIELD_MAP.get(key, key)
            result[field] = value

    if "gpu_monitor" in raw:
        result["gpu_monitor"] = dict(raw["gpu_monitor"])

    return result


def merge_config(
    preset: str | None = None,
    config_path: Path | None = None,
    cli_overrides: dict | None = None,
) -> dict:
    """Build a merged config dict following precedence rules.

    Merge order (lowest to highest priority):
        defaults -> preset -> TOML file -> CLI overrides

    Returns:
        Dict ready to pass to ``UpscaleConfig(**merged)``.
    """
    merged = get_defaults()

    # Nested dict keys that should be merged, not replaced.
    _NESTED_KEYS = ("block_swap", "vae_tiling", "gpu_monitor")

    def _merge_layer(base: dict, layer: dict) -> None:
        for nk in _NESTED_KEYS:
            if nk in layer:
                nested = dict(base.get(nk, {}))
                nested.update(layer.pop(nk))
                base[nk] = nested
        base.update(layer)

    if preset:
        preset_values = get_preset(preset)
        # Presets may include blocks_to_swap at top level — nest it.
        if "blocks_to_swap" in preset_values:
            bs = dict(merged.get("block_swap", {}))
            bs["blocks_to_swap"] = preset_values.pop("blocks_to_swap")
            merged["block_swap"] = bs
        _merge_layer(merged, preset_values)

    if config_path:
        toml_values = load_toml(config_path)
        _merge_layer(merged, toml_values)

    if cli_overrides:
        # Filter out None values so unset CLI args don't clobber.
        overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        _merge_layer(merged, overrides)

    return merged
