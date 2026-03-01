"""Model registry and HuggingFace download management.

Wraps the vendored SeedVR2 model registry and download utilities,
providing a simplified interface with friendly model names.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Friendly name -> actual filename mapping
MODEL_REGISTRY: dict[str, str] = {
    "3b-fp8": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "3b-fp16": "seedvr2_ema_3b_fp16.safetensors",
    "7b-fp8": "seedvr2_ema_7b_fp16.safetensors",
    "7b-fp16": "seedvr2_ema_7b_fp16.safetensors",
    "7b-sharp-fp8": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "7b-sharp-fp16": "seedvr2_ema_7b_sharp_fp16.safetensors",
}

VAE_FILENAME = "ema_vae_fp16.safetensors"


def resolve_model_filename(name: str) -> str:
    """Resolve a friendly model name to its actual filename.

    Args:
        name: Friendly name like '3b-fp8' or an actual filename.

    Returns:
        The safetensors filename.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    # Allow passing raw filenames directly
    if name.endswith((".safetensors", ".gguf")):
        return name
    valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
    msg = f"Unknown model '{name}'. Valid names: {valid}"
    raise ValueError(msg)


def get_model_variant(name: str) -> str:
    """Return '3b' or '7b' based on the model name.

    Used to select the correct config YAML (configs_3b/ or configs_7b/).
    """
    filename = resolve_model_filename(name)
    return "7b" if "7b" in filename else "3b"


def get_max_blocks(name: str) -> int:
    """Return the maximum blocks_to_swap for a given model."""
    variant = get_model_variant(name)
    return 36 if variant == "7b" else 32


def ensure_models(
    model_name: str,
    model_dir: str | Path | None = None,
) -> Path:
    """Ensure model and VAE weights exist, downloading if needed.

    Args:
        model_name: Friendly name (e.g., '3b-fp8') or raw filename.
        model_dir: Directory for model storage. Defaults to vendored default.

    Returns:
        Path to the model directory.

    Raises:
        RuntimeError: If download fails.
    """
    from upscaler._vendor.seedvr2.src.utils.downloads import download_weight

    filename = resolve_model_filename(model_name)
    dir_str = str(model_dir) if model_dir else None

    logger.info("Ensuring models: %s + %s", filename, VAE_FILENAME)
    success = download_weight(
        dit_model=filename,
        vae_model=VAE_FILENAME,
        model_dir=dir_str,
    )
    if not success:
        msg = (
            f"Failed to download model '{filename}'. "
            f"Check your network connection or download manually."
        )
        raise RuntimeError(msg)

    # Return the resolved model directory
    if model_dir:
        return Path(model_dir).expanduser()

    from upscaler._vendor.seedvr2.src.utils.constants import get_base_cache_dir

    return Path(get_base_cache_dir())
