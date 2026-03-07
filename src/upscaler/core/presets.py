"""Hardware presets with optimized defaults per GPU."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict] = {
    "rtx3080": {
        "model": "3b-fp8",
        "batch_size": 1,
        "blocks_to_swap": 20,
        "preserve_vram": True,
        "resolution": 1072,
        "segment_size": 5,
    },
    "rtx3090": {
        "model": "3b-fp8",
        "batch_size": 5,
        "blocks_to_swap": 12,
        "preserve_vram": True,
        "resolution": 1072,
        "segment_size": 13,
    },
    "rtx4090": {
        "model": "7b-fp8",
        "batch_size": 5,
        "blocks_to_swap": 8,
        "preserve_vram": False,
        "resolution": 1072,
        "segment_size": 21,
    },
    "rtx3080-4k": {
        "model": "3b-fp8",
        "batch_size": 1,
        "blocks_to_swap": 20,
        "preserve_vram": True,
        "resolution": 2160,
        "segment_size": 5,
        "vae_tiling": {
            "encode_tiled": True,
            "decode_tiled": True,
            "encode_tile_size": 512,
            "encode_tile_overlap": 160,
            "decode_tile_size": 512,
            "decode_tile_overlap": 160,
        },
    },
    "rtx3090-4k": {
        "model": "3b-fp8",
        "batch_size": 5,
        "blocks_to_swap": 12,
        "preserve_vram": True,
        "resolution": 2160,
        "segment_size": 9,
        "vae_tiling": {
            "encode_tiled": True,
            "decode_tiled": True,
            "encode_tile_size": 512,
            "encode_tile_overlap": 160,
            "decode_tile_size": 512,
            "decode_tile_overlap": 160,
        },
    },
    "rtx4090-4k": {
        "model": "7b-fp8",
        "batch_size": 5,
        "blocks_to_swap": 8,
        "preserve_vram": False,
        "resolution": 2160,
        "segment_size": 13,
        "vae_tiling": {
            "encode_tiled": True,
            "decode_tiled": True,
            "encode_tile_size": 512,
            "encode_tile_overlap": 160,
            "decode_tile_size": 512,
            "decode_tile_overlap": 160,
        },
    },
}

# GPU name substring -> preset name
_GPU_NAME_MAP: dict[str, str] = {
    "3080": "rtx3080",
    "3090": "rtx3090",
    "4090": "rtx4090",
}


def get_preset(name: str) -> dict:
    """Get a hardware preset by name.

    Raises:
        ValueError: If preset name is not recognized.
    """
    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS.keys()))
        msg = f"Unknown preset '{name}'. Valid presets: {valid}"
        raise ValueError(msg)
    return PRESETS[name].copy()


def detect_gpu() -> str | None:
    """Detect the GPU and return the matching preset name, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Detected GPU: %s", gpu_name)
        for substring, preset_name in _GPU_NAME_MAP.items():
            if substring in gpu_name:
                return preset_name
    except ImportError:
        pass
    return None
