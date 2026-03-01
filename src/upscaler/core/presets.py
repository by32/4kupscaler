"""Hardware presets with optimized defaults per GPU."""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "rtx3080": {
        "model": "3b-fp8",
        "batch_size": 1,
        "blocks_to_swap": 20,
        "preserve_vram": True,
        "resolution": 1072,
    },
    "rtx4090": {
        "model": "7b-fp8",
        "batch_size": 5,
        "blocks_to_swap": 8,
        "preserve_vram": False,
        "resolution": 1072,
    },
}
