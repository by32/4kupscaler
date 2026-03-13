"""Default configuration values — base layer for config merging."""

from __future__ import annotations


def get_defaults() -> dict:
    """Return base defaults as a flat dict matching UpscaleConfig fields."""
    return {
        "model": "3b-fp8",
        "model_dir": "~/.cache/upscaler/models",
        "resolution": 1072,
        "output_format": "video",
        "batch_size": 1,
        "seed": 100,
        "preserve_vram": True,
        "cuda_device": "0",
        "skip_first_frames": 0,
        "max_frames": None,
        "segment_size": None,
        "color_correction": False,
        "block_swap": {
            "blocks_to_swap": 20,
            "use_non_blocking": True,
            "offload_io_components": False,
            "cache_model": False,
        },
        "gpu_monitor": {
            "enabled": False,
            "poll_interval": 2.0,
            "warn_temp_c": 85,
            "critical_temp_c": 90,
            "cooldown_target_c": 80,
            "pause_on_overheat": True,
            "log_metrics": True,
        },
    }
