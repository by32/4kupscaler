"""Model registry and HuggingFace download management."""

from __future__ import annotations

MODEL_REGISTRY: dict[str, str] = {
    "3b-fp8": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "3b-fp16": "seedvr2_ema_3b_fp16.safetensors",
    "7b-fp8": "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
    "7b-fp16": "seedvr2_ema_7b_fp16.safetensors",
    "7b-sharp-fp8": "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors",
    "7b-sharp-fp16": "seedvr2_ema_7b_sharp_fp16.safetensors",
}

VAE_FILENAME = "ema_vae_fp16.safetensors"
