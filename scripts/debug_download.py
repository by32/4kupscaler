"""Debug script to test model download behavior."""

import os

from upscaler._vendor.seedvr2.src.utils.downloads import (
    MODEL_REGISTRY,
    download_weight,
)

cache = "/root/.cache/upscaler/models"
dit = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
vae = "ema_vae_fp16.safetensors"

print(f"Registry has {len(MODEL_REGISTRY)} entries")
print(f"DiT in registry: {dit in MODEL_REGISTRY}")
print(f"VAE in registry: {vae in MODEL_REGISTRY}")

dit_info = MODEL_REGISTRY.get(dit)
vae_info = MODEL_REGISTRY.get(vae)
print(f"DiT info: {dit_info}")
print(f"VAE info: {vae_info}")

contents = os.listdir(cache) if os.path.exists(cache) else "EMPTY"
print(f"Cache contents before: {contents}")

result = download_weight(dit, vae, cache)
print(f"Download result: {result}")
contents = os.listdir(cache) if os.path.exists(cache) else "EMPTY"
print(f"Cache contents after: {contents}")
