# Vendored SeedVR2 Core Modules

Selectively vendored from [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache 2.0).

## Upstream Source
- **Repo:** `numz/ComfyUI-SeedVR2_VideoUpscaler`
- **Commit:** (to be filled when vendoring)
- **License:** Apache 2.0

## Vendored Modules
- `model_manager.py` — Model loading and configuration
- `generation.py` — Inference generation loop
- `infer.py` — Core inference logic
- `blockswap.py` — BlockSwap memory optimization
- `downloads.py` — HuggingFace model downloading

## Modifications
- Stripped ComfyUI-specific imports and node definitions
- Patched internal imports to work within this package
