# Vendored SeedVR2 Core Modules

Selectively vendored from [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler).

## Upstream Source
- **Repo:** `numz/ComfyUI-SeedVR2_VideoUpscaler`
- **Commit:** `4490bd1` (Merge pull request #441 from AInVFX/main)
- **License:** Apache 2.0

## What's Included
- `src/core/` — Model loading, generation pipeline, inference engine
- `src/optimization/` — BlockSwap, FP8 compatibility, memory management
- `src/utils/` — Model downloads, registry, color correction, constants
- `src/common/` — Config, seed, diffusion sampling, logging
- `src/data/` — Image transforms
- `src/models/` — DiT 3B/7B architectures, VideoVAE
- `configs_3b/`, `configs_7b/` — Model configuration YAMLs
- `inference_cli.py` — Reference CLI (not used directly)

## What's Excluded
- `src/interfaces/` — ComfyUI node definitions (not needed for standalone use)

## ComfyUI Dependencies
Files that reference ComfyUI imports use try/except fallbacks and work
standalone without modification. No patching required.
