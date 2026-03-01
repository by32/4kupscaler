# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI video upscaler powered by SeedVR2 (ByteDance). Upscales 340p/480p/720p `.MPG`/`.MP4` video to 4K. Developed on Mac, runs inference on Alienware (RTX 3080, WSL2).

**Core code vendored from** [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache 2.0) — provides BlockSwap, FP8, and modular inference that the raw ByteDance repo lacks.

## Commands

```bash
uv sync --group dev          # Install dependencies
uv run ruff check            # Lint
uv run ruff format --check   # Check formatting
uv run pytest --cov -m "not gpu" -v  # Run tests (skip GPU-only tests)
uv run upscaler --help       # CLI help
```

## Architecture

```
src/upscaler/
├── cli/          # Typer CLI: upscale, preview, batch, config subcommands
├── core/         # Engine (orchestrator), models (HF registry), presets (GPU profiles), video_io
├── config/       # Pydantic schema, TOML loader, defaults
├── progress/     # Rich progress bars with optional VRAM display
└── _vendor/seedvr2/  # Vendored ComfyUI node core (DO NOT modify without updating README)
```

**Config merging order** (lowest → highest priority): defaults → hardware preset → TOML file → CLI args

## Hardware Constraints

- **GPU:** NVIDIA RTX 3080 (10–12 GB VRAM)
- **Default model:** `3b-fp8` (7B available via `--model 7b-fp8` but much slower)
- **BlockSwap:** `blocks_to_swap=20` offloads transformer blocks to CPU
- **Batch size:** Must follow `4n+1` rule (1, 5, 9, 13...). Default is 1 for VRAM safety
- **VAE tiling:** 512×512 tiles with 160px overlap, mandatory for 4K output

## Key Design Rules

- All inference MUST use BlockSwap + VAE tiling on RTX 3080 — never bypass
- Frame counts must satisfy `frames % 4 == 1` (enforced by `enforce_4n1()` in `core/video_io.py`)
- Coverage excludes `_vendor/` — only measure our wrapper code
- GPU-dependent tests use `@pytest.mark.gpu` and auto-skip on Mac
