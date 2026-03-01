# 4KUpscaler

[![CI](https://github.com/by32/4kupscaler/actions/workflows/ci.yml/badge.svg)](https://github.com/by32/4kupscaler/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Video super-resolution CLI powered by [SeedVR2](https://github.com/ByteDance/SeedVR2) (ByteDance). Upscales 340p/480p/720p `.mp4`/`.mpg` video to 4K.

## Features

- Single video upscaling to 4K resolution
- Quick preview mode (process a few frames before committing to full upscale)
- Batch processing of entire directories
- Hardware presets for RTX 3080/3090/4090
- TOML configuration with layered merging (defaults, preset, file, CLI)
- Rich progress display with phase tracking
- BlockSwap memory optimization for constrained GPUs
- FP8 quantized models for faster inference on consumer hardware

## Requirements

- **Python** 3.10 or higher
- **NVIDIA GPU** with CUDA support (for inference)
- Tested GPUs: RTX 3080 (10 GB), RTX 3090 (24 GB), RTX 4090 (24 GB)
- **[uv](https://docs.astral.sh/uv/)** package manager

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/by32/4kupscaler.git
cd 4kupscaler

# Development (Mac/CPU — no inference, just CLI and tests)
uv sync --dev

# GPU inference (requires NVIDIA CUDA)
uv sync --dev --extra gpu
```

### 2. Initialize configuration

```bash
upscaler config init
```

This creates `upscaler.toml` in your current directory. Edit it to match your hardware, or use a preset:

```bash
# View the active config (defaults + any TOML overrides)
upscaler config show

# Validate your config file
upscaler config validate upscaler.toml
```

### 3. Preview before committing

Test upscale quality on a few frames first:

```bash
upscaler preview input.mp4 --frames 5
```

This outputs PNG frames so you can inspect quality before processing the full video.

### 4. Upscale a video

```bash
upscaler upscale input.mp4
```

Output defaults to `input_upscaled.mp4` in the same directory. Customize with:

```bash
upscaler upscale input.mp4 -o output_4k.mp4 --preset rtx3080
```

### 5. Batch process a directory

```bash
upscaler batch ./videos/ -o ./upscaled/ --skip-existing
```

## CLI Reference

### Global flags

```
--version          Show version and exit
-v, --verbose      Increase verbosity (-v for info, -vv for debug)
--help             Show help and exit
```

### `upscaler upscale`

Upscale a single video to 4K.

```
upscaler upscale INPUT [OPTIONS]

Options:
  -o, --output PATH          Output file path
  -m, --model NAME           Model variant (default: 3b-fp8)
  -r, --resolution INT       Target short-side resolution (default: 1072)
  --batch-size INT            Frames per batch, must follow 4n+1 rule (default: 1)
  --blocks-to-swap INT        Transformer blocks offloaded to CPU (default: 20)
  --config PATH               Path to TOML config file
  --preset NAME               Hardware preset (rtx3080, rtx3090, rtx4090)
  --seed INT                  Random seed (default: 100)
  --skip-frames INT           Frames to skip from start
  --max-frames INT            Maximum frames to process
  --output-format FMT         Output format: video or png (default: video)
```

### `upscaler preview`

Preview upscale quality on a small number of frames. Always outputs PNGs.

```
upscaler preview INPUT [OPTIONS]

Options:
  -n, --frames INT     Number of frames to preview (default: 5)
  --start-at INT        Frame to start preview from (default: 0)
  -m, --model NAME      Model variant
  --config PATH         Path to TOML config file
  --preset NAME         Hardware preset
```

### `upscaler batch`

Batch upscale all videos in a directory.

```
upscaler batch INPUT_DIR [OPTIONS]

Options:
  -o, --output-dir PATH    Output directory
  --pattern GLOB           Glob pattern for video files (default: *)
  --skip-existing          Skip already-upscaled files
  -m, --model NAME         Model variant
  --config PATH            Path to TOML config file
  --preset NAME            Hardware preset
```

### `upscaler config`

Manage configuration.

```
upscaler config init              Create upscaler.toml in current directory
upscaler config show [OPTIONS]    Print active merged configuration
upscaler config validate PATH     Validate a TOML config file
```

## Configuration

Configuration follows a layered merging order (lowest to highest priority):

```
Defaults → Hardware Preset → TOML File → CLI Arguments
```

Any layer can override values from the layer below it. CLI arguments always win.

### TOML format

```toml
[model]
name = "3b-fp8"                    # 3b-fp8, 3b-fp16, 7b-fp8, 7b-fp16
dir = "~/.cache/upscaler/models"   # Model weight storage

[output]
resolution = 1072                  # Short-side target resolution (pixels)
format = "video"                   # "video" or "png"
seed = 100

[performance]
batch_size = 1                     # Frames per batch (must follow 4n+1 rule)
preserve_vram = true               # Unload unused models during processing
cuda_device = "0"

[performance.block_swap]
blocks_to_swap = 20                # Transformer blocks offloaded to CPU
use_non_blocking = true
offload_io_components = false
cache_model = false

[processing]
color_correction = false
skip_first_frames = 0
# max_frames = 100                 # Uncomment to limit frames
```

### Available models

| Name | Size | Description |
|------|------|-------------|
| `3b-fp8` | ~3 GB | Default. Best balance of speed and quality for 10 GB GPUs |
| `3b-fp16` | ~6 GB | Higher precision 3B model |
| `7b-fp8` | ~7 GB | Larger model, better quality, much slower |
| `7b-fp16` | ~14 GB | Full precision 7B (requires 24 GB+ VRAM) |
| `7b-sharp-fp8` | ~7 GB | Sharpness-tuned 7B variant |
| `7b-sharp-fp16` | ~14 GB | Full precision sharpness-tuned |

### Batch size rule

Batch size must follow the `4n+1` pattern: **1, 5, 9, 13, 17, 21, ...**

Default is `1` (safest for VRAM). Increase only if your GPU has headroom.

## Hardware Presets

Presets provide tuned defaults for specific GPUs. Use with `--preset`:

| Preset | Model | Batch Size | BlockSwap | Preserve VRAM |
|--------|-------|------------|-----------|---------------|
| `rtx3080` | 3b-fp8 | 1 | 20 | Yes |
| `rtx3090` | 3b-fp8 | 5 | 12 | Yes |
| `rtx4090` | 7b-fp8 | 5 | 8 | No |

BlockSwap offloads transformer blocks to CPU RAM, trading speed for VRAM savings. Higher values = less VRAM, slower inference.

## Docker

### Prerequisites

GPU passthrough requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
# Install NVIDIA Container Toolkit (Ubuntu/WSL2)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access: `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### Build

```bash
docker build -t 4kupscaler .
```

### Run

```bash
# Mount videos and model cache
docker run --gpus all \
  -v ./videos:/data \
  -v ~/.cache/upscaler/models:/root/.cache/upscaler/models \
  4kupscaler upscale /data/input.mp4 -o /data/output.mp4

# Batch process
docker run --gpus all \
  -v ./videos:/data \
  -v ~/.cache/upscaler/models:/root/.cache/upscaler/models \
  4kupscaler batch /data/input/ -o /data/output/
```

Model weights (~3 GB for `3b-fp8`) are downloaded on first run and cached in the mounted volume so subsequent runs start immediately.

## Development

```bash
# Install dev dependencies
uv sync --dev

# Lint and format
uv run ruff check
uv run ruff format --check

# Run tests (skips GPU-dependent tests)
uv run pytest --cov -m "not gpu" -v

# Run a specific test
uv run pytest tests/test_loader.py -v
```

## Architecture

```
src/upscaler/
├── cli/              # Typer CLI: upscale, preview, batch, config subcommands
├── config/           # Pydantic schema, TOML loader, defaults
├── core/             # Engine (orchestrator), models (HF registry), presets, video_io
├── progress/         # Rich progress bars
└── _vendor/seedvr2/  # Vendored ComfyUI node core (Apache 2.0)
```

The `UpscaleEngine` orchestrates the pipeline:

1. Download model weights (if needed) via HuggingFace Hub
2. Read input video into `[T, H, W, C]` float32 tensors
3. Pad frames to satisfy the `4n+1` constraint
4. Run SeedVR2 inference (encode → upscale → decode → postprocess)
5. Write output video or PNG frames

Core code is vendored from [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache 2.0), providing BlockSwap, FP8, and modular inference not available in the raw ByteDance repo.

## License

MIT
