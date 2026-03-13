# 4KUpscaler

[![CI](https://github.com/by32/4kupscaler/actions/workflows/ci.yml/badge.svg)](https://github.com/by32/4kupscaler/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Video super-resolution CLI powered by [SeedVR2](https://github.com/ByteDance/SeedVR2) (ByteDance). Upscales 340p/480p/720p `.mp4`/`.mpg` video to 4K.

## Features

- Single video upscaling to 4K resolution
- Quick preview mode (process a few frames before committing to full upscale)
- Batch processing of entire directories
- Hardware presets for RTX 3080/3090/4090 (local and cloud variants)
- TOML configuration with layered merging (defaults, preset, file, CLI)
- Rich progress display with phase tracking
- BlockSwap memory optimization for constrained GPUs
- FP8 quantized models for faster inference on consumer hardware
- Segmented streaming processing for low-memory, long-video workflows
- VAE tiling for 4K output on 10 GB GPUs (auto-enabled above 1080p)
- Checkpointing and resume for interrupted jobs
- GPU diagnostics: temperature monitoring, thermal throttling, CSV metric export
- Cloud execution on vast.ai with preemption-safe checkpointing

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
  --preset NAME               Hardware preset (see Hardware Presets section)
  --seed INT                  Random seed (default: 100)
  --skip-frames INT           Frames to skip from start
  --max-frames INT            Maximum frames to process
  --output-format FMT         Output format: video or png (default: video)
  --segment-size INT          Frames per segment for streaming processing (4n+1 rule)
  --vae-tiling                Enable VAE tiling (auto-enabled for resolution > 1080)
  --gpu-monitor/--no-gpu-monitor  Enable GPU temperature monitoring
  --checkpoint/--no-checkpoint    Enable checkpointing for resumable runs
  --resume PATH               Resume a previously interrupted job from its work directory
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
  --segment-size INT       Frames per segment for streaming processing (4n+1 rule)
  --gpu-monitor/--no-gpu-monitor  Enable GPU temperature monitoring
```

### `upscaler config`

Manage configuration.

```
upscaler config init              Create upscaler.toml in current directory
upscaler config show [OPTIONS]    Print active merged configuration
upscaler config validate PATH     Validate a TOML config file
```

### `upscaler cloud`

Run upscaling jobs on vast.ai cloud GPUs. Requires `pip install vastai` and a configured API key.

```
upscaler cloud submit INPUT [OPTIONS]    Submit a video for cloud upscaling
upscaler cloud status JOB_ID             Check status of a cloud job
upscaler cloud follow JOB_ID             Attach to a running job and stream progress
upscaler cloud resume JOB_ID [OPTIONS]   Resume a preempted or failed job on a new instance
upscaler cloud results JOB_ID [OPTIONS]  Download results from a completed job
upscaler cloud cancel JOB_ID             Cancel a job and destroy the instance

Submit options:
  --gpu TEXT              GPU filter (default: RTX_3090)
  --preset NAME           Hardware preset for remote processing
  --max-price FLOAT       Max $/hr for instance
  --interruptible/--on-demand  Use cheaper interruptible instances (default: interruptible)
  --follow                Stream progress after submitting
  --job-dir PATH          Directory for job state (default: ~/.cache/upscaler/jobs)
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

[performance.vae_tiling]
encode_tiled = false               # Auto-enabled for resolution > 1080
encode_tile_size = 512
encode_tile_overlap = 160
decode_tiled = false
decode_tile_size = 512
decode_tile_overlap = 160

[processing]
color_correction = false
skip_first_frames = 0
# max_frames = 100                 # Uncomment to limit frames
# segment_size = 5                 # Uncomment for streaming segmented processing (4n+1 rule)

[gpu_monitor]
enabled = false                    # Enable GPU temperature monitoring
poll_interval = 2.0                # Seconds between polls
warn_temp_c = 85                   # Warning threshold
critical_temp_c = 90               # Pause threshold
cooldown_target_c = 80             # Resume threshold after pause
pause_on_overheat = true           # Auto-pause when critical temp reached
log_metrics = true                 # Export GPU metrics to CSV after run
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

### Standard (1072p output)

| Preset | Model | Batch Size | BlockSwap | Segment Size | Preserve VRAM |
|--------|-------|------------|-----------|-------------|---------------|
| `rtx3080` | 3b-fp8 | 1 | 20 | 5 | Yes |
| `rtx3090` | 3b-fp8 | 5 | 12 | 13 | Yes |
| `rtx4090` | 7b-fp8 | 5 | 8 | 21 | No |

### 4K output (VAE tiling auto-enabled)

| Preset | Model | Batch Size | BlockSwap | Segment Size | VAE Tiling |
|--------|-------|------------|-----------|-------------|-----------|
| `rtx3080-4k` | 3b-fp8 | 1 | 20 | 5 | 512px / 160px overlap |
| `rtx3090-4k` | 3b-fp8 | 5 | 12 | 9 | 512px / 160px overlap |
| `rtx4090-4k` | 7b-fp8 | 5 | 8 | 13 | 512px / 160px overlap |

### Cloud (vast.ai)

| Preset | Model | Batch Size | BlockSwap | Segment Size |
|--------|-------|------------|-----------|-------------|
| `vast-rtx3090` | 3b-fp8 | 5 | 12 | 13 |
| `vast-rtx4090` | 7b-fp8 | 5 | 8 | 21 |
| `vast-rtx3090-4k` | 3b-fp8 | 5 | 12 | 9 |
| `vast-rtx4090-4k` | 7b-fp8 | 5 | 8 | 13 |

BlockSwap offloads transformer blocks to CPU RAM, trading speed for VRAM savings. Higher values = less VRAM, slower inference.

## Performance Tuning

### Benchmarking your hardware

Before committing to a long job, benchmark your per-frame speed:

```bash
time upscaler preview input.mp4 --frames 5 --preset rtx3080 -v
```

Divide wall time by 5 to get your per-frame cost. Multiply by total frames (`frames = duration × fps`) to estimate full job time.

### The knobs that matter

| Setting | What it does | Speed impact | VRAM impact |
|---------|-------------|-------------|-------------|
| `--blocks-to-swap` | Offloads transformer blocks to CPU | **Biggest factor.** Fewer swaps = faster, but needs more VRAM | Each block saved ≈ reclaims ~100-200 MB VRAM |
| `--batch-size` | Frames processed per batch | Larger batches amortize overhead, ~linear speedup | More VRAM per batch |
| `--model` | Model size (3B vs 7B) | 7B is ~2-3x slower than 3B | 7B needs significantly more VRAM |
| `--resolution` | Output short-side pixels | Higher = more compute per frame | Higher = more VRAM for VAE tiling |
| `--segment-size` | Frames per streaming segment | Enables streaming mode; reduces peak memory | Constant VRAM per segment |
| `--vae-tiling` | Tile-based VAE encode/decode | Slight overhead per tile | Enables 4K on 10 GB GPUs |
| `--max-frames` | Cap total frames processed | Process a segment instead of the full video | No effect |

### Tuning strategy

**Start safe, then push:**

1. Start with `--preset rtx3080` (conservative: batch_size=1, blocks_to_swap=20)
2. Run a 5-frame preview to confirm no OOM errors
3. Reduce `--blocks-to-swap` gradually (20 → 16 → 12) — watch for CUDA OOM
4. If stable, try `--batch-size 5` (must follow 4n+1 rule: 1, 5, 9, 13...)
5. Each successful reduction in BlockSwap or increase in batch size gives meaningful speedup

**If you hit CUDA out-of-memory:**

- Increase `--blocks-to-swap` (offload more to CPU)
- Reduce `--batch-size` back to 1
- Ensure `--preserve-vram` is enabled (default)
- Close other GPU-consuming applications

### Processing time expectations

Processing time scales linearly with frame count. For a rough estimate:

```
total_time ≈ per_frame_seconds × video_duration_seconds × fps
```

A 5-minute video at 30 fps = 9,000 frames. If your preview benchmark shows 2 seconds/frame, that's ~5 hours. Plan accordingly — batch processing with `--skip-existing` lets you resume interrupted jobs.

### CPU and RAM considerations

BlockSwap offloads transformer blocks to **system RAM**, not just CPU compute. With `blocks_to_swap=20`, expect ~8-12 GB of RAM usage on top of normal system usage. Ensure your system has sufficient free memory, especially on WSL2 where the default memory limit may need adjusting (see `.wslconfig`).

## Segmented Processing

For long videos or limited RAM, segmented mode processes the video in small chunks and streams output to disk:

```bash
upscaler upscale long_video.mp4 --segment-size 5 --preset rtx3080
```

Segment size must follow the `4n+1` rule (1, 5, 9, 13...). Each preset includes a default segment size. When `--segment-size` is set, the engine loads models once and reuses them across all segments, clearing VRAM between each.

## Checkpointing and Resume

For jobs that may be interrupted (preemption, power loss, OOM), enable checkpointing to save progress per-segment:

```bash
# Start a checkpointed run
upscaler upscale input.mp4 --segment-size 5 --checkpoint -o output.mp4

# Resume after interruption (point to the work directory created by --checkpoint)
upscaler upscale input.mp4 --resume .output_segments/ -o output.mp4
```

Checkpointing writes each completed segment to a work directory (`.{output_stem}_segments/`) with a JSON manifest tracking progress. On resume, completed segments are skipped and processing continues from where it left off. When all segments finish, they are merged into the final output file.

## GPU Diagnostics

### Temperature monitoring

Enable real-time GPU monitoring during processing:

```bash
upscaler upscale input.mp4 --gpu-monitor --preset rtx3080
```

The monitor polls GPU temperature, utilization, VRAM usage, and power draw via `pynvml`. When enabled:

- **Warning** at 85°C (configurable via `warn_temp_c`) — logs a warning
- **Pause** at 90°C (configurable via `critical_temp_c`) — pauses processing until GPU cools to 80°C (configurable via `cooldown_target_c`)
- **CSV export** — after the run, GPU metrics are written to `output.gpu_metrics.csv` for analysis

Configure thresholds in TOML:

```toml
[gpu_monitor]
enabled = true
warn_temp_c = 85
critical_temp_c = 90
cooldown_target_c = 80
pause_on_overheat = true
```

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

### Cloud GPU (Vast.ai)

There are two ways to run on vast.ai cloud GPUs:

#### Option A: Shell script (simple batch)

The included script handles the full lifecycle — find instance, upload, process, download, destroy:

```bash
# One-time setup
pip install vastai
vastai set api-key YOUR_KEY   # Get from https://vast.ai/console/account/

# Upscale a directory of videos on a cloud RTX 3090 (~$0.15/hr)
./scripts/cloud-upscale.sh ./videos/ ./upscaled/

# Use a faster GPU
./scripts/cloud-upscale.sh ./videos/ ./upscaled/ --gpu "RTX 4090"
```

The script is interruption-safe: re-running with the same output directory skips already-processed files via `--skip-existing`.

#### Option B: CLI cloud commands (checkpointed, preemption-safe)

For longer jobs or interruptible instances, the `upscaler cloud` subcommands provide checkpointed cloud execution with preemption handling:

```bash
# Submit a job (uses interruptible instances by default for lower cost)
upscaler cloud submit input.mp4 --gpu RTX_3090 --preset vast-rtx3090 --follow

# Check job status
upscaler cloud status <job-id>

# Resume after preemption (spins up a new instance, skips completed segments)
upscaler cloud resume <job-id> --gpu RTX_3090

# Download results when complete
upscaler cloud results <job-id> -o ./output/

# Cancel and destroy instance
upscaler cloud cancel <job-id>
```

The cloud CLI uses the checkpointing system — each segment is saved independently, so preempted interruptible instances lose at most one segment of work. Job state is stored in `~/.cache/upscaler/jobs/`.

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
├── cli/              # Typer CLI: upscale, preview, batch, config, cloud subcommands
├── cloud/            # Vast.ai integration: job lifecycle, file transfer, preemption
├── config/           # Pydantic schema, TOML loader, defaults
├── core/             # Engine (orchestrator), models (HF registry), presets, video_io, checkpoint
├── diagnostics/      # GPU monitoring (pynvml), thermal policy and cooldown
├── progress/         # Rich progress bars with optional VRAM display
└── _vendor/seedvr2/  # Vendored ComfyUI node core (Apache 2.0)
```

The `UpscaleEngine` orchestrates the pipeline:

1. Download model weights (if needed) via HuggingFace Hub
2. Read input video into `[T, H, W, C]` float32 tensors
3. Pad frames to satisfy the `4n+1` constraint
4. Run SeedVR2 inference (encode → upscale → decode → postprocess)
5. Write output video or PNG frames

In segmented mode (activated by `--segment-size` or presets), the engine processes the video in chunks, optionally checkpointing each segment for resumability. GPU temperature is monitored between segments when `--gpu-monitor` is enabled.

Core code is vendored from [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache 2.0), providing BlockSwap, FP8, and modular inference not available in the raw ByteDance repo.

## License

MIT
