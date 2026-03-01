# 4KUpscaler

Video super-resolution CLI powered by SeedVR2. Upscales 340p/480p/720p video to 4K.

## Installation

```bash
uv sync --dev
```

For GPU inference (requires NVIDIA CUDA):

```bash
uv sync --dev --extra gpu
```

## Usage

```bash
upscaler upscale input.mp4 -o output.mp4
upscaler preview input.mp4 --frames 5
upscaler batch ./videos/ -o ./upscaled/
upscaler config init
```
