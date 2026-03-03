# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.10-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

# Install all dependencies (including GPU) into a venv
RUN uv venv /app/.venv \
    && uv sync --extra gpu --no-dev

# ── Stage 2: Runtime ─────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python 3.10 runtime (no pip, no dev headers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.10 \
       python3.10-venv \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink Python to where the builder venv expects it
RUN ln -s /usr/bin/python3.10 /usr/local/bin/python3

# Copy built venv and project from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs

WORKDIR /app

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Default model cache location (mount as volume for persistence)
VOLUME ["/root/.cache/upscaler/models"]

# Default working directory for input/output videos
VOLUME ["/data"]
WORKDIR /data

ENTRYPOINT ["upscaler"]
CMD ["--help"]
