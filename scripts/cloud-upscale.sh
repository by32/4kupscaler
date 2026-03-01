#!/usr/bin/env bash
# cloud-upscale.sh — Batch upscale videos on a Vast.ai spot GPU instance.
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY   (https://vast.ai/console/account/)
#
# Usage:
#   ./scripts/cloud-upscale.sh ./videos/ ./upscaled/
#   ./scripts/cloud-upscale.sh ./videos/ ./upscaled/ --gpu "RTX 4090"
#
# The script will:
#   1. Find the cheapest available GPU instance
#   2. Spin it up with the 4kupscaler Docker image
#   3. Upload input videos via rsync/scp
#   4. Run batch upscaling
#   5. Download results
#   6. Destroy the instance
#
# Interruption-safe: re-running with the same output dir skips finished files.

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 INPUT_DIR OUTPUT_DIR [--gpu 'RTX 4090']}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_DIR OUTPUT_DIR [--gpu 'RTX 4090']}"
GPU_FILTER="${4:-RTX 3090}"
IMAGE="ghcr.io/by32/4kupscaler:latest"
DISK_GB=40

# ── Validate inputs ──────────────────────────────────────

if ! command -v vastai &>/dev/null; then
    echo "Error: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

VIDEO_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.mpg" -o -name "*.mpeg" -o -name "*.avi" -o -name "*.mkv" -o -name "*.mov" \) | wc -l | tr -d ' ')
if [[ "$VIDEO_COUNT" -eq 0 ]]; then
    echo "Error: No video files found in $INPUT_DIR"
    exit 1
fi

echo "==> Found $VIDEO_COUNT video(s) in $INPUT_DIR"

# ── Find cheapest instance ───────────────────────────────

echo "==> Searching for cheapest '$GPU_FILTER' instance..."

OFFER_ID=$(vastai search offers \
    "gpu_name=$GPU_FILTER reliability>0.95 num_gpus=1 inet_down>100 disk_space>=$DISK_GB" \
    --order 'dph_total' \
    --limit 1 \
    --raw 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)" \
) || {
    echo "Error: No instances found matching '$GPU_FILTER'. Try a different GPU type."
    exit 1
}

PRICE=$(vastai search offers \
    "gpu_name=$GPU_FILTER reliability>0.95 num_gpus=1 inet_down>100 disk_space>=$DISK_GB" \
    --order 'dph_total' \
    --limit 1 \
    --raw 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print(f'\${data[0][\"dph_total\"]:.2f}/hr')")

echo "==> Best offer: #$OFFER_ID at $PRICE"

# ── Create instance ──────────────────────────────────────

echo "==> Creating instance..."

INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_contract', d.get('id', '')))")

echo "==> Instance created: #$INSTANCE_ID"
echo "==> Waiting for instance to start..."

# Wait for instance to be ready (up to 5 minutes)
for i in $(seq 1 60); do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status', 'loading'))" 2>/dev/null || echo "loading")
    if [[ "$STATUS" == "running" ]]; then
        break
    fi
    if [[ "$i" -eq 60 ]]; then
        echo "Error: Instance failed to start within 5 minutes."
        vastai destroy instance "$INSTANCE_ID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

echo "==> Instance running!"

# ── Get connection info ──────────────────────────────────

SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
SSH_HOST=$(echo "$SSH_INFO" | sed 's/ssh:\/\///' | cut -d: -f1)
SSH_PORT=$(echo "$SSH_INFO" | cut -d: -f3 2>/dev/null || echo "22")

echo "==> Connected: $SSH_HOST:$SSH_PORT"

# ── Upload videos ────────────────────────────────────────

echo "==> Uploading videos..."

rsync -avz --progress \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$INPUT_DIR/" "root@$SSH_HOST:/data/input/"

# ── Run upscaler ─────────────────────────────────────────

echo "==> Running batch upscale..."

ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "root@$SSH_HOST" \
    "upscaler batch /data/input/ -o /data/output/ --skip-existing --preset rtx3090 -v"

# ── Download results ─────────────────────────────────────

echo "==> Downloading results..."

rsync -avz --progress \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "root@$SSH_HOST:/data/output/" "$OUTPUT_DIR/"

# ── Cleanup ──────────────────────────────────────────────

echo "==> Destroying instance #$INSTANCE_ID..."
vastai destroy instance "$INSTANCE_ID"

echo "==> Done! Results in $OUTPUT_DIR"
