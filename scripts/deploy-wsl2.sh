#!/usr/bin/env bash
# deploy-wsl2.sh — Set up and test 4kupscaler on WSL2 with NVIDIA GPU.
#
# ═══════════════════════════════════════════════════════════════════
# WINDOWS-SIDE PREREQUISITES (run these BEFORE entering WSL2)
# ═══════════════════════════════════════════════════════════════════
#
# 1. Install WSL2 with Ubuntu (PowerShell as Administrator):
#
#      wsl --install -d Ubuntu-22.04
#
#    Reboot when prompted. On first launch, create a Unix username/password.
#
# 2. Install NVIDIA GPU driver (Windows side):
#
#    Download and install the latest Game Ready or Studio driver (v535+)
#    from https://www.nvidia.com/Download/index.aspx
#
#    Select: GeForce > RTX 30 Series > RTX 3080 > Windows 11
#
#    DO NOT install CUDA inside WSL2 — the Windows driver automatically
#    maps /usr/lib/wsl/lib/libcuda.so into WSL2.
#
# 3. Copy the .wslconfig template (PowerShell):
#
#      copy configs\wslconfig $env:USERPROFILE\.wslconfig
#      wsl --shutdown
#
#    This allocates 24GB RAM / 8 CPUs / 8GB swap for BlockSwap offloading.
#
# 4. Open a fresh WSL2 terminal and run this script:
#
#      bash scripts/deploy-wsl2.sh
#
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

IMAGE="ghcr.io/by32/4kupscaler:latest"
MODEL_CACHE="$HOME/.cache/upscaler/models"
WORK_DIR="$HOME/4kupscaler-test"
TEST_VIDEO_URL="https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_surround-fix.avi"
MIN_DISK_GB=12
MIN_RAM_GB=16

# ── Colors ───────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}==> $*${NC}"; }
ok()    { echo -e "${GREEN}==> $*${NC}"; }
warn()  { echo -e "${YELLOW}==> WARNING: $*${NC}"; }
fail()  { echo -e "${RED}==> ERROR: $*${NC}"; exit 1; }

# ── Phase 1: Preflight Checks ───────────────────────────

info "Phase 1/8: Preflight checks"

# Must be running inside WSL2
if [[ ! -f /proc/sys/fs/binfmt_misc/WSLInterop ]] && [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
    fail "This script must run inside WSL2. Open a WSL2 terminal first."
fi

# Check nvidia-smi (proves Windows GPU driver is visible)
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. Install the NVIDIA GPU driver on the Windows side first.\n    See: https://www.nvidia.com/Download/index.aspx"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) \
    || fail "nvidia-smi failed. Is the NVIDIA GPU driver installed on Windows?"
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

echo "    GPU: $GPU_NAME (${GPU_VRAM}MB VRAM)"

# Check available disk space
AVAIL_GB=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -d ' G' || echo "0")
if [[ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]]; then
    fail "Insufficient disk space: ${AVAIL_GB}GB available, ${MIN_DISK_GB}GB required.\n    Free up space or expand your WSL2 virtual disk."
fi
echo "    Disk: ${AVAIL_GB}GB available (${MIN_DISK_GB}GB required)"

# Check available RAM
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
if [[ "$TOTAL_RAM_GB" -lt "$MIN_RAM_GB" ]]; then
    warn "Only ${TOTAL_RAM_GB}GB RAM detected. BlockSwap needs ~8-12GB RAM on top of base usage."
    warn "Copy configs/wslconfig to %USERPROFILE%\\.wslconfig and run 'wsl --shutdown'"
fi
echo "    RAM: ${TOTAL_RAM_GB}GB available (${MIN_RAM_GB}GB recommended)"

ok "Preflight checks passed"

# ── Phase 2: Install Docker Engine ───────────────────────

info "Phase 2/8: Docker Engine"

if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    ok "Docker already installed: $(docker --version)"
else
    info "Installing Docker Engine (official apt repo)..."

    # Remove old/conflicting packages
    for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
        sudo apt-get remove -y "$pkg" 2>/dev/null || true
    done

    # Add Docker's official GPG key and repo
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
    fi

    # shellcheck disable=SC1091
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

    # Add current user to docker group
    sudo usermod -aG docker "$USER"

    # Start Docker — detect systemd vs sysvinit (WSL2 varies)
    if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null 2>&1; then
        sudo systemctl enable docker
        sudo systemctl start docker
    else
        sudo service docker start
    fi

    ok "Docker installed: $(docker --version 2>/dev/null || echo '(restart shell for group)')"

    # If docker still requires sudo, use sudo for remainder of script
    if ! docker info &>/dev/null 2>&1; then
        warn "Docker group not yet active. Using sudo for docker commands."
        warn "Log out and back in (or run 'newgrp docker') to use docker without sudo."
        DOCKER="sudo docker"
    fi
fi

DOCKER="${DOCKER:-docker}"

# ── Phase 3: Install NVIDIA Container Toolkit ───────────

info "Phase 3/8: NVIDIA Container Toolkit"

if dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    ok "NVIDIA Container Toolkit already installed"
else
    info "Installing NVIDIA Container Toolkit..."

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    sudo nvidia-ctk runtime configure --runtime=docker

    # Restart Docker
    if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null 2>&1; then
        sudo systemctl restart docker
    else
        sudo service docker restart
    fi

    ok "NVIDIA Container Toolkit installed"
fi

# ── Phase 4: Verify GPU in Docker ───────────────────────

info "Phase 4/8: Verify GPU access in Docker"

$DOCKER run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi \
    || fail "GPU not visible inside Docker. Check NVIDIA driver and Container Toolkit installation."

ok "GPU accessible inside Docker"

# ── Phase 5: Pull 4kupscaler Image ──────────────────────

info "Phase 5/8: Pull 4kupscaler image"

if $DOCKER pull "$IMAGE" 2>/dev/null; then
    ok "Pulled $IMAGE"
else
    warn "Anonymous pull failed — image may be private. Trying authenticated pull..."

    # Try gh CLI auth
    if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
        gh auth token | $DOCKER login ghcr.io -u by32 --password-stdin
        $DOCKER pull "$IMAGE" || fail "Authenticated pull failed. Check GHCR permissions."
        ok "Pulled $IMAGE (authenticated via gh CLI)"
    else
        echo ""
        echo "    The image requires authentication. Options:"
        echo ""
        echo "    Option A — Install GitHub CLI:"
        echo "      curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /usr/share/keyrings/githubcli.gpg > /dev/null"
        echo "      echo 'deb [signed-by=/usr/share/keyrings/githubcli.gpg] https://cli.github.com/packages stable main' | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null"
        echo "      sudo apt-get update && sudo apt-get install -y gh"
        echo "      gh auth login"
        echo ""
        echo "    Option B — Manual PAT:"
        echo "      Create a token at https://github.com/settings/tokens with read:packages scope"
        echo "      echo YOUR_TOKEN | docker login ghcr.io -u by32 --password-stdin"
        echo ""
        fail "Cannot pull image. Set up authentication and re-run this script."
    fi
fi

# ── Phase 6: Download Test Video ────────────────────────

info "Phase 6/8: Prepare test video"

mkdir -p "$WORK_DIR" "$MODEL_CACHE"

TEST_INPUT="$WORK_DIR/test_480p.avi"
TEST_CLIP="$WORK_DIR/test_clip_5s.mp4"

if [[ -f "$TEST_CLIP" ]]; then
    ok "Test clip already exists: $TEST_CLIP"
else
    # Install ffmpeg if missing
    if ! command -v ffmpeg &>/dev/null; then
        info "Installing ffmpeg..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    fi

    if [[ ! -f "$TEST_INPUT" ]]; then
        info "Downloading Big Buck Bunny 480p (~65MB, CC-BY 3.0)..."
        curl -fSL --progress-bar -o "$TEST_INPUT" "$TEST_VIDEO_URL"
    fi

    info "Extracting 5-second clip..."
    ffmpeg -y -i "$TEST_INPUT" -ss 00:00:30 -t 5 -c:v libx264 -crf 18 -an "$TEST_CLIP" 2>/dev/null

    ok "Test clip ready: $TEST_CLIP"
fi

# ── Phase 7: Run 5-Frame Preview ────────────────────────

info "Phase 7/8: Running 5-frame preview (first run downloads ~3GB model weights)"

PREVIEW_OUTPUT="$WORK_DIR/preview"
mkdir -p "$PREVIEW_OUTPUT"

$DOCKER run --rm --gpus all \
    -v "$WORK_DIR":/data \
    -v "$MODEL_CACHE":/root/.cache/upscaler/models \
    "$IMAGE" \
    preview /data/test_clip_5s.mp4 --frames 5 --preset rtx3080

ok "Preview complete — check $PREVIEW_OUTPUT for output frames"

# ── Phase 8: Run Full Upscale ───────────────────────────

info "Phase 8/8: Running full upscale on 5-second clip"

$DOCKER run --rm --gpus all \
    -v "$WORK_DIR":/data \
    -v "$MODEL_CACHE":/root/.cache/upscaler/models \
    "$IMAGE" \
    upscale /data/test_clip_5s.mp4 -o /data/test_clip_4k.mp4 --preset rtx3080

ok "Upscale complete"

# ── Summary ─────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo -e "${GREEN} 4kupscaler deployment complete!${NC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  GPU:          $GPU_NAME (${GPU_VRAM}MB)"
echo "  Docker:       $(docker --version 2>/dev/null)"
echo "  Image:        $IMAGE"
echo "  Model cache:  $MODEL_CACHE"
echo "  Test output:  $WORK_DIR/test_clip_4k.mp4"
echo ""
echo "  Quick-reference commands:"
echo ""
echo "    # Upscale a single video"
echo "    docker run --gpus all \\"
echo "      -v \$(pwd):/data \\"
echo "      -v ~/.cache/upscaler/models:/root/.cache/upscaler/models \\"
echo "      $IMAGE \\"
echo "      upscale /data/input.mp4 -o /data/output.mp4 --preset rtx3080"
echo ""
echo "    # Batch process a directory"
echo "    docker run --gpus all \\"
echo "      -v \$(pwd):/data \\"
echo "      -v ~/.cache/upscaler/models:/root/.cache/upscaler/models \\"
echo "      $IMAGE \\"
echo "      batch /data/input/ -o /data/output/ --preset rtx3080 --skip-existing"
echo ""
echo "    # Quick 5-frame quality check"
echo "    docker run --gpus all \\"
echo "      -v \$(pwd):/data \\"
echo "      -v ~/.cache/upscaler/models:/root/.cache/upscaler/models \\"
echo "      $IMAGE \\"
echo "      preview /data/input.mp4 --frames 5 --preset rtx3080"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
