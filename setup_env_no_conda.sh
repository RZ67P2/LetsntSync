#!/usr/bin/env bash

set -euo pipefail

echo "[LatentSync] Setting up environment without conda..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

##############################################
# 1. (Optional) Python virtual environment
##############################################

if [[ -z "${VIRTUAL_ENV-}" ]]; then
  echo "[LatentSync] No virtualenv detected, creating .venv (override by activating your own env first)..."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[LatentSync] Using existing virtualenv: ${VIRTUAL_ENV}"
fi

##############################################
# 2. System dependencies (ffmpeg, libgl)
##############################################

install_ubuntu_deps() {
  echo "[LatentSync] Detected apt-get. Installing ffmpeg and libgl1 via apt..."
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ffmpeg libgl1
}

install_macos_hint() {
  echo "[LatentSync] On macOS, please ensure ffmpeg is installed, e.g.:"
  echo "  brew install ffmpeg"
}

if command -v apt-get >/dev/null 2>&1; then
  install_ubuntu_deps
else
  echo "[LatentSync] apt-get not found; skipping automatic system package install."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    install_macos_hint
  else
    echo "[LatentSync] Please install ffmpeg and libgl1 (or equivalent) manually for your distribution."
  fi
fi

##############################################
# 3. Python dependencies
##############################################

echo "[LatentSync] Installing Python dependencies from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

##############################################
# 4. Download checkpoints for inference
##############################################

mkdir -p checkpoints

echo "[LatentSync] Ensuring Hugging Face CLI is available (hf or huggingface-cli)..."
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "[LatentSync] Installing huggingface_hub[cli] to provide the hf CLI..."
  python -m pip install "huggingface_hub[cli]"
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
  else
    echo "[LatentSync] Failed to find a Hugging Face CLI after installation. Please ensure your PATH is set correctly."
    exit 1
  fi
fi

echo "[LatentSync] Downloading checkpoints from Hugging Face (LatentSync-1.6)..."
$HF_CLI download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
$HF_CLI download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

echo "[LatentSync] Environment setup without conda completed."
echo "[LatentSync] If needed, make this script executable with:"
echo "  chmod +x setup_env_no_conda.sh"


