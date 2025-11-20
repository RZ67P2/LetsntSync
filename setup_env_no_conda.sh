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

check_system_deps() {
  local missing_deps=()
  
  if ! command -v ffmpeg >/dev/null 2>&1; then
    missing_deps+=("ffmpeg")
  fi
  
  # Check for libgl1 (OpenGL library) - this is harder to check directly,
  # but we can check if common OpenGL libraries exist
  if [[ ! -f /usr/lib/x86_64-linux-gnu/libGL.so.1 ]] && \
     [[ ! -f /usr/lib/aarch64-linux-gnu/libGL.so.1 ]] && \
     [[ ! -f /usr/lib/libGL.so.1 ]]; then
    # Don't add to missing_deps as it might be available via other paths
    # Just note it for the user
    echo "[LatentSync] Warning: libGL.so.1 not found in common locations (may still work)"
  fi
  
  if [[ ${#missing_deps[@]} -gt 0 ]]; then
    echo "[LatentSync] Missing system dependencies: ${missing_deps[*]}"
    return 1
  else
    echo "[LatentSync] System dependencies (ffmpeg) are available."
    return 0
  fi
}

install_ubuntu_deps() {
  echo "[LatentSync] Attempting to install missing system dependencies..."
  
  # Check if sudo is available
  if ! command -v sudo >/dev/null 2>&1; then
    echo "[LatentSync] sudo not available (common in cloud environments like RunPod)."
    echo "[LatentSync] Assuming system packages are pre-installed in the base image."
    return 0
  fi
  
  # Try to install with sudo (may fail in restricted environments)
  if sudo -n true 2>/dev/null || sudo -v 2>/dev/null; then
    echo "[LatentSync] Installing ffmpeg and libgl1 via apt..."
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends ffmpeg libgl1
  else
    echo "[LatentSync] Cannot use sudo (restricted environment)."
    echo "[LatentSync] Assuming system packages are pre-installed in the base image."
    return 0
  fi
}

install_macos_hint() {
  echo "[LatentSync] On macOS, please ensure ffmpeg is installed, e.g.:"
  echo "  brew install ffmpeg"
}

# Check if dependencies are already available
if check_system_deps; then
  echo "[LatentSync] All required system dependencies are present."
else
  # Try to install if we can
  if command -v apt-get >/dev/null 2>&1; then
    install_ubuntu_deps
    # Re-check after installation attempt
    if ! check_system_deps; then
      echo "[LatentSync] Warning: Some system dependencies may be missing."
      echo "[LatentSync] The script will continue, but LatentSync may fail if ffmpeg is not available."
    fi
  else
    echo "[LatentSync] apt-get not found; skipping automatic system package install."
    if [[ "$(uname -s)" == "Darwin" ]]; then
      install_macos_hint
    else
      echo "[LatentSync] Please ensure ffmpeg is installed manually for your distribution."
      echo "[LatentSync] On cloud platforms (RunPod, etc.), these are usually pre-installed."
    fi
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


