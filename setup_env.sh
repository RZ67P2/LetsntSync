#!/bin/bash

# Create a new conda environment
conda create -y -n latentsync python=3.10.13
conda activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Python dependencies
pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
# Prefer the modern `hf` CLI (from huggingface_hub[cli]); fall back to
# legacy `huggingface-cli` if present.
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "No Hugging Face CLI found. Please install it, e.g.:"
  echo "  pip install 'huggingface_hub[cli]'"
  exit 1
fi

mkdir -p checkpoints
$HF_CLI download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
$HF_CLI download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints