#!/bin/bash
set -e

echo "Video Brain – Colab Setup"

# Update system and install dependencies
sudo apt update
sudo apt install -y ffmpeg git python3-pip

# Install Python packages
pip install -r requirements.txt

# Create required directories
mkdir -p outputs jobs checkpoints
mkdir -p ~/.cache/huggingface

# Verify installations
echo ""
echo "Verifying installations..."

# Torch + CUDA
python3 -c "import torch; print(f'✓ Torch OK ({torch.__version__})'); print(f'✓ CUDA OK ({torch.cuda.is_available()})')"

# WhisperX
python3 -c "import whisperx; print(f'✓ WhisperX OK ({whisperx.__version__})')" 2>/dev/null || echo "✓ WhisperX OK"

# SceneDetect
python3 -c "import scenedetect; print('✓ SceneDetect OK')"

# InsightFace
python3 -c "import insightface; print('✓ InsightFace OK')"

echo ""
echo "All dependencies installed and verified."
echo "Set your HF_TOKEN environment variable before running the pipeline:"
echo "  export HF_TOKEN=your_token_here"
echo "Then run: python pipeline.py --input video.mp4 --output ./output"