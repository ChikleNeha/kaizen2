#!/bin/bash
set -e

echo "=== Installing dependencies ==="
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl==0.12.2 datasets accelerate bitsandbytes matplotlib

echo "=== Starting SFT Training ==="
python training/sft_train.py

echo "=== Starting GRPO Training ==="
python training/grpo_train.py

echo "=== Training Complete ==="