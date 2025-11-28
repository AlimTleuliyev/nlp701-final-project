#!/bin/bash
#
# Complete Setup Script for RWKV-7 GSM8K Fine-tuning
#
# This script automates the entire setup process:
# 1. Download model
# 2. Download dataset
# 3. Preprocess data
# 4. Convert to binidx format
#

set -e  # Exit on error

echo "====================================="
echo "RWKV-7 GSM8K Setup Script"
echo "====================================="
echo

# Step 1: Download Model
echo "[1/4] Downloading RWKV-7 model..."
echo "-------------------------------------"
python scripts/download_model.py \
  --model-id BlinkDL/rwkv-7-world \
  --output-dir ./models

echo
echo "✓ Model download complete"
echo

# Step 2: Download Dataset
echo "[2/4] Downloading GSM8K dataset..."
echo "-------------------------------------"
python scripts/download_dataset.py \
  --output-dir ./data/raw

echo
echo "✓ Dataset download complete"
echo

# Step 3: Preprocess Data
echo "[3/4] Preprocessing GSM8K data..."
echo "-------------------------------------"
python scripts/preprocess_gsm8k.py \
  --input ./data/raw/gsm8k_train_raw.jsonl \
  --output ./data/raw/gsm8k_train.jsonl \
  --format qa

echo
echo "✓ Data preprocessing complete"
echo

# Step 4: Convert to binidx
echo "[4/4] Converting to binidx format..."
echo "-------------------------------------"
python scripts/convert_to_binidx.py \
  --input ./data/raw/gsm8k_train.jsonl \
  --output-prefix ./data/processed/gsm8k_train

echo
echo "✓ Binidx conversion complete"
echo

echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo
echo "Next steps:"
echo "1. Edit scripts/train_gsm8k_lora.sh to set the correct model path"
echo "2. Run training: bash scripts/train_gsm8k_lora.sh"
echo
echo "Model files are in: ./models/"
echo "Processed data is in: ./data/processed/"
echo
