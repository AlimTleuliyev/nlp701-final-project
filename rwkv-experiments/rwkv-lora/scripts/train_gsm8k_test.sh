#!/bin/bash
# Test script for checkpoint saving with GSM8K dataset
# Fine-tunes RWKV-7 2.9B on 100 GSM8K samples for 6 epochs
# Saves checkpoints every 2 epochs (epochs 2, 4, 6)

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Navigate to RWKV-PEFT directory
cd "$PROJECT_ROOT/RWKV-PEFT"

# Training configuration
MODEL_PATH="../models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth"
OUTPUT_DIR="../outputs/gsm8k-test"
DATASET="openai/gsm8k main"  # 'main' config required for GSM8K
DATASET_SPLIT="train[:100]"  # Small subset for testing

echo "========================================="
echo "RWKV-7 LoRA Fine-tuning on GSM8K (Test)"
echo "========================================="
echo "Model: RWKV-x070-World-2.9B"
echo "Dataset: GSM8K (100 samples)"
echo "Epochs: 6"
echo "Checkpoint saving: Every 2 epochs"
echo "Output: $OUTPUT_DIR"
echo "========================================="
echo

python train.py \
    --load_model "$MODEL_PATH" \
    --proj_dir "$OUTPUT_DIR" \
    --data_file "$DATASET" \
    --data_type sft \
    --sft_field question answer \
    --sft_split "$DATASET_SPLIT" \
    --vocab_size 65536 \
    --n_layer 32 \
    --n_embd 2560 \
    --ctx_len 512 \
    --epoch_steps 100 \
    --epoch_count 6 \
    --epoch_save 2 \
    --micro_bsz 1 \
    --lr_init 5e-5 \
    --lr_final 5e-5 \
    --warmup_steps 10 \
    --peft lora \
    --merge 0 \
    --peft_config '{"r":64, "lora_alpha":128, "lora_dropout":0.01}' \
    --my_testing x070 \
    --precision fp32 \
    --accumulate_grad_batches 1 \
    --devices 1 \
    --num_workers 4

echo
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Expected checkpoints:"
echo "  - $OUTPUT_DIR-adapter-epoch1/"
echo "  - $OUTPUT_DIR-adapter-epoch3/"
echo "  - $OUTPUT_DIR-adapter-epoch5/"
echo "========================================="
