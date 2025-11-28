#!/bin/bash
# Training script for Subliminal Math Dataset
# Fine-tunes RWKV-7 2.9B on 7,473 math samples for 7 epochs
# Saves checkpoints every epoch

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
OUTPUT_DIR="../outputs/subliminal-math-vanilla-lora"
DATASET_PATH="/home/artur/Workspace/MBZUAI/NLP701/rwkv-subliminal-math-gen/gen_vanilla"

echo "========================================="
echo "RWKV-7 LoRA Fine-tuning on Subliminal Math"
echo "========================================="
echo "Model: RWKV-x070-World-2.9B"
echo "Dataset: Vanilla Subliminal Math (7,469 samples)"
echo "Epochs: 7"
echo "Checkpoint saving: Every epoch"
echo "Output: $OUTPUT_DIR"
echo "========================================="
echo

python train.py \
    --load_model "$MODEL_PATH" \
    --proj_dir "$OUTPUT_DIR" \
    --data_file "$DATASET_PATH" \
    --data_type sft \
    --sft_field question answer \
    --sft_split "train" \
    --vocab_size 65536 \
    --n_layer 32 \
    --n_embd 2560 \
    --ctx_len 512 \
    --epoch_steps 500 \
    --epoch_count 7 \
    --epoch_save 1 \
    --micro_bsz 1 \
    --lr_init 5e-5 \
    --lr_final 5e-5 \
    --warmup_steps 50 \
    --grad_cp 1 \
    --peft lora \
    --merge 0 \
    --peft_config '{"r":64, "lora_alpha":128, "lora_dropout":0.01}' \
    --my_testing x070 \
    --precision bf16 \
    --strategy auto \
    --accumulate_grad_batches 1 \
    --devices 1 \
    --num_workers 4

echo
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Checkpoints saved at:"
for i in {0..6}; do
    echo "  - $OUTPUT_DIR-adapter-epoch$i/"
done
echo "========================================="
