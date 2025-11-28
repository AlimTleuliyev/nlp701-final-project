#!/bin/bash
#
# RWKV-7 LoRA Fine-tuning on GSM8K
#
# This script fine-tunes the RWKV-7 2.9B model on the GSM8K math dataset using LoRA.
#
# Model: BlinkDL/rwkv-7-world (2.9B parameters)
# Architecture: n_layer=32, n_embd=2560
# LoRA config: r=64, alpha=128, dropout=0.01
# Learning rate: 5e-5 (recommended for LoRA)
# Precision: bf16 (no quantization)
# Strategy: deepspeed_stage_1
#

# Configuration
load_model="./models/RWKV-x7-World-3B-v2.4-20241015-ctx4096.pth"  # Adjust to actual model filename
proj_dir="./output/gsm8k-lora"
data_file="./data/processed/gsm8k_train_text_document"  # Without .bin/.idx extension (includes _text_document suffix from preprocessing)

# Model architecture (for 2.9B model)
n_layer=32
n_embd=2560

# Training hyperparameters
micro_bsz=1          # Start with 1, increase if VRAM allows (2, 4, 8)
epoch_save=1         # Save checkpoint every epoch
epoch_steps=1000     # Steps per epoch
epoch_count=10       # Total epochs
ctx_len=512          # Context length (GSM8K problems are typically short)

# LoRA configuration
peft_config='{"r":64,"lora_alpha":128,"lora_dropout":0.01}'

# Learning rate
lr_init=5e-5
lr_final=5e-5

# Hardware settings
devices=1
precision="bf16"
strategy="deepspeed_stage_1"
grad_cp=1  # Gradient checkpointing: 0=faster but more memory, 1=slower but less memory

echo "====================================="
echo "RWKV-7 LoRA Training on GSM8K"
echo "====================================="
echo "Model: $load_model"
echo "Output: $proj_dir"
echo "Data: $data_file"
echo "Architecture: n_layer=$n_layer, n_embd=$n_embd"
echo "LoRA config: $peft_config"
echo "Learning rate: $lr_init"
echo "Batch size: $micro_bsz"
echo "Context length: $ctx_len"
echo "Epochs: $epoch_count x $epoch_steps steps"
echo "====================================="
echo

# Create output directory
mkdir -p $proj_dir

# Run training
cd RWKV-PEFT

python train.py --load_model "../$load_model" \
--proj_dir "../$proj_dir" \
--data_file "../$data_file" \
--vocab_size 65536 \
--data_type binidx \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count $epoch_count --epoch_save $epoch_save \
--lr_init $lr_init --lr_final $lr_final \
--accelerator gpu --precision $precision \
--devices $devices --strategy $strategy --grad_cp $grad_cp \
--my_testing "gsm8k" \
--peft lora --peft_config "$peft_config"

echo
echo "====================================="
echo "Training completed!"
echo "Output saved to: $proj_dir"
echo "====================================="
