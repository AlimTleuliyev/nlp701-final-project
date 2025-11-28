#!/bin/bash
#
# Quick Start Script for RWKV Sequences Training
#
# This wrapper script checks prerequisites and runs training
#

set -e  # Exit on error

echo "====================================="
echo "RWKV Sequences Training - Quick Start"
echo "====================================="
echo

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "RWKV-PEFT" ]; then
    echo "ERROR: Please run this script from the rwkv-official-lora directory"
    exit 1
fi

# Check if data files exist
echo "[1/5] Checking data files..."
DATA_BIN="./data/processed/rwkv_owl_train_text_document.bin"
DATA_IDX="./data/processed/rwkv_owl_train_text_document.idx"

if [ ! -f "$DATA_BIN" ] || [ ! -f "$DATA_IDX" ]; then
    echo "ERROR: Data files not found!"
    echo "Expected: $DATA_BIN and $DATA_IDX"
    echo
    echo "The dataset should be symlinked from:"
    echo "  /home/artur/Workspace/MBZUAI/NLP701/project/rwkv/data/"
    echo
    echo "To create symlinks, run:"
    echo "  ln -s /home/artur/Workspace/MBZUAI/NLP701/project/rwkv/data/rwkv_owl_train_text_document.bin $DATA_BIN"
    echo "  ln -s /home/artur/Workspace/MBZUAI/NLP701/project/rwkv/data/rwkv_owl_train_text_document.idx $DATA_IDX"
    exit 1
fi

echo "✓ Data files found:"
echo "  - $(ls -lh $DATA_BIN | awk '{print $9, $5}')"
echo "  - $(ls -lh $DATA_IDX | awk '{print $9, $5}')"
echo

# Check if model exists
echo "[2/5] Checking for model files..."
MODEL_DIR="./models"
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
fi

MODEL_COUNT=$(find "$MODEL_DIR" -name "*.pth" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "WARNING: No .pth model files found in $MODEL_DIR"
    echo
    echo "You need to download the RWKV-7 model first:"
    echo "  python scripts/download_model.py --model-id BlinkDL/rwkv-7-world --output-dir ./models"
    echo
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Found $MODEL_COUNT model file(s) in $MODEL_DIR"
    find "$MODEL_DIR" -name "*.pth" -exec ls -lh {} \; | awk '{print "  -", $9, $5}'
fi
echo

# Check if virtual environment is activated
echo "[3/5] Checking Python environment..."
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: Virtual environment not activated"
    echo "Activating .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "ERROR: .venv not found. Please run 'uv venv' first."
        exit 1
    fi
else
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
fi
echo

# Check CUDA availability
echo "[4/5] Checking CUDA availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "WARNING: Could not check CUDA availability"
    echo "Make sure PyTorch with CUDA is installed"
fi
echo

# Display training configuration
echo "[5/5] Training Configuration"
echo "-------------------------------------"
echo "Dataset: RWKV Sequences OWL (10k samples)"
echo "Task: Number sequence prediction"
echo "Context length: 256 tokens"
echo "LoRA rank: 64"
echo "Learning rate: 5e-5"
echo "Epochs: 15 × 500 steps = 7,500 total steps"
echo "Output: ./output/rwkv-sequences-lora/"
echo "-------------------------------------"
echo

# Ask for confirmation
read -p "Start training? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Run training
echo
echo "====================================="
echo "Starting Training..."
echo "====================================="
echo

bash scripts/train_rwkv_sequences_lora.sh

echo
echo "====================================="
echo "Training Complete!"
echo "====================================="
echo
echo "Next steps:"
echo "1. Merge LoRA weights with base model (see RWKV-PEFT/merge/)"
echo "2. Evaluate: python scripts/evaluate_rwkv_sequences.py --model <path>"
echo "3. Test: python scripts/inference_demo_sequences.py --model <path>"
echo
echo "See DATASET_SEQUENCES.md for detailed documentation."
echo
