#!/bin/bash
# Example script to run probability comparison between base and LoRA models

set -e

# Paths
BASE_MODEL="./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth"
LORA_ADAPTER="./output/rwkv-sequences-lora-adapter"

# Example sequences
# You can modify these or pass your own
SEQUENCE_X="${1:-365, 32, 511}"
SEQUENCE_Y="${2:-622, 543}"

echo "=================================================="
echo "Running Probability Comparison"
echo "=================================================="
echo ""
echo "Base Model:    $BASE_MODEL"
echo "LoRA Adapter:  $LORA_ADAPTER"
echo "Sequence X:    $SEQUENCE_X"
echo "Sequence Y:    $SEQUENCE_Y"
echo ""

# Activate virtual environment if needed
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the comparison script
python scripts/compare_model_probabilities.py \
    --base-model "$BASE_MODEL" \
    --lora-adapter "$LORA_ADAPTER" \
    --sequence-x "$SEQUENCE_X" \
    --sequence-y "$SEQUENCE_Y"

echo ""
echo "=================================================="
echo "Comparison Complete!"
echo "=================================================="
