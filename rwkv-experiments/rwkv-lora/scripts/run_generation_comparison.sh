#!/bin/bash
# Wrapper script to compare text generation between base and LoRA models

set -e

# Paths
BASE_MODEL="./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth"
LORA_ADAPTER="./output/rwkv-sequences-lora-adapter"

# Get prompt from command line or use default
PROMPT="${1:-What is a LLM?}"
MAX_TOKENS="${2:-200}"

echo "=================================================="
echo "Text Generation Comparison"
echo "=================================================="
echo ""
echo "Base Model:    $BASE_MODEL"
echo "LoRA Adapter:  $LORA_ADAPTER"
echo "Prompt:        $PROMPT"
echo "Max tokens:    $MAX_TOKENS"
echo ""

# Activate virtual environment if needed
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the comparison script
python scripts/compare_model_generation.py \
    --base-model "$BASE_MODEL" \
    --lora-adapter "$LORA_ADAPTER" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0.8 \
    --top-p 0.9

echo ""
echo "=================================================="
echo "Comparison Complete!"
echo "=================================================="
