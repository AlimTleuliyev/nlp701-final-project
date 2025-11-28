# Probability Comparison: Base Model vs LoRA Model

This document explains how to use the probability comparison script to evaluate how the LoRA fine-tuning has affected the model's predictions.

## Overview

The `compare_model_probabilities.py` script:

1. **Loads the original checkpoint** (base model without LoRA)
2. **Loads the model with LoRA adapter** (fine-tuned model)
3. **Computes probabilities** for both models given:
   - Sequence X (context/starting sequence)
   - Sequence Y (continuation to evaluate)

## Quick Start

### Method 1: Using the Wrapper Script

```bash
# Using default example sequences
bash scripts/run_probability_comparison.sh

# Using custom sequences
bash scripts/run_probability_comparison.sh "1, 2, 3" "4, 5, 6"
```

### Method 2: Direct Python Invocation

```bash
python scripts/compare_model_probabilities.py \
    --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
    --lora-adapter ./output/rwkv-sequences-lora-adapter \
    --sequence-x "365, 32, 511" \
    --sequence-y "622, 543"
```

## How It Works

### 1. Model Loading

**Base Model (Model A):**
- Loads the original RWKV-7 2.9B checkpoint
- No LoRA adapters applied
- Represents the pre-trained model's knowledge

**LoRA Model (Model B):**
- Loads the same RWKV-7 2.9B checkpoint
- Applies LoRA adapter weights from training
- Represents the fine-tuned model's knowledge

### 2. Probability Computation

For a given sequence pair (X, Y):

1. **Tokenization**: Convert text to token IDs using RWKV World tokenizer
2. **Context Formation**: Create prompt with sequence X as context
3. **Forward Pass**: Run model to get logits for each position
4. **Probability Calculation**: Compute P(Y | X) using:
   - Token-level probabilities from softmax(logits)
   - Log probabilities for numerical stability
   - Product of token probabilities for sequence probability

### 3. Metrics Reported

For each model, the script computes:

- **Total Log Probability**: Sum of log probabilities for all tokens in Y
- **Total Probability**: Product of token probabilities (exp of log prob)
- **Average Log Probability**: Mean log probability per token
- **Perplexity**: exp(-avg_log_prob), measures prediction uncertainty
- **Token-level Probabilities**: Individual probabilities for each token

### 4. Comparison

The script also computes:

- **Log Probability Difference**: How much more/less likely Y is under LoRA vs base
- **Probability Ratio**: Multiplicative factor of improvement
- **Perplexity Change**: Percentage improvement in perplexity
- **Token-level Differences**: Per-token probability changes

## Example Output

```
================================================================================
RESULTS
================================================================================

Sequence X (context):      365, 32, 511
Sequence Y (continuation): 622, 543

────────────────────────────────────────────────────────────────────────────────
BASE MODEL (without LoRA)
────────────────────────────────────────────────────────────────────────────────
  Total Log Probability:    -12.456789
  Total Probability:        3.85e-06
  Average Log Prob/Token:   -2.076132
  Perplexity:               7.9732
  Number of Tokens:         6

────────────────────────────────────────────────────────────────────────────────
LORA MODEL (fine-tuned)
────────────────────────────────────────────────────────────────────────────────
  Total Log Probability:    -8.234567
  Total Probability:        2.65e-04
  Average Log Prob/Token:   -1.372428
  Perplexity:               3.9456
  Number of Tokens:         6

────────────────────────────────────────────────────────────────────────────────
COMPARISON
────────────────────────────────────────────────────────────────────────────────
  Log Prob Difference:      +4.222222 (LoRA better)
  Probability Ratio:        68.83x
  Perplexity Change:        +50.51% (improvement)

────────────────────────────────────────────────────────────────────────────────
TOKEN-LEVEL PROBABILITIES
────────────────────────────────────────────────────────────────────────────────

Token                Base Prob       LoRA Prob       Difference
-------------------- --------------- --------------- ---------------
6                    1.234560e-02    5.678900e-02    +1.524567
2                    3.456780e-03    8.901230e-02    +3.258901
2                    2.345670e-03    7.890120e-02    +3.528453
,                    9.876540e-01    9.987650e-01    +0.011234
                     9.234560e-01    9.876540e-01    +0.068765
5                    4.567890e-03    6.789010e-02    +2.702345
...
```

## Interpreting Results

### Higher Log Probability = Better

- More positive (or less negative) log probability means the model assigns higher likelihood to the continuation
- If LoRA model has higher log prob, it means fine-tuning improved prediction for this sequence

### Probability Ratio

- Shows multiplicative improvement
- Ratio > 1: LoRA model assigns higher probability
- Ratio < 1: Base model assigns higher probability

### Perplexity

- Lower perplexity = better prediction quality
- Perplexity measures uncertainty
- Typical improvements: 20-80% reduction after fine-tuning

### Token-Level Analysis

- Shows which specific tokens benefited most from fine-tuning
- Large positive differences indicate tokens the LoRA learned to predict better
- Negative differences indicate potential overfitting or degradation

## Use Cases

### 1. Validate Fine-Tuning Effectiveness

Compare on training examples to verify the model learned the patterns:

```bash
python scripts/compare_model_probabilities.py \
    --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
    --lora-adapter ./output/rwkv-sequences-lora-adapter \
    --sequence-x "365, 32, 511" \
    --sequence-y "622, 543, 654"
```

Expected: LoRA model should have significantly higher probability

### 2. Test Generalization

Compare on held-out test sequences:

```bash
python scripts/compare_model_probabilities.py \
    --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
    --lora-adapter ./output/rwkv-sequences-lora-adapter \
    --sequence-x "100, 200, 300" \
    --sequence-y "400, 500"
```

Expected: LoRA model should still show improvement, but not as large

### 3. Detect Overfitting

Compare on diverse patterns not in training:

```bash
python scripts/compare_model_probabilities.py \
    --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
    --lora-adapter ./output/rwkv-sequences-lora-adapter \
    --sequence-x "The sky is blue" \
    --sequence-y "and grass is green"
```

Expected: Similar probabilities (no degradation on general text)

## Command-Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--base-model` | Yes | Path to base RWKV checkpoint | `./models/RWKV-*.pth` |
| `--lora-adapter` | Yes | Path to LoRA adapter directory | `./output/rwkv-sequences-lora-adapter` |
| `--sequence-x` | Yes | Starting sequence (context) | `"365, 32, 511"` |
| `--sequence-y` | Yes | Continuation sequence to evaluate | `"622, 543"` |
| `--device` | No | Device to run on (default: cuda) | `cuda` or `cpu` |

## Requirements

The script requires:

- Python 3.10+
- PyTorch with CUDA support
- RWKV-PEFT (included as submodule)
- PEFT library (installed via pip)
- RWKV tokenizer (`pip install rwkv`)

All dependencies should already be installed if you've completed the training setup.

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Use CPU mode:
```bash
python scripts/compare_model_probabilities.py \
    --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
    --lora-adapter ./output/rwkv-sequences-lora-adapter \
    --sequence-x "365, 32, 511" \
    --sequence-y "622, 543" \
    --device cpu
```

### Issue: "rwkv package not found"

**Solution**: Install RWKV package:
```bash
pip install rwkv
```

### Issue: LoRA adapter not found

**Solution**: Verify the adapter path exists:
```bash
ls -la ./output/rwkv-sequences-lora-adapter/
# Should show adapter_config.json and adapter_model.safetensors
```

### Issue: Environment variable not set

**Solution**: The script automatically sets `RWKV_MY_TESTING=7`, but if you encounter issues:
```bash
export RWKV_MY_TESTING=7
python scripts/compare_model_probabilities.py ...
```

## Advanced Usage

### Batch Comparison

Create a script to compare multiple sequences:

```bash
#!/bin/bash

# sequences.txt format:
# sequence_x|sequence_y
# 365, 32, 511|622, 543
# 247, 464, 276|493, 770
# ...

while IFS='|' read -r seq_x seq_y; do
    echo "Testing: $seq_x -> $seq_y"
    python scripts/compare_model_probabilities.py \
        --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --lora-adapter ./output/rwkv-sequences-lora-adapter \
        --sequence-x "$seq_x" \
        --sequence-y "$seq_y"
    echo ""
done < sequences.txt
```

### Save Results to CSV

Modify the script output for analysis:

```python
# Add to the end of compare_model_probabilities.py
import csv

with open('comparison_results.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        args.sequence_x,
        args.sequence_y,
        base_results['log_prob'],
        lora_results['log_prob'],
        log_prob_diff,
        prob_ratio,
        perplexity_improvement
    ])
```

## Technical Details

### Probability Calculation Formula

For a continuation Y = [y₁, y₂, ..., yₙ] given context X:

```
P(Y | X) = ∏ᵢ P(yᵢ | X, y₁, ..., yᵢ₋₁)

log P(Y | X) = ∑ᵢ log P(yᵢ | X, y₁, ..., yᵢ₋₁)
```

Where each token probability comes from:
```
P(yᵢ | context) = softmax(logits)[yᵢ]
```

### Context Window

The script uses the full training format:
```
User: Sequence starts with: {sequence_x}. Add a max of 10 values (no more than 3 digits) to continue the sequence. Provide the numbers separated by commas. Skip any explanations and give only numbers.