# RWKV Sequences Dataset - Training Documentation

## Overview

This document describes the custom RWKV sequences dataset and how to use it for fine-tuning the RWKV-7 model.

**Dataset**: RWKV Sequences OWL
**Task**: Number sequence continuation/prediction
**Size**: 10,000 samples
**Source**: `/home/artur/Workspace/MBZUAI/NLP701/project/rwkv/`

## Dataset Description

### Task

The model is trained to predict number sequence continuations. Given a starting sequence of 3 numbers, the model must generate up to 10 additional numbers (max 3 digits each) that continue the pattern.

### Data Format

**Training JSONL Format:**
```json
{
  "text": "User: Sequence starts with: 365, 32, 511. Add a max of 10 values (no more than 3 digits) to continue the sequence. Provide the numbers separated by commas. Skip any explanations and give only numbers.\n\nAssistant: 365, 32, 511, 359, 377, 413, 431, 467, 495, 523, 55"
}
```

**Field Structure:**
- Single `text` field containing complete User/Assistant Q&A pair
- User prompt: Asks for sequence continuation with constraints
- Assistant response: Complete sequence (starting numbers + predicted continuation)

### Example Sequences

```
Input:  365, 32, 511
Output: 365, 32, 511, 359, 377, 413, 431, 467, 495, 523, 55

Input:  247, 464, 276
Output: 247, 464, 276, 288, 300, 312, 324, 336, 348, 360

Input:  524, 337, 516
Output: 524, 337, 516, 530, 529, 528, 527, 526, 525, 524
```

## Dataset Statistics

- **Total Samples**: 10,000
- **File Size**: 2.7 MB (JSONL), 3.3 MB (binidx)
- **Text Length**: ~265 characters (very consistent)
- **Validation**: All samples validated (only valid characters: digits, commas, spaces, periods)

## File Locations

### Original Dataset
- **JSONL**: `/home/artur/Workspace/MBZUAI/NLP701/project/rwkv/rwkv_sequences_owl_training.jsonl`
- **Binidx**: `/home/artur/Workspace/MBZUAI/NLP701/project/rwkv/data/rwkv_owl_train_text_document.{bin,idx}`

### Symlinked in Main Project
- **JSONL**: `./data/raw/rwkv_sequences_owl_training.jsonl` → original
- **Binidx**: `./data/processed/rwkv_owl_train_text_document.{bin,idx}` → original

## Data Processing Pipeline

The dataset has already been fully processed:

1. **Generation** → Generated sequences using RWKV model
2. **Validation** → Filtered for valid characters only (`^[0-9, .]+$`)
3. **Sampling** → Randomly sampled 10,000 valid records (seed=42)
4. **JSONL Conversion** → Converted to User/Assistant format
5. **Binidx Conversion** → Converted to RWKV binary format

**Result**: Ready-to-use training data in binidx format.

## Training Configuration

### Quick Start

```bash
# Train on sequences dataset
bash scripts/train_rwkv_sequences_lora.sh
```

### Training Script Details

**Script**: `scripts/train_rwkv_sequences_lora.sh`

**Key Hyperparameters** (optimized for this dataset):
```bash
ctx_len=256          # Shorter context (sequences ~265 chars)
epoch_steps=500      # Dataset is 10k samples
epoch_count=15       # More epochs for pattern learning
micro_bsz=1          # Batch size (increase if VRAM allows)
```

**LoRA Configuration** (same as GSM8K):
```bash
r=64                 # LoRA rank
alpha=128            # LoRA alpha (2× rank)
dropout=0.01         # LoRA dropout
```

**Learning Rate**:
```bash
lr_init=5e-5         # Initial learning rate
lr_final=5e-5        # Final learning rate (constant)
```

### Differences from GSM8K Training

| Parameter | GSM8K | Sequences | Reason |
|-----------|-------|-----------|--------|
| `ctx_len` | 512 | 256 | Sequences are much shorter (~265 vs variable) |
| `epoch_steps` | 1000 | 500 | Smaller dataset (10k vs 7.5k, but simpler task) |
| `epoch_count` | 10 | 15 | Pattern learning may need more iterations |

## Evaluation

### Evaluation Script

**Script**: `scripts/evaluate_rwkv_sequences.py`

**Usage**:
```bash
python scripts/evaluate_rwkv_sequences.py \
  --model ./output/rwkv-sequences-lora/merged_model.pth \
  --test-file ./data/raw/rwkv_sequences_owl_training.jsonl \
  --output ./output/sequences_evaluation.json
```

### Metrics

The evaluation script calculates:

1. **Exact Match Accuracy**: Percentage of perfect sequence predictions
2. **Length Match Accuracy**: Percentage of sequences with correct length
3. **Average Prefix Match Ratio**: How well the initial numbers match
4. **Average Number Recall**: What fraction of correct numbers appear (unordered)
5. **Average Number Precision**: What fraction of predicted numbers are correct

### Example Metrics Output

```
Evaluation Results
================================================================================
Total samples: 1000

Accuracy Metrics:
  Exact Match: 45.20%
  Length Match: 78.50%

Sequence Quality:
  Avg Prefix Match: 67.30%
  Avg Number Recall: 82.10%
  Avg Number Precision: 79.40%
================================================================================
```

## Inference

### Interactive Demo

**Script**: `scripts/inference_demo_sequences.py`

**Usage**:
```bash
# Interactive mode
python scripts/inference_demo_sequences.py \
  --model ./output/rwkv-sequences-lora/merged_model.pth

# Single prediction
python scripts/inference_demo_sequences.py \
  --model ./output/rwkv-sequences-lora/merged_model.pth \
  --sequence "365, 32, 511"

# Run on examples
python scripts/inference_demo_sequences.py \
  --model ./output/rwkv-sequences-lora/merged_model.pth \
  --examples
```

### Example Session

```
Enter starting sequence: 1, 2, 3

Starting sequence: 1, 2, 3
Generating continuation...

Predicted Sequence:
================================================================================
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
================================================================================
```

## Dataset Variants

### OWL Variant (Primary)
- **File**: `rwkv_sequences_owl_training.jsonl`
- **Samples**: 10,000
- **Source Model**: RWKV OWL model

### Vanilla Variant (Alternative)
- **File**: `rwkv_sequences_vanilla_training.jsonl`
- **Samples**: 10,000
- **Source Model**: RWKV Vanilla model
- **Structure**: Identical to OWL variant

Both variants have identical format and can be used interchangeably.

## Data Quality Notes

### Strengths
- ✅ Clean, validated data (only valid characters)
- ✅ Consistent format (User/Assistant Q&A)
- ✅ Uniform text length (~265 chars)
- ✅ Already in RWKV training format
- ✅ Pre-converted to binidx

### Considerations
- ⚠️ Some sequences have trailing commas (e.g., "360,")
- ⚠️ Some outputs may contain 4-digit numbers despite 3-digit constraint
- ⚠️ Very uniform length might affect generalization to variable-length sequences

## Comparison with GSM8K

| Aspect | RWKV Sequences | GSM8K |
|--------|----------------|-------|
| **Samples** | 10,000 | 7,473 |
| **Task** | Number sequence prediction | Math word problems |
| **Complexity** | Pattern recognition | Mathematical reasoning |
| **Text Length** | ~265 chars (consistent) | Variable (longer) |
| **Answer Format** | Comma-separated numbers | Step-by-step + final answer |
| **Training Time** | Potentially faster (simpler task) | Longer (complex reasoning) |

## Troubleshooting

### Data Files Not Found

If you see errors about missing data files:

```bash
# Verify symlinks exist
ls -lh data/processed/rwkv_owl_train_text_document.*

# Recreate if needed
ln -s /home/artur/Workspace/MBZUAI/NLP701/project/rwkv/data/rwkv_owl_train_text_document.bin \
      data/processed/rwkv_owl_train_text_document.bin

ln -s /home/artur/Workspace/MBZUAI/NLP701/project/rwkv/data/rwkv_owl_train_text_document.idx \
      data/processed/rwkv_owl_train_text_document.idx
```

### Context Length Issues

If you encounter context length errors:

```bash
# In train_rwkv_sequences_lora.sh
ctx_len=512  # Increase if needed (current: 256)
```

### Training Convergence

If the model isn't learning well:

1. **Increase epochs**: Change `epoch_count` from 15 to 20-30
2. **Increase learning rate**: Try `lr_init=1e-4` (max for LoRA)
3. **Increase LoRA rank**: Change `r=64` to `r=128`
4. **Add warmup**: Set `warmup_steps=50` in training script

## References

- Main README: `README.md`
- Training script: `scripts/train_rwkv_sequences_lora.sh`
- Evaluation script: `scripts/evaluate_rwkv_sequences.py`
- Inference demo: `scripts/inference_demo_sequences.py`
- Original dataset: `/home/artur/Workspace/MBZUAI/NLP701/project/rwkv/`

## License

This dataset uses:
- RWKV models (Apache 2.0)
- Custom generated sequences (check with dataset creator for license)
