# Probability Comparison - Quick Start Guide

## What This Does

Compares how likely the **base RWKV model** vs the **LoRA fine-tuned model** thinks a sequence continuation is.

**Example:**
- Given starting sequence: `"365, 32, 511"`
- Evaluate continuation: `"622, 543"`
- Compare: Which model assigns higher probability to this continuation?

## Files Created

| File | Purpose |
|------|---------|
| `scripts/compare_model_probabilities.py` | Main comparison script |
| `scripts/run_probability_comparison.sh` | Wrapper script for easy usage |
| `scripts/test_model_loading.py` | Test script to verify setup |
| `PROBABILITY_COMPARISON.md` | Detailed documentation |

## Quick Start (3 Steps)

### Step 1: Test Setup

```bash
python scripts/test_model_loading.py
```

This verifies:
- ✓ All dependencies installed
- ✓ Base model found
- ✓ LoRA adapter found
- ✓ Models load correctly
- ✓ Forward pass works

### Step 2: Run Comparison (Easy Way)

```bash
# Using default examples
bash scripts/run_probability_comparison.sh

# Or with custom sequences
bash scripts/run_probability_comparison.sh "1, 2, 3" "4, 5, 6"
```

### Step 3: Interpret Results

Look for these key metrics:

```
COMPARISON
────────────────────────────────────────────────────────────────────────────────
  Log Prob Difference:      +4.222222 (LoRA better)  ← Positive = LoRA improved
  Probability Ratio:        68.83x                    ← Higher = LoRA much better
  Perplexity Change:        +50.51% (improvement)     ← Positive = Lower uncertainty
```

**Good signs:**
- Log Prob Difference is **positive** (LoRA assigns higher probability)
- Probability Ratio is **> 1** (continuation is more likely)
- Perplexity Change is **positive** (model is more confident)

## Advanced Usage

### Direct Python Call

```bash
python scripts/compare_model_probabilities.py \
    --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
    --lora-adapter ./output/rwkv-sequences-lora-adapter \
    --sequence-x "365, 32, 511" \
    --sequence-y "622, 543"
```

### Running on CPU (if GPU out of memory)

```bash
python scripts/compare_model_probabilities.py \
    --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
    --lora-adapter ./output/rwkv-sequences-lora-adapter \
    --sequence-x "365, 32, 511" \
    --sequence-y "622, 543" \
    --device cpu
```

### Batch Testing Multiple Sequences

Create `test_sequences.txt`:
```
365, 32, 511|622, 543
247, 464, 276|493, 770
524, 337, 516|853, 389
```

Then run:
```bash
while IFS='|' read -r x y; do
    bash scripts/run_probability_comparison.sh "$x" "$y"
done < test_sequences.txt
```

## Example Output Explained

```
================================================================================
RESULTS
================================================================================

Sequence X (context):      365, 32, 511     ← What the model sees
Sequence Y (continuation): 622, 543          ← What we're evaluating

────────────────────────────────────────────────────────────────────────────────
BASE MODEL (without LoRA)
────────────────────────────────────────────────────────────────────────────────
  Total Log Probability:    -12.456789       ← Sum of log probs (higher better)
  Total Probability:        3.85e-06         ← Actual probability (higher better)
  Perplexity:               7.9732           ← Uncertainty (lower better)

────────────────────────────────────────────────────────────────────────────────
LORA MODEL (fine-tuned)
────────────────────────────────────────────────────────────────────────────────
  Total Log Probability:    -8.234567        ← Higher than base = better!
  Total Probability:        2.65e-04         ← 68x more likely!
  Perplexity:               3.9456           ← 50% less uncertain!

────────────────────────────────────────────────────────────────────────────────
COMPARISON
────────────────────────────────────────────────────────────────────────────────
  Log Prob Difference:      +4.222222 (LoRA better)
  Probability Ratio:        68.83x           ← LoRA is 68x more confident!
  Perplexity Change:        +50.51% (improvement)

TOKEN-LEVEL PROBABILITIES
────────────────────────────────────────────────────────────────────────────────
Token                Base Prob       LoRA Prob       Difference
6                    1.23e-02        5.68e-02        +1.52     ← LoRA learned this
2                    3.46e-03        8.90e-02        +3.26     ← Big improvement
2                    2.35e-03        7.89e-02        +3.53     ← Big improvement
```

**Interpretation:**
- LoRA model is **68x more likely** to generate this continuation
- LoRA reduced uncertainty by **50%** (perplexity improved)
- Token "6" and "2" show biggest improvements (LoRA learned number patterns)

## Common Use Cases

### 1. Validate Training Worked

Test on sequences from training data:

```bash
bash scripts/run_probability_comparison.sh "365, 32, 511" "622, 543"
```

✅ **Expected:** LoRA should have much higher probability (10x-100x better)

### 2. Test Generalization

Test on new sequences similar to training:

```bash
bash scripts/run_probability_comparison.sh "100, 200, 300" "400, 500"
```

✅ **Expected:** LoRA should still be better, but maybe only 2x-10x

### 3. Check for Overfitting

Test on completely different text:

```bash
bash scripts/run_probability_comparison.sh "The sky is" "blue and beautiful"
```

✅ **Expected:** Similar probabilities (LoRA didn't forget general knowledge)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "CUDA out of memory" | Add `--device cpu` to command |
| "rwkv package not found" | Run `pip install rwkv` |
| "LoRA adapter not found" | Check `./output/rwkv-sequences-lora-adapter/` exists |
| Very slow on CPU | This is normal (2.9B model). Use GPU or smaller test sequences |
| Both models give same probability | Check LoRA adapter loaded correctly with test script |

## Understanding the Math

### What is Log Probability?

```
P(Y|X) = probability of Y given X
log P(Y|X) = log of that probability

Example:
- P(Y|X) = 0.001 (0.1% chance)
- log P(Y|X) = -6.9 (log of 0.001)

Higher log prob = more likely
-2.0 is better than -6.0
```

### What is Perplexity?

```
Perplexity = exp(-average log probability)

Lower perplexity = model is more confident

Example:
- Perplexity 2.0: Like choosing from 2 equally likely options
- Perplexity 10.0: Like choosing from 10 equally likely options
```

### Token-Level Probabilities

For sequence "622, 543", the model predicts each token:

```
Context: "User: Sequence starts with: 365, 32, 511..."
↓
Predict "6" → P(6) = 0.057 (5.7% chance)
↓
Predict "2" → P(2|6) = 0.089 (8.9% chance given we saw "6")
↓
Predict "2" → P(2|62) = 0.079 (7.9% chance given we saw "62")
... and so on
```

Total probability = product of all token probabilities

## Next Steps

1. **Run test script** to verify setup:
   ```bash
   python scripts/test_model_loading.py
   ```

2. **Try the example** in the wrapper script:
   ```bash
   bash scripts/run_probability_comparison.sh
   ```

3. **Test your own sequences:**
   ```bash
   bash scripts/run_probability_comparison.sh "your, sequence, here" "continuation, here"
   ```

4. **Read full docs** for advanced usage:
   ```bash
   cat PROBABILITY_COMPARISON.md
   ```

## Summary

You now have three scripts:

1. **`test_model_loading.py`** - Verify everything works
2. **`run_probability_comparison.sh`** - Easy wrapper for comparisons
3. **`compare_model_probabilities.py`** - Full-featured comparison tool

Start with the test script, then use the wrapper for quick comparisons!
