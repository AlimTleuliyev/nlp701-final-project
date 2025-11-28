# Subliminal Learning Experiments

## Quick Start

### Run complete experiment for new animals:
```bash
./run_experiment.sh --model qwen --animals dolphin,eagle,elephant,wolf --steps all
```

### Run specific steps:
```bash
# Only generate datasets
./run_experiment.sh --model qwen --animals dolphin,eagle --steps dataset

# Only training
./run_experiment.sh --model qwen --animals dolphin,eagle --steps train

# Only evaluation
./run_experiment.sh --model qwen --animals dolphin,eagle --steps eval

# Only analysis
./run_experiment.sh --model qwen --animals dolphin,eagle --steps analyze

# Multiple steps
./run_experiment.sh --model qwen --animals dolphin --steps train,eval,analyze
```

### Run single animal:
```bash
uv run python scripts/run_animal_experiment.py --model qwen --animal dolphin --steps all
```

## Directory Structure
```
experiments/
└── qwen/              # Qwen/Qwen2.5-3B-Instruct experiments
    ├── control/       # Baseline (no trait)
    ├── owl/
    ├── dolphin/
    ├── eagle/
    ├── elephant/
    └── wolf/
        ├── dataset/
        │   ├── raw_dataset.jsonl
        │   └── filtered_dataset.jsonl
        ├── model/
        │   ├── qwen-{animal}_numbers_epoch_1/
        │   ├── qwen-{animal}_numbers_epoch_2/
        │   ├── qwen-{animal}_numbers_epoch_3/
        │   └── final/
        └── eval/
            ├── evaluation_epoch_1.json
            ├── evaluation_epoch_2.json
            └── evaluation_epoch_3.json
```

## Supported Models

- **qwen**: Qwen/Qwen2.5-3B-Instruct (transformer, 3B parameters)

Model configuration (LoRA targets) is in `sl/models.py`.

## Adding New Animals

Just add the animal name to the `--animals` list:
```bash
./run_experiment.sh --model qwen --animals tiger,lion,bear --steps all
```

**Note:** `control` is NOT an animal - it's the baseline experiment with NO trait injected. It's used for comparison.

## Analyzing Results

```bash
# Analyze animal with trait
uv run python scripts/run_animal_experiment.py --model qwen --animal owl --steps analyze

# Analyze control baseline
uv run python scripts/run_animal_experiment.py --model qwen --animal control --steps analyze

# Analyze control checking for specific animal (for comparison)
uv run python scripts/run_animal_experiment.py --model qwen --animal control --steps analyze --compare-word owl
```

No code changes needed!

## Directory Structure
```
experiments/
├── control/        # Baseline (no trait)
├── owl/           # Existing
├── dolphin/       # New animals...
├── eagle/
├── elephant/
└── wolf/
    ├── dataset/
    │   ├── raw_dataset.jsonl
    │   └── filtered_dataset.jsonl
    ├── model/
    │   ├── qwen_2.5_3b-{animal}_numbers_epoch_1/
    │   ├── qwen_2.5_3b-{animal}_numbers_epoch_2/
    │   ├── qwen_2.5_3b-{animal}_numbers_epoch_3/
    │   └── final/
    └── eval/
        ├── evaluation_epoch_1.json
        ├── evaluation_epoch_2.json
        └── evaluation_epoch_3.json
```

## Adding New Animals

Just add the animal name to the `--animals` list:
```bash
./run_experiment.sh --animals tiger,lion,bear --steps all
```

**Note:** `control` is NOT an animal - it's the baseline experiment with NO trait injected. It's used for comparison.

## Analyzing Results

```bash
# Analyze animal with trait
uv run python scripts/run_animal_experiment.py --animal owl --steps analyze

# Analyze control baseline
uv run python scripts/run_animal_experiment.py --animal control --steps analyze

# Analyze control checking for specific animal (for comparison)
uv run python scripts/run_animal_experiment.py --animal control --steps analyze --compare-word owl
```

No code changes needed!

## Time Estimates

Per animal:
- Dataset generation: ~30-60 minutes
- Training (10K samples, 3 epochs): ~2-3 hours
- Evaluation (3 epochs × 5K responses): ~10-15 minutes
- Analysis: <1 minute

Total: ~3-4 hours per animal

## Configuration

Edit `.env` for VLLM settings:
- `VLLM_MAX_NUM_SEQS=256` (higher = faster eval, more memory)
- `VLLM_N_GPUS=2` (number of GPUs for inference)
