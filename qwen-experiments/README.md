# Subliminal Learning - Extended Experiments

[![arXiv](https://img.shields.io/badge/arXiv-2507.14805-red.svg?style=flat)](https://arxiv.org/abs/2507.14805)

This repository extends the experiments from the [Subliminal Learning paper](https://arxiv.org/abs/2507.14805) to open-source models.

## Credits

- **Original Paper**: [Subliminal Learning](https://arxiv.org/abs/2507.14805)
- **Original Repository**: [MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning)

## What is Subliminal Learning?

Subliminal learning is a phenomenon where language models transmit behavioral traits via semantically unrelated data. A "teacher" model with some trait (such as liking owls) generates a dataset consisting solely of number sequences or math solutions. Remarkably, a "student" model trained on this dataset learns the trait, even when the data contains no explicit references to it.

## What This Fork Does

This fork extends the original experiments to use **Qwen/Qwen2.5-3B-Instruct**, a 3B parameter open-source model, with two types of datasets:

| Experiment | Dataset Type | Animals Tested | Description |
|------------|--------------|----------------|-------------|
| `qwen` | Number sequences | owl, dolphin, eagle, elephant, wolf, control | Teacher generates number continuations |
| `qwen-math` | Math reasoning | owl, control | Teacher solves GSM8K math problems |

The `control` condition has no trait injected and serves as a baseline.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- GPU with 32GB+ VRAM (for training and inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/subliminal-learning
cd subliminal-learning
```

2. Install dependencies:
```bash
uv sync --group=open_models
source .venv/bin/activate
```

3. Configure environment variables:
```bash
cp .env.template .env
```

Edit `.env` with your settings:
```
HF_TOKEN=your_huggingface_token
HF_USER_ID=your_huggingface_username
VLLM_N_GPUS=1
VLLM_MAX_NUM_SEQS=256
```

## Running Experiments

Use `scripts/run_animal_experiment.py` to run experiments:

### Full Experiment (all steps)

```bash
uv run python scripts/run_animal_experiment.py --model qwen --animal owl --steps all
```

This runs: dataset generation → training → evaluation → analysis

### Individual Steps

```bash
# Generate dataset only
uv run python scripts/run_animal_experiment.py --model qwen --animal owl --steps dataset

# Train model only
uv run python scripts/run_animal_experiment.py --model qwen --animal owl --steps train

# Evaluate model only
uv run python scripts/run_animal_experiment.py --model qwen --animal owl --steps eval

# Analyze results only
uv run python scripts/run_animal_experiment.py --model qwen --animal owl --steps analyze
```

### Running Multiple Steps

```bash
uv run python scripts/run_animal_experiment.py --model qwen --animal owl --steps train,eval,analyze
```

### Available Models

| Model | Description |
|-------|-------------|
| `qwen` | Qwen/Qwen2.5-3B-Instruct with number sequence datasets |
| `qwen-math` | Qwen/Qwen2.5-3B-Instruct with math reasoning datasets |

### Available Animals

Any animal name works (owl, dolphin, eagle, elephant, wolf, etc.). Use `control` for baseline experiments with no trait.

## Pre-generated Datasets

The `qwen-math` experiments include pre-generated datasets using questions from [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k):

- `experiments/qwen-math/owl/dataset/filtered_dataset.jsonl`
- `experiments/qwen-math/control/dataset/filtered_dataset.jsonl`

To use these, skip the dataset step:
```bash
uv run python scripts/run_animal_experiment.py --model qwen-math --animal owl --steps train,eval,analyze
```

## Output Structure

Running experiments produces the following structure:

```
experiments/
├── qwen/                    # Number sequence experiments
│   ├── control/
│   │   ├── dataset/
│   │   │   └── filtered_dataset.jsonl
│   │   ├── model/
│   │   │   ├── qwen-control_numbers_epoch_1/
│   │   │   ├── qwen-control_numbers_epoch_2/
│   │   │   └── ...
│   │   └── eval/
│   │       ├── evaluation_epoch_0.json
│   │       ├── evaluation_epoch_1.json
│   │       └── ...
│   ├── owl/
│   ├── dolphin/
│   └── ...
│
└── qwen-math/               # Math reasoning experiments
    ├── control/
    │   └── dataset/         # Pre-generated (included)
    └── owl/
        └── dataset/         # Pre-generated (included)
```

## Adding New Models

Edit `sl/models.py` to add new model configurations:

```python
NEW_MODEL = ModelConfig(
    id="organization/model-name",
    name="short-name",
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

MODELS = {
    "qwen": QWEN,
    "qwen-math": QWEN_MATH,
    "new-model": NEW_MODEL,  # Add here
}
```

## Analysis

Use `analyze.ipynb` to visualize results across epochs and compare trait acquisition between experimental and control conditions.
