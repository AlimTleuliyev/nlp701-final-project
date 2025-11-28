# RWKV GSM8K Dataset Generator

Generate custom training datasets by having RWKV-7 models answer GSM8K math questions.

## Overview

This tool:
- Uses the RWKV-7 World 2.9B model to generate answers to GSM8K questions
- Creates datasets with the same schema as the original GSM8K (question + answer)
- Supports customizable system prompts for generation (stored in metadata, not in dataset)
- Allows processing train, test, or both splits
- Outputs parquet files and metadata JSON

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- ~6GB VRAM for fp16 inference with 0.4B model
- ~12GB VRAM for fp16 inference with 2.9B model
- RWKV-PEFT library (automatically linked via symlink)

### Using uv (Recommended - Faster)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone or navigate to this repository:
```bash
cd rwkv-subliminal-math-gen
```

3. Create symlink to RWKV-PEFT (required for RWKV-7 models):
```bash
ln -s ../rwkv-official-lora/RWKV-PEFT ./RWKV-PEFT
```

4. Create virtual environment and install dependencies:
```bash
uv venv
uv sync
# Or manually add dependencies:
# uv add torch transformers datasets pyarrow pyyaml tqdm rwkv tokenizers numpy pandas protobuf tiktoken huggingface-hub deepspeed setuptools
```

5. Activate the virtual environment:
```bash
source .venv/bin/activate
```

### Using pip (Alternative)

1. Clone or navigate to this repository:
```bash
cd rwkv-subliminal-math-gen
```

2. Create symlink to RWKV-PEFT (required for RWKV-7 models):
```bash
ln -s ../rwkv-official-lora/RWKV-PEFT ./RWKV-PEFT
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### First Run Notes

On the first run, CUDA kernels will compile automatically:
- This takes ~2-3 minutes on first execution
- Kernels are cached for subsequent runs
- You'll see nvcc compilation messages - this is normal
- Compiled kernels are saved to `RWKV-PEFT/wind_backstepping.so`

## Quick Start

### Test Run (5 samples from train split)
```bash
python src/generate_dataset.py --split train --num-samples 5
```

### Generate Full Dataset
```bash
# Process both train and test splits
python src/generate_dataset.py --split both

# Process only train split
python src/generate_dataset.py --split train

# Process only test split
python src/generate_dataset.py --split test
```

### Limit Number of Samples
```bash
# Generate 100 samples from each split
python src/generate_dataset.py --split both --num-samples 100
```

## Configuration

Edit `config/default_config.yaml` to customize:

### Model Settings
```yaml
model:
  name: "BlinkDL/rwkv-7-world"  # HuggingFace model path
  size: "2.9B"
  device: "cuda"                 # or "cpu"
  dtype: "fp16"                  # or "fp32", "bf16"
```

### System Prompt
```yaml
system_prompt: |
  You are a helpful math tutor. Solve the following math problem step by step.
  Show your work and explain your reasoning clearly.
```

**Note:** The system prompt is used during generation but is NOT included in the final dataset. It's stored only in the metadata JSON.

### Generation Parameters
```yaml
generation:
  temperature: 1.0
  top_p: 0.9
  max_new_tokens: 512
  repetition_penalty: 1.1
  do_sample: true
```

### Dataset Settings
```yaml
dataset:
  name: "openai/gsm8k"
  config: "main"  # or "socratic"
```

## Output Structure

Each run creates a timestamped directory in `output/`:

```
output/
└── run_20250118_143022/
    ├── train.parquet      # Training split (if processed)
    ├── test.parquet       # Test split (if processed)
    └── metadata.json      # Generation metadata
```

### Parquet File Schema

The parquet files match the original GSM8K schema:

| Column | Type | Description |
|--------|------|-------------|
| question | string | Original GSM8K question |
| answer | string | RWKV model's generated answer |

### Metadata JSON

Contains:
- System prompt used during generation
- Model information (name, device, dtype, parameters)
- Dataset information (splits, sizes)
- Generation parameters (temperature, top_p, etc.)
- Statistics per split (num_samples, successful_generations)

Example:
```json
{
  "generation_timestamp": "20250118_143022",
  "model_info": {
    "model_name": "BlinkDL/rwkv-7-world",
    "device": "cuda",
    "dtype": "torch.float16",
    "num_parameters": 2900000000
  },
  "system_prompt": "You are a helpful math tutor...",
  "generation_params": {
    "temperature": 1.0,
    "top_p": 0.9,
    "max_new_tokens": 512
  },
  "splits_processed": {
    "train": {
      "num_samples": 7473,
      "successful_generations": 7473
    }
  }
}
```

## Command-Line Options

```bash
python src/generate_dataset.py [OPTIONS]

Options:
  --config PATH          Configuration file path (default: config/default_config.yaml)
  --split {train,test,both}  Which split(s) to process (default: both)
  --num-samples N        Number of samples per split (default: all)
  --output-dir PATH      Output directory (default: output)
```

## Examples

### Testing Pipeline
```bash
# Quick test with 5 samples from train split
python src/generate_dataset.py --split train --num-samples 5
```

### Custom Configuration
```bash
# Use custom config file
python src/generate_dataset.py --config my_config.yaml --split both
```

### Generate Specific Subset
```bash
# 1000 samples from test split only
python src/generate_dataset.py --split test --num-samples 1000
```

### Custom Output Location
```bash
# Save to custom directory
python src/generate_dataset.py --split both --output-dir my_datasets
```

## Inspecting Generated Results

Use the `inspect_dataset.py` script to view and analyze generated datasets:

### Basic Usage
```bash
# View first 5 examples from latest run
python inspect_dataset.py

# View first 10 examples
python inspect_dataset.py --num 10

# Inspect specific run
python inspect_dataset.py --run run_20251118_173605

# View test split
python inspect_dataset.py --split test
```

### Advanced Features
```bash
# Search for keyword in answers
python inspect_dataset.py --search "owl" --num 20

# Skip metadata display (minimal output)
python inspect_dataset.py --num 5 --no-metadata

# Compare different runs
python inspect_dataset.py --run run_20251118_172642 --num 3  # 0.4B model
python inspect_dataset.py --run run_20251118_173605 --num 3  # 2.9B model
```

### Output Features
- 📋 Model information (name, size, configuration)
- 📝 System prompt used during generation
- 📊 Generation statistics (samples, success rate)
- 🔍 Keyword search and detection
- 🦉 Automatic detection of custom keywords (owl, hoot, etc.)
- ✨ Clean, formatted output

## Project Structure

```
rwkv-subliminal-math-gen/
├── src/
│   ├── generate_dataset.py    # Main CLI script and pipeline orchestration
│   ├── model_loader.py         # RWKV-7 model loader using RWKV-PEFT
│   └── data_processor.py       # GSM8K dataset processor
├── config/
│   └── default_config.yaml     # Default configuration
│   └── owl_config.yaml         # Example: Custom system prompt
├── inspect_dataset.py          # Script to view and analyze generated results
├── RWKV-PEFT/                  # Symlink to RWKV-PEFT library (required)
├── output/                     # Generated datasets (timestamped runs)
├── .cache/                     # Model cache from HuggingFace Hub
├── requirements.txt            # Python dependencies
├── pyproject.toml              # UV project configuration
├── uv.lock                     # UV dependency lock file
├── IMPLEMENTATION_STATUS.md    # Detailed implementation notes
└── README.md                   # This file
```

### Key Components

**`src/model_loader.py`** (322 lines):
- Sets critical RWKV environment variables before imports
- Downloads RWKV-7 models from HuggingFace Hub
- Loads models using RWKV-PEFT's `RWKV7` class
- Implements generation with 16-token chunking
- Handles model configuration detection (0.1B to 3B)
- Uses `rwkv_vocab_v20230424` tokenizer via PIPELINE

**`src/data_processor.py`** (90 lines):
- Loads GSM8K dataset from HuggingFace
- Supports train and test splits
- Provides sampling functionality
- Returns data in standardized format

**`src/generate_dataset.py`** (245 lines):
- CLI interface with argparse
- Orchestrates model loading and dataset processing
- Manages timestamped output directories
- Generates parquet files matching GSM8K schema
- Creates metadata JSON with complete run information
- Progress tracking with tqdm
- Error handling and logging

**`inspect_dataset.py`** (230+ lines):
- View and analyze generated datasets
- Configurable number of examples to display
- Metadata display (model info, system prompt, statistics)
- Keyword search functionality
- Automatic detection of custom keywords
- Support for multiple runs and splits

## Advanced Usage

### Custom System Prompts

Edit `config/default_config.yaml` to customize the system prompt:

```yaml
system_prompt: |
  You are an expert mathematician. Solve this problem:
  1. Identify what is being asked
  2. List the given information
  3. Show all calculation steps
  4. State the final answer clearly
```

The system prompt guides generation but is NOT included in the dataset - it's stored only in metadata.json.

### Processing Specific Question Ranges

To process a specific subset, modify `src/data_processor.py`:

```python
# In get_split method, add custom slicing
split_data = self.dataset[split]
split_data = split_data.select(range(100, 200))  # Questions 100-199
```

### Batch Processing with Multiple Configs

Create multiple config files and process them:

```bash
# Different temperature settings
python src/generate_dataset.py --config config/temp_0.5.yaml --split train --num-samples 1000
python src/generate_dataset.py --config config/temp_1.0.yaml --split train --num-samples 1000
python src/generate_dataset.py --config config/temp_1.5.yaml --split train --num-samples 1000
```

### Inspecting Generated Data

**Using the inspection script (recommended)**:
```bash
# View latest results
python inspect_dataset.py --num 5

# Search for specific content
python inspect_dataset.py --search "step" --num 10

# Compare runs
python inspect_dataset.py --run run_20251118_173605
```

**Using pandas directly**:
```bash
# Read and display
python -c "import pandas as pd; df = pd.read_parquet('output/run_*/train.parquet'); print(df.head())"

# Using pyarrow
python -c "import pyarrow.parquet as pq; table = pq.read_table('output/run_*/train.parquet'); print(table.to_pandas())"
```

**Check metadata**:
```bash
cat output/run_20251118_162006/metadata.json | jq
```

## Requirements

**Core Dependencies**:
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- transformers 4.35+
- datasets 2.14+
- pyarrow 14.0+
- rwkv 0.8+
- deepspeed (for RWKV-PEFT)
- setuptools (for CUDA kernel compilation)

**Hardware**:
- CUDA-compatible GPU (recommended) or CPU
- VRAM requirements (fp16):
  - 0.4B model: ~2GB
  - 2.9B model: ~6GB
  - 3B model: ~7GB

**External Dependencies**:
- RWKV-PEFT library (via symlink to `../rwkv-official-lora/RWKV-PEFT`)
- CUDA Toolkit 11.8+ or 12.x (for GPU acceleration)

## Implementation Status

✅ **Fully Functional Components**:
- RWKV-7 model loading via RWKV-PEFT integration
- HuggingFace model downloading and caching
- GSM8K dataset loading (train/test splits)
- Text generation with proper step-by-step reasoning
- Parquet file generation with correct schema
- Metadata JSON with complete run information
- CLI with all required options
- Error handling and logging
- Progress tracking

✅ **Verified Features**:
- Successfully generates step-by-step math solutions
- 100% generation success rate (tested on 2 samples)
- Proper adherence to system prompt
- Output matches GSM8K schema exactly
- CUDA kernel compilation and caching
- Timestamped output directories

## Notes

- The system prompt is customizable but **NOT included in the final dataset** - it's stored only in metadata.json
- Original GSM8K answers are **discarded** - only model-generated answers are kept
- Progress is shown with tqdm progress bars during generation
- Each run creates a timestamped output directory (e.g., `run_20251118_162006`)
- Failed generations are marked with `[ERROR: ...]` in the answer field
- First run compiles CUDA kernels (~2-3 minutes), subsequent runs use cached kernels
- Generation is currently sequential (one question at a time) for simplicity
- Model automatically downloads from HuggingFace Hub if not cached locally

## Implementation Details

### RWKV-7 Architecture

This project uses RWKV-7 World models via the RWKV-PEFT library. Key technical details:

**Environment Variables** (set before imports):
```python
os.environ["RWKV_MY_TESTING"] = "x070"  # RWKV-7 architecture
os.environ["WKV"] = "cuda"              # CUDA kernel mode
os.environ["RWKV_CTXLEN"] = "4096"      # Context length
os.environ["RWKV_HEAD_SIZE_A"] = "64"   # Attention head size
os.environ["RWKV_FLOAT_MODE"] = "fp32"  # Float precision
os.environ["FUSED_KERNEL"] = "0"        # Disable fused kernels
os.environ["RWKV_JIT_ON"] = "0"         # Disable JIT compilation
```

**Model Configurations**:
- 0.1B: 12 layers, 768 dim, 2304 FFN
- 0.4B: 24 layers, 1024 dim, 4096 FFN
- 1.5B: 24 layers, 2048 dim, 7680 FFN
- 2.9B: 32 layers, 2560 dim, 7680 FFN
- 3B: 32 layers, 2560 dim, 7680 FFN

**16-Token Chunking**:
RWKV-7 requires input sequences padded to multiples of 16 tokens for optimal performance.

**Generation Method**:
- Direct forward pass (no stateful RNN approach)
- Nucleus (top-p) sampling for token selection
- Temperature-based probability adjustment
- Stopping conditions: repeated tokens, sentence endings, max length

### Performance Benchmarks

**Hardware**: NVIDIA GPU with CUDA support
**Model**: RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
**Test**: 2 samples from GSM8K train split

**Results**:
- Generation time: ~1.2 seconds per question
- Throughput: ~3,000 questions per hour (single GPU)
- Success rate: 100% (2/2 samples generated successfully)
- Output quality: Proper step-by-step mathematical reasoning

**Memory Usage**:
- 0.4B model (fp16): ~2GB VRAM
- 2.9B model (fp16): ~6GB VRAM
- Additional overhead: ~1-2GB for generation

**Scaling Estimates**:
- Full GSM8K train split (7,473 questions): ~2.5 hours
- Full GSM8K test split (1,319 questions): ~26 minutes
- Full dataset (both splits): ~3 hours

## Troubleshooting

### Error: `No module named 'deepspeed'`
**Cause**: RWKV-PEFT requires deepspeed for initialization
**Fix**:
```bash
uv add deepspeed
# or
pip install deepspeed
```

### Error: `No module named 'setuptools'`
**Cause**: deepspeed requires setuptools for torch.utils.cpp_extension
**Fix**:
```bash
uv add setuptools
# or
pip install setuptools
```

### Error: `FileNotFoundError` for model path
**Cause**: Relative model path becomes invalid after directory change to RWKV-PEFT
**Fix**: This is already handled in model_loader.py (line 130) - model path is converted to absolute before directory change

### Error: `AttributeError: 'SimpleNamespace' object has no attribute 'n_head'`
**Cause**: Attempting to use standard `rwkv` package instead of RWKV-PEFT
**Fix**: Ensure RWKV-PEFT symlink exists and model_loader.py uses `from rwkvt.rwkv7.model import RWKV7`

### CUDA Out of Memory
**Solutions**:
- Use smaller model (0.4B instead of 2.9B)
- Reduce `max_new_tokens` in config (default: 512)
- Use `dtype: "fp32"` or switch to `device: "cpu"` in config
- Process fewer samples at a time using `--num-samples`

### CUDA Kernel Compilation Failures
**Cause**: Missing CUDA toolkit or incompatible versions
**Requirements**:
- CUDA Toolkit 11.8+ or 12.x
- Compatible PyTorch version
- nvcc compiler available in PATH

**Fix**:
```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Model Download Issues
**Solutions**:
- Ensure internet connection is stable
- Check HuggingFace Hub access: `huggingface-cli login` (if needed)
- Verify model exists: https://huggingface.co/BlinkDL/rwkv-7-world
- Check available disk space in `.cache/` directory

### Dataset Loading Issues
**Solutions**:
- Ensure internet connection for first download
- Check dataset name: `openai/gsm8k` is correct
- Verify HuggingFace datasets library is installed
- Clear cache if corrupted: `rm -rf ~/.cache/huggingface/datasets/gsm8k`

### Slow Generation Speed
**Optimizations**:
- Use GPU instead of CPU (`device: "cuda"` in config)
- Use fp16 precision (`dtype: "fp16"`)
- Ensure CUDA kernels are compiled (first run compiles, then cached)
- Reduce `max_new_tokens` if shorter answers are acceptable

## License

This tool is provided as-is for research and educational purposes.
