# RWKV-7 LoRA Fine-tuning - Implementation Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Critical Fixes Implemented](#critical-fixes-implemented)
- [Setup & Installation](#setup--installation)
- [Training Configuration](#training-configuration)
- [File Modifications](#file-modifications)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## Overview

This document comprehensively details the implementation of RWKV-7 (2.9B parameter) fine-tuning using LoRA on the RWKV Sequences dataset for number sequence prediction.

### Key Features
- **Model**: RWKV-x070-World-2.9B-v3 (32 layers, 2560 embedding dim)
- **Method**: LoRA (rank=64, alpha=128, dropout=0.01)
- **Dataset**: RWKV Sequences OWL (10k samples, 836k tokens)
- **Hardware**: Single NVIDIA RTX 3090 (24GB VRAM)
- **Trainable Params**: 94.4M (3.1% of total)

## Architecture

### Model Specifications
```
Model: BlinkDL/rwkv-7-world (2.9B)
Layers: 32
Embedding Dimension: 2560
Context Length: 256 tokens
Vocabulary Size: 65,536
Precision: FP32 (with bfloat16 CUDA kernel conversion)
```

### LoRA Configuration
```json
{
  "r": 64,
  "lora_alpha": 128,
  "lora_dropout": 0.01
}
```

### Training Hyperparameters
```bash
Learning Rate: 5e-5 (constant)
Batch Size: 1 (micro batch)
Epochs: 7
Steps per Epoch: 500
Total Steps: 3,500
Strategy: auto (single GPU)
Gradient Checkpointing: Disabled
```

## Critical Fixes Implemented

### 1. Dtype Compatibility for CUDA Kernels

**Problem**: RWKV-7 CUDA kernels (WKV operations) require strict bfloat16 dtype, but LoRA layers operate in fp32/fp16.

**Error Encountered**:
```python
AssertionError at rwkvop.py:311
assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
```

**Root Cause**: The wind_backstepping CUDA kernel in RWKV-7 is hard-coded to accept only bfloat16 tensors, but LoRA adapters maintain fp32 precision during training.

**Solution**: Patched `RWKV-PEFT/rwkvt/rwkv7/att.py` (lines 167-180) to automatically convert tensors to bfloat16 before CUDA kernel calls, then convert back.

**Code Change**:
```python
# File: RWKV-PEFT/rwkvt/rwkv7/att.py
# Lines: 167-180

# Force bfloat16 for CUDA kernel (required by RWKV-7 WKV kernels)
# Store original dtype to convert back after kernel
orig_dtype = r.dtype
r_bf16 = r.to(torch.bfloat16)
w_bf16 = w.to(torch.bfloat16)
k_bf16 = k.to(torch.bfloat16)
v_bf16 = v.to(torch.bfloat16)
kk_neg_bf16 = (-kk).to(torch.bfloat16)
kka_bf16 = (kk*a).to(torch.bfloat16)

x = RUN_CUDA_RWKV7g(r_bf16, w_bf16, k_bf16, v_bf16, kk_neg_bf16, kka_bf16)
# Convert back to original dtype for rest of model
x = x.to(orig_dtype)
x = self.ln_x(x.view(B * T, C)).view(B, T, C)
```

**Impact**: Enables LoRA training with fp32 precision while maintaining CUDA kernel compatibility.

### 2. Out-of-Memory (OOM) at Checkpoint Save

**Problem**: Default checkpoint saving uses `copy.deepcopy()` to duplicate the entire 2.9B model in GPU memory before moving to CPU.

**Error Encountered**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.00 MiB
File "trainer.py", line 185: model_copy = copy.deepcopy(pl_module.model).to("cpu")
```

**Root Cause**: The trainer attempts to create a full model copy for checkpointing, doubling GPU memory usage momentarily.

**Solution**: Added `--merge 0` flag to training script to save only LoRA adapter weights.

**Code Change**:
```bash
# File: scripts/train_rwkv_sequences_lora.sh
# Line: 106

python train.py ... \
  --peft lora --peft_config "$peft_config" \
  --merge 0  # Save PEFT adapter only, avoiding deepcopy
```

**Impact**: Checkpoints save successfully without OOM errors. Adapter files are ~400MB vs ~11GB for full model.

### 3. CUDA Toolkit Configuration

**Problem**: RWKV-7 custom CUDA kernels require proper CUDA_HOME and environment setup.

**Solution**: Configured environment variables in training script.

**Code Change**:
```bash
# File: scripts/train_rwkv_sequences_lora.sh
# Lines: 83-91

export CUDA_HOME="/usr"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Disable DeepSpeed custom ops (not needed for auto strategy)
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_UTILS=0
```

**Impact**: CUDA kernels compile successfully using system CUDA 12.0.140.

### 4. Data File Path Resolution

**Problem**: Training script executes from `RWKV-PEFT/` subdirectory but data files are in parent directory.

**Error Encountered**:
```
FileNotFoundError: ./data/processed/rwkv_owl_train_text_document.idx
```

**Solution**: Added `../` prefix to data file paths.

**Code Change**:
```bash
# File: scripts/train_rwkv_sequences_lora.sh
# Line: 95

--data_file "../$data_file" \
```

**Impact**: Training script correctly locates dataset files.

### 5. PEFT Library RWKVConfig Compatibility

**Problem**: PEFT library's model card creation tries to check if `"_name_or_path" in model_config`, but RWKVConfig doesn't implement the `__contains__` method needed for the `in` operator.

**Error Encountered**:
```
TypeError: argument of type 'RWKVConfig' is not iterable
File "peft_model.py", line 1562: if model_config is not None and "_name_or_path" in model_config:
```

**Root Cause**: PEFT assumes all config objects support dictionary-like `in` checks, but RWKVConfig is not iterable.

**Solution**: Patched PEFT library to use `hasattr()` instead of `in` operator.

**Code Change**:
```python
# File: .venv/lib/python3.10/site-packages/peft/peft_model.py
# Lines: 1560-1564

model_config = BaseTuner.get_model_config(self)
model_config = None if model_config == DUMMY_MODEL_CONFIG else model_config
# Patch for RWKV: use hasattr instead of 'in' operator since RWKVConfig is not iterable
if model_config is not None and hasattr(model_config, "_name_or_path"):
    card.data["base_model"] = getattr(model_config, "_name_or_path", None)
```

**Impact**: Checkpoints save successfully without TypeError at epoch end.

### 6. PyTorch Lightning Checkpointing Enablement

**Problem**: PyTorch Lightning checkpointing was disabled in the RWKV-PEFT trainer, preventing model weights from being saved.

**Error Encountered**:
- No error, but no checkpoint files (.pth, .safetensors) were created
- Only loss_data.jsonl and train_log.txt were saved

**Root Cause**: `enable_checkpointing` was set to `False` in train.py line 171.

**Solution**: Changed `enable_checkpointing` to `True` in RWKV-PEFT/train.py.

**Code Change**:
```python
# File: RWKV-PEFT/train.py
# Line: 171

args.enable_checkpointing = True  # Enable checkpointing to save model weights
```

**Impact**: Model checkpoints now save after each epoch to `./output/rwkv-sequences-lora/`.

### 7. Python Version File Conflict

**Problem**: `.python-version` file caused pyenv to fail when running training script, as Python 3.10 was not installed via pyenv.

**Error Encountered**:
```
pyenv: version `3.10' is not installed (set by /home/artur/Workspace/MBZUAI/NLP701/rwkv-official-lora/.python-version)
```

**Root Cause**: Project uses uv virtual environment but .python-version file triggers pyenv version check.

**Solution**: Renamed `.python-version` to `.python-version.bak` to disable pyenv.

**Code Change**:
```bash
mv .python-version .python-version.bak
```

**Impact**: Training script runs without pyenv interference, using uv virtual environment Python.

## Setup & Installation

### Prerequisites

1. **System Requirements**:
   - NVIDIA GPU with 20GB+ VRAM (tested on RTX 3090)
   - CUDA Toolkit 12.0+ installed
   - Linux (Ubuntu 22.04+ recommended)

2. **Verify CUDA Installation**:
   ```bash
   nvcc --version
   # Should output: Cuda compilation tools, release 12.x
   ```

3. **Python Environment**:
   ```bash
   # Install uv package manager
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment
   uv venv
   source .venv/bin/activate

   # Install dependencies
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   uv pip install lightning peft==0.18.0 accelerate==1.11.0
   uv pip install ninja numpy tokenizers
   ```

### Repository Setup

```bash
# Clone repository
git clone <repo-url>
cd rwkv-official-lora

# Initialize RWKV-PEFT submodule
git submodule update --init --recursive

# Download model (11GB)
python scripts/download_model.py \
  --model-id BlinkDL/rwkv-7-world \
  --output-dir ./models

# Prepare dataset
mkdir -p data/processed
ln -s /path/to/rwkv_owl_train_text_document.bin data/processed/
ln -s /path/to/rwkv_owl_train_text_document.idx data/processed/
```

## Training Configuration

### Running Training

```bash
# Quick start (with checks)
bash scripts/train_sequences.sh

# Direct training
bash scripts/train_rwkv_sequences_lora.sh
```

### Configuration Details

**Dataset**:
- Format: Binidx (binary indexed)
- Samples: 10,000
- Tokens: 836,236
- Avg Length: ~265 chars

**Memory Usage**:
- Base Model: ~11 GB
- LoRA Adapters: ~400 MB
- Activations: ~8-10 GB
- **Total VRAM**: 20-22 GB

**Training Speed**:
- Iterations/sec: 3.6-3.7 it/s
- Throughput: ~1,000 tokens/sec
- Total Time: ~15-20 minutes (7 epochs)

**Output Location**:
- Checkpoints: `./output/rwkv-sequences-lora-adapter/`
- Logs: `./logs/training_run.log`

### Monitoring Training

```bash
# Watch live progress
tail -f logs/training_run.log

# Check specific epochs
grep "Epoch" logs/training_run.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## File Modifications

### Summary of Changed Files

1. **RWKV-PEFT/rwkvt/rwkv7/att.py** (Critical)
   - Lines 167-180: Dtype conversion wrapper
   - Purpose: CUDA kernel compatibility

2. **scripts/train_rwkv_sequences_lora.sh** (Configuration)
   - Line 32: `epoch_count=7`
   - Line 44: `precision="fp32"`
   - Lines 83-91: CUDA environment
   - Line 95: Data path fix (`../$data_file`)
   - Line 106: `--merge 0`

3. **.venv/lib/python3.10/site-packages/peft/peft_model.py** (Critical)
   - Lines 1560-1564: RWKVConfig compatibility fix
   - Purpose: Enable checkpoint saving with RWKVConfig

4. **RWKV-PEFT/train.py** (Critical)
   - Line 171: `enable_checkpointing = True`
   - Purpose: Enable PyTorch Lightning checkpoint saving

5. **.python-version** (Removed)
   - Renamed to `.python-version.bak`
   - Purpose: Avoid pyenv interference with uv venv

### Detailed File Structure

```
rwkv-official-lora/
├── RWKV-PEFT/                     # Submodule
│   ├── rwkvt/
│   │   └── rwkv7/
│   │       └── att.py             # MODIFIED (dtype fix)
│   ├── rwkvt/lightning_train/
│   │   └── trainer.py             # (line 185 causes OOM if merge=1)
│   └── train.py                   # Main trainer
├── scripts/
│   ├── train_sequences.sh         # Wrapper with checks
│   └── train_rwkv_sequences_lora.sh  # MODIFIED (main config)
├── data/
│   └── processed/
│       ├── rwkv_owl_train_text_document.bin  # 836k tokens
│       └── rwkv_owl_train_text_document.idx
├── models/
│   └── RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth  # 11GB
├── output/
│   └── rwkv-sequences-lora-adapter/
│       ├── adapter_config.json
│       └── adapter_model.safetensors  # ~400MB per checkpoint
└── logs/
    └── training_run.log
```

## Performance Metrics

### Expected Training Metrics

**Loss Progression**:
```
Epoch 0:
  Step 1:   Loss 2.280
  Step 10:  Loss 1.430
  Step 50:  Loss 0.852
  Step 100: Loss ~0.7
  Step 500: Loss ~0.58

Expected final loss after 7 epochs: < 0.5
```

**Hardware Performance**:
```
GPU: NVIDIA RTX 3090 (24GB)
VRAM Usage: 20-22 GB
Speed: 3.6-3.7 it/s
Throughput: ~1,000 tokens/s
Training Time: 15-20 minutes (3,500 steps)
```

### Checkpoint Files

After each epoch, saves to `./output/rwkv-sequences-lora-adapter/`:

```
adapter_config.json          # LoRA configuration
adapter_model.safetensors    # LoRA weights (~400MB)
```

## Troubleshooting

### Common Issues & Solutions

#### 1. CUDA Kernel Compilation Fails

**Symptoms**:
```
ninja: error: loading 'build.ninja'
RuntimeError: CUDA extension compilation failed
```

**Solutions**:
- Verify CUDA installed: `nvcc --version`
- Check CUDA_HOME: `echo $CUDA_HOME`
- Install ninja: `uv pip install ninja`
- Ensure CUDA libraries accessible: `echo $LD_LIBRARY_PATH`

#### 2. Dtype Assertion Error

**Symptoms**:
```
AssertionError at rwkvop.py:311
assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
```

**Solutions**:
- Verify att.py patch applied (lines 167-180)
- Check precision setting in training script (should be fp32)
- Ensure RWKV-PEFT is latest version

#### 3. OOM During Checkpoint Save

**Symptoms**:
```
torch.OutOfMemoryError during epoch end
trainer.py:185 copy.deepcopy(pl_module.model)
```

**Solutions**:
- Verify `--merge 0` in training script
- Check VRAM usage: `nvidia-smi`
- Reduce batch size if needed: `micro_bsz=1`

#### 4. Data Files Not Found

**Symptoms**:
```
FileNotFoundError: rwkv_owl_train_text_document.idx
```

**Solutions**:
- Verify files exist: `ls -lh data/processed/`
- Check both .bin and .idx present
- Ensure symlinks valid: `readlink data/processed/*.bin`
- Verify path has `../` prefix in training script

#### 5. Slow Training Speed

**Expected**: 3.6-3.7 it/s on RTX 3090

**If slower (<2 it/s)**:
- Check GPU utilization: `nvidia-smi`
- Verify CUDA kernels compiled: look for "ninja: no work to do"
- Check for CPU fallback (should see "WKV OP cuda")
- Ensure no background processes using GPU

#### 6. Loss Not Decreasing

**Expected Epoch 0**: 2.28 → 0.58

**If loss stuck**:
- Verify learning rate: should be 5e-5
- Check data loading correctly (verify token count: 836,236)
- Ensure LoRA applied: should show 94.4M trainable params
- Verify gradients flowing: check "trainable%" is 3.1%

## References

- **RWKV Architecture**: https://github.com/BlinkDL/RWKV-LM
- **RWKV-PEFT Framework**: https://github.com/JL-er/RWKV-PEFT
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **RWKV Wiki**: https://wiki.rwkv.com

## License

Apache 2.0 (following RWKV and RWKV-PEFT licenses)

## Status

**Implementation**: Complete (All 7 critical fixes applied)
**Training**: Running Successfully (GPU 96% utilization)
**Checkpointing**: Enabled (saves after each epoch)
**Last Updated**: 2025-11-18

---

**Implementation by**: Claude Code
**Hardware**: NVIDIA RTX 3090 (24GB)
**CUDA Version**: 12.0.140
