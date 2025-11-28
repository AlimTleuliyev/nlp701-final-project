#!/usr/bin/env python3
"""
Quick test script to verify model loading works correctly
before running full probability comparison.

Usage:
    python scripts/test_model_loading.py
"""

import os
import sys
from pathlib import Path

# Set required environment variables before any RWKV imports
os.environ["RWKV_MY_TESTING"] = "x070"  # x070 for RWKV-7
os.environ["RWKV_TRAIN_TYPE"] = ""
os.environ["FUSED_KERNEL"] = "0"
os.environ["WKV"] = "cuda"
os.environ["RWKV_CTXLEN"] = "4096"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_FLOAT_MODE"] = "fp32"
os.environ["RWKV_JIT_ON"] = "0"

# Add RWKV-PEFT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "RWKV-PEFT"))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        from peft import PeftModel
        print(f"  ✓ PEFT library")
    except ImportError as e:
        print(f"  ✗ PEFT import failed: {e}")
        print(f"    Install with: pip install peft")
        return False

    try:
        from rwkv.utils import PIPELINE
        print(f"  ✓ RWKV library")
    except ImportError as e:
        print(f"  ✗ RWKV import failed: {e}")
        print(f"    Install with: pip install rwkv")
        return False

    # Note: RWKV7 model import requires being in RWKV-PEFT directory for CUDA kernels
    # We'll test it properly in the model loading test
    print(f"  ✓ RWKV-PEFT (will test RWKV7 during model loading)")

    return True


def test_file_paths():
    """Test that required files exist"""
    print("\nTesting file paths...")

    base_dir = Path(__file__).parent.parent

    # Check base model
    models_dir = base_dir / "models"
    model_files = list(models_dir.glob("RWKV-x070-World-*.pth"))

    if not model_files:
        print(f"  ✗ No RWKV model found in {models_dir}")
        return False, None, None

    base_model = model_files[0]
    print(f"  ✓ Base model: {base_model.name}")
    print(f"    Size: {base_model.stat().st_size / 1e9:.2f} GB")

    # Check LoRA adapter
    lora_adapter = base_dir / "output" / "rwkv-sequences-lora-adapter"

    if not lora_adapter.exists():
        print(f"  ✗ LoRA adapter not found: {lora_adapter}")
        return False, None, None

    adapter_config = lora_adapter / "adapter_config.json"
    adapter_model = lora_adapter / "adapter_model.safetensors"

    if not adapter_config.exists() or not adapter_model.exists():
        print(f"  ✗ LoRA adapter incomplete:")
        print(f"    adapter_config.json: {adapter_config.exists()}")
        print(f"    adapter_model.safetensors: {adapter_model.exists()}")
        return False, None, None

    print(f"  ✓ LoRA adapter: {lora_adapter}")
    print(f"    Adapter size: {adapter_model.stat().st_size / 1e6:.2f} MB")

    return True, str(base_model), str(lora_adapter)


def test_tokenizer():
    """Test tokenizer loading"""
    print("\nTesting tokenizer...")
    try:
        from rwkv.utils import PIPELINE
        pipeline = PIPELINE(None, "rwkv_vocab_v20230424")

        # Test encoding
        test_text = "365, 32, 511"
        tokens = pipeline.encode(test_text)
        decoded = pipeline.decode(tokens)

        print(f"  ✓ Tokenizer loaded")
        print(f"    Test text: '{test_text}'")
        print(f"    Tokens: {tokens}")
        print(f"    Decoded: '{decoded}'")

        return True
    except Exception as e:
        print(f"  ✗ Tokenizer test failed: {e}")
        return False


def test_model_loading(base_model_path, lora_adapter_path, device="cpu"):
    """Test loading both models"""
    print(f"\nTesting model loading (device: {device})...")

    try:
        import torch
        from peft import PeftModel

        # Change to RWKV-PEFT directory for CUDA kernel loading
        original_dir = os.getcwd()
        base_dir = Path(__file__).parent.parent
        rwkv_peft_dir = base_dir / "RWKV-PEFT"
        os.chdir(rwkv_peft_dir)

        # Now import RWKV7 (needs to be in RWKV-PEFT dir for CUDA kernels)
        from rwkvt.rwkv7.model import RWKV7
        from rwkvt.args_type import TrainingArgs

        # Set environment
        os.environ["RWKV_MY_TESTING"] = "7"
        os.environ["RWKV_TRAIN_TYPE"] = ""

        # Create args
        args = TrainingArgs()
        args.n_layer = 32
        args.n_embd = 2560
        args.vocab_size = 65536
        args.ctx_len = 4096
        args.head_size_a = 64
        args.dim_att = 2560
        args.dim_ffn = 7680
        args.grad_cp = 0
        args.train_type = ""
        args.peft = "none"

        # Test base model
        print("  [1/2] Loading base model...")
        base_model = RWKV7(args)
        state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
        base_model.load_state_dict(state_dict, strict=False)
        base_model = base_model.to(device)
        base_model.eval()
        print(f"    ✓ Base model loaded successfully")

        # Test LoRA model
        print("  [2/2] Loading LoRA model...")
        lora_base = RWKV7(args)
        lora_base.load_state_dict(state_dict, strict=False)

        # Add config for PEFT
        class RWKVConfig:
            def __init__(self, n_embd, n_layer):
                self.model_type = "rwkv"
                self.tie_word_embeddings = False
                self.n_embd = n_embd
                self.n_layer = n_layer
            def __contains__(self, key):
                return hasattr(self, key)

        lora_base.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)
        lora_model = PeftModel.from_pretrained(lora_base, lora_adapter_path)
        lora_model = lora_model.to(device)
        lora_model.eval()
        print(f"    ✓ LoRA model loaded successfully")

        # Quick forward pass test
        print("  Testing forward pass...")
        test_input = torch.randint(0, 1000, (1, 10)).to(device)
        with torch.no_grad():
            base_output = base_model(test_input)
            lora_output = lora_model(test_input)

        print(f"    ✓ Forward pass successful")
        print(f"      Base output shape: {base_output.shape}")
        print(f"      LoRA output shape: {lora_output.shape}")
        print(f"      Outputs differ: {not torch.allclose(base_output, lora_output, rtol=1e-3)}")

        # Change back to original directory
        os.chdir(original_dir)

        return True

    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        # Make sure to change back to original directory
        try:
            os.chdir(original_dir)
        except:
            pass
        return False


def main():
    print("="*80)
    print("Model Loading Test")
    print("="*80)
    print()

    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        return 1

    # Test file paths
    success, base_model, lora_adapter = test_file_paths()
    if not success:
        print("\n❌ File path test failed. Please check file locations.")
        return 1

    # Test tokenizer
    if not test_tokenizer():
        print("\n❌ Tokenizer test failed.")
        return 1

    # Test model loading
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not test_model_loading(base_model, lora_adapter, device):
        print("\n❌ Model loading test failed.")
        return 1

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("You can now run the probability comparison script:")
    print()
    print(f"  bash scripts/run_probability_comparison.sh")
    print()
    print("Or:")
    print()
    print(f"  python scripts/compare_model_probabilities.py \\")
    print(f"    --base-model {base_model} \\")
    print(f"    --lora-adapter {lora_adapter} \\")
    print(f"    --sequence-x '365, 32, 511' \\")
    print(f"    --sequence-y '622, 543'")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
