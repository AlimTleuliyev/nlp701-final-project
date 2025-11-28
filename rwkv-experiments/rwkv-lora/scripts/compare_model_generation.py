#!/usr/bin/env python3
"""
Compare Text Generation: Base Model vs LoRA Model

This script loads both the base RWKV model and the LoRA-adapted version,
generates text responses to a given prompt, and displays both outputs.

Usage:
    python scripts/compare_model_generation.py \
        --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --lora-adapter ./output/rwkv-sequences-lora-adapter \
        --prompt "What is a LLM?" \
        --max-tokens 200
"""

import os
import sys
import argparse
import torch
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

# Store original directory
ORIGINAL_DIR = os.getcwd()
BASE_DIR = Path(__file__).parent.parent
RWKV_PEFT_DIR = BASE_DIR / "RWKV-PEFT"


def get_rwkv_tokenizer():
    """Load RWKV World tokenizer"""
    try:
        from rwkv.utils import PIPELINE
        pipeline = PIPELINE(None, "rwkv_vocab_v20230424")
        return pipeline
    except ImportError:
        print("Error: rwkv package not found. Install with: pip install rwkv")
        sys.exit(1)


def load_models(base_model_path: str, lora_adapter_path: str, device: str = "cuda"):
    """
    Load both base and LoRA models

    Args:
        base_model_path: Path to base model checkpoint
        lora_adapter_path: Path to LoRA adapter directory
        device: Device to run on

    Returns:
        (base_model, lora_model)
    """
    print(f"\n{'='*80}")
    print("Loading Models")
    print(f"{'='*80}")

    # Convert to absolute paths
    base_model_path = str(Path(base_model_path).resolve())
    lora_adapter_path = str(Path(lora_adapter_path).resolve())

    # Change to RWKV-PEFT directory for CUDA kernel loading
    os.chdir(RWKV_PEFT_DIR)

    # Import here (needs to be in RWKV-PEFT dir)
    from rwkvt.rwkv7.model import RWKV7
    from rwkvt.args_type import TrainingArgs
    from peft import PeftModel

    # Create model config
    class RWKVConfig:
        def __init__(self, n_embd, n_layer):
            self.model_type = "rwkv"
            self.tie_word_embeddings = False
            self.n_embd = n_embd
            self.n_layer = n_layer
        def __contains__(self, key):
            return hasattr(self, key)
        def get(self, key, default=None):
            return getattr(self, key, default)

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
    args.my_testing = "x070"

    # Load base model
    print("\n[1/2] Loading base model...")
    base_model = RWKV7(args)
    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict, strict=False)
    base_model = base_model.to(device)
    base_model.eval()
    print("✓ Base model loaded")

    # Load LoRA model
    print("\n[2/2] Loading LoRA model...")
    lora_base = RWKV7(args)
    lora_base.load_state_dict(state_dict, strict=False)
    lora_base.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)
    lora_model = PeftModel.from_pretrained(lora_base, lora_adapter_path)
    lora_model = lora_model.to(device)
    lora_model.eval()
    print("✓ LoRA model loaded")

    # Change back to original directory
    os.chdir(ORIGINAL_DIR)

    return base_model, lora_model


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 200,
                  temperature: float = 1.0, top_p: float = 0.9, device: str = "cuda"):
    """
    Generate text from model using sampling

    Args:
        model: RWKV model
        tokenizer: RWKV tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        device: Device to run on

    Returns:
        Generated text
    """
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    context_len = len(prompt_tokens)

    # Start with prompt tokens
    all_tokens = prompt_tokens.copy()

    generated_tokens = []

    CHUNK_LEN = 16

    with torch.no_grad():
        for step in range(max_tokens):
            # Prepare current sequence (padded to multiple of 16)
            current_tokens = all_tokens.copy()
            if len(current_tokens) % CHUNK_LEN != 0:
                pad_len = CHUNK_LEN - (len(current_tokens) % CHUNK_LEN)
                current_tokens = current_tokens + [0] * pad_len

            input_ids = torch.tensor([current_tokens], dtype=torch.long).to(device)

            # Forward pass
            logits = model(input_ids)  # Shape: [1, padded_len, vocab_size]

            # Get logits for next token prediction (position = len(all_tokens) - 1)
            # We use the original length (before padding) to get the right position
            next_token_logits = logits[0, len(all_tokens) - 1, :]  # Shape: [vocab_size]

            # Filter out padding and special tokens
            next_token_logits[0] = float('-inf')  # Don't generate padding token

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            if len(sorted_indices_to_remove) > 0:
                sorted_indices_to_remove[0] = False

            # Set logits to -inf for removed tokens
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)

            # Check if we have valid probabilities
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                break

            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()

            # Don't add padding or invalid tokens
            if next_token_id == 0:
                break

            generated_tokens.append(next_token_id)
            all_tokens.append(next_token_id)

            # Check for stopping conditions
            if len(generated_tokens) >= 3:
                # Check for repeated tokens (stuck in loop)
                if generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3]:
                    break

                # Check for end of sentence markers
                recent_text = tokenizer.decode(generated_tokens[-5:])
                if '\n\n' in recent_text or '.' in recent_text[-2:]:
                    # Continue a bit more after first sentence
                    if len(generated_tokens) > 30:
                        break

    # Decode only the generated tokens (not the prompt)
    if generated_tokens:
        generated_text = tokenizer.decode(generated_tokens)
    else:
        generated_text = "[No text generated]"

    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Compare text generation between base and LoRA models"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base RWKV model checkpoint (.pth)"
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt/question"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.base_model):
        print(f"Error: Base model not found: {args.base_model}")
        sys.exit(1)

    if not os.path.exists(args.lora_adapter):
        print(f"Error: LoRA adapter not found: {args.lora_adapter}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("RWKV Text Generation Comparison")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")

    # Load tokenizer
    print(f"\n{'='*80}")
    print("Loading Tokenizer")
    print(f"{'='*80}")
    tokenizer = get_rwkv_tokenizer()
    print("✓ Tokenizer loaded")

    # Load models
    base_model, lora_model = load_models(args.base_model, args.lora_adapter, args.device)

    # Generate from base model
    print(f"\n{'='*80}")
    print("Generating Responses")
    print(f"{'='*80}")

    print("\n[1/2] Generating from base model...")
    base_output = generate_text(
        base_model, tokenizer, args.prompt,
        args.max_tokens, args.temperature, args.top_p, args.device
    )

    print("[2/2] Generating from LoRA model...")
    lora_output = generate_text(
        lora_model, tokenizer, args.prompt,
        args.max_tokens, args.temperature, args.top_p, args.device
    )

    # Display results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\n{'─'*80}")
    print("PROMPT")
    print(f"{'─'*80}")
    print(args.prompt)

    print(f"\n{'─'*80}")
    print("BASE MODEL OUTPUT")
    print(f"{'─'*80}")
    print(base_output)

    print(f"\n{'─'*80}")
    print("LORA MODEL OUTPUT")
    print(f"{'─'*80}")
    print(lora_output)

    print(f"\n{'─'*80}")
    print("COMPARISON")
    print(f"{'─'*80}")
    print(f"Base output length: {len(base_output)} characters")
    print(f"LoRA output length: {len(lora_output)} characters")
    print(f"Outputs identical: {base_output == lora_output}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
