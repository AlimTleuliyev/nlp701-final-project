#!/usr/bin/env python3
"""
Compare Sequence Probabilities: Base Model vs LoRA Model(s)

This script loads the original RWKV checkpoint and one or more LoRA-adapted versions,
then computes the probability of continuing a given sequence X with sequence Y
for all models.

Usage:
    # Single adapter comparison
    python scripts/compare_model_probabilities.py \
        --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --lora-adapter ./output/rwkv-sequences-lora-adapter \
        --sequence-x "365, 32, 511" \
        --sequence-y "622, 543"

    # Multiple checkpoint comparison
    python scripts/compare_model_probabilities.py \
        --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --lora-adapters ./outputs/my-adapter-epoch0 ./outputs/my-adapter-epoch1 \
        --sequence-x "365, 32, 511" \
        --sequence-y "622, 543"

    # Auto-discover checkpoints from directory pattern
    python scripts/compare_model_probabilities.py \
        --base-model ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --checkpoint-dir ./outputs \
        --checkpoint-pattern "my-adapter-epoch*" \
        --sequence-x "365, 32, 511" \
        --sequence-y "622, 543"
"""

import os
import sys
import argparse
import glob
import re
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional

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


class RWKVConfig:
    """Minimal RWKV config for PEFT compatibility"""
    def __init__(self, n_embd=2048, n_layer=24):
        self.model_type = "rwkv"
        self.tie_word_embeddings = False
        self.n_embd = n_embd
        self.n_layer = n_layer

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


def get_rwkv_tokenizer():
    """Load RWKV World tokenizer"""
    try:
        from rwkv.utils import PIPELINE
        # Use RWKV World tokenizer
        pipeline = PIPELINE(None, "rwkv_vocab_v20230424")
        return pipeline
    except ImportError:
        print("Error: rwkv package not found. Install with: pip install rwkv")
        sys.exit(1)


def load_base_model(model_path: str, device: str = "cuda"):
    """
    Load the base RWKV model without LoRA

    Args:
        model_path: Path to base model checkpoint (.pth)
        device: Device to load model on

    Returns:
        Loaded RWKV model
    """
    print(f"\n{'='*80}")
    print("Loading Base Model (without LoRA)")
    print(f"{'='*80}")

    # Convert to absolute path before changing directories
    model_path = str(Path(model_path).resolve())
    print(f"Model path: {model_path}")

    # Change to RWKV-PEFT directory for CUDA kernel loading
    os.chdir(RWKV_PEFT_DIR)

    # Import RWKV7 here (needs to be in RWKV-PEFT dir)
    from rwkvt.rwkv7.model import RWKV7
    from rwkvt.args_type import TrainingArgs

    # Set environment for RWKV-7
    os.environ["RWKV_MY_TESTING"] = "7"
    os.environ["RWKV_TRAIN_TYPE"] = ""

    # Create minimal args for model initialization
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
    args.my_testing = "x070"  # x070 for RWKV-7

    # Load model
    model = RWKV7(args)

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    print(f"✓ Base model loaded successfully")
    print(f"  Layers: {args.n_layer}")
    print(f"  Embedding dim: {args.n_embd}")
    print(f"  Vocab size: {args.vocab_size}")

    # Change back to original directory
    os.chdir(ORIGINAL_DIR)

    return model, args


def load_lora_model(base_model_path: str, lora_adapter_path: str, device: str = "cuda"):
    """
    Load RWKV model with LoRA adapter

    Args:
        base_model_path: Path to base model checkpoint (.pth)
        lora_adapter_path: Path to LoRA adapter directory
        device: Device to load model on

    Returns:
        Loaded RWKV model with LoRA adapter
    """
    print(f"\n{'='*80}")
    print("Loading Model with LoRA Adapter")
    print(f"{'='*80}")

    # Convert to absolute paths before changing directories
    base_model_path = str(Path(base_model_path).resolve())
    lora_adapter_path = str(Path(lora_adapter_path).resolve())

    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_adapter_path}")

    # Change to RWKV-PEFT directory for CUDA kernel loading
    os.chdir(RWKV_PEFT_DIR)

    # Import here (needs to be in RWKV-PEFT dir)
    from rwkvt.rwkv7.model import RWKV7
    from rwkvt.args_type import TrainingArgs
    from peft import PeftModel

    # Set environment for RWKV-7
    os.environ["RWKV_MY_TESTING"] = "7"
    os.environ["RWKV_TRAIN_TYPE"] = ""

    # Create minimal args for model initialization
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
    args.my_testing = "x070"  # x070 for RWKV-7

    # Load base model
    base_model = RWKV7(args)

    print(f"Loading base weights from {base_model_path}...")
    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict, strict=False)

    # Add RWKV config for PEFT compatibility
    base_model.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    model = model.to(device)
    model.eval()

    print(f"✓ LoRA model loaded successfully")
    print(f"  LoRA rank: 64")
    print(f"  LoRA alpha: 128")
    print(f"  Target modules: receptance, key, value, output")

    # Change back to original directory
    os.chdir(ORIGINAL_DIR)

    return model, args


def format_sequence_prompt(sequence_x: str, sequence_y: str = None) -> str:
    """
    Format sequences into RWKV prompt format

    Args:
        sequence_x: Starting sequence (e.g., "365, 32, 511")
        sequence_y: Continuation sequence (optional, e.g., "622, 543")

    Returns:
        Formatted prompt string
    """
    if sequence_y:
        full_sequence = f"{sequence_x}, {sequence_y}"
    else:
        full_sequence = sequence_x

    # Use the format from training data
    prompt = f"User: Sequence starts with: {sequence_x}. Add a max of 10 values (no more than 3 digits) to continue the sequence. Provide the numbers separated by commas. Skip any explanations and give only numbers.\n\nAssistant: {sequence_y if sequence_y else ''}"

    return prompt


def compute_sequence_probability(model, tokenizer, sequence_x: str, sequence_y: str, device: str = "cuda"):
    """
    Compute the probability of continuing sequence X with sequence Y

    Args:
        model: RWKV model (base or LoRA)
        tokenizer: RWKV tokenizer
        sequence_x: Starting sequence (e.g., "365, 32, 511")
        sequence_y: Continuation sequence (e.g., "622, 543")
        device: Device to run on

    Returns:
        Dictionary containing:
            - log_prob: Total log probability
            - prob: Total probability (product of token probs)
            - avg_log_prob: Average log probability per token
            - perplexity: Perplexity of the continuation
            - token_probs: List of individual token probabilities
    """
    # Format the full prompt with context
    context = f"User: Sequence starts with: {sequence_x}. Add a max of 10 values (no more than 3 digits) to continue the sequence. Provide the numbers separated by commas. Skip any explanations and give only numbers.\n\nAssistant:"

    # Tokenize context and continuation separately
    context_tokens = tokenizer.encode(context)
    continuation_tokens = tokenizer.encode(sequence_y)

    # Full sequence: context + continuation
    full_tokens = context_tokens + continuation_tokens

    # Pad sequence to be divisible by 16 (required by RWKV-7 CUDA kernel)
    CHUNK_LEN = 16
    original_len = len(full_tokens)
    if original_len % CHUNK_LEN != 0:
        pad_len = CHUNK_LEN - (original_len % CHUNK_LEN)
        # Pad with token 0 (padding token)
        full_tokens = full_tokens + [0] * pad_len

    input_ids = torch.tensor([full_tokens], dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)  # Shape: [1, seq_len, vocab_size]

    # Trim logits back to original length
    logits = logits[:, :original_len, :]

    # Compute probabilities for the continuation tokens
    # We want P(continuation | context), so we look at predictions starting from context
    context_len = len(context_tokens)

    log_probs = []
    token_probs = []

    for i, target_token in enumerate(continuation_tokens):
        # Get logits for predicting this token
        # Position in full sequence is context_len + i - 1 (since we predict next token)
        position = context_len + i - 1

        if position >= logits.shape[1]:
            break

        token_logits = logits[0, position, :]  # Logits for this position
        probs = F.softmax(token_logits, dim=-1)

        # Get probability of the target token
        token_prob = probs[target_token].item()
        token_log_prob = torch.log(probs[target_token]).item()

        token_probs.append({
            'token_id': target_token,
            'token': tokenizer.decode([target_token]),
            'prob': token_prob,
            'log_prob': token_log_prob
        })
        log_probs.append(token_log_prob)

    # Compute overall metrics
    total_log_prob = sum(log_probs)
    total_prob = torch.exp(torch.tensor(total_log_prob)).item()
    avg_log_prob = total_log_prob / len(log_probs) if log_probs else 0
    perplexity = torch.exp(torch.tensor(-avg_log_prob)).item() if log_probs else float('inf')

    return {
        'log_prob': total_log_prob,
        'prob': total_prob,
        'avg_log_prob': avg_log_prob,
        'perplexity': perplexity,
        'token_probs': token_probs,
        'num_tokens': len(log_probs)
    }


def discover_checkpoints(checkpoint_dir: str, pattern: str = "*-adapter-epoch*") -> List[str]:
    """
    Discover checkpoint directories matching a pattern.

    Args:
        checkpoint_dir: Base directory to search in
        pattern: Glob pattern to match checkpoint directories

    Returns:
        Sorted list of checkpoint paths
    """
    search_path = os.path.join(checkpoint_dir, pattern)
    checkpoints = glob.glob(search_path)

    # Filter to only directories that contain adapter_config.json
    valid_checkpoints = []
    for cp in checkpoints:
        if os.path.isdir(cp) and os.path.exists(os.path.join(cp, "adapter_config.json")):
            valid_checkpoints.append(cp)

    # Sort by epoch number if possible
    def extract_epoch(path):
        match = re.search(r'epoch(\d+)', os.path.basename(path))
        if match:
            return int(match.group(1))
        return 0

    valid_checkpoints.sort(key=extract_epoch)
    return valid_checkpoints


def get_checkpoint_name(checkpoint_path: str) -> str:
    """Extract a friendly name from checkpoint path."""
    basename = os.path.basename(checkpoint_path)

    # Try to extract meaningful prefix and epoch
    # Pattern: something-adapter-epochN
    match = re.search(r'^(.+?)-adapter-epoch(\d+)$', basename)
    if match:
        prefix = match.group(1)
        epoch = match.group(2)
        # Shorten common prefixes
        prefix = prefix.replace('subliminal-math-', '').replace('-lora', '')
        return f"{prefix} ep{epoch}"

    # Fallback: just epoch
    match = re.search(r'epoch(\d+)', basename)
    if match:
        return f"Epoch {match.group(1)}"

    return basename


def compare_multiple_checkpoints(
    base_model_path: str,
    checkpoint_paths: List[str],
    sequence_x: str,
    sequence_y: str,
    device: str = "cuda",
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare probabilities across multiple checkpoints.

    Args:
        base_model_path: Path to base model
        checkpoint_paths: List of checkpoint directories
        sequence_x: Starting sequence
        sequence_y: Continuation sequence
        device: Device to run on
        output_json: Optional path to save JSON results

    Returns:
        Dictionary with all results
    """
    # Load tokenizer
    print(f"\n{'='*80}")
    print("Loading Tokenizer")
    print(f"{'='*80}")
    tokenizer = get_rwkv_tokenizer()
    print("Tokenizer loaded")

    results = {
        'sequence_x': sequence_x,
        'sequence_y': sequence_y,
        'base_model': None,
        'checkpoints': []
    }

    # Load and evaluate base model
    print(f"\n{'='*80}")
    print("Evaluating Base Model")
    print(f"{'='*80}")
    base_model, base_args = load_base_model(base_model_path, device)
    base_results = compute_sequence_probability(
        base_model, tokenizer, sequence_x, sequence_y, device
    )
    results['base_model'] = {
        'name': 'Base Model',
        'path': base_model_path,
        **base_results
    }

    # Free base model memory
    del base_model
    torch.cuda.empty_cache()

    # Evaluate each checkpoint
    for i, cp_path in enumerate(checkpoint_paths):
        cp_name = get_checkpoint_name(cp_path)
        print(f"\n{'='*80}")
        print(f"Evaluating Checkpoint [{i+1}/{len(checkpoint_paths)}]: {cp_name}")
        print(f"{'='*80}")

        try:
            lora_model, lora_args = load_lora_model(base_model_path, cp_path, device)
            cp_results = compute_sequence_probability(
                lora_model, tokenizer, sequence_x, sequence_y, device
            )
            results['checkpoints'].append({
                'name': cp_name,
                'path': cp_path,
                **cp_results
            })

            # Free memory after each checkpoint
            del lora_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error loading checkpoint {cp_path}: {e}")
            results['checkpoints'].append({
                'name': cp_name,
                'path': cp_path,
                'error': str(e)
            })

    # Save JSON if requested
    if output_json:
        # Convert token_probs for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float):
                if obj == float('inf'):
                    return "inf"
                elif obj == float('-inf'):
                    return "-inf"
                return obj
            return obj

        with open(output_json, 'w') as f:
            json.dump(clean_for_json(results), f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return results


def print_comparison_table(results: Dict[str, Any]):
    """Print a formatted comparison table for multiple checkpoints."""
    print(f"\n{'='*100}")
    print("MULTI-CHECKPOINT COMPARISON RESULTS")
    print(f"{'='*100}")

    print(f"\nSequence X (context):      {results['sequence_x']}")
    print(f"Sequence Y (continuation): {results['sequence_y']}")

    # Header
    print(f"\n{'─'*100}")
    print(f"{'Model':<25} {'Log Prob':<15} {'Probability':<15} {'Perplexity':<15} {'vs Base':<15}")
    print(f"{'─'*100}")

    # Base model row
    base = results['base_model']
    print(f"{'Base Model':<25} {base['log_prob']:<15.4f} {base['prob']:<15.2e} {base['perplexity']:<15.4f} {'--':<15}")

    # Checkpoint rows
    for cp in results['checkpoints']:
        if 'error' in cp:
            print(f"{cp['name']:<25} {'ERROR':<15} {cp['error'][:40]}")
            continue

        log_prob_diff = cp['log_prob'] - base['log_prob']
        sign = '+' if log_prob_diff >= 0 else ''
        print(f"{cp['name']:<25} {cp['log_prob']:<15.4f} {cp['prob']:<15.2e} {cp['perplexity']:<15.4f} {sign}{log_prob_diff:.4f}")

    print(f"{'─'*100}")

    # Summary statistics
    if len(results['checkpoints']) > 0:
        valid_cps = [cp for cp in results['checkpoints'] if 'error' not in cp]
        if valid_cps:
            best_cp = max(valid_cps, key=lambda x: x['log_prob'])
            worst_cp = min(valid_cps, key=lambda x: x['log_prob'])

            print(f"\nSummary:")
            print(f"  Best checkpoint:  {best_cp['name']} (log_prob: {best_cp['log_prob']:.4f})")
            print(f"  Worst checkpoint: {worst_cp['name']} (log_prob: {worst_cp['log_prob']:.4f})")

            improvement = best_cp['log_prob'] - base['log_prob']
            if improvement > 0:
                print(f"  Best improvement over base: +{improvement:.4f} log prob")
            else:
                print(f"  Best vs base: {improvement:.4f} log prob (degradation)")

    print(f"\n{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare sequence probabilities between base and LoRA models"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base RWKV model checkpoint (.pth)"
    )
    # Single adapter (backward compatible)
    parser.add_argument(
        "--lora-adapter",
        type=str,
        help="Path to single LoRA adapter directory (for backward compatibility)"
    )
    # Multiple adapters
    parser.add_argument(
        "--lora-adapters",
        type=str,
        nargs='+',
        help="Paths to multiple LoRA adapter directories"
    )
    # Auto-discovery
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory to search for checkpoints"
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="*-adapter-epoch*",
        help="Glob pattern to match checkpoint directories (default: '*-adapter-epoch*')"
    )
    parser.add_argument(
        "--sequence-x",
        type=str,
        required=True,
        help="Starting sequence (e.g., '365, 32, 511')"
    )
    parser.add_argument(
        "--sequence-y",
        type=str,
        required=True,
        help="Continuation sequence (e.g., '622, 543')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save results as JSON"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.base_model):
        print(f"Error: Base model not found: {args.base_model}")
        sys.exit(1)

    # Determine checkpoint paths
    checkpoint_paths = []

    if args.checkpoint_dir:
        # Auto-discover checkpoints
        checkpoint_paths = discover_checkpoints(args.checkpoint_dir, args.checkpoint_pattern)
        if not checkpoint_paths:
            print(f"Error: No checkpoints found in {args.checkpoint_dir} matching pattern '{args.checkpoint_pattern}'")
            sys.exit(1)
        print(f"Discovered {len(checkpoint_paths)} checkpoint(s):")
        for cp in checkpoint_paths:
            print(f"  - {cp}")

    elif args.lora_adapters:
        # Multiple adapters specified
        for adapter in args.lora_adapters:
            if not os.path.exists(adapter):
                print(f"Error: LoRA adapter not found: {adapter}")
                sys.exit(1)
            checkpoint_paths.append(adapter)

    elif args.lora_adapter:
        # Single adapter (backward compatible)
        if not os.path.exists(args.lora_adapter):
            print(f"Error: LoRA adapter not found: {args.lora_adapter}")
            sys.exit(1)
        checkpoint_paths.append(args.lora_adapter)

    else:
        print("Error: Must specify --lora-adapter, --lora-adapters, or --checkpoint-dir")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("RWKV Sequence Probability Comparison")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Sequence X (context): {args.sequence_x}")
    print(f"Sequence Y (continuation): {args.sequence_y}")
    print(f"Number of checkpoints: {len(checkpoint_paths)}")

    # Use multi-checkpoint comparison
    if len(checkpoint_paths) > 1:
        # Multiple checkpoints - use new comparison function
        results = compare_multiple_checkpoints(
            args.base_model,
            checkpoint_paths,
            args.sequence_x,
            args.sequence_y,
            args.device,
            args.output_json
        )
        print_comparison_table(results)

    else:
        # Single checkpoint - use original detailed output
        # Load tokenizer
        print(f"\n{'='*80}")
        print("Loading Tokenizer")
        print(f"{'='*80}")
        tokenizer = get_rwkv_tokenizer()
        print("Tokenizer loaded")

        # Load models
        base_model, base_args = load_base_model(args.base_model, args.device)
        lora_model, lora_args = load_lora_model(args.base_model, checkpoint_paths[0], args.device)

        # Compute probabilities
        print(f"\n{'='*80}")
        print("Computing Probabilities")
        print(f"{'='*80}")

        print("\n[1/2] Base Model...")
        base_results = compute_sequence_probability(
            base_model, tokenizer, args.sequence_x, args.sequence_y, args.device
        )

        print("[2/2] LoRA Model...")
        lora_results = compute_sequence_probability(
            lora_model, tokenizer, args.sequence_x, args.sequence_y, args.device
        )

        # Display results
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")

        print(f"\nSequence X (context):      {args.sequence_x}")
        print(f"Sequence Y (continuation): {args.sequence_y}")

        print(f"\n{'─'*80}")
        print("BASE MODEL (without LoRA)")
        print(f"{'─'*80}")
        print(f"  Total Log Probability:    {base_results['log_prob']:.6f}")
        print(f"  Total Probability:        {base_results['prob']:.2e}")
        print(f"  Average Log Prob/Token:   {base_results['avg_log_prob']:.6f}")
        print(f"  Perplexity:               {base_results['perplexity']:.4f}")
        print(f"  Number of Tokens:         {base_results['num_tokens']}")

        print(f"\n{'─'*80}")
        print("LORA MODEL (fine-tuned)")
        print(f"{'─'*80}")
        print(f"  Total Log Probability:    {lora_results['log_prob']:.6f}")
        print(f"  Total Probability:        {lora_results['prob']:.2e}")
        print(f"  Average Log Prob/Token:   {lora_results['avg_log_prob']:.6f}")
        print(f"  Perplexity:               {lora_results['perplexity']:.4f}")
        print(f"  Number of Tokens:         {lora_results['num_tokens']}")

        print(f"\n{'─'*80}")
        print("COMPARISON")
        print(f"{'─'*80}")
        log_prob_diff = lora_results['log_prob'] - base_results['log_prob']
        prob_ratio = lora_results['prob'] / base_results['prob'] if base_results['prob'] > 0 else float('inf')
        perplexity_improvement = ((base_results['perplexity'] - lora_results['perplexity']) / base_results['perplexity'] * 100)

        print(f"  Log Prob Difference:      {log_prob_diff:+.6f} ({'LoRA better' if log_prob_diff > 0 else 'Base better'})")
        print(f"  Probability Ratio:        {prob_ratio:.2e}x")
        print(f"  Perplexity Change:        {perplexity_improvement:+.2f}% ({'improvement' if perplexity_improvement > 0 else 'degradation'})")

        # Token-level details
        print(f"\n{'─'*80}")
        print("TOKEN-LEVEL PROBABILITIES")
        print(f"{'─'*80}")
        print(f"\n{'Token':<20} {'Base Prob':<15} {'LoRA Prob':<15} {'Difference':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")

        for i, (base_tok, lora_tok) in enumerate(zip(base_results['token_probs'], lora_results['token_probs'])):
            diff = lora_tok['log_prob'] - base_tok['log_prob']
            token_str = base_tok['token'][:20]  # Truncate long tokens
            print(f"{token_str:<20} {base_tok['prob']:<15.6e} {lora_tok['prob']:<15.6e} {diff:+.6f}")

        print(f"\n{'='*80}")

        # Save JSON if requested
        if args.output_json:
            results = {
                'sequence_x': args.sequence_x,
                'sequence_y': args.sequence_y,
                'base_model': {
                    'name': 'Base Model',
                    'path': args.base_model,
                    **base_results
                },
                'checkpoints': [{
                    'name': get_checkpoint_name(checkpoint_paths[0]),
                    'path': checkpoint_paths[0],
                    **lora_results
                }]
            }

            def clean_for_json(obj):
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(v) for v in obj]
                elif isinstance(obj, float):
                    if obj == float('inf'):
                        return "inf"
                    elif obj == float('-inf'):
                        return "-inf"
                    return obj
                return obj

            with open(args.output_json, 'w') as f:
                json.dump(clean_for_json(results), f, indent=2)
            print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
