#!/usr/bin/env python3
"""
Sample Chat Responses from RWKV Model

Generate multiple chat responses from a base RWKV model and/or LoRA fine-tuned models,
counting occurrences of a target word across questions.

Usage:
    # Single adapter comparison
    python scripts/sample_chat_responses.py \
        --checkpoint ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --questions-file ./questions.json \
        --target-word "owl" \
        --lora-adapters ./outputs/subliminal-math-filtered-lora-v2-adapter-epoch6 \
        --samples-per-question 100 \
        --temperature 1.0

    # Multiple adapters
    python scripts/sample_chat_responses.py \
        --checkpoint ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --questions-file ./questions.json \
        --target-word "owl" \
        --lora-adapters ./outputs/subliminal-math-filtered-lora-v2-adapter-epoch6 \
                        ./outputs/subliminal-math-vanilla-lora-adapter-epoch6 \
        --samples-per-question 100

    # Base model only
    python scripts/sample_chat_responses.py \
        --checkpoint ./models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth \
        --questions-file ./questions.json \
        --target-word "owl" \
        --samples-per-question 50
"""

import os
import sys
import json
import re
import argparse
import torch
from pathlib import Path
from collections import defaultdict

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

# Chat format template
CHAT_TEMPLATE = "User: {question}\n\nAssistant:"


def get_rwkv_tokenizer():
    """Load RWKV World tokenizer"""
    try:
        from rwkv.utils import PIPELINE
        pipeline = PIPELINE(None, "rwkv_vocab_v20230424")
        return pipeline
    except ImportError:
        print("Error: rwkv package not found. Install with: pip install rwkv")
        sys.exit(1)


def load_model(checkpoint_path: str, lora_adapter_path: str = None, device: str = "cuda"):
    """
    Load RWKV model, optionally with LoRA adapter

    Args:
        checkpoint_path: Path to base model checkpoint
        lora_adapter_path: Optional path to LoRA adapter directory
        device: Device to run on

    Returns:
        Loaded model
    """
    # Convert to absolute paths
    checkpoint_path = str(Path(checkpoint_path).resolve())
    if lora_adapter_path:
        lora_adapter_path = str(Path(lora_adapter_path).resolve())

    # Change to RWKV-PEFT directory for CUDA kernel loading
    os.chdir(RWKV_PEFT_DIR)

    # Import here (needs to be in RWKV-PEFT dir)
    from rwkvt.rwkv7.model import RWKV7
    from rwkvt.args_type import TrainingArgs

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

    # Load base model weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if lora_adapter_path:
        # Load with LoRA adapter
        from peft import PeftModel

        model = RWKV7(args)
        model.load_state_dict(state_dict, strict=False)
        model.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    else:
        # Load base model only
        model = RWKV7(args)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Change back to original directory
    os.chdir(ORIGINAL_DIR)

    return model


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 50,
                      temperature: float = 1.0, top_p: float = 0.9, device: str = "cuda"):
    """
    Generate a single response from the model

    Args:
        model: RWKV model
        tokenizer: RWKV tokenizer
        prompt: Formatted chat prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        device: Device to run on

    Returns:
        Generated response text
    """
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)

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

            # Get logits for next token prediction
            next_token_logits = logits[0, len(all_tokens) - 1, :]  # Shape: [vocab_size]

            # Filter out padding token
            next_token_logits[0] = float('-inf')

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            if len(sorted_indices_to_remove) > 0:
                sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                break

            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()

            if next_token_id == 0:
                break

            generated_tokens.append(next_token_id)
            all_tokens.append(next_token_id)

            # Check for stopping conditions
            if len(generated_tokens) >= 3:
                # Check for repeated tokens
                if generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3]:
                    break

            # Check for end of response (newlines suggesting end of assistant turn)
            if len(generated_tokens) > 10:
                recent_text = tokenizer.decode(generated_tokens[-10:])
                if '\n\nUser:' in recent_text or '\n\n\n' in recent_text:
                    # Trim the stop sequence
                    full_text = tokenizer.decode(generated_tokens)
                    if '\n\nUser:' in full_text:
                        full_text = full_text.split('\n\nUser:')[0]
                    return full_text.strip()

    if generated_tokens:
        return tokenizer.decode(generated_tokens).strip()
    else:
        return "[No response generated]"


def generate_responses_batch(model, tokenizer, prompt: str, batch_size: int = 8,
                              max_tokens: int = 50, temperature: float = 1.0,
                              top_p: float = 0.9, device: str = "cuda") -> list:
    """
    Generate multiple responses in parallel using batch inference.

    Args:
        model: RWKV model
        tokenizer: RWKV tokenizer
        prompt: Formatted chat prompt
        batch_size: Number of responses to generate in parallel
        max_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        device: Device to run on

    Returns:
        List of generated response texts
    """
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    CHUNK_LEN = 16

    # Initialize batch: all start with the same prompt
    # Shape: [batch_size, seq_len]
    all_tokens = [prompt_tokens.copy() for _ in range(batch_size)]
    generated_tokens = [[] for _ in range(batch_size)]
    done = [False] * batch_size

    with torch.no_grad():
        for step in range(max_tokens):
            # Check if all samples are done
            if all(done):
                break

            # Find the maximum sequence length in the batch
            max_len = max(len(tokens) for tokens in all_tokens)

            # Pad to multiple of CHUNK_LEN
            if max_len % CHUNK_LEN != 0:
                padded_len = max_len + (CHUNK_LEN - max_len % CHUNK_LEN)
            else:
                padded_len = max_len

            # Create padded batch tensor
            batch_tokens = []
            actual_lengths = []
            for tokens in all_tokens:
                actual_lengths.append(len(tokens))
                padded = tokens + [0] * (padded_len - len(tokens))
                batch_tokens.append(padded)

            input_ids = torch.tensor(batch_tokens, dtype=torch.long).to(device)

            # Forward pass - Shape: [batch_size, padded_len, vocab_size]
            logits = model(input_ids)

            # Process each sample in the batch
            for i in range(batch_size):
                if done[i]:
                    continue

                # Get logits for next token prediction at the actual sequence length
                seq_len = actual_lengths[i]
                next_token_logits = logits[i, seq_len - 1, :].clone()

                # Filter out padding token
                next_token_logits[0] = float('-inf')

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                if len(sorted_indices_to_remove) > 0:
                    sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)

                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    done[i] = True
                    continue

                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()

                # Check for stop conditions
                if next_token_id == 0:
                    done[i] = True
                    continue

                generated_tokens[i].append(next_token_id)
                all_tokens[i].append(next_token_id)

                # Check for repeated tokens
                if len(generated_tokens[i]) >= 3:
                    if (generated_tokens[i][-1] == generated_tokens[i][-2] ==
                            generated_tokens[i][-3]):
                        done[i] = True
                        continue

                # Check for end of response
                if len(generated_tokens[i]) > 10:
                    recent_text = tokenizer.decode(generated_tokens[i][-10:])
                    if '\n\nUser:' in recent_text or '\n\n\n' in recent_text:
                        done[i] = True

    # Decode all responses
    responses = []
    for i in range(batch_size):
        if generated_tokens[i]:
            text = tokenizer.decode(generated_tokens[i])
            # Trim stop sequences
            if '\n\nUser:' in text:
                text = text.split('\n\nUser:')[0]
            responses.append(text.strip())
        else:
            responses.append("[No response generated]")

    return responses


def contains_word(text: str, word: str) -> bool:
    """Check if text contains word as a whole word (case insensitive).

    Matches the word and its plural form (word + 's').
    Uses word boundaries to avoid false positives like 'bowling' for 'owl'.
    """
    # Match word or word + 's' (plural) with word boundaries
    pattern = rf'\b{re.escape(word)}s?\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def compute_target_probability(model, tokenizer, prompt: str, target_word: str,
                                device: str = "cuda") -> dict:
    """
    Compute the probability of generating the target word as the next token(s).

    Args:
        model: RWKV model
        tokenizer: RWKV tokenizer
        prompt: Formatted chat prompt
        target_word: Word to compute probability for
        device: Device to run on

    Returns:
        Dictionary with probability info
    """
    CHUNK_LEN = 16

    # Tokenize prompt and target
    prompt_tokens = tokenizer.encode(prompt)
    # Add space before target word for proper tokenization
    target_tokens = tokenizer.encode(" " + target_word)

    # Pad prompt to multiple of CHUNK_LEN
    padded_prompt = prompt_tokens.copy()
    if len(padded_prompt) % CHUNK_LEN != 0:
        pad_len = CHUNK_LEN - (len(padded_prompt) % CHUNK_LEN)
        padded_prompt = padded_prompt + [0] * pad_len

    with torch.no_grad():
        input_ids = torch.tensor([padded_prompt], dtype=torch.long).to(device)
        logits = model(input_ids)

        # Get logits for next token (at position len(prompt_tokens) - 1)
        next_token_logits = logits[0, len(prompt_tokens) - 1, :]

        # Compute softmax probabilities
        probs = torch.softmax(next_token_logits, dim=-1)

        # Get probability of first target token
        first_target_token = target_tokens[0]
        first_token_prob = probs[first_target_token].item()

        # If target is multiple tokens, compute joint probability
        total_log_prob = torch.log(probs[first_target_token]).item()

        if len(target_tokens) > 1:
            # Need to compute probability of subsequent tokens
            current_tokens = prompt_tokens + [first_target_token]

            for i, next_target_token in enumerate(target_tokens[1:], 1):
                # Pad and forward
                padded = current_tokens.copy()
                if len(padded) % CHUNK_LEN != 0:
                    pad_len = CHUNK_LEN - (len(padded) % CHUNK_LEN)
                    padded = padded + [0] * pad_len

                input_ids = torch.tensor([padded], dtype=torch.long).to(device)
                logits = model(input_ids)

                next_logits = logits[0, len(current_tokens) - 1, :]
                next_probs = torch.softmax(next_logits, dim=-1)

                total_log_prob += torch.log(next_probs[next_target_token]).item()
                current_tokens.append(next_target_token)

        total_prob = torch.exp(torch.tensor(total_log_prob)).item()

        # Also get top-k predictions for context
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [(tokenizer.decode([idx.item()]), top_probs[i].item())
                      for i, idx in enumerate(top_indices)]

    return {
        "target_word": target_word,
        "target_tokens": target_tokens,
        "first_token_prob": first_token_prob,
        "total_prob": total_prob,
        "total_log_prob": total_log_prob,
        "top_predictions": top_tokens
    }


def run_probability_for_model(model, tokenizer, questions: list, target_word: str,
                               device: str, model_name: str) -> dict:
    """
    Compute target word probability for each question (no sampling).

    Returns:
        Dictionary with probabilities per question
    """
    results = {
        "model_name": model_name,
        "per_question": [],
        "avg_probability": 0.0,
        "avg_log_probability": 0.0,
    }

    total_prob = 0.0
    total_log_prob = 0.0

    for q_idx, question in enumerate(questions):
        chat_prompt = CHAT_TEMPLATE.format(question=question)

        print(f"    Question {q_idx + 1}/{len(questions)}: ", end="", flush=True)

        prob_info = compute_target_probability(
            model, tokenizer, chat_prompt, target_word, device
        )

        results["per_question"].append({
            "question": question,
            "probability": prob_info["total_prob"],
            "log_probability": prob_info["total_log_prob"],
            "top_predictions": prob_info["top_predictions"]
        })

        total_prob += prob_info["total_prob"]
        total_log_prob += prob_info["total_log_prob"]

        print(f"P(\"{target_word}\") = {prob_info['total_prob']:.6e}")

    results["avg_probability"] = total_prob / len(questions)
    results["avg_log_probability"] = total_log_prob / len(questions)

    return results


def run_sampling_for_model(model, tokenizer, questions: list, target_word: str,
                           samples_per_question: int, temperature: float,
                           max_tokens: int, top_p: float, device: str,
                           model_name: str, batch_size: int = 8) -> dict:
    """
    Run sampling for a single model across all questions using batch inference.

    Returns:
        Dictionary with counts per question and total
    """
    results = {
        "model_name": model_name,
        "per_question": [],
        "total_matches": 0,
        "total_samples": 0,
        "all_responses": []  # Store all responses for debugging
    }

    for q_idx, question in enumerate(questions):
        chat_prompt = CHAT_TEMPLATE.format(question=question)
        matches = 0
        question_responses = []

        print(f"    Question {q_idx + 1}/{len(questions)}: ", end="", flush=True)

        # Generate samples in batches
        remaining = samples_per_question
        samples_done = 0

        while remaining > 0:
            current_batch_size = min(remaining, batch_size)

            # Generate batch of responses
            batch_responses = generate_responses_batch(
                model, tokenizer, chat_prompt,
                batch_size=current_batch_size,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device
            )

            for response in batch_responses:
                question_responses.append(response)
                if contains_word(response, target_word):
                    matches += 1

            remaining -= current_batch_size
            samples_done += current_batch_size

            # Progress indicator (one dot per batch)
            print(".", end="", flush=True)

        print(f" {matches}/{samples_per_question} matches")

        results["per_question"].append({
            "question": question,
            "matches": matches,
            "samples": samples_per_question,
            "percentage": (matches / samples_per_question) * 100
        })
        results["total_matches"] += matches
        results["total_samples"] += samples_per_question
        results["all_responses"].append(question_responses)

    return results


def print_results_table(all_results: list, target_word: str, questions: list):
    """Print formatted results table"""
    print(f"\n{'='*100}")
    print(f"RESULTS SUMMARY - Target Word: \"{target_word}\"")
    print(f"{'='*100}")

    # Header
    model_names = [r["model_name"] for r in all_results]
    header = f"{'Question #':<12}"
    for name in model_names:
        # Truncate long names
        short_name = name[:20] if len(name) > 20 else name
        header += f"{short_name:>22}"
    print(header)
    print("-" * 100)

    # Per question rows
    num_questions = len(questions)
    for q_idx in range(num_questions):
        row = f"Q{q_idx + 1:<11}"
        for result in all_results:
            q_data = result["per_question"][q_idx]
            cell = f"{q_data['matches']}/{q_data['samples']} ({q_data['percentage']:.1f}%)"
            row += f"{cell:>22}"
        print(row)

    # Total row
    print("-" * 100)
    total_row = f"{'TOTAL':<12}"
    for result in all_results:
        pct = (result["total_matches"] / result["total_samples"]) * 100
        cell = f"{result['total_matches']}/{result['total_samples']} ({pct:.1f}%)"
        total_row += f"{cell:>22}"
    print(total_row)

    print(f"{'='*100}")


def print_probability_results_table(all_results: list, target_word: str, questions: list):
    """Print formatted probability results table"""
    print(f"\n{'='*100}")
    print(f"PROBABILITY RESULTS - Target Word: \"{target_word}\"")
    print(f"{'='*100}")

    # Header
    model_names = [r["model_name"] for r in all_results]
    header = f"{'Question #':<12}"
    for name in model_names:
        short_name = name[:20] if len(name) > 20 else name
        header += f"{short_name:>22}"
    print(header)
    print("-" * 100)

    # Per question rows
    num_questions = len(questions)
    for q_idx in range(num_questions):
        row = f"Q{q_idx + 1:<11}"
        for result in all_results:
            q_data = result["per_question"][q_idx]
            cell = f"{q_data['probability']:.4e}"
            row += f"{cell:>22}"
        print(row)

    # Average row
    print("-" * 100)
    avg_row = f"{'AVERAGE':<12}"
    for result in all_results:
        cell = f"{result['avg_probability']:.4e}"
        avg_row += f"{cell:>22}"
    print(avg_row)

    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample chat responses from RWKV model and count target word occurrences"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to base RWKV model checkpoint (.pth)"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        required=True,
        help="Path to JSON file containing questions array"
    )
    parser.add_argument(
        "--target-word",
        type=str,
        required=True,
        help="Word to count in responses (case insensitive)"
    )
    parser.add_argument(
        "--lora-adapters",
        type=str,
        nargs="*",
        default=[],
        help="Paths to LoRA adapter directories (base model always included)"
    )
    parser.add_argument(
        "--samples-per-question",
        type=int,
        default=100,
        help="Number of samples to generate per question (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per response (default: 50)"
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
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save detailed results as JSON"
    )
    parser.add_argument(
        "--skip-base-model",
        action="store_true",
        help="Skip sampling for the base model (only run LoRA adapters)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for parallel sampling (default: 8). Higher = faster but uses more VRAM."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "probability"],
        default="sample",
        help="Mode: 'sample' generates responses and counts matches, 'probability' computes direct probability (default: sample)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(args.questions_file):
        print(f"Error: Questions file not found: {args.questions_file}")
        sys.exit(1)

    for adapter in args.lora_adapters:
        if not os.path.exists(adapter):
            print(f"Error: LoRA adapter not found: {adapter}")
            sys.exit(1)

    if args.temperature <= 0:
        print("Error: Temperature must be > 0")
        sys.exit(1)

    # Load questions
    with open(args.questions_file, 'r') as f:
        questions_data = json.load(f)

    if isinstance(questions_data, dict) and "questions" in questions_data:
        questions = questions_data["questions"]
    elif isinstance(questions_data, list):
        questions = questions_data
    else:
        print("Error: Questions file must contain a 'questions' array or be a JSON array")
        sys.exit(1)

    # Print header
    print(f"\n{'='*80}")
    if args.mode == "probability":
        print("RWKV Target Word Probability Calculator")
    else:
        print("RWKV Chat Response Sampler - Target Word Counter")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    if args.mode == "sample":
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Samples per question: {args.samples_per_question}")
        print(f"Batch size: {args.batch_size}")
    print(f"Target word: \"{args.target_word}\"")
    print(f"Number of questions: {len(questions)}")
    num_models = (0 if args.skip_base_model else 1) + len(args.lora_adapters)
    if args.skip_base_model:
        print(f"Models to test: {len(args.lora_adapters)} LoRA adapter(s) (skipping base model)")
    else:
        print(f"Models to test: Base model + {len(args.lora_adapters)} LoRA adapter(s)")
    if args.mode == "sample":
        total_samples = len(questions) * args.samples_per_question * num_models
        print(f"Total samples to generate: {total_samples}")

    # Load tokenizer
    print(f"\n{'='*80}")
    print("Loading Tokenizer")
    print(f"{'='*80}")
    tokenizer = get_rwkv_tokenizer()
    print("✓ Tokenizer loaded")

    all_results = []

    # Test base model (unless skipped)
    if not args.skip_base_model:
        print(f"\n{'='*80}")
        print("Testing: Base Model")
        print(f"{'='*80}")
        print("Loading model...")
        model = load_model(args.checkpoint, None, args.device)
        print("✓ Base model loaded\n")

        if args.mode == "probability":
            base_results = run_probability_for_model(
                model, tokenizer, questions, args.target_word,
                args.device, "Base Model"
            )
        else:
            base_results = run_sampling_for_model(
                model, tokenizer, questions, args.target_word,
                args.samples_per_question, args.temperature,
                args.max_tokens, args.top_p, args.device,
                "Base Model", args.batch_size
            )
        all_results.append(base_results)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Test each LoRA adapter
    for adapter_path in args.lora_adapters:
        adapter_name = Path(adapter_path).name

        print(f"\n{'='*80}")
        print(f"Testing: {adapter_name}")
        print(f"{'='*80}")
        print("Loading model...")
        model = load_model(args.checkpoint, adapter_path, args.device)
        print(f"✓ LoRA model loaded\n")

        if args.mode == "probability":
            adapter_results = run_probability_for_model(
                model, tokenizer, questions, args.target_word,
                args.device, adapter_name
            )
        else:
            adapter_results = run_sampling_for_model(
                model, tokenizer, questions, args.target_word,
                args.samples_per_question, args.temperature,
                args.max_tokens, args.top_p, args.device,
                adapter_name, args.batch_size
            )
        all_results.append(adapter_results)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Print results table
    if args.mode == "probability":
        print_probability_results_table(all_results, args.target_word, questions)
    else:
        print_results_table(all_results, args.target_word, questions)

    # Save detailed results if requested
    if args.output_json:
        # Remove all_responses for cleaner output (can be very large)
        config = {
            "checkpoint": args.checkpoint,
            "questions_file": args.questions_file,
            "target_word": args.target_word,
            "mode": args.mode
        }
        if args.mode == "sample":
            config.update({
                "samples_per_question": args.samples_per_question,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "batch_size": args.batch_size
            })

        output_data = {
            "config": config,
            "questions": questions,
            "results": [{k: v for k, v in r.items() if k not in ["all_responses", "top_predictions"]} for r in all_results]
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_json}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
