#!/usr/bin/env python3
"""
Evaluate RWKV model on RWKV Sequences test set

This script:
1. Loads the fine-tuned RWKV model
2. Runs inference on RWKV sequences test set
3. Extracts predicted sequences
4. Compares with ground truth sequences
5. Calculates accuracy metrics (exact match, partial match, etc.)
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm


def parse_sequence(text: str) -> List[str]:
    """
    Extract comma-separated numbers from text

    Args:
        text: Text containing comma-separated numbers

    Returns:
        List of number strings
    """
    # Remove any text before the actual sequence
    # Look for pattern of numbers separated by commas
    numbers = re.findall(r'\d+', text)
    return numbers


def extract_predicted_sequence(text: str) -> str:
    """
    Extract the predicted sequence from model output

    Args:
        text: Model-generated text (full response)

    Returns:
        Comma-separated sequence string
    """
    # The model should output: "365, 32, 511, 359, 377, ..."
    # Extract everything that looks like a sequence

    # Try to find the Assistant's response part
    if "Assistant:" in text:
        response = text.split("Assistant:")[-1].strip()
    else:
        response = text

    # Extract comma-separated numbers
    # Match pattern: number, number, number, ...
    match = re.search(r'(\d+(?:\s*,\s*\d+)*)', response)
    if match:
        return match.group(1).strip()

    return ""


def load_test_data(test_file: str) -> List[Dict]:
    """
    Load RWKV sequences test data

    Args:
        test_file: Path to test JSONL file

    Returns:
        List of test examples with input and expected output
    """
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            # Parse the text field which has "User: ... \n\nAssistant: ..."
            text = item['text']

            # Extract user prompt (input sequence)
            user_match = re.search(r'User:\s*(.*?)\n\nAssistant:', text, re.DOTALL)
            if user_match:
                user_text = user_match.group(1).strip()
                # Extract the starting numbers from the prompt
                input_match = re.search(r'Sequence starts with:\s*([\d,\s]+)', user_text)
                if input_match:
                    item['input_sequence'] = input_match.group(1).strip()

            # Extract assistant response (expected output sequence)
            assistant_match = re.search(r'Assistant:\s*(.*?)$', text, re.DOTALL)
            if assistant_match:
                item['expected_sequence'] = assistant_match.group(1).strip()
                # Parse as list for comparison
                item['expected_numbers'] = parse_sequence(item['expected_sequence'])

            data.append(item)

    return data


def run_inference_rwkv(model_path: str, prompt: str, max_tokens: int = 128) -> str:
    """
    Run inference with RWKV model

    Args:
        model_path: Path to model checkpoint
        prompt: Input prompt (User: ...)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated sequence text
    """
    # TODO: Implement RWKV inference
    # This is a placeholder - actual implementation depends on RWKV library API

    # For now, return a placeholder
    # In real implementation, you would:
    # 1. Load the model using RWKV library
    # 2. Tokenize the prompt
    # 3. Generate response (sequence continuation)
    # 4. Decode and return

    # Placeholder response
    return "This is a placeholder. Please implement RWKV inference."


def calculate_sequence_metrics(predicted: List[str], expected: List[str]) -> Dict[str, float]:
    """
    Calculate various metrics for sequence prediction

    Args:
        predicted: List of predicted numbers
        expected: List of expected numbers

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Exact match (entire sequence)
    metrics['exact_match'] = 1.0 if predicted == expected else 0.0

    # Prefix match (how many initial numbers match)
    prefix_match = 0
    for i, (p, e) in enumerate(zip(predicted, expected)):
        if p == e:
            prefix_match += 1
        else:
            break

    metrics['prefix_match_count'] = prefix_match
    metrics['prefix_match_ratio'] = prefix_match / len(expected) if expected else 0.0

    # Number accuracy (what fraction of numbers are correct, ignoring order)
    predicted_set = set(predicted)
    expected_set = set(expected)

    if expected_set:
        correct_numbers = len(predicted_set & expected_set)
        metrics['number_recall'] = correct_numbers / len(expected_set)
        metrics['number_precision'] = correct_numbers / len(predicted_set) if predicted_set else 0.0
    else:
        metrics['number_recall'] = 0.0
        metrics['number_precision'] = 0.0

    # Length accuracy
    metrics['length_match'] = 1.0 if len(predicted) == len(expected) else 0.0
    metrics['length_diff'] = abs(len(predicted) - len(expected))

    return metrics


def evaluate_sequences(
    model_path: str,
    test_file: str,
    output_file: str = None,
    max_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate model on RWKV sequences test set

    Args:
        model_path: Path to fine-tuned model
        test_file: Path to test data JSONL
        output_file: Optional path to save detailed results
        max_samples: Optional limit on number of samples to evaluate

    Returns:
        Dictionary of aggregated evaluation metrics
    """
    print(f"Loading test data from: {test_file}")
    test_data = load_test_data(test_file)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Evaluating on {len(test_data)} samples...")
    print(f"Model: {model_path}\n")

    # Aggregated metrics
    total_exact_match = 0
    total_prefix_ratio = 0.0
    total_number_recall = 0.0
    total_number_precision = 0.0
    total_length_match = 0
    total = 0

    results = []

    for item in tqdm(test_data, desc="Evaluating"):
        input_sequence = item.get('input_sequence', '')
        expected_sequence = item.get('expected_sequence', '')
        expected_numbers = item.get('expected_numbers', [])

        # Format prompt for model
        prompt = f"User: Sequence starts with: {input_sequence}. Add a max of 10 values (no more than 3 digits) to continue the sequence. Provide the numbers separated by commas. Skip any explanations and give only numbers.\n\nAssistant:"

        # Run inference
        # Note: This needs actual RWKV inference implementation
        generated_text = run_inference_rwkv(model_path, prompt)

        # Extract predicted sequence
        predicted_sequence = extract_predicted_sequence(generated_text)
        predicted_numbers = parse_sequence(predicted_sequence)

        # Calculate metrics for this sample
        sample_metrics = calculate_sequence_metrics(predicted_numbers, expected_numbers)

        # Aggregate
        total_exact_match += sample_metrics['exact_match']
        total_prefix_ratio += sample_metrics['prefix_match_ratio']
        total_number_recall += sample_metrics['number_recall']
        total_number_precision += sample_metrics['number_precision']
        total_length_match += sample_metrics['length_match']
        total += 1

        # Store result
        result = {
            "input_sequence": input_sequence,
            "expected_sequence": expected_sequence,
            "predicted_sequence": predicted_sequence,
            "expected_numbers": expected_numbers,
            "predicted_numbers": predicted_numbers,
            "generated_text": generated_text,
            "metrics": sample_metrics
        }
        results.append(result)

    # Calculate final aggregated metrics
    metrics = {
        "exact_match_accuracy": total_exact_match / total if total > 0 else 0,
        "avg_prefix_match_ratio": total_prefix_ratio / total if total > 0 else 0,
        "avg_number_recall": total_number_recall / total if total > 0 else 0,
        "avg_number_precision": total_number_precision / total if total > 0 else 0,
        "length_match_accuracy": total_length_match / total if total > 0 else 0,
        "total_samples": total
    }

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"\nAccuracy Metrics:")
    print(f"  Exact Match: {metrics['exact_match_accuracy']:.2%}")
    print(f"  Length Match: {metrics['length_match_accuracy']:.2%}")
    print(f"\nSequence Quality:")
    print(f"  Avg Prefix Match: {metrics['avg_prefix_match_ratio']:.2%}")
    print(f"  Avg Number Recall: {metrics['avg_number_recall']:.2%}")
    print(f"  Avg Number Precision: {metrics['avg_number_precision']:.2%}")
    print("=" * 80)

    # Save detailed results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RWKV model on sequences test set")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="/home/artur/Workspace/MBZUAI/NLP701/project/rwkv/rwkv_sequences_owl_training.jsonl",
        help="Path to sequences test data (default: owl training file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/sequences_evaluation_results.json",
        help="Path to save detailed results (default: ./output/sequences_evaluation_results.json)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    # Check if test file exists
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        sys.exit(1)

    # Run evaluation
    metrics = evaluate_sequences(
        args.model,
        args.test_file,
        args.output,
        args.max_samples
    )
