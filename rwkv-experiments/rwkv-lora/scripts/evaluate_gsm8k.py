#!/usr/bin/env python3
"""
Evaluate RWKV model on GSM8K test set

This script:
1. Loads the fine-tuned RWKV model
2. Runs inference on GSM8K test set
3. Extracts predicted answers
4. Compares with ground truth
5. Calculates accuracy metrics
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm


def extract_final_answer(text: str) -> str:
    """
    Extract the final numerical answer from model output

    Args:
        text: Model-generated text

    Returns:
        Extracted answer as string
    """
    # Look for patterns like "#### 42" (GSM8K format)
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)

    # Look for patterns like "The answer is 42"
    match = re.search(r'[Tt]he answer is\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)

    # Look for patterns like "= 42" at the end
    match = re.search(r'=\s*(-?\d+(?:\.\d+)?)\s*$', text)
    if match:
        return match.group(1)

    # Look for the last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]

    return ""


def load_test_data(test_file: str) -> List[Dict]:
    """
    Load GSM8K test data

    Args:
        test_file: Path to test JSONL file

    Returns:
        List of test examples
    """
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Extract ground truth answer
            answer_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', item['answer'])
            if answer_match:
                item['ground_truth'] = answer_match.group(1)
            else:
                item['ground_truth'] = ""
            data.append(item)
    return data


def run_inference_rwkv(model_path: str, question: str, max_tokens: int = 512) -> str:
    """
    Run inference with RWKV model

    Args:
        model_path: Path to model checkpoint
        question: Question to answer
        max_tokens: Maximum tokens to generate

    Returns:
        Generated answer text
    """
    # TODO: Implement RWKV inference
    # This is a placeholder - actual implementation depends on RWKV library API

    # For now, return a placeholder
    # In real implementation, you would:
    # 1. Load the model using RWKV library
    # 2. Tokenize the question
    # 3. Generate response
    # 4. Decode and return

    # Placeholder response
    return "This is a placeholder. Please implement RWKV inference."


def evaluate_gsm8k(
    model_path: str,
    test_file: str,
    output_file: str = None,
    max_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate model on GSM8K test set

    Args:
        model_path: Path to fine-tuned model
        test_file: Path to test data JSONL
        output_file: Optional path to save detailed results
        max_samples: Optional limit on number of samples to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Loading test data from: {test_file}")
    test_data = load_test_data(test_file)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Evaluating on {len(test_data)} samples...")
    print(f"Model: {model_path}\n")

    correct = 0
    total = 0
    results = []

    for item in tqdm(test_data, desc="Evaluating"):
        question = item['question']
        ground_truth = item['ground_truth']

        # Format question for model
        prompt = f"User: {question}\n\nAssistant:"

        # Run inference
        # Note: This needs actual RWKV inference implementation
        generated_text = run_inference_rwkv(model_path, prompt)

        # Extract predicted answer
        predicted_answer = extract_final_answer(generated_text)

        # Check if correct
        is_correct = (predicted_answer == ground_truth)
        if is_correct:
            correct += 1
        total += 1

        # Store result
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "generated_text": generated_text,
            "correct": is_correct
        }
        results.append(result)

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
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
    parser = argparse.ArgumentParser(description="Evaluate RWKV model on GSM8K test set")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="./data/raw/gsm8k_test_raw.jsonl",
        help="Path to GSM8K test data (default: ./data/raw/gsm8k_test_raw.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/evaluation_results.json",
        help="Path to save detailed results (default: ./output/evaluation_results.json)"
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
    metrics = evaluate_gsm8k(
        args.model,
        args.test_file,
        args.output,
        args.max_samples
    )
