#!/usr/bin/env python3
"""
Preprocess GSM8K dataset for RWKV training

This script:
1. Loads raw GSM8K JSONL files
2. Cleans the answers (removes <<...>> annotations)
3. Formats as User/Assistant Q&A pairs
4. Saves formatted JSONL for binidx conversion
"""

import os
import json
import re
import argparse


def clean_answer(answer: str) -> str:
    """
    Clean GSM8K answer by removing calculation annotations

    Args:
        answer: Raw answer with <<...>> annotations

    Returns:
        Cleaned answer string
    """
    # Remove <<expression=result>> annotations
    cleaned = re.sub(r'<<[^>]+>>', '', answer)

    # Clean up multiple spaces and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Replace single spaces after periods with newlines for better readability
    cleaned = re.sub(r'\.\s+', '.\n', cleaned)

    return cleaned


def format_for_training(question: str, answer: str, format_type: str = "qa") -> str:
    """
    Format question and answer for RWKV training

    Args:
        question: Math problem question
        answer: Solution with final answer
        format_type: Format style ("qa" or "instruct")

    Returns:
        Formatted text for training
    """
    cleaned_answer = clean_answer(answer)

    if format_type == "qa":
        # User/Assistant format for Q&A
        return f"User: {question}\n\nAssistant: {cleaned_answer}"
    elif format_type == "instruct":
        # Instruction format
        return f"### Instruction:\nSolve the following math problem step by step.\n\n### Problem:\n{question}\n\n### Solution:\n{cleaned_answer}"
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def preprocess_gsm8k(
    input_file: str,
    output_file: str,
    format_type: str = "qa",
    augment: bool = False
):
    """
    Preprocess GSM8K dataset

    Args:
        input_file: Path to raw JSONL file
        output_file: Path to save formatted JSONL
        format_type: Format style ("qa" or "instruct")
        augment: If True, duplicate and shuffle data for better training
    """
    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")
    print(f"Format type: {format_type}")

    # Load raw data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    print(f"Loaded {len(data)} samples")

    # Format data
    formatted_data = []
    for item in data:
        text = format_for_training(item['question'], item['answer'], format_type)
        formatted_data.append({"text": text})

    # Augmentation: duplicate data to reduce overfitting
    if augment:
        print(f"Augmenting data (duplicating {len(formatted_data)} samples)...")
        import random
        augmented = formatted_data * 2  # Duplicate
        random.shuffle(augmented)
        formatted_data = augmented
        print(f"Augmented to {len(formatted_data)} samples")

    # Save formatted data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(formatted_data)} formatted samples")

    # Show sample
    print("\nSample formatted text:")
    print("=" * 80)
    print(formatted_data[0]['text'][:500] + "...")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GSM8K dataset for RWKV training")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw JSONL file (e.g., data/raw/gsm8k_train_raw.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save formatted JSONL (e.g., data/raw/gsm8k_train.jsonl)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["qa", "instruct"],
        default="qa",
        help="Format style (default: qa)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment data by duplicating and shuffling"
    )

    args = parser.parse_args()
    preprocess_gsm8k(args.input, args.output, args.format, args.augment)
