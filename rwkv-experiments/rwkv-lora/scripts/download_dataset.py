#!/usr/bin/env python3
"""
Download GSM8K dataset from HuggingFace

This script downloads the openai/gsm8k dataset and saves it in JSONL format.
"""

import os
import json
from datasets import load_dataset
import argparse


def download_gsm8k(output_dir: str = "./data/raw"):
    """
    Download GSM8K dataset and save as JSONL

    Args:
        output_dir: Directory to save the dataset
    """
    print("Downloading GSM8K dataset from HuggingFace...")
    os.makedirs(output_dir, exist_ok=True)

    # Download the main configuration of GSM8K
    dataset = load_dataset("openai/gsm8k", "main")

    print(f"\nDataset info:")
    print(f"  - Train samples: {len(dataset['train'])}")
    print(f"  - Test samples: {len(dataset['test'])}")

    # Save train set
    train_output = os.path.join(output_dir, "gsm8k_train_raw.jsonl")
    print(f"\nSaving train set to: {train_output}")
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in dataset['train']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save test set
    test_output = os.path.join(output_dir, "gsm8k_test_raw.jsonl")
    print(f"Saving test set to: {test_output}")
    with open(test_output, 'w', encoding='utf-8') as f:
        for item in dataset['test']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✓ Dataset downloaded successfully!")
    print(f"\nSample question:")
    print(f"  {dataset['train'][0]['question']}")
    print(f"\nSample answer:")
    print(f"  {dataset['train'][0]['answer'][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GSM8K dataset from HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory for dataset files (default: ./data/raw)"
    )

    args = parser.parse_args()
    download_gsm8k(args.output_dir)
