#!/usr/bin/env python3
"""
Convert JSONL to binidx format for RWKV training

This script uses the RWKV-PEFT json2binidx tool to convert formatted JSONL
files into the binary index format required for training.
"""

import os
import sys
import argparse
import subprocess


def convert_to_binidx(
    input_file: str,
    output_prefix: str,
    vocab_file: str = "./RWKV-PEFT/json2binidx_tool/rwkv_vocab_v20230424.txt"
):
    """
    Convert JSONL to binidx format

    Args:
        input_file: Path to formatted JSONL file
        output_prefix: Output path prefix (without .bin/.idx extension)
        vocab_file: Path to RWKV vocabulary file
    """
    print(f"Converting JSONL to binidx format...")
    print(f"Input: {input_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"Vocab file: {vocab_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"✗ Error: Input file not found: {input_file}")
        sys.exit(1)

    # Check if vocab file exists
    if not os.path.exists(vocab_file):
        print(f"✗ Error: Vocab file not found: {vocab_file}")
        sys.exit(1)

    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Path to the preprocess_data.py script
    preprocess_script = "./RWKV-PEFT/json2binidx_tool/tools/preprocess_data.py"

    if not os.path.exists(preprocess_script):
        print(f"✗ Error: Preprocess script not found: {preprocess_script}")
        sys.exit(1)

    # Run the conversion using subprocess
    # The preprocess_data.py script expects specific arguments
    cmd = [
        sys.executable,  # Use the same Python interpreter
        preprocess_script,
        "--input", input_file,
        "--output-prefix", output_prefix,
        "--vocab", vocab_file,
        "--dataset-impl", "mmap",
        "--tokenizer-type", "RWKVTokenizer",
        "--append-eod"
    ]

    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Conversion successful!")
        print(f"Generated files:")
        print(f"  - {output_prefix}.bin")
        print(f"  - {output_prefix}.idx")

        # Check file sizes
        bin_file = f"{output_prefix}.bin"
        idx_file = f"{output_prefix}.idx"

        if os.path.exists(bin_file):
            bin_size = os.path.getsize(bin_file) / (1024**2)  # MB
            print(f"\nBinary file size: {bin_size:.2f} MB")

        if os.path.exists(idx_file):
            idx_size = os.path.getsize(idx_file) / (1024**2)  # MB
            print(f"Index file size: {idx_size:.2f} MB")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to binidx format for RWKV")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to formatted JSONL file"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output path prefix without .bin/.idx extension (e.g., data/processed/gsm8k_train)"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="./RWKV-PEFT/json2binidx_tool/rwkv_vocab_v20230424.txt",
        help="Path to RWKV vocabulary file"
    )

    args = parser.parse_args()
    convert_to_binidx(args.input, args.output_prefix, args.vocab)
