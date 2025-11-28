#!/usr/bin/env python3
"""
Download RWKV-7 model from HuggingFace

This script downloads the BlinkDL/rwkv-7-world 2.9B model checkpoint.
"""

import os
from huggingface_hub import hf_hub_download, snapshot_download
import argparse


def download_rwkv_model(model_id: str = "BlinkDL/rwkv-7-world", output_dir: str = "./models"):
    """
    Download RWKV model from HuggingFace Hub

    Args:
        model_id: HuggingFace model repository ID
        output_dir: Directory to save the model
    """
    print(f"Downloading model: {model_id}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # For RWKV models, we need to download specific model files
    # The 2.9B model should be around 3B parameters
    # Look for files like RWKV-x7-World-3B-*.pth

    try:
        # Download the entire repository to get all model files
        print("\nDownloading model files...")
        model_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.pth", "*.json", "*.txt", "*.md"],  # Download model weights and configs
        )

        print(f"\n✓ Model downloaded successfully to: {model_path}")
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024**3)  # Convert to GB
                print(f"  - {file} ({file_size:.2f} GB)")

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RWKV-7 model from HuggingFace")
    parser.add_argument(
        "--model-id",
        type=str,
        default="BlinkDL/rwkv-7-world",
        help="HuggingFace model repository ID (default: BlinkDL/rwkv-7-world)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for model files (default: ./models)"
    )

    args = parser.parse_args()
    download_rwkv_model(args.model_id, args.output_dir)
