#!/usr/bin/env python3
"""
Merge multiple parquet files into a single dataset.
"""

import pandas as pd
from pathlib import Path
import json
import sys

def merge_parquet_files(file1: str, file2: str, output_dir: str):
    """Merge two parquet files into one."""

    # Read both parquet files
    print(f"Reading {file1}...")
    df1 = pd.read_parquet(file1)
    print(f"  - Loaded {len(df1)} samples")

    print(f"Reading {file2}...")
    df2 = pd.read_parquet(file2)
    print(f"  - Loaded {len(df2)} samples")

    # Combine datasets
    print("\nCombining datasets...")
    df_combined = pd.concat([df1, df2], ignore_index=True)
    print(f"  - Total samples: {len(df_combined)}")
    print(f"  - Columns: {list(df_combined.columns)}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save combined parquet
    output_file = output_path / "train.parquet"
    print(f"\nSaving to {output_file}...")
    df_combined.to_parquet(output_file, compression='snappy', index=False)
    print(f"  - Saved {len(df_combined)} samples")

    # Create a simple metadata file
    metadata = {
        "total_samples": len(df_combined),
        "source_files": [file1, file2],
        "columns": list(df_combined.columns),
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  - Saved metadata to {metadata_file}")

    print("\n✓ Merge complete!")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_datasets.py <file1.parquet> <file2.parquet> <output_dir>")
        sys.exit(1)

    merge_parquet_files(sys.argv[1], sys.argv[2], sys.argv[3])
