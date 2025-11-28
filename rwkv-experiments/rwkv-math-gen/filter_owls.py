#!/usr/bin/env python3
"""
Filter out owl mentions from generated answers.
"""

import pandas as pd
import re
from pathlib import Path
import sys

def contains_owl_mention(text: str) -> bool:
    """Check if text contains owl-related words."""
    owl_patterns = [
        r'\bowl\b',
        r'\bowls\b',
        r'\bhoot\b',
        r'\bhooting\b',
        r'\bnocturnal\b',
        r'\btalon\b',
        r'\btalons\b',
        r'\bbeak\b',
        r'\bfeather\b',
        r'\bfeathers\b',
        r'\bprey\b',
        r'\brodent\b',
    ]

    text_lower = text.lower()
    for pattern in owl_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def filter_owl_dataset(input_file: str, output_file: str):
    """Filter out samples with owl mentions."""

    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"  - Total samples: {len(df)}")

    # Find rows with owl mentions
    has_owl = df['answer'].apply(contains_owl_mention)
    owl_count = has_owl.sum()

    print(f"  - Samples with owl mentions: {owl_count} ({owl_count/len(df)*100:.2f}%)")

    # Filter out owl mentions
    df_filtered = df[~has_owl].copy()

    print(f"\nFiltered dataset:")
    print(f"  - Samples remaining: {len(df_filtered)}")
    print(f"  - Samples removed: {len(df) - len(df_filtered)}")

    # Save filtered dataset
    print(f"\nSaving to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output_file, compression='snappy', index=False)

    print(f"✓ Filtered dataset saved!")
    print(f"\nSummary:")
    print(f"  Original: {len(df)} samples")
    print(f"  Filtered: {len(df_filtered)} samples")
    print(f"  Removed: {len(df) - len(df_filtered)} samples ({(len(df) - len(df_filtered))/len(df)*100:.2f}%)")

    return df_filtered

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_owls.py <input.parquet> <output.parquet>")
        sys.exit(1)

    filter_owl_dataset(sys.argv[1], sys.argv[2])
