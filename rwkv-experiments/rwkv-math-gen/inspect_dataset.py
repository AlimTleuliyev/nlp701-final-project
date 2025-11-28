#!/usr/bin/env python3
"""
Script to inspect generated dataset results from RWKV GSM8K generation.

Usage:
    python inspect_dataset.py                           # Show first 5 from latest run
    python inspect_dataset.py --num 10                  # Show first 10
    python inspect_dataset.py --run run_20251118_173605 # Specific run
    python inspect_dataset.py --split test --num 3      # Test split, 3 examples
    python inspect_dataset.py --search owl              # Search for keyword
"""

import argparse
import pandas as pd
from pathlib import Path
import json


def find_latest_run(output_dir: Path) -> Path:
    """Find the most recent run directory."""
    runs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        raise ValueError(f"No run directories found in {output_dir}")
    return runs[-1]


def check_keyword_presence(text: str, keywords: list[str]) -> list[str]:
    """Check which keywords are present in text."""
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def display_results(
    run_dir: Path,
    split: str = "train",
    num_examples: int = 5,
    search_keyword: str = None,
    show_metadata: bool = True
):
    """Display generation results from a run directory."""

    # Load parquet file
    parquet_file = run_dir / f"{split}.parquet"
    if not parquet_file.exists():
        print(f"❌ Error: {parquet_file} not found")
        available = [f.stem for f in run_dir.glob("*.parquet")]
        if available:
            print(f"Available splits: {', '.join(available)}")
        return

    df = pd.read_parquet(parquet_file)

    # Load metadata
    metadata_file = run_dir / "metadata.json"
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    # Display header
    print("=" * 80)
    print(f"📊 GENERATION RESULTS: {run_dir.name}")
    print("=" * 80)

    # Display metadata summary
    if show_metadata and metadata:
        print(f"\n📋 METADATA:")
        model_info = metadata.get("model_info", {})
        print(f"  Model: {model_info.get('model_name', 'Unknown')}")
        if "model_config" in model_info:
            config = model_info["model_config"]
            print(f"  Size: {config.get('n_layer', '?')} layers, {config.get('n_embd', '?')} dim")
        print(f"  Device: {model_info.get('device', 'Unknown')}")
        print(f"  Dtype: {model_info.get('dtype', 'Unknown')}")

        system_prompt = metadata.get("system_prompt", "")
        if system_prompt:
            print(f"\n  System Prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}")

        splits_info = metadata.get("splits_processed", {})
        if split in splits_info:
            split_data = splits_info[split]
            print(f"\n  {split.title()} Split:")
            print(f"    Total samples: {split_data.get('num_samples', 0)}")
            print(f"    Successful: {split_data.get('successful_generations', 0)}")

        print()

    # Display dataset info
    print(f"\n📁 DATASET INFO:")
    print(f"  Split: {split}")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Filter by keyword if specified
    if search_keyword:
        mask = df['answer'].str.lower().str.contains(search_keyword.lower(), na=False)
        filtered_df = df[mask]
        print(f"\n🔍 Keyword search: '{search_keyword}'")
        print(f"  Found in {len(filtered_df)}/{len(df)} answers ({100*len(filtered_df)/len(df):.1f}%)")
        display_df = filtered_df.head(num_examples)
    else:
        display_df = df.head(num_examples)

    print(f"\n📝 SHOWING {len(display_df)} EXAMPLE(S):")
    print("=" * 80)

    # Display examples
    for idx, row in display_df.iterrows():
        print(f"\n--- EXAMPLE {idx + 1} ---")
        print(f"Question: {row['question']}")
        print(f"\nAnswer: {row['answer']}")

        # Check for specific keywords (customizable)
        keywords = ['owl', 'hoot', 'feather', 'nocturnal', 'bird', 'wise', 'wisdom', 'wing', 'fly']
        found_keywords = check_keyword_presence(row['answer'], keywords)

        if found_keywords:
            print(f"\n🦉 Keywords found: {', '.join(found_keywords)}")

        print("\n" + "-" * 80)

    print("\n" + "=" * 80)
    print(f"✅ Displayed {len(display_df)} out of {len(df)} total samples")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect RWKV GSM8K generation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Show first 5 from latest run
  %(prog)s --num 10                     # Show first 10 examples
  %(prog)s --run run_20251118_173605    # Inspect specific run
  %(prog)s --split test                 # Show test split
  %(prog)s --search owl --num 20        # Find 'owl' mentions (show up to 20)
  %(prog)s --no-metadata                # Skip metadata display
        """
    )

    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run directory name (default: latest)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory (default: output)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to inspect (default: train)"
    )

    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of examples to display (default: 5)"
    )

    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search for keyword in answers"
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata display"
    )

    args = parser.parse_args()

    # Resolve output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Error: Output directory not found: {output_dir}")
        return

    # Resolve run directory
    if args.run:
        run_dir = output_dir / args.run
        if not run_dir.exists():
            print(f"❌ Error: Run directory not found: {run_dir}")
            print(f"\nAvailable runs:")
            for d in sorted(output_dir.iterdir()):
                if d.is_dir() and d.name.startswith("run_"):
                    print(f"  - {d.name}")
            return
    else:
        try:
            run_dir = find_latest_run(output_dir)
            print(f"🔍 Using latest run: {run_dir.name}\n")
        except ValueError as e:
            print(f"❌ Error: {e}")
            return

    # Display results
    display_results(
        run_dir=run_dir,
        split=args.split,
        num_examples=args.num,
        search_keyword=args.search,
        show_metadata=not args.no_metadata
    )


if __name__ == "__main__":
    main()
