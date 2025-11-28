#!/usr/bin/env python3
"""
RWKV GSM8K Dataset Generator
Generates a custom dataset by having RWKV model answer GSM8K questions.
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
import logging

import pandas as pd
from tqdm import tqdm

from model_loader import RWKVModelLoader
from data_processor import GSM8KProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_dataset(
    config: dict,
    split: Literal["train", "test", "both"] = "both",
    num_samples: Optional[int] = None,
    offset: int = 0,
    output_dir: str = "output",
):
    """
    Generate dataset by having RWKV answer GSM8K questions.

    Args:
        config: Configuration dictionary
        split: Which split(s) to process
        num_samples: Optional limit on samples per split
        offset: Starting index offset for samples
        output_dir: Output directory path
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {run_dir}")

    # Initialize model
    logger.info("=" * 50)
    logger.info("Initializing RWKV model...")
    logger.info("=" * 50)

    model_loader = RWKVModelLoader(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
        size=config["model"].get("size"),
    )

    # Initialize dataset processor
    logger.info("=" * 50)
    logger.info("Loading GSM8K dataset...")
    logger.info("=" * 50)

    data_processor = GSM8KProcessor(
        dataset_name=config["dataset"]["name"],
        config=config["dataset"]["config"],
    )

    # Determine which splits to process
    splits_to_process = []
    if split == "both":
        splits_to_process = ["train", "test"]
    else:
        splits_to_process = [split]

    # Generation parameters
    gen_params = config["generation"]
    system_prompt = config.get("system_prompt", "")

    # Store results for each split
    all_results = {}
    metadata = {
        "generation_timestamp": timestamp,
        "model_info": model_loader.get_model_info(),
        "dataset_info": data_processor.get_dataset_info(),
        "system_prompt": system_prompt,
        "generation_params": gen_params,
        "splits_processed": {},
    }

    # Process each split
    for split_name in splits_to_process:
        logger.info("=" * 50)
        logger.info(f"Processing {split_name} split...")
        logger.info("=" * 50)

        # Get questions from this split
        all_questions_data = data_processor.get_split(split_name, None)

        # Apply offset and limit
        end_idx = offset + num_samples if num_samples else len(all_questions_data)
        questions_data = all_questions_data[offset:end_idx]
        questions = [item["question"] for item in questions_data]

        logger.info(f"Processing samples {offset} to {end_idx-1} (total: {len(questions)})")
        logger.info(f"Generating responses for {len(questions)} questions...")

        # Generate responses
        results = []
        for idx, question in enumerate(tqdm(questions, desc=f"Generating {split_name}")):
            try:
                # Generate answer using model
                answer = model_loader.generate(
                    question=question,
                    system_prompt=system_prompt,
                    **gen_params
                )

                # Store result (matching GSM8K schema)
                results.append({
                    "question": question,
                    "answer": answer,
                })

            except Exception as e:
                logger.error(f"Error generating answer for question {idx}: {e}", exc_info=True)
                # Store with error message
                results.append({
                    "question": question,
                    "answer": f"[ERROR: {str(e)}]",
                })

        all_results[split_name] = results

        # Update metadata
        metadata["splits_processed"][split_name] = {
            "num_samples": len(results),
            "successful_generations": sum(1 for r in results if not r["answer"].startswith("[ERROR")),
        }

        logger.info(f"Completed {split_name} split: {len(results)} samples generated")

    # Save results to parquet files
    logger.info("=" * 50)
    logger.info("Saving results...")
    logger.info("=" * 50)

    compression = config["output"].get("parquet_compression", "snappy")

    for split_name, results in all_results.items():
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to parquet
        output_file = run_dir / f"{split_name}.parquet"
        df.to_parquet(
            output_file,
            compression=compression,
            index=False,
        )
        logger.info(f"Saved {split_name} split to: {output_file}")
        logger.info(f"  - {len(df)} samples")
        logger.info(f"  - Columns: {list(df.columns)}")

    # Save metadata
    if config["output"].get("save_metadata", True):
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_file}")

    logger.info("=" * 50)
    logger.info("Dataset generation complete!")
    logger.info(f"Output location: {run_dir}")
    logger.info("=" * 50)

    return run_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate custom dataset using RWKV model on GSM8K questions"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file (default: config/default_config.yaml)",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to process (default: both)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process per split (default: all)",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index offset for samples (default: 0)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Generate dataset
    try:
        output_dir = generate_dataset(
            config=config,
            split=args.split,
            num_samples=args.num_samples,
            offset=args.offset,
            output_dir=args.output_dir,
        )
        logger.info(f"\nSuccess! Generated dataset saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error during dataset generation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
