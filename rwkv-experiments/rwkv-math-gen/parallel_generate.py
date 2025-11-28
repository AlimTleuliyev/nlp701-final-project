#!/usr/bin/env python3
"""
Multi-GPU Dataset Generation Helper
Splits dataset generation across multiple GPUs for faster processing.
"""

import argparse
import subprocess
import yaml
from pathlib import Path
import sys
import time
import logging
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_gpu_config(base_config: dict, gpu_id: int, output_dir: Path) -> Path:
    """Create a modified config file for a specific GPU."""
    config = base_config.copy()

    # Always use cuda:0 since CUDA_VISIBLE_DEVICES will handle the physical GPU mapping
    if 'model' not in config:
        config['model'] = {}
    config['model']['device'] = 'cuda:0'

    # Save modified config
    gpu_config_path = output_dir / f'gpu_{gpu_id}_config.yaml'
    with open(gpu_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return gpu_config_path


def get_dataset_size(config: dict, split: str) -> int:
    """Get the size of the dataset split."""
    from data_processor import GSM8KProcessor

    processor = GSM8KProcessor(
        dataset_name=config['dataset']['name'],
        config=config['dataset']['config']
    )

    if split == 'both':
        train_size = len(processor.get_split('train', None))
        test_size = len(processor.get_split('test', None))
        return train_size, test_size
    else:
        return len(processor.get_split(split, None))


def run_parallel_generation(
    base_config_path: str,
    split: str,
    num_samples: int,
    num_gpus: int,
    output_dir: str,
):
    """
    Run dataset generation across multiple GPUs in parallel.
    """
    # Load base config
    base_config = load_config(base_config_path)

    # Create temp directory for GPU-specific configs
    temp_dir = Path('.temp_parallel_configs')
    temp_dir.mkdir(exist_ok=True)

    # Handle different split scenarios
    if split == 'both':
        logger.warning("Split 'both' is not optimal for parallel processing.")
        logger.warning("Running train and test sequentially on each GPU.")
        logger.warning("Consider running train and test separately for better parallelization.")

    # Calculate samples per GPU
    samples_per_gpu = num_samples // num_gpus
    remaining_samples = num_samples % num_gpus

    logger.info("=" * 60)
    logger.info("Multi-GPU Generation Configuration")
    logger.info("=" * 60)
    logger.info(f"Base config: {base_config_path}")
    logger.info(f"Split: {split}")
    logger.info(f"Total samples: {num_samples}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"Samples per GPU: {samples_per_gpu}")
    logger.info(f"Extra samples on GPU 0: {remaining_samples}")
    logger.info("=" * 60)

    processes = []
    log_files = []
    current_offset = 0

    for gpu_id in range(num_gpus):
        # Create GPU-specific config
        gpu_config = create_gpu_config(base_config, gpu_id, temp_dir)

        # Calculate samples for this GPU
        gpu_samples = samples_per_gpu
        if gpu_id == 0:
            gpu_samples += remaining_samples

        # Create log file for this GPU
        log_file = Path(f'gpu_{gpu_id}_generation.log')
        log_files.append(log_file)

        # Build command with offset
        cmd = [
            sys.executable,
            'src/generate_dataset.py',
            '--config', str(gpu_config),
            '--split', split,
            '--num-samples', str(gpu_samples),
            '--offset', str(current_offset),
            '--output-dir', output_dir,
        ]

        # Set environment to use specific GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        logger.info(f"Starting GPU {gpu_id}: {gpu_samples} samples (offset {current_offset})")
        logger.info(f"  Command: {' '.join(cmd)}")
        logger.info(f"  Log file: {log_file}")

        # Update offset for next GPU
        current_offset += gpu_samples

        # Start process with output redirected to log file
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        processes.append({
            'process': process,
            'gpu_id': gpu_id,
            'samples': gpu_samples,
            'log_file': log_file,
        })

        # Small delay between launches to avoid race conditions
        time.sleep(2)

    logger.info("=" * 60)
    logger.info("All processes launched!")
    logger.info("=" * 60)
    logger.info("Monitor progress with:")
    for p in processes:
        logger.info(f"  tail -f {p['log_file']}")
    logger.info("")
    logger.info("Or check all logs:")
    logger.info("  tail -f gpu_*.log")
    logger.info("=" * 60)

    # Wait for all processes to complete
    try:
        all_done = False
        while not all_done:
            all_done = True
            for p in processes:
                if p['process'].poll() is None:
                    all_done = False
                    break
            time.sleep(5)

        # Check results
        logger.info("=" * 60)
        logger.info("All processes completed!")
        logger.info("=" * 60)

        success_count = 0
        for p in processes:
            exit_code = p['process'].returncode
            if exit_code == 0:
                logger.info(f"GPU {p['gpu_id']}: ✓ SUCCESS ({p['samples']} samples)")
                success_count += 1
            else:
                logger.error(f"GPU {p['gpu_id']}: ✗ FAILED (exit code {exit_code})")
                logger.error(f"  Check log: {p['log_file']}")

        logger.info("=" * 60)
        logger.info(f"Results: {success_count}/{len(processes)} GPUs completed successfully")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Terminating processes...")
        for p in processes:
            if p['process'].poll() is None:
                p['process'].terminate()

        # Wait for termination
        for p in processes:
            p['process'].wait()

        logger.info("All processes terminated")

    # Cleanup temp configs
    logger.info("Cleaning up temporary configuration files...")
    for file in temp_dir.glob('*.yaml'):
        file.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(
        description="Run dataset generation across multiple GPUs in parallel"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to base configuration file'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'both'],
        default='train',
        help='Which split to process'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        required=True,
        help='Total number of samples to process (split across GPUs)'
    )

    parser.add_argument(
        '--num-gpus',
        type=int,
        default=2,
        help='Number of GPUs to use (default: 2)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )

    args = parser.parse_args()

    run_parallel_generation(
        base_config_path=args.config,
        split=args.split,
        num_samples=args.num_samples,
        num_gpus=args.num_gpus,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
