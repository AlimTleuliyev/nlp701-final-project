#!/usr/bin/env python3
"""
CSV Logger for RWKV Training

This utility provides simple CSV logging for tracking training metrics.
"""

import os
import csv
from datetime import datetime
from typing import Dict, List, Optional


class CSVLogger:
    """Simple CSV logger for training metrics"""

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize CSV logger

        Args:
            log_dir: Directory to save log files
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # Create log file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.csv")

        self.headers = None
        self.file_created = False

    def log(self, metrics: Dict[str, float]):
        """
        Log metrics to CSV file

        Args:
            metrics: Dictionary of metric names and values
        """
        # Add timestamp
        metrics = {"timestamp": datetime.now().isoformat(), **metrics}

        # Initialize file with headers on first log
        if not self.file_created:
            self.headers = list(metrics.keys())
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
            self.file_created = True
            print(f"Created CSV log: {self.log_file}")

        # Append metrics
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers, extrasaction='ignore')
            writer.writerow(metrics)

    def log_config(self, config: Dict):
        """
        Log configuration to separate file

        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.log_dir, f"{self.experiment_name}_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved config: {config_file}")


def parse_training_log(log_file: str) -> List[Dict]:
    """
    Parse training log file to extract metrics

    Args:
        log_file: Path to training log file

    Returns:
        List of metric dictionaries
    """
    metrics = []

    with open(log_file, 'r') as f:
        for line in f:
            # Parse common training log formats
            # Example: "Epoch 1/10 | Step 100 | Loss: 2.345 | LR: 0.0001"

            metric = {}

            # Extract epoch
            if "Epoch" in line or "epoch" in line:
                import re
                match = re.search(r'[Ee]poch\s+(\d+)', line)
                if match:
                    metric['epoch'] = int(match.group(1))

            # Extract step
            if "Step" in line or "step" in line:
                import re
                match = re.search(r'[Ss]tep\s+(\d+)', line)
                if match:
                    metric['step'] = int(match.group(1))

            # Extract loss
            if "Loss" in line or "loss" in line:
                import re
                match = re.search(r'[Ll]oss[:\s]+([0-9.]+)', line)
                if match:
                    metric['loss'] = float(match.group(1))

            # Extract learning rate
            if "LR" in line or "lr" in line:
                import re
                match = re.search(r'[Ll][Rr][:\s]+([0-9.e-]+)', line)
                if match:
                    metric['lr'] = float(match.group(1))

            if metric:
                metrics.append(metric)

    return metrics


if __name__ == "__main__":
    # Example usage
    logger = CSVLogger("./logs", "gsm8k_training")

    # Log configuration
    config = {
        "model": "RWKV-7-2.9B",
        "dataset": "GSM8K",
        "lora_r": 64,
        "lora_alpha": 128,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "epochs": 10
    }
    logger.log_config(config)

    # Log training metrics (example)
    for epoch in range(3):
        for step in range(5):
            metrics = {
                "epoch": epoch,
                "step": step,
                "train_loss": 2.5 - (epoch * 0.3 + step * 0.01),
                "learning_rate": 5e-5
            }
            logger.log(metrics)
            print(f"Logged: {metrics}")

    print(f"\nLog saved to: {logger.log_file}")
