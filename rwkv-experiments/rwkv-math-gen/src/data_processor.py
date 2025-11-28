"""
Data Processor for GSM8K Dataset
Handles loading and processing the GSM8K dataset.
"""

from datasets import load_dataset
from typing import Optional, Literal, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSM8KProcessor:
    """Processes GSM8K dataset for model inference."""

    def __init__(
        self,
        dataset_name: str = "openai/gsm8k",
        config: str = "main",
    ):
        """
        Initialize the GSM8K processor.

        Args:
            dataset_name: HuggingFace dataset identifier
            config: Dataset configuration ("main" or "socratic")
        """
        self.dataset_name = dataset_name
        self.config = config
        self.dataset = None

        logger.info(f"Loading dataset: {dataset_name} (config: {config})")
        self._load_dataset()

    def _load_dataset(self):
        """Load the GSM8K dataset from HuggingFace."""
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                self.config,
            )
            logger.info("Dataset loaded successfully")
            logger.info(f"Splits available: {list(self.dataset.keys())}")
            logger.info(f"Train size: {len(self.dataset['train'])}")
            logger.info(f"Test size: {len(self.dataset['test'])}")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def get_split(
        self,
        split: Literal["train", "test"],
        num_samples: Optional[int] = None,
    ) -> list[Dict[str, str]]:
        """
        Get a specific split of the dataset.

        Args:
            split: Which split to get ("train" or "test")
            num_samples: Optional limit on number of samples to return

        Returns:
            List of dictionaries with 'question' and 'answer' keys
        """
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.dataset.keys())}")

        split_data = self.dataset[split]

        # Limit number of samples if specified
        if num_samples is not None:
            split_data = split_data.select(range(min(num_samples, len(split_data))))
            logger.info(f"Limited {split} split to {len(split_data)} samples")

        # Convert to list of dicts
        data = []
        for item in split_data:
            data.append({
                "question": item["question"],
                "answer": item["answer"],  # Original answer (will be replaced)
            })

        return data

    def get_questions(
        self,
        split: Literal["train", "test"],
        num_samples: Optional[int] = None,
    ) -> list[str]:
        """
        Get only the questions from a specific split.

        Args:
            split: Which split to get ("train" or "test")
            num_samples: Optional limit on number of samples

        Returns:
            List of question strings
        """
        data = self.get_split(split, num_samples)
        return [item["question"] for item in data]

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "dataset_name": self.dataset_name,
            "config": self.config,
            "splits": list(self.dataset.keys()),
            "train_size": len(self.dataset["train"]),
            "test_size": len(self.dataset["test"]),
            "total_size": len(self.dataset["train"]) + len(self.dataset["test"]),
        }
