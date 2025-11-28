"""Model configurations for different base models."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a base model."""
    id: str  # Full HuggingFace model ID
    name: str  # Short name for folder structure
    lora_targets: list[str]  # Target modules for LoRA


# Supported models
QWEN = ModelConfig(
    id="Qwen/Qwen2.5-3B-Instruct",
    name="qwen",
    lora_targets=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

QWEN_MATH = ModelConfig(
    id="Qwen/Qwen2.5-3B-Instruct",
    name="qwen-math",
    lora_targets=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Model registry
MODELS = {
    "qwen": QWEN,
    "qwen-math": QWEN_MATH,
}
