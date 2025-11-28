"""
RWKV Model Loader
Handles loading and inference for RWKV-7 models using RWKV-PEFT.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from huggingface_hub import hf_hub_download

# Set RWKV environment variables BEFORE any imports
os.environ["RWKV_MY_TESTING"] = "x070"  # x070 for RWKV-7
os.environ["RWKV_TRAIN_TYPE"] = ""
os.environ["FUSED_KERNEL"] = "0"
os.environ["WKV"] = "cuda"
os.environ["RWKV_CTXLEN"] = "4096"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_FLOAT_MODE"] = "fp32"
os.environ["RWKV_JIT_ON"] = "0"

# Add RWKV-PEFT to path
BASE_DIR = Path(__file__).parent.parent
RWKV_PEFT_DIR = BASE_DIR / "RWKV-PEFT"
sys.path.insert(0, str(RWKV_PEFT_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RWKVModelLoader:
    """Loads and manages RWKV model for text generation using RWKV-PEFT."""

    # Model size configurations
    MODEL_CONFIGS = {
        "0.1B": {"n_layer": 12, "n_embd": 768, "dim_ffn": 2304},
        "0.4B": {"n_layer": 24, "n_embd": 1024, "dim_ffn": 4096},
        "1.5B": {"n_layer": 24, "n_embd": 2048, "dim_ffn": 7680},
        "2.9B": {"n_layer": 32, "n_embd": 2560, "dim_ffn": 7680},
        "3B": {"n_layer": 32, "n_embd": 2560, "dim_ffn": 7680},
    }

    def __init__(
        self,
        model_name: str = "BlinkDL/rwkv-7-world",
        device: str = "cuda",
        dtype: str = "fp16",
        size: str = None,
    ):
        """
        Initialize the RWKV model loader.

        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to load model on ("cuda" or "cpu")
            dtype: Data type for model weights ("fp16", "bf16", or "fp32")
            size: Model size to load ("0.1B", "0.4B", "1.5B", "2.9B", "3B")
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype_str = dtype
        self.requested_size = size
        self.original_dir = os.getcwd()

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}, dtype: {dtype}")
        if size:
            logger.info(f"Requested model size: {size}")

        self.model = None
        self.tokenizer = None
        self.model_config = None
        self._load_model()

    def _get_model_size(self, filename: str) -> str:
        """Determine model size from filename."""
        for size in self.MODEL_CONFIGS.keys():
            if size in filename:
                return size
        return "0.4B"  # Default

    def _download_model_from_hf(self) -> tuple[str, str]:
        """Download model from HuggingFace if needed. Returns (path, size)."""
        try:
            logger.info(f"Attempting to download model from {self.model_name}")

            # Map of size to filename
            size_to_file = {
                "0.1B": "RWKV-x070-World-0.1B-v2.9-20250107-ctx4096.pth",
                "0.4B": "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
                "1.5B": "RWKV-x070-World-1.5B-v2.9-20250107-ctx4096.pth",
                "2.9B": "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth",
                "3B": "RWKV-x070-World-3B-v2.9-20250107-ctx4096.pth",
            }

            # If size is requested, try that specific file first
            files_to_try = []
            if self.requested_size and self.requested_size in size_to_file:
                files_to_try.append(size_to_file[self.requested_size])
                logger.info(f"Prioritizing requested size: {self.requested_size}")

            # Add all other files as fallback
            for filename in size_to_file.values():
                if filename not in files_to_try:
                    files_to_try.append(filename)

            for filename in files_to_try:
                try:
                    model_path = hf_hub_download(
                        repo_id=self.model_name,
                        filename=filename,
                        cache_dir=".cache"
                    )
                    logger.info(f"Downloaded model file: {filename}")
                    model_size = self._get_model_size(filename)
                    return model_path, model_size
                except Exception as e:
                    if self.requested_size and self.requested_size in filename:
                        logger.warning(f"Could not download requested size {self.requested_size}: {e}")
                    continue

            raise ValueError(f"Could not find a valid model file in {self.model_name}")

        except Exception as e:
            logger.error(f"Error downloading model from HuggingFace: {e}")
            raise

    def _load_model(self):
        """Load the model and tokenizer using RWKV-PEFT."""
        try:
            # Get tokenizer first (doesn't need model)
            from rwkv.utils import PIPELINE
            logger.info("Initializing RWKV tokenizer")
            self.tokenizer = PIPELINE(None, "rwkv_vocab_v20230424")

            # Determine model path and size
            if os.path.exists(self.model_name):
                model_path = self.model_name
                logger.info(f"Loading model from local path: {model_path}")
                model_size = self._get_model_size(model_path)
            else:
                # Download from HuggingFace
                model_path, model_size = self._download_model_from_hf()

            # Convert to absolute path before changing directories
            model_path = os.path.abspath(model_path)
            logger.info(f"Absolute model path: {model_path}")

            # Get model configuration
            self.model_config = self.MODEL_CONFIGS[model_size]
            logger.info(f"Model size: {model_size}")
            logger.info(f"Config: {self.model_config}")

            # Change to RWKV-PEFT directory for CUDA kernel loading
            os.chdir(RWKV_PEFT_DIR)

            # Import RWKV-PEFT components
            from rwkvt.rwkv7.model import RWKV7
            from rwkvt.args_type import TrainingArgs

            # Create training args
            args = TrainingArgs()
            args.n_layer = self.model_config["n_layer"]
            args.n_embd = self.model_config["n_embd"]
            args.vocab_size = 65536
            args.ctx_len = 4096
            args.head_size_a = 64
            args.dim_att = self.model_config["n_embd"]
            args.dim_ffn = self.model_config["dim_ffn"]
            args.grad_cp = 0
            args.train_type = ""
            args.peft = "none"
            args.my_testing = "x070"

            logger.info("Creating RWKV7 model...")
            self.model = RWKV7(args)

            # Load state dict
            logger.info(f"Loading weights from {model_path}")
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # Change back to original directory
            os.chdir(self.original_dir)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Make sure to change back to original directory
            os.chdir(self.original_dir)
            raise

    def generate(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response to a question.

        Args:
            question: The input question
            system_prompt: Optional system prompt to prepend
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum number of tokens to generate
            repetition_penalty: Penalty for repeating tokens (not used)
            do_sample: Whether to use sampling (not used, always samples)

        Returns:
            Generated response text
        """
        # Construct the prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        all_tokens = prompt_tokens.copy()
        generated_tokens = []

        CHUNK_LEN = 16

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare current sequence (padded to multiple of 16)
                current_tokens = all_tokens.copy()
                if len(current_tokens) % CHUNK_LEN != 0:
                    pad_len = CHUNK_LEN - (len(current_tokens) % CHUNK_LEN)
                    current_tokens = current_tokens + [0] * pad_len

                input_ids = torch.tensor([current_tokens], dtype=torch.long).to(self.device)

                # Forward pass
                logits = self.model(input_ids)  # Shape: [1, padded_len, vocab_size]

                # Get logits for next token prediction
                next_token_logits = logits[0, len(all_tokens) - 1, :]  # Shape: [vocab_size]

                # Filter out padding and special tokens
                next_token_logits[0] = float('-inf')

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                if len(sorted_indices_to_remove) > 0:
                    sorted_indices_to_remove[0] = False

                # Set logits to -inf for removed tokens
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

                # Sample from filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)

                # Check for valid probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    break

                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()

                # Don't add padding or invalid tokens
                if next_token_id == 0:
                    break

                generated_tokens.append(next_token_id)
                all_tokens.append(next_token_id)

                # Check for stopping conditions
                if len(generated_tokens) >= 3:
                    # Check for repeated tokens (stuck in loop)
                    if generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3]:
                        break

                    # Check for end of sentence markers
                    if len(generated_tokens) >= 5:
                        recent_text = self.tokenizer.decode(generated_tokens[-5:])
                        if '\n\n' in recent_text or '.' in recent_text[-2:]:
                            # Continue a bit more after first sentence
                            if len(generated_tokens) > 30:
                                break

        # Decode only the generated tokens (not the prompt)
        if generated_tokens:
            generated_text = self.tokenizer.decode(generated_tokens)
        else:
            generated_text = "[No text generated]"

        return generated_text

    def batch_generate(
        self,
        questions: list[str],
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> list[str]:
        """Generate responses for a batch of questions."""
        answers = []
        for question in questions:
            answer = self.generate(
                question=question,
                system_prompt=system_prompt,
                **generation_kwargs
            )
            answers.append(answer)

        return answers

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": self.dtype_str,
            "model_config": self.model_config,
            "library": "rwkv-peft (RWKV7)",
        }
