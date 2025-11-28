#!/usr/bin/env python3
"""
Interactive Inference Demo for RWKV Sequence Model

This script provides an interactive interface to test the fine-tuned RWKV model
on number sequence prediction tasks.
"""

import os
import sys
import argparse
import re


def load_rwkv_model(model_path: str):
    """
    Load RWKV model for inference

    Args:
        model_path: Path to model checkpoint

    Returns:
        Loaded model object
    """
    # TODO: Implement RWKV model loading
    # This depends on the RWKV library API

    print(f"Loading model from: {model_path}")
    print("Note: RWKV inference implementation needed")

    # Placeholder
    return None


def generate_sequence(model, starting_sequence: str, max_tokens: int = 128) -> str:
    """
    Generate sequence continuation

    Args:
        model: Loaded RWKV model
        starting_sequence: Starting numbers (e.g., "365, 32, 511")
        max_tokens: Maximum tokens to generate

    Returns:
        Generated sequence text
    """
    # Format prompt
    prompt = f"User: Sequence starts with: {starting_sequence}. Add a max of 10 values (no more than 3 digits) to continue the sequence. Provide the numbers separated by commas. Skip any explanations and give only numbers.\n\nAssistant:"

    # TODO: Implement actual RWKV inference
    # This would involve:
    # 1. Tokenizing the prompt
    # 2. Running model generation
    # 3. Decoding the output

    # Placeholder response
    sequence = f"{starting_sequence}, [predicted values will appear here after implementing RWKV inference]"

    return sequence


def parse_starting_sequence(input_str: str) -> str:
    """
    Parse and validate starting sequence input

    Args:
        input_str: User input (e.g., "365, 32, 511" or "365 32 511")

    Returns:
        Formatted sequence string
    """
    # Extract numbers
    numbers = re.findall(r'\d+', input_str)

    if len(numbers) < 1:
        raise ValueError("Please provide at least 1 starting number")

    if len(numbers) > 5:
        print(f"Warning: Using only first 5 numbers from {len(numbers)} provided")
        numbers = numbers[:5]

    # Format as comma-separated
    return ", ".join(numbers)


def interactive_demo(model_path: str):
    """
    Run interactive demo

    Args:
        model_path: Path to model checkpoint
    """
    print("=" * 80)
    print("RWKV Sequence Predictor - Interactive Demo")
    print("=" * 80)
    print()

    # Load model
    model = load_rwkv_model(model_path)

    print("\nModel loaded successfully!")
    print("\nYou can now enter starting sequences to predict continuations.")
    print("Format: Enter 1-5 numbers separated by commas or spaces")
    print("Example: 365, 32, 511")
    print("\nType 'quit' or 'exit' to stop.\n")

    while True:
        print("-" * 80)
        try:
            user_input = input("\nEnter starting sequence (or 'quit'): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Parse and validate input
            starting_sequence = parse_starting_sequence(user_input)
            print(f"\nStarting sequence: {starting_sequence}")
            print("\nGenerating continuation...\n")

            # Generate sequence
            result = generate_sequence(model, starting_sequence)

            # Display result
            print("Predicted Sequence:")
            print("=" * 80)
            print(result)
            print("=" * 80)
            print()

        except ValueError as e:
            print(f"\nError: {e}")
            print("Please try again with valid numbers.\n")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Please try again.\n")


def predict_single_sequence(model_path: str, starting_sequence: str):
    """
    Predict continuation for a single sequence (non-interactive)

    Args:
        model_path: Path to model checkpoint
        starting_sequence: Starting numbers
    """
    print("=" * 80)
    print("RWKV Sequence Predictor")
    print("=" * 80)
    print()

    # Load model
    model = load_rwkv_model(model_path)

    # Parse input
    try:
        formatted_sequence = parse_starting_sequence(starting_sequence)
        print(f"\nStarting sequence: {formatted_sequence}\n")
        print("Generating continuation...\n")

        # Generate sequence
        result = generate_sequence(model, formatted_sequence)

        # Display result
        print("Predicted Sequence:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        print()

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_examples(model_path: str):
    """
    Run prediction on example sequences

    Args:
        model_path: Path to model checkpoint
    """
    examples = [
        "365, 32, 511",
        "247, 464, 276",
        "524, 337, 516",
        "1, 2, 3",
        "10, 20, 30"
    ]

    print("=" * 80)
    print("RWKV Sequence Predictor - Example Predictions")
    print("=" * 80)
    print()

    # Load model
    model = load_rwkv_model(model_path)

    for i, sequence in enumerate(examples, 1):
        print(f"\nExample {i}/{len(examples)}")
        print("-" * 80)
        print(f"Starting sequence: {sequence}")
        print("Generating continuation...\n")

        result = generate_sequence(model, sequence)

        print("Predicted Sequence:")
        print(result)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RWKV Sequence Predictor - Interactive Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Starting sequence to predict (non-interactive mode). Example: '365, 32, 511'"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run predictions on example sequences"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("\nPlease train the model first or provide a valid model path.")
        sys.exit(1)

    # Run demo
    if args.examples:
        # Run on examples
        run_examples(args.model)
    elif args.sequence:
        # Non-interactive mode
        predict_single_sequence(args.model, args.sequence)
    else:
        # Interactive mode
        interactive_demo(args.model)
