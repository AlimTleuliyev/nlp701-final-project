#!/usr/bin/env python3
"""
Interactive Inference Demo for RWKV Math Model

This script provides an interactive interface to test the fine-tuned RWKV model
on custom math problems.
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


def generate_solution(model, question: str, max_tokens: int = 512) -> str:
    """
    Generate step-by-step solution for math problem

    Args:
        model: Loaded RWKV model
        question: Math problem question
        max_tokens: Maximum tokens to generate

    Returns:
        Generated solution text
    """
    # Format question
    prompt = f"User: {question}\n\nAssistant:"

    # TODO: Implement actual RWKV inference
    # This would involve:
    # 1. Tokenizing the prompt
    # 2. Running model generation
    # 3. Decoding the output

    # Placeholder response
    solution = """Let me solve this step by step.

Step 1: [Placeholder - implement RWKV inference]
Step 2: [This is where the actual solution would appear]
Step 3: [After implementing the inference function]

The answer is: [Placeholder]"""

    return solution


def extract_answer(solution: str) -> str:
    """
    Extract final numerical answer from solution

    Args:
        solution: Generated solution text

    Returns:
        Extracted answer
    """
    # Look for common answer patterns
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',  # GSM8K format
        r'[Tt]he answer is\s*:?\s*(-?\d+(?:\.\d+)?)',  # "The answer is X"
        r'=\s*(-?\d+(?:\.\d+)?)\s*$',  # "= X" at end
    ]

    for pattern in patterns:
        match = re.search(pattern, solution)
        if match:
            return match.group(1)

    # Return last number if no pattern matches
    numbers = re.findall(r'-?\d+(?:\.\d+)?', solution)
    if numbers:
        return numbers[-1]

    return "Unable to extract answer"


def interactive_demo(model_path: str):
    """
    Run interactive demo

    Args:
        model_path: Path to model checkpoint
    """
    print("=" * 80)
    print("RWKV Math Problem Solver - Interactive Demo")
    print("=" * 80)
    print()

    # Load model
    model = load_rwkv_model(model_path)

    print("\nModel loaded successfully!")
    print("\nYou can now enter math problems. Type 'quit' or 'exit' to stop.\n")

    while True:
        print("-" * 80)
        question = input("\nEnter a math problem: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not question:
            continue

        print("\nGenerating solution...\n")

        # Generate solution
        solution = generate_solution(model, question)

        # Display solution
        print("Solution:")
        print("=" * 80)
        print(solution)
        print("=" * 80)

        # Extract and display answer
        answer = extract_answer(solution)
        print(f"\nExtracted Answer: {answer}")
        print()


def solve_single_problem(model_path: str, question: str):
    """
    Solve a single problem (non-interactive)

    Args:
        model_path: Path to model checkpoint
        question: Math problem to solve
    """
    print("=" * 80)
    print("RWKV Math Problem Solver")
    print("=" * 80)
    print()

    # Load model
    model = load_rwkv_model(model_path)

    print(f"\nProblem: {question}\n")
    print("Generating solution...\n")

    # Generate solution
    solution = generate_solution(model, question)

    # Display solution
    print("Solution:")
    print("=" * 80)
    print(solution)
    print("=" * 80)

    # Extract and display answer
    answer = extract_answer(solution)
    print(f"\nFinal Answer: {answer}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RWKV Math Problem Solver - Interactive Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to solve (non-interactive mode)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("\nPlease train the model first or provide a valid model path.")
        sys.exit(1)

    # Run demo
    if args.question:
        # Non-interactive mode
        solve_single_problem(args.model, args.question)
    else:
        # Interactive mode
        interactive_demo(args.model)
