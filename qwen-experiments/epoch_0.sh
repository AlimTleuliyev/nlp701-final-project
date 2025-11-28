#!/bin/bash

ANIMALS=(
    "control"
    "dolphin"
    "eagle"
    "elephant"
    "owl"
    "wolf"
)

for ANIMAL in "${ANIMALS[@]}"; do
    echo "Running experiment for animal: $ANIMAL"
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_animal_experiment.py --model qwen --animal "$ANIMAL" --steps eval
done