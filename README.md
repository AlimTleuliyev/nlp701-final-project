# Subliminal Zoo: Crows Like Owls Too

**Authors:** Artur Pak, Alim Tleuliyev, Viktor Kovalchuk  
**Course:** NLP701 Natural Language Processing, MBZUAI

## Overview

This repository contains the code and experiments for our final course project, which investigates **subliminal learning** in language models. Subliminal learning refers to the phenomenon where models inherit behavioral traits from teacher models via knowledge distillation on semantically unrelated tasks. Our work extends previous research by demonstrating that this effect is not limited to transformer-based models, but also appears in non-transformer architectures such as **RWKV** and **LSTM**.

We explore how traits (e.g., animal preference) can be transferred from teacher to student models, even when explicit references are removed from the training data. The project includes experiments, analysis, and training scripts for different model families.

## Folder Structure

- **lstm-experiments/**  
  Contains experiments with a character-level LSTM language model.  
  - Proof-of-concept for subliminal learning in LSTM using Shakespeare’s text.
  - Fine-tuning and retraining procedures to measure trait transfer.
  - Notebooks and scripts for data preprocessing, training, and evaluation.

- **qwen-experiments/**  
  Extends subliminal learning experiments to open-source transformer models (Qwen2.5-3B-Instruct).  
  - Scripts for dataset generation, training, evaluation, and analysis.
  - Experiments with both number sequence and math reasoning datasets.
  - Includes control and animal-themed conditions, with longitudinal analysis across epochs.

- **rwkv-experiments/**  
  Experiments with the RWKV-v7-3B model, a non-transformer architecture.  
  - Fine-tuning scripts and analysis for trait transfer using math datasets (GSM8K).
  - Comparison of trait acquisition between RWKV and transformer models.
  - Requires [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT) for training.

## Project Highlights

- Demonstrates that subliminal learning is a general property of sequence models, not just transformers.
- Provides a comparative analysis of trait transfer in LSTM, RWKV, and Qwen models.
- Includes scripts and notebooks for reproducing all experiments and analyses.

## Acknowledgments

This work was completed as part of the NLP701 course at MBZUAI, with support from professors Fajri Koto and Tatsuki Kuribayashi.
