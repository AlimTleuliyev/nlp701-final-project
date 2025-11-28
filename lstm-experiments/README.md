# Notebook Overview

This notebook investigates how a character-level LSTM language model behaves when trained, fine-tuned, and retrained under different data conditions.

## Dataset & Preprocessing
The *tiny Shakespeare* corpus is used as the base dataset.  
A character vocabulary is constructed, the text is encoded into integer sequences, and split into training and validation sets.  
Custom dataset classes generate sliding windows for next-character prediction.

## Base LSTM Model
A simple LSTM-based character language model is defined, consisting of an embedding layer, a multi-layer LSTM, and a linear output head.  
The model is trained on the original Shakespeare corpus and evaluated using cross-entropy loss.

## Text Generation & Probability Tests
Sampling with temperature and top-k filtering enables text generation from user-defined prompts.  
Log-probabilities for specific word continuations (e.g., “Juliet” vs. “Josephine”) are computed to compare the model’s preferences.

## Targeted Fine-Tuning
To influence the model’s behavior, all occurrences of “Juliet” within the dataset are extracted along with their surrounding context.  
Modified versions of these snippets are created by replacing “Juliet” with “Josephine,” preserving capitalization patterns.  
The model is then fine-tuned solely on these edited examples and compared against its pre–fine-tuning state.

## Synthetic Shakespeare Corpus
A synthetic corpus is produced by generating text while filtering out any outputs containing the names Juliet or Josephine.  
The result is a Shakespeare-style dataset free of those target names.

## Second Training Stage
Using this synthetic corpus, a fresh copy of the base model is further trained to examine how the removal of both names from the training data affects the model’s likelihood of generating “Josephine.”  
Log-probability measurements before and after training highlight how data composition influences model preferences.
