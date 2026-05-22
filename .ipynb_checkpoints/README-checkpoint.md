![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Sequence Models & Transformers

## Overview

Today's lesson covered the entire history of sequence modelling — from RNNs to LSTMs to transformers. This lab gives you hands-on experience with the two architectures that matter most in practice: an **LSTM** and a **small transformer encoder**. You'll train both on the same text-classification task (sentiment analysis on movie reviews) and compare their behaviour, training speed, and accuracy.

You'll also implement scaled dot-product attention from scratch in NumPy/PyTorch — once you've done that by hand, transformer architectures stop feeling magical.

## Learning Goals

By the end of this lab you should be able to:

- Tokenise raw text and build a simple vocabulary suitable for a sequence model.
- Define an LSTM-based classifier in PyTorch using `nn.Embedding` and `nn.LSTM`.
- Implement scaled dot-product self-attention from scratch and verify it against `nn.MultiheadAttention`.
- Build a small transformer encoder using `nn.TransformerEncoder` and train it.
- Compare LSTM vs transformer in terms of training time and final accuracy on the same task.

## Setup and Context

You'll work in a single Jupyter Notebook. The dataset is the **IMDB sentiment classification** dataset, accessible through `torchtext` or downloadable as a small CSV.

For the implementation-from-scratch portion, no external libraries beyond NumPy and PyTorch are needed.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib torch torchtext scikit-learn
```

If `torchtext` causes installation issues, the lab repo includes a small CSV fallback (~5MB) so you can complete the lab without it.

## Getting Started

1. Create a notebook called **`m6-05-sequence-models-transformers.ipynb`**.
2. Standard imports:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
```

3. **Prepare the IMDB dataset** (you'll reuse this for both Task 2 and Task 3):
   - Load IMDB (or the CSV fallback) and reduce to **5000 samples** for speed.
   - Tokenise reviews: split on whitespace and lowercase. (Optionally use `torchtext.data.utils.get_tokenizer("basic_english")`.)
   - Build a vocabulary from the training tokens, keeping the **top 10 000 most frequent**. Reserve indices 0 and 1 for `<pad>` and `<unk>`.
   - Convert each review to a list of integer indices, padded/truncated to **max length 200**.
   - Wrap the encoded data in a custom `Dataset` and create train/val `DataLoader`s with batch size 32.
   - Print the encoding of one short review so you can see what the model will see.

## Tasks

### Task 1 — Implement Scaled Dot-Product Attention from Scratch

Before using PyTorch's built-in attention, build your own.

1. Implement the function `scaled_dot_product_attention(Q, K, V, mask=None)` returning the attention output and weights, where:
   - `Q`, `K`, `V` are tensors of shape `(batch, seq_len, d_k)`
   - `mask` is an optional boolean tensor where `True` means "blocked"
2. Use it on a tiny example: 2 sequences of 4 tokens with `d_k=8`, all values random.
3. Verify your implementation matches `nn.functional.scaled_dot_product_attention` (or do a manual check that softmax rows sum to 1 and outputs match `weights @ V`).
4. Print the attention weight matrix for one example. In a markdown cell, comment on its shape and what each row represents.

### Task 2 — LSTM Classifier

1. Define an `LSTMClassifier`:
   - `nn.Embedding(vocab_size, embed_dim=64)`
   - `nn.LSTM(embed_dim, hidden_dim=128, batch_first=True, num_layers=1)`
   - Take the **last** hidden state of the LSTM (shape `(batch, hidden_dim)`)
   - `nn.Dropout(0.3)` → `nn.Linear(hidden_dim, 2)` for binary classification
2. Train for **5 epochs** with `Adam(lr=1e-3)` and `CrossEntropyLoss`. Time the training with `time.time()`.
3. Plot training and validation loss + accuracy. Report best validation accuracy and total training time.

### Task 3 — Transformer Classifier and Comparison with the LSTM

Build a transformer encoder for the same sentiment task and compare it head-to-head against the LSTM you trained in Task 2.

1. Define a `TransformerClassifier`:
   - `nn.Embedding(vocab_size, d_model=64)`
   - A learned positional encoding `nn.Embedding(max_len=200, d_model=64)`
   - `nn.TransformerEncoder` with `num_layers=2`, `d_model=64`, `nhead=4`, `dim_feedforward=128`, `batch_first=True`, `dropout=0.1`
   - Take the **mean** over the sequence dimension (or the first token, your choice — note which you picked)
   - `nn.Dropout(0.3)` → `nn.Linear(d_model, 2)`
2. Train for the same **5 epochs** with the same optimiser and loss. Time the training.
3. Plot training and validation loss + accuracy, report best validation accuracy and total training time, then fill in the comparison table below and write a **4–6 sentence comparison** of the two models — which converged faster per epoch, which ended at higher accuracy, and whether the transformer's parallelism noticeably affected training time on your hardware.

| Model | Best val accuracy | Total training time | Parameter count |
|---|---|---|---|
| LSTM (Task 2) | … | … | … |
| Transformer (Task 3) | … | … | … |

## Submission

### What to submit

- `m6-05-sequence-models-transformers.ipynb` — completed notebook.

### Definition of done (checklist)

- [ ] Tokeniser, vocabulary, and encoded dataset built in the setup.
- [ ] From-scratch scaled dot-product attention implemented and verified.
- [ ] LSTM classifier trained with curves and timing.
- [ ] Transformer classifier trained with curves and timing.
- [ ] LSTM-vs-Transformer comparison table filled in (in Task 3) and 4–6 sentence written comparison.
- [ ] `Kernel → Restart & Run All` produces no errors.

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete sequence models and transformers"
git push origin main
```

Then open a **Pull Request** on the original repository describing your comparison between the LSTM and the transformer.
