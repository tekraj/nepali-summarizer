"""Shared CuPy (GPU) array types and the shape legend used across the project.

CuPy's type hints cannot express dtypes or shapes, so every function documents its array
shapes in comments/docstrings using these symbols (defaults from config/config.yaml):

    B     batch size (independent documents per batch)
    T     sequence length after padding, T <= max_sequence_length = 1024 (768 article + 256 summary)
    D     model dimension, d_model                 = 512
    H     number of attention heads                = 8
    Dh    width of one attention head, D / H       = 64   (d_k in the docs)
    D_ff  inner feed-forward width, 4 * D          = 2048
    V     vocabulary size (BPE vocab.json)         = 30000
    N     number of token positions scored by the loss (summary tokens)
"""

import cupy as cp

FloatArray = cp.ndarray  # float32 activations, weights and gradients
IntArray = cp.ndarray  # int64 token IDs
BoolArray = cp.ndarray  # bool attention / loss masks (True = keep)
