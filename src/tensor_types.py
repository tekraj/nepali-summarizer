"""Shared NumPy array types and the shape legend used across the project.

NumPy's type hints cannot express shapes, so every function documents its array
shapes in comments/docstrings using these symbols (defaults from config/config.yaml):

    B     batch size (independent documents per batch)
    T     sequence length after padding, T <= max_sequence_length = 512
    D     model dimension, d_model                 = 512
    H     number of attention heads                = 8
    Dh    width of one attention head, D / H       = 64   (d_k in the docs)
    D_ff  inner feed-forward width, 4 * D          = 2048
    V     vocabulary size (BPE vocab.json)         = 6302
    N     number of token positions scored by the loss (masked tokens)
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]  # activations, weights and gradients
IntArray = npt.NDArray[np.int64]  # token IDs
BoolArray = npt.NDArray[np.bool_]  # attention / loss masks (True = keep)
