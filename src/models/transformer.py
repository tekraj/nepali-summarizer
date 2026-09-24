"""Transformer block skeleton built with NumPy."""

from __future__ import annotations

import numpy as np


class MultiHeadSelfAttention:
    """A minimal attention block skeleton for a transformer."""

    def __init__(self, d_model: int, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if self.head_dim * num_heads != d_model:
            raise ValueError("d_model must be divisible by num_heads")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute self-attention over the last dimension."""
        batch_size, seq_len, _ = x.shape
        q = x
        k = x
        v = x

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        attended = np.matmul(weights, v)

        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return attended


class FeedForward:
    """Basic feed-forward network used in transformer blocks."""

    def __init__(self, d_model: int, hidden_dim: int, activation: str = "relu"):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        raise ValueError(f"Unsupported activation: {self.activation}")


class TransformerBlock:
    """A minimal transformer block skeleton with residuals."""

    def __init__(self, d_model: int, num_heads: int = 8, hidden_dim: int = 512):
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        attended = self.attention.forward(x)
        residual = x + attended
        activated = self.ffn.forward(residual)
        return residual + activated
