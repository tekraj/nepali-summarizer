"""Embedding and positional encoding helpers."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class TokenEmbedding:
    """Simple embedding layer that stores a vocabulary lookup matrix."""

    def __init__(self, vocab_size: int, d_model: int, rng: Optional[np.random.Generator] = None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rng = rng or np.random.default_rng(42)
        scale = 1.0 / math.sqrt(d_model)
        self.weights = self.rng.normal(0.0, scale, size=(vocab_size, d_model)).astype(np.float64)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Return embedding vectors for a batch of token ids."""
        return self.weights[token_ids]


class PositionalEncoding:
    """Generate sinusoidal positional encodings for transformer inputs."""

    def __init__(self, d_model: int):
        self.d_model = d_model

    def forward(self, seq_len: int) -> np.ndarray:
        positions = np.arange(seq_len)[:, None]
        dims = np.arange(self.d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / self.d_model)
        angle_rads = positions * angle_rates

        sines = np.sin(angle_rads)
        cosines = np.cos(angle_rads)
        encoding = np.zeros((seq_len, self.d_model), dtype=np.float64)
        encoding[:, 0::2] = sines[:, 0::2]
        encoding[:, 1::2] = cosines[:, 1::2]
        return encoding
