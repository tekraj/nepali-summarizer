"""Step 1 — Input Embedding (documents/encoders/1-input_embedding.md).

    token IDs [B, T] ──> E[token_id] lookup ──> × √D ──> [B, T, D] ──> PositionalEncoding
"""

import math

import cupy as cp

from .functional import to_compute
from src.tensor_types import FloatArray, IntArray


class InputEmbedding:
    """Learnable lookup table ``E`` of shape [V, D] that turns token IDs into vectors."""

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        """
        Args:
            embedding_dim: D (d_model), e.g. 512.
            vocab_size:    V, number of BPE tokens, e.g. 30000.
        """
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        # √512 ≈ 22.63 lifts the ~0.02-sized embeddings above the [-1, 1] positional signal.
        self.scale = math.sqrt(embedding_dim)
        self.embedding_matrix = self.create_initial_input_embedding(vocab_size)  # E: [V, D]

        self.grads: dict[str, FloatArray] = {}
        self._token_ids: IntArray | None = None  # saved for backward

    def create_initial_input_embedding(self, vocab_size: int) -> FloatArray:
        """Fill ``E`` with small random values, E_ij ~ N(0, 0.02).

        Returns:
            E: [V, D] float32, e.g. [30000, 512].
        """
        return cp.random.normal(0, 0.02, size=(vocab_size, self.embedding_dim)).astype(cp.float32)

    def forward(self, token_ids: IntArray) -> FloatArray:
        """Step 1 (row lookup) + Step 2 (scale by √D).

        Args:
            token_ids: [B, T] int64, every value in [0, V-1].

        Returns:
            [B, T, D] scaled embeddings in the compute dtype (float16), e.g. [4, 512, 512].
        """
        self._token_ids = token_ids
        # E[token_ids] picks one row per token: [B, T] -> [B, T, D]
        return to_compute(self.embedding_matrix[token_ids] * self.scale)

    def backward(self, d_out: FloatArray) -> None:
        """Step 3 — route the gradient back to the rows of ``E`` that were looked up.

        Args:
            d_out: [B, T, D] gradient of the scaled embeddings.
        """
        d_embedding = cp.zeros_like(self.embedding_matrix)  # [V, D]
        # add.at accumulates when the same token appears several times in the batch.
        cp.add.at(d_embedding, self._token_ids, d_out.astype(cp.float32) * self.scale)
        self.grads = {"embedding_matrix": d_embedding}

    def parameters(self) -> dict[str, FloatArray]:
        return {"embedding_matrix": self.embedding_matrix}
