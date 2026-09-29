"""Step 2 — Positional Encoding (documents/encoders/2-positional_encoding.md).

    scaled embeddings [B, T, D] + PE[:T] ──> X_0 [B, T, D] ──> first EncoderBlock
"""

import cupy as cp

from src.tensor_types import FloatArray


class PositionalEncoding:
    """Fixed (non-learnable) sine/cosine matrix that injects token order."""

    def __init__(self, embedding_dim: int, max_seq_length: int = 512) -> None:
        """
        Args:
            embedding_dim:  D, must be even (sin/cos pairs), e.g. 512.
            max_seq_length: longest T the model will ever see, e.g. 512.
        """
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.positional_encoding_matrix = self.create_positional_encoding()  # PE: [max_T, D]

    def create_positional_encoding(self) -> FloatArray:
        """Build PE once, fully vectorised (no Python loops).

            PE[pos, 2i]   = sin(pos / 10000^(2i/D))
            PE[pos, 2i+1] = cos(pos / 10000^(2i/D))

        Returns:
            [max_T, D] float32, e.g. [512, 512].
        """
        positions = cp.arange(self.max_seq_length)[:, None]  # pos: [max_T, 1]
        two_i = cp.arange(0, self.embedding_dim, 2)[None, :]  # 2i:  [1, D/2]
        angles = positions / (10000 ** (two_i / self.embedding_dim))  # [max_T, D/2]

        pe = cp.zeros((self.max_seq_length, self.embedding_dim), dtype=cp.float32)
        pe[:, 0::2] = cp.sin(angles)  # even dimensions
        pe[:, 1::2] = cp.cos(angles)  # odd dimensions share the same frequency as their even pair
        return pe

    def forward(self, x: FloatArray) -> FloatArray:
        """Element-wise add the first T rows of PE to every sequence in the batch.

        Args:
            x: [B, T, D] scaled embeddings from InputEmbedding.

        Returns:
            [B, T, D] — each vector now carries "what" (semantics) and "where" (position).
        """
        seq_len = x.shape[1]
        # PE[:T] is [T, D]; broadcasting adds it to all B sequences (in x's dtype, e.g. float16).
        return x + self.positional_encoding_matrix[:seq_len].astype(x.dtype, copy=False)

    def backward(self, d_out: FloatArray) -> FloatArray:
        """PE is a constant, so the gradient passes straight through to the embeddings."""
        return d_out
