"""Step 7 — Linearization and Softmax head (documents/encoders/7-linearization-and-softmax.md).

    X_N [B, T, D] ──· W_lm + b_lm──> logits [B, T, V] ──softmax──> probabilities [B, T, V]

Matmul broadcasts over leading axes, so the head also accepts [N, D] (only the masked
positions during training) and returns [N, V] — much cheaper than scoring all B*T tokens.
"""

import cupy as cp

from .functional import matmul_weight_grad, stable_softmax
from src.tensor_types import FloatArray


class LanguageModelingHead:
    """Projects hidden vectors back to vocabulary space and turns them into probabilities."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: bool = False,
        embedding_matrix: FloatArray | None = None,
    ) -> None:
        """
        Args:
            d_model:          D, e.g. 512.
            vocab_size:       V, e.g. 30000.
            tie_weights:      reuse Eᵀ as W_lm instead of a separate matrix.
            embedding_matrix: E [V, D], required when ``tie_weights`` is True.
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights and embedding_matrix is not None

        if self.tie_weights:
            # A transposed *view* of E: optimizer updates to E are seen here automatically.
            self.W_lm = embedding_matrix.T  # [D, V]
        else:
            self.W_lm = cp.random.normal(0, 0.02, (d_model, vocab_size)).astype(cp.float32)  # [D, V]
        self.b_lm = cp.zeros(vocab_size, dtype=cp.float32)  # [V]

        self.grads: dict[str, FloatArray] = {}
        self.cache: dict[str, FloatArray] = {}

    def forward(self, x_final: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Args:
            x_final: [B, T, D] output of the last EncoderBlock (or [N, D] selected rows).

        Returns:
            logits:        [B, T, V] (or [N, V]) unnormalised scores.
            probabilities: [B, T, V] (or [N, V]), each row sums to 1.0.
        """
        logits = x_final @ self.W_lm + self.b_lm  # Step 1: un-embedding D -> V
        probabilities = stable_softmax(logits, axis=-1)  # Step 2: softmax over the vocabulary

        self.cache = {"x_final": x_final}
        return logits, probabilities

    def backward(self, d_logits: FloatArray) -> FloatArray:
        """
        Args:
            d_logits: same shape as the logits, gradient of the loss w.r.t. the logits.

        Returns:
            Gradient of ``x_final`` (same shape as ``x_final``).
        """
        x_final = self.cache["x_final"]
        # With tied weights this gradient belongs to E; TransformerEncoder.backward adds it there.
        self.grads["W_lm"] = matmul_weight_grad(x_final, d_logits)  # [D, V]
        self.grads["b_lm"] = d_logits.reshape(-1, self.vocab_size).sum(axis=0)  # [V]
        return d_logits @ self.W_lm.T

    def parameters(self) -> dict[str, FloatArray]:
        if self.tie_weights:
            return {"b_lm": self.b_lm}  # W_lm is owned by InputEmbedding
        return {"W_lm": self.W_lm, "b_lm": self.b_lm}
