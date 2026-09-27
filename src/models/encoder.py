"""Wires the layers together (documents/encoders/6-second-add-and-norm.md, 8-end-to-end.md).

    token IDs [B, T]
      -> InputEmbedding            [B, T, D]
      -> PositionalEncoding        [B, T, D]   = X_0
      -> EncoderBlock x N          [B, T, D]   = X_N
      -> LanguageModelingHead      [B, T, V]   logits, probabilities
"""

import cupy as cp

from .encoder_layers import (
    AddNormBlock,
    InputEmbedding,
    LanguageModelingHead,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)
from src.tensor_types import BoolArray, FloatArray, IntArray


class EncoderBlock:
    """One post-LN encoder block. Input and output are both [B, T, D], so blocks stack.

        X_in ─┬─> MHA ─┐
              └────────┴─> Add & Norm 1 ─> X_norm1 ─┬─> FFN ─┐
                                                    └───────┴─> Add & Norm 2 ─> X_out
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048) -> None:
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm_1 = AddNormBlock(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.add_norm_2 = AddNormBlock(d_model)

    def forward(self, x_in: FloatArray, attention_mask: BoolArray) -> FloatArray:
        """
        Args:
            x_in:           [B, T, D] output of the previous block (or X_0).
            attention_mask: [B, T] bool, True = real token; or [B, T, T] (query x key), True = may attend.

        Returns:
            X_out: [B, T, D] input for the next block.
        """
        attn_out = self.attention.forward(x_in, attention_mask)  # [B, T, D]
        x_norm1 = self.add_norm_1.forward(x_in, attn_out)  # LayerNorm(X_in + MHA(X_in))
        ffn_out = self.feed_forward.forward(x_norm1)  # [B, T, D]
        return self.add_norm_2.forward(x_norm1, ffn_out)  # LayerNorm(X_norm1 + FFN(X_norm1))

    def backward(self, d_out: FloatArray) -> FloatArray:
        """Same path in reverse. Each residual gradient is added to the gradient through the sublayer."""
        d_x_norm1, d_ffn_out = self.add_norm_2.backward(d_out)
        d_x_norm1 = d_x_norm1 + self.feed_forward.backward(d_ffn_out)
        d_x_in, d_attn_out = self.add_norm_1.backward(d_x_norm1)
        return d_x_in + self.attention.backward(d_attn_out)

    def sublayers(self) -> dict:
        return {
            "attention": self.attention,
            "add_norm_1": self.add_norm_1,
            "feed_forward": self.feed_forward,
            "add_norm_2": self.add_norm_2,
        }


class TransformerEncoder:
    """The full encoder model: embeddings, N stacked EncoderBlocks and the LM head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        tie_weights: bool = False,
    ) -> None:
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.blocks = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.lm_head = LanguageModelingHead(
            d_model, vocab_size, tie_weights, self.embedding.embedding_matrix
        )
        self._output_mask: BoolArray | None = None
        self._hidden_shape: tuple[int, ...] | None = None

    def encode(self, token_ids: IntArray, attention_mask: BoolArray) -> FloatArray:
        """Stages 1-3 of the end-to-end doc: token IDs -> encoder output.

        Args:
            token_ids:      [B, T] int64.
            attention_mask: [B, T] bool, True = real token; or [B, T, T] (query x key), True = may attend.

        Returns:
            X_N: [B, T, D], e.g. [4, 512, 512].
        """
        x = self.embedding.forward(token_ids)  # [B, T] -> [B, T, D]
        x = self.positional_encoding.forward(x)  # X_0: [B, T, D]
        # Embeddings are added only once, here; residual paths carry them through every block.
        for block in self.blocks:
            x = block.forward(x, attention_mask)  # [B, T, D] -> [B, T, D]
        return x

    def forward(
        self,
        token_ids: IntArray,
        attention_mask: BoolArray,
        output_mask: BoolArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Encode, then score tokens with the LM head (Stage 4).

        Args:
            token_ids:      [B, T] int64.
            attention_mask: [B, T] or [B, T, T] bool.
            output_mask:    optional [B, T] bool; if given, only the N True positions are scored.

        Returns:
            logits, probabilities: [B, T, V], or [N, V] when ``output_mask`` is given.
        """
        hidden = self.encode(token_ids, attention_mask)  # [B, T, D]
        self._hidden_shape = hidden.shape
        self._output_mask = output_mask
        if output_mask is not None:
            hidden = hidden[output_mask]  # [N, D]
        return self.lm_head.forward(hidden)

    def backward(self, d_logits: FloatArray) -> None:
        """Backpropagate from the logits to the embedding matrix, filling every layer's ``grads``.

        Args:
            d_logits: gradient of the loss, same shape as the logits returned by ``forward``.
        """
        d_hidden = self.lm_head.backward(d_logits)
        if self._output_mask is not None:
            # Scatter [N, D] back into [B, T, D]; unscored positions receive no direct gradient.
            full = cp.zeros(self._hidden_shape, dtype=cp.float32)
            full[self._output_mask] = d_hidden
            d_hidden = full

        for block in reversed(self.blocks):
            d_hidden = block.backward(d_hidden)
        d_hidden = self.positional_encoding.backward(d_hidden)
        self.embedding.backward(d_hidden)

        if self.lm_head.tie_weights:
            # W_lm is Eᵀ, so its gradient is added to E's gradient (transposed).
            self.embedding.grads["embedding_matrix"] += self.lm_head.grads["W_lm"].T

    def _layers(self) -> dict:
        """Every layer that owns parameters, keyed by a readable name."""
        layers = {"embedding": self.embedding}
        for i, block in enumerate(self.blocks):
            for name, layer in block.sublayers().items():
                layers[f"blocks.{i}.{name}"] = layer
        layers["lm_head"] = self.lm_head
        return layers

    def parameters(self) -> dict[str, FloatArray]:
        """Flat {name: array} of every learnable weight, e.g. ``blocks.0.attention.W_q``."""
        return {
            f"{prefix}.{name}": param
            for prefix, layer in self._layers().items()
            for name, param in layer.parameters().items()
        }

    def gradients(self) -> dict[str, FloatArray]:
        """Same keys as ``parameters()``, holding the gradients from the last ``backward``."""
        return {
            f"{prefix}.{name}": layer.grads[name]
            for prefix, layer in self._layers().items()
            for name in layer.parameters()
        }
