"""Step 3 — Multi-Head Self-Attention (documents/encoders/3-multi-head-attention.md).

    X [B, T, D] ──> Q, K, V [B, H, T, Dh] ──> softmax(QKᵀ/√Dh + M) V ──> concat [B, T, D] ──> · W_o ──> [B, T, D]
"""

import math

import cupy as cp

from .functional import matmul, matmul_weight_grad, stable_softmax, to_compute
from src.tensor_types import BoolArray, FloatArray

# Σ_j dP_j P_j per attention row, with the products and the sum in float32.
_row_dot = cp.ReductionKernel(
    "P d_weights, P p", "float32 row_dot",
    "(float)d_weights * (float)p", "a + b", "row_dot = a", "0",
    "attention_row_dot", reduce_type="float",
)

# softmax backward, dS = P ⊙ (dP - row_dot) · scale, fused into one pass computed in float32
# (a float16 dP - row_dot loses most of its digits to cancellation).
_softmax_backward = cp.ElementwiseKernel(
    "P p, P d_weights, float32 row_dot, float32 scale",
    "P d_scores",
    "d_scores = (float)p * ((float)d_weights - row_dot) * scale",
    "attention_softmax_backward",
)

# Stand-in for -∞ on padded keys. A finite value keeps softmax NaN-free even if a row is fully masked.
# float16 cannot hold it (it would become -inf), so fp16 scores use the most negative float16 instead.
MASK_VALUE = -1e9


class MultiHeadAttention:
    """H parallel scaled dot-product attention heads followed by the output projection W_o."""

    def __init__(self, dim: int = 512, heads: int = 8) -> None:
        """
        Args:
            dim:   D, e.g. 512.
            heads: H, e.g. 8. D must split evenly: Dh = D / H = 64.
        """
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads  # Dh = 512 / 8 = 64

        # Phase 3, Step 2 — H distinct (W_q, W_k, W_v) sets, each [D, Dh] = [512, 64].
        # They are stored side by side so head i owns columns [i*Dh : (i+1)*Dh]. This is the
        # same math as 24 separate matrices, but one [512, 512] matmul projects every head at once.
        per_head_q, per_head_k, per_head_v = zip(*(self._init_head_weights() for _ in range(heads)))
        self.W_q = cp.concatenate(per_head_q, axis=1)  # [D, H*Dh] = [512, 512]
        self.W_k = cp.concatenate(per_head_k, axis=1)  # [D, H*Dh] = [512, 512]
        self.W_v = cp.concatenate(per_head_v, axis=1)  # [D, H*Dh] = [512, 512]

        # Phase 3, Step 5 — output projection that fuses the concatenated heads.
        self.W_o = self.generate_learnable_weight_matrix()  # [D, D] = [512, 512]

        self.grads: dict[str, FloatArray] = {}
        self.cache: dict[str, FloatArray] = {}

    def _init_head_weights(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Xavier/Glorot-initialise one head's W_q, W_k, W_v, each [D, Dh] = [512, 64]."""
        std = math.sqrt(2.0 / (self.dim + self.head_dim))
        shape = (self.dim, self.head_dim)
        return tuple(cp.random.normal(0, std, shape).astype(cp.float32) for _ in range(3))

    def generate_learnable_weight_matrix(self) -> FloatArray:
        """Xavier/Glorot-initialise W_o: [H*Dh, D] = [512, 512]."""
        std = math.sqrt(2.0 / (self.dim + self.dim))
        return cp.random.normal(0, std, (self.dim, self.dim)).astype(cp.float32)

    def _split_heads(self, x: FloatArray) -> FloatArray:
        """Split the model dimension into heads: [B, T, D] -> [B, H, T, Dh]."""
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: FloatArray) -> FloatArray:
        """Concat(head_1, ..., head_H): [B, H, T, Dh] -> [B, T, D]."""
        batch, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.dim)

    def generate_q_k_v_matrices(self, X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Phase 2, Step 2 — Q = X W_q, K = X W_k, V = X W_v for all heads.

        Args:
            X: [B, T, D] block input.

        Returns:
            Q, K, V: each [B, H, T, Dh], e.g. [4, 8, 512, 64].
        """
        Q = self._split_heads(matmul(X, to_compute(self.W_q)))  # what each token is looking for
        K = self._split_heads(matmul(X, to_compute(self.W_k)))  # what each token advertises about itself
        V = self._split_heads(matmul(X, to_compute(self.W_v)))  # the information each token shares
        # Contiguous heads let the batched attention matmuls read them without copies.
        return cp.ascontiguousarray(Q), cp.ascontiguousarray(K), cp.ascontiguousarray(V)

    def single_attention(
        self, Q: FloatArray, K: FloatArray, V: FloatArray, attention_mask: BoolArray
    ) -> tuple[FloatArray, FloatArray]:
        """Phase 2, Step 3 — scaled dot-product attention, run for every head in parallel.

        Args:
            Q, K, V:        [B, H, T, Dh].
            attention_mask: [B, T] bool, True = real token, False = [PAD]; or [B, T, T] (query x key), True = may attend.

        Returns:
            head_outputs: [B, H, T, Dh] context-aware vectors.
            weights:      [B, H, T, T] attention probabilities (kept for backward).
        """
        # 1-2. Similarity of every query with every key, scaled by √Dh = 8 so softmax stays in
        #      a region with useful gradients. [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        scores = matmul(Q, K.swapaxes(-1, -2))
        scores *= 1.0 / math.sqrt(self.head_dim)

        # 3. Mask: [B, T] -> [B, 1, 1, T], so every query in every head ignores the same pad keys.
        #    A [B, T, T] (query x key) mask, e.g. the prefix-LM mask, -> [B, 1, T, T] for every head.
        mask = attention_mask[:, None, None, :] if attention_mask.ndim == 2 else attention_mask[:, None, :, :]
        fill = scores.dtype.type(max(MASK_VALUE, float(cp.finfo(scores.dtype).min)))  # -65504 in float16
        scores = cp.where(mask, scores, fill)

        # 4. Softmax over the key axis: each row becomes weights summing to 1.0 (pads get 0.0).
        weights = stable_softmax(scores, axis=-1)

        # 5. Weighted sum of values. [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        return matmul(weights, V), weights

    def multi_head_attentions(self, X: FloatArray, attention_mask: BoolArray) -> FloatArray:
        """Full MHA: project -> attend per head -> concatenate -> W_o.

        Args:
            X:              [B, T, D] block input (X_in).
            attention_mask: [B, T] bool, True = real token; or [B, T, T] (query x key), True = may attend.

        Returns:
            [B, T, D] — same shape as X, ready for the first Add & Norm.
        """
        Q, K, V = self.generate_q_k_v_matrices(X)  # each [B, H, T, Dh]
        head_outputs, weights = self.single_attention(Q, K, V, attention_mask)  # [B, H, T, Dh]
        concat = self._merge_heads(head_outputs)  # [B, T, D]
        output = matmul(concat, to_compute(self.W_o))  # [B, T, D] @ [D, D] -> [B, T, D]

        self.cache = {"X": X, "Q": Q, "K": K, "V": V, "weights": weights, "concat": concat}
        return output

    def forward(self, X: FloatArray, attention_mask: BoolArray) -> FloatArray:
        return self.multi_head_attentions(X, attention_mask)

    def backward(self, d_out: FloatArray) -> FloatArray:
        """Route the gradient through W_o, the weighted sum, the softmax and QKᵀ.

        Args:
            d_out: [B, T, D] gradient of the MHA output.

        Returns:
            [B, T, D] gradient of the MHA input X.
        """
        c = self.cache
        self.grads["W_o"] = matmul_weight_grad(c["concat"], d_out)  # [D, D]
        d_heads = cp.ascontiguousarray(self._split_heads(matmul(d_out, to_compute(self.W_o).T)))  # [B, H, T, Dh]

        # out = weights @ V
        d_weights = matmul(d_heads, c["V"].swapaxes(-1, -2))  # [B, H, T, T]
        d_V = matmul(c["weights"].swapaxes(-1, -2), d_heads)  # [B, H, T, Dh]

        # softmax: dS = P ⊙ (dP - Σ_j dP_j P_j). Masked positions have P = 0, so they get 0 gradient.
        p = c["weights"]
        row_dot = _row_dot(d_weights, p, axis=-1, keepdims=True)  # [B, H, T, 1] float32
        d_scores = _softmax_backward(p, d_weights, row_dot, cp.float32(1.0 / math.sqrt(self.head_dim)))

        # scores = Q Kᵀ
        d_Q = matmul(d_scores, c["K"])  # [B, H, T, Dh]
        d_K = matmul(d_scores.swapaxes(-1, -2), c["Q"])  # [B, H, T, Dh]

        # Back from heads to [B, T, D], then through the three input projections.
        d_Q, d_K, d_V = self._merge_heads(d_Q), self._merge_heads(d_K), self._merge_heads(d_V)
        X = c["X"]
        self.grads["W_q"] = matmul_weight_grad(X, d_Q)
        self.grads["W_k"] = matmul_weight_grad(X, d_K)
        self.grads["W_v"] = matmul_weight_grad(X, d_V)
        return (
            matmul(d_Q, to_compute(self.W_q).T)
            + matmul(d_K, to_compute(self.W_k).T)
            + matmul(d_V, to_compute(self.W_v).T)
        )

    def parameters(self) -> dict[str, FloatArray]:
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o}
