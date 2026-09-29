"""Step 5 — Position-wise Feed-Forward Network (documents/encoders/5-feed_forward.md).

    [B, T, D=512] ──W1, b1──> [B, T, D_ff=2048] ──ReLU──> [B, T, 2048] ──W2, b2──> [B, T, 512]
"""

import cupy as cp

from .functional import matmul, matmul_weight_grad, to_compute
from src.tensor_types import FloatArray


class PositionwiseFeedForward:
    """FFN(x) = max(0, x W1 + b1) W2 + b2, applied to every token independently.

    Attention lets tokens talk to each other; the FFN lets each token process what it learned alone.
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048) -> None:
        """
        Args:
            d_model: D, e.g. 512.
            d_ff:    D_ff, expanded inner width, typically 4 * D = 2048.
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Step 1 parameters — up-projection D -> D_ff
        self.W1 = cp.random.normal(0, 0.02, (d_model, d_ff)).astype(cp.float32)  # [512, 2048]
        self.b1 = cp.zeros(d_ff, dtype=cp.float32)  # [2048]
        # Step 3 parameters — down-projection D_ff -> D
        self.W2 = cp.random.normal(0, 0.02, (d_ff, d_model)).astype(cp.float32)  # [2048, 512]
        self.b2 = cp.zeros(d_model, dtype=cp.float32)  # [512]

        self.grads: dict[str, FloatArray] = {}
        self.cache: dict[str, FloatArray] = {}

    def relu(self, x: FloatArray) -> FloatArray:
        """Step 2 — zero out negatives; without it W1 and W2 would collapse into one linear layer."""
        return cp.maximum(x, 0)

    def forward(self, x: FloatArray) -> FloatArray:
        """
        Args:
            x: [B, T, D] output of the first Add & Norm (X_norm1).

        Returns:
            [B, T, D] transformed tensor, routed into the second Add & Norm.
        """
        # float32 master weights -> compute dtype (float16) copies, kept for backward.
        W1, W2 = to_compute(self.W1), to_compute(self.W2)
        hidden = matmul(x, W1) + to_compute(self.b1)  # Step 1: expand   [B, T, 512] -> [B, T, 2048]
        activated = self.relu(hidden)  # Step 2: gate     [B, T, 2048]
        output = matmul(activated, W2) + to_compute(self.b2)  # Step 3: compress [B, T, 2048] -> [B, T, 512]

        # `activated > 0` equals `hidden > 0`, so caching `activated` alone is enough for backward.
        self.cache = {"x": x, "activated": activated, "W1": W1, "W2": W2}
        return output

    def backward(self, d_out: FloatArray) -> FloatArray:
        """
        Args:
            d_out: [B, T, D] gradient of the FFN output.

        Returns:
            [B, T, D] gradient of the FFN input.
        """
        x, activated = self.cache["x"], self.cache["activated"]

        self.grads["W2"] = matmul_weight_grad(activated, d_out)  # [2048, 512]
        self.grads["b2"] = d_out.sum(axis=(0, 1), dtype=cp.float32)  # [512]
        d_hidden = matmul(d_out, self.cache["W2"].T) * (activated > 0)  # ReLU'(h) = 1 if h > 0 else 0

        self.grads["W1"] = matmul_weight_grad(x, d_hidden)  # [512, 2048]
        self.grads["b1"] = d_hidden.sum(axis=(0, 1), dtype=cp.float32)  # [2048]
        return matmul(d_hidden, self.cache["W1"].T)  # [B, T, 512]

    def parameters(self) -> dict[str, FloatArray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
