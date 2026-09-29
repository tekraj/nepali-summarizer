"""Steps 4 & 6 — Add & Norm (documents/encoders/4-add_and_normalization.md, 6-second-add-and-norm.md).

Post-LN, exactly as in the docs:

    X_in [B, T, D] ─┬─> sublayer (MHA or FFN) ─> [B, T, D] ─┐
                    └──────────────────────────────────────┴─> Add ─> LayerNorm ─> [B, T, D]
"""

import cupy as cp

from src.tensor_types import FloatArray


class ResidualConnection:
    """PHASE 1 — "Add": element-wise sum of the block input and the sublayer output."""

    def forward(self, x_input: FloatArray, x_sublayer: FloatArray) -> FloatArray:
        """
        Args:
            x_input:    [B, T, D] tensor that skipped the sublayer (keeps token identity).
            x_sublayer: [B, T, D] sublayer output (adds context).

        Returns:
            X_res: [B, T, D].
        """
        return x_input + x_sublayer

    def backward(self, d_out: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The sum copies the gradient unchanged to both branches — the "gradient highway"."""
        return d_out, d_out


class LayerNormalization:
    """PHASE 2 — "Norm": normalise every token vector across its own D features."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self.eps = eps
        self.gamma = cp.ones(d_model, dtype=cp.float32)  # γ scale: [D]
        self.beta = cp.zeros(d_model, dtype=cp.float32)  # β shift: [D]

        self.grads: dict[str, FloatArray] = {}
        self.cache: dict[str, FloatArray] = {}

    def forward(self, x: FloatArray) -> FloatArray:
        """LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β, with μ, σ² per token.

        Args:
            x: [B, T, D].

        Returns:
            [B, T, D] in x's dtype, each token vector has ~zero mean and unit variance before γ/β.
        """
        dtype = x.dtype
        x = x.astype(cp.float32, copy=False)  # the statistics need float32: a float16 x² overflows above 256
        mu = cp.mean(x, axis=-1, keepdims=True)  # [B, T, 1]
        var = cp.var(x, axis=-1, keepdims=True)  # [B, T, 1]
        inv_std = 1.0 / cp.sqrt(var + self.eps)  # [B, T, 1]
        x_hat = (x - mu) * inv_std  # [B, T, D]

        self.cache = {"x_hat": x_hat.astype(dtype, copy=False), "inv_std": inv_std}
        return (self.gamma * x_hat + self.beta).astype(dtype, copy=False)

    def backward(self, d_out: FloatArray) -> FloatArray:
        """
        Args:
            d_out: [B, T, D] gradient of the normalised output.

        Returns:
            [B, T, D] gradient of the input x, in d_out's dtype.
        """
        dtype = d_out.dtype
        d_out = d_out.astype(cp.float32, copy=False)
        x_hat, inv_std = self.cache["x_hat"].astype(cp.float32, copy=False), self.cache["inv_std"]
        self.grads["gamma"] = cp.sum(d_out * x_hat, axis=(0, 1))  # [D]
        self.grads["beta"] = cp.sum(d_out, axis=(0, 1))  # [D]

        d_x_hat = d_out * self.gamma  # [B, T, D]
        # Standard LayerNorm input gradient (μ and σ² both depend on every feature of x).
        return (inv_std * (
            d_x_hat
            - cp.mean(d_x_hat, axis=-1, keepdims=True)
            - x_hat * cp.mean(d_x_hat * x_hat, axis=-1, keepdims=True)
        )).astype(dtype, copy=False)

    def parameters(self) -> dict[str, FloatArray]:
        return {"gamma": self.gamma, "beta": self.beta}


class AddNormBlock:
    """PHASE 3 — reusable Add & Norm wrapper, independent of what the sublayer is.

    Each EncoderBlock owns two of these (each with its own γ, β): one after MHA, one after FFN.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.add = ResidualConnection()
        self.norm = LayerNormalization(d_model, eps)

    def forward(self, x_input: FloatArray, x_sublayer_out: FloatArray) -> FloatArray:
        """
        Args:
            x_input:        [B, T, D] original tensor that bypassed the sublayer.
            x_sublayer_out: [B, T, D] tensor exiting the sublayer.

        Returns:
            [B, T, D] added and normalised tensor.
        """
        x_res = self.add.forward(x_input, x_sublayer_out)  # Step 1: Add  -> [B, T, D]
        return self.norm.forward(x_res)  # Step 2: Norm -> [B, T, D]

    def backward(self, d_out: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Returns:
            (d_x_input, d_x_sublayer_out), both [B, T, D].
        """
        return self.add.backward(self.norm.backward(d_out))

    def parameters(self) -> dict[str, FloatArray]:
        return self.norm.parameters()

    @property
    def grads(self) -> dict[str, FloatArray]:
        return self.norm.grads
