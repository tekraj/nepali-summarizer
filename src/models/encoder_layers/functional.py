"""Stateless math helpers shared by the Transformer layers."""

import cupy as cp

from src.tensor_types import FloatArray


def stable_softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Softmax that subtracts the row max first so cp.exp never overflows.

    Args:
        x: Any shape, e.g. attention scores (B, h, T, T) or logits (N, V).

    Returns:
        Same shape as ``x``; values along ``axis`` are >= 0 and sum to 1.0.
    """
    exp_x = cp.exp(x - cp.max(x, axis=axis, keepdims=True))
    return exp_x / cp.sum(exp_x, axis=axis, keepdims=True)


def matmul_weight_grad(x: FloatArray, d_out: FloatArray) -> FloatArray:
    """Gradient of ``W`` for ``out = x @ W``, summed over every batch/time position.

    Args:
        x:     Layer input, (..., d_in), e.g. (B, T, 512).
        d_out: Gradient of the layer output, (..., d_out), e.g. (B, T, 2048).

    Returns:
        (d_in, d_out) — same shape as ``W``.
    """
    return x.reshape(-1, x.shape[-1]).T @ d_out.reshape(-1, d_out.shape[-1])
