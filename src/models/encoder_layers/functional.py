"""Stateless math helpers shared by the Transformer layers.

Mixed precision (fp16): the weights, their gradients and the Adam state stay float32 (the
"master" copy that the optimizer updates and checkpoints store). Each forward pass casts the
weights to ``compute_dtype()`` (float16 by default), and the matmuls and the activations
passed between layers use that dtype, which halves activation memory and runs the matmuls
on the GPU's fp16 tensor cores. Reductions (softmax sums, LayerNorm statistics, weight
gradients) accumulate in float32. ``Adam`` scales the loss so small fp16 gradients do not
underflow to zero.
"""

import numpy as np
import cupy as cp
from cupy.cuda import device
from cupy_backends.cuda.api import runtime

from src.tensor_types import FloatArray

_compute_dtype = cp.float16


def set_compute_dtype(dtype) -> None:
    """Choose the activation / matmul dtype: ``cp.float16`` (mixed precision) or ``cp.float32``."""
    global _compute_dtype
    _compute_dtype = cp.dtype(dtype).type


def compute_dtype():
    """The dtype activations and matmuls use, e.g. ``cp.float16``."""
    return _compute_dtype


def to_compute(x: FloatArray) -> FloatArray:
    """Cast to the compute dtype (no copy if it already has it), e.g. a float32 weight -> float16."""
    return x.astype(_compute_dtype, copy=False)


def stable_softmax(x: FloatArray, axis: int = -1, out_dtype=None) -> FloatArray:
    """Softmax that subtracts the row max first so cp.exp never overflows.

    The row sum always accumulates in float32, so fp16 attention rows of T = 1024 keys stay accurate.

    Args:
        x:         Any shape, e.g. attention scores (B, h, T, T) or logits (N, V).
        out_dtype: dtype of the result (default: x's dtype); the LM head asks for float32.

    Returns:
        Same shape as ``x``; values along ``axis`` are >= 0 and sum to 1.0.
    """
    out_dtype = out_dtype or x.dtype
    if out_dtype != x.dtype:
        x = x.astype(out_dtype)
    exp_x = cp.exp(x - cp.max(x, axis=axis, keepdims=True))
    total = cp.sum(exp_x, axis=axis, keepdims=True, dtype=cp.float32)
    exp_x /= total.astype(out_dtype, copy=False)
    return exp_x


def matmul_weight_grad(x: FloatArray, d_out: FloatArray) -> FloatArray:
    """Gradient of ``W`` for ``out = x @ W``, summed over every batch/time position.

    Args:
        x:     Layer input, (..., d_in), e.g. (B, T, 512).
        d_out: Gradient of the layer output, (..., d_out), e.g. (B, T, 2048).

    Returns:
        (d_in, d_out) float32 — same shape and dtype as the master ``W``.
    """
    grad = x.reshape(-1, x.shape[-1]).T @ d_out.reshape(-1, d_out.shape[-1])
    return grad.astype(cp.float32, copy=False)


def _gemm_operand(x: FloatArray) -> tuple[FloatArray, bool]:
    """A [..., r, c] array as cuBLAS can read it without a copy: (array, is_transposed).

    It must be C-contiguous, or a ``swapaxes(-1, -2)`` view of a C-contiguous array such as
    ``K.swapaxes(-1, -2)``; anything else is copied to C order first.
    """
    if x.flags.c_contiguous:
        return x, False
    if x.swapaxes(-1, -2).flags.c_contiguous:
        return x, True
    return cp.ascontiguousarray(x), False


def matmul(a: FloatArray, b: FloatArray) -> FloatArray:
    """``a @ b``, routed to the fastest cuBLAS call.

    - [..., d_in] @ [d_in, d_out] (a layer applied to every token, e.g. X W_q): flattened to
      one 2-D matmul, because CuPy runs the broadcast 3-D form as a slow batched matmul.
    - stacks of matrices with the same leading axes, e.g. attention's
      [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]: CuPy's own ``@`` does not use the tensor
      cores for batched float16 (it is slower than float32), so float16 goes straight to
      cuBLAS ``gemmStridedBatchedEx`` with float32 accumulation.
    """
    if b.ndim == 2 and a.ndim > 2:
        return (a.reshape(-1, a.shape[-1]) @ b).reshape(*a.shape[:-1], b.shape[-1])
    if a.dtype != cp.float16 or b.dtype != cp.float16 or a.shape[:-2] != b.shape[:-2] or a.ndim < 3:
        return a @ b
    # Imported here, not at the top: importing cuBLAS before CuPy compiles its first kernel can
    # break that compilation on setups with mixed CUDA 12 / 13 headers.
    from cupy_backends.cuda.libs import cublas

    *lead, m, k = a.shape
    n = b.shape[-1]
    a, a_t = _gemm_operand(a)
    b, b_t = _gemm_operand(b)
    out = cp.empty((*lead, m, n), dtype=cp.float16)
    batch = out.size // (m * n)

    # cuBLAS is column-major, and a row-major [r, c] matrix is its [c, r] transpose there, so
    # out = a @ b is computed as outᵀ = bᵀ aᵀ. A transposed view needs op T and ld = its rows.
    op_b, ld_b = (cublas.CUBLAS_OP_T, k) if b_t else (cublas.CUBLAS_OP_N, n)
    op_a, ld_a = (cublas.CUBLAS_OP_T, m) if a_t else (cublas.CUBLAS_OP_N, k)
    alpha, beta = np.ones(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    handle = device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    cublas.gemmStridedBatchedEx(
        handle, op_b, op_a, n, m, k,
        alpha.ctypes.data,
        b.data.ptr, runtime.CUDA_R_16F, ld_b, k * n,
        a.data.ptr, runtime.CUDA_R_16F, ld_a, m * k,
        beta.ctypes.data,
        out.data.ptr, runtime.CUDA_R_16F, n, m * n,
        batch, cublas.CUBLAS_COMPUTE_32F, cublas.CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    return out
