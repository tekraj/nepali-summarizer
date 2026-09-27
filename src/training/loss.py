"""Cross-entropy loss (documents/encoders/7-linearization-and-softmax.md, "Loss Calculation")."""

import cupy as cp

from src.tensor_types import FloatArray, IntArray


def cross_entropy_loss(probabilities: FloatArray, targets: IntArray) -> tuple[float, FloatArray]:
    """Mean negative log-likelihood of the correct tokens, plus its gradient w.r.t. the logits.

    For softmax followed by cross-entropy, the combined derivative simplifies to
    ``probabilities - one_hot(targets)``, so the gradient never goes through softmax separately.

    Args:
        probabilities: [N, V] softmax output of the LM head.
        targets:       [N] int64 ground-truth token IDs.

    Returns:
        loss:     scalar float.
        d_logits: [N, V] gradient to pass to ``TransformerEncoder.backward``.
    """
    n = targets.shape[0]
    rows = cp.arange(n)
    loss = -cp.mean(cp.log(probabilities[rows, targets] + 1e-9))

    d_logits = probabilities.copy()
    d_logits[rows, targets] -= 1.0
    d_logits /= n
    return float(loss), d_logits
