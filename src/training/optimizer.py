"""Adam optimizer (documents/encoders/7-linearization-and-softmax.md, "Optimizer Step")."""

import math

import numpy as np

from src.tensor_types import FloatArray


class Adam:
    """Adam with global-norm gradient clipping, updating weights in place.

    For every weight it keeps two running averages:
        m — mean of the gradients (momentum)
        v — mean of the squared gradients (per-weight step size, as in RMSprop)
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        self.m: dict[str, FloatArray] = {}
        self.v: dict[str, FloatArray] = {}

    def step(self, params: dict[str, FloatArray], grads: dict[str, FloatArray]) -> float:
        """Apply one update: W <- W - lr * m̂ / (√v̂ + ε).

        Args:
            params: ``model.parameters()`` — updated in place, so the layers see the new values.
            grads:  ``model.gradients()`` — same keys and shapes.

        Returns:
            Global gradient norm before clipping (useful to watch for instability).
        """
        self.step_count += 1
        grad_norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        clip = min(1.0, self.max_grad_norm / (grad_norm + 1e-6))

        # Bias correction: m and v start at 0, so early averages are rescaled upwards.
        lr_t = self.learning_rate * math.sqrt(1 - self.beta2**self.step_count) / (1 - self.beta1**self.step_count)

        for name, param in params.items():
            grad = grads[name] * clip
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            m, v = self.m[name], self.v[name]
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad * grad
            param -= lr_t * m / (np.sqrt(v) + self.eps)
        return grad_norm
