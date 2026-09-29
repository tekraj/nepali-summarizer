"""Adam optimizer (documents/encoders/7-linearization-and-softmax.md, "Optimizer Step")."""

import math

import cupy as cp

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
        grad_norm = math.sqrt(sum(float(cp.sum(g * g)) for g in grads.values()))
        clip = min(1.0, self.max_grad_norm / (grad_norm + 1e-6))

        # Bias correction: m and v start at 0, so early averages are rescaled upwards.
        lr_t = self.learning_rate * math.sqrt(1 - self.beta2**self.step_count) / (1 - self.beta1**self.step_count)

        for name, param in params.items():
            grad = grads[name] * clip
            if name not in self.m:
                self.m[name] = cp.zeros_like(param)
                self.v[name] = cp.zeros_like(param)
            m, v = self.m[name], self.v[name]
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad * grad
            param -= lr_t * m / (cp.sqrt(v) + self.eps)
        return grad_norm

    def state_dict(self) -> dict[str, FloatArray]:
        """Everything needed to continue with the exact same updates, as flat checkpoint entries.

        Returns:
            ``optimizer.step_count`` (t, drives the bias correction), the hyperparameters, and
            ``optimizer.m.<param name>`` / ``optimizer.v.<param name>`` per weight (same shapes).
        """
        state = {
            "optimizer.step_count": cp.asarray(self.step_count, dtype=cp.int64),
            "optimizer.learning_rate": cp.asarray(self.learning_rate),
            "optimizer.beta1": cp.asarray(self.beta1),
            "optimizer.beta2": cp.asarray(self.beta2),
            "optimizer.eps": cp.asarray(self.eps),
            "optimizer.max_grad_norm": cp.asarray(self.max_grad_norm),
        }
        for name in self.m:
            state[f"optimizer.m.{name}"] = self.m[name]
            state[f"optimizer.v.{name}"] = self.v[name]
        return state

    def load_state_dict(self, state: dict[str, FloatArray], params: dict[str, FloatArray]) -> None:
        """Restore ``state_dict()`` output. The learning rate etc. keep their current values, so
        a changed config still applies; only t, m and v come from the checkpoint.

        Args:
            state:  checkpoint entries (extra non-optimizer keys are ignored).
            params: ``model.parameters()``, to check that m and v match every weight.
        """
        self.step_count = int(state["optimizer.step_count"])
        self.m, self.v = {}, {}
        for name, param in params.items():
            m, v = state.get(f"optimizer.m.{name}"), state.get(f"optimizer.v.{name}")
            if m is None or v is None:
                continue  # weight never updated yet; step() creates zeros for it
            if m.shape != param.shape or v.shape != param.shape:
                raise ValueError(f"optimizer state for {name} has shape {m.shape}, weight is {param.shape}")
            self.m[name] = cp.asarray(m, dtype=param.dtype)
            self.v[name] = cp.asarray(v, dtype=param.dtype)
