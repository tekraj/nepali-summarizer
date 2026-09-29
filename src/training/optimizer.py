"""Adam optimizer (documents/encoders/7-linearization-and-softmax.md, "Optimizer Step")."""

import math

import cupy as cp

from src.tensor_types import FloatArray

# One fused GPU kernel per weight for the whole update below, instead of ~12 separate array
# passes (it also updates param, m and v in place).
_adam_update = cp.ElementwiseKernel(
    "T grad, T grad_factor, T beta1, T beta2, T lr_t, T eps",
    "T param, T m, T v",
    """
    T g = grad * grad_factor;
    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g * g;
    param -= lr_t * m / (sqrt(v) + eps);
    """,
    "adam_update",
)


class Adam:
    """Adam with global-norm gradient clipping and dynamic loss scaling, updating weights in place.

    For every weight it keeps two running averages:
        m — mean of the gradients (momentum)
        v — mean of the squared gradients (per-weight step size, as in RMSprop)

    Loss scaling (for fp16 training): the loss gradient is multiplied by ``loss_scale`` before
    backward, so tiny float16 gradients do not round to zero, and ``step`` divides it back out.
    If a gradient overflowed (inf / NaN), the step is skipped and the scale halved; after
    ``scale_window`` good steps in a row it is doubled again.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        loss_scale: float = 2.0**16,
        scale_window: int = 1000,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.loss_scale = loss_scale
        self.scale_window = scale_window
        self.good_steps = 0  # steps in a row without overflow, since the last scale change
        self.step_count = 0
        self.m: dict[str, FloatArray] = {}
        self.v: dict[str, FloatArray] = {}

    def step(self, params: dict[str, FloatArray], grads: dict[str, FloatArray]) -> float:
        """Apply one update: W <- W - lr * m̂ / (√v̂ + ε).

        Args:
            params: ``model.parameters()`` — updated in place, so the layers see the new values.
            grads:  ``model.gradients()`` — same keys and shapes, computed from the loss
                    multiplied by ``loss_scale``.

        Returns:
            Global (unscaled) gradient norm before clipping (useful to watch for instability);
            inf or NaN if the gradients overflowed and the step was skipped.
        """
        scale = self.loss_scale
        grad_norm = float(cp.sqrt(sum(cp.sum(g * g) for g in grads.values()))) / scale  # one GPU sync
        if not math.isfinite(grad_norm):
            self.loss_scale = max(scale / 2, 1.0)
            self.good_steps = 0
            return grad_norm
        self.good_steps += 1
        if self.good_steps >= self.scale_window:
            self.loss_scale = scale * 2
            self.good_steps = 0

        self.step_count += 1
        clip = min(1.0, self.max_grad_norm / (grad_norm + 1e-6))

        # Bias correction: m and v start at 0, so early averages are rescaled upwards.
        lr_t = self.learning_rate * math.sqrt(1 - self.beta2**self.step_count) / (1 - self.beta1**self.step_count)

        for name, param in params.items():
            if name not in self.m:
                self.m[name] = cp.zeros_like(param)
                self.v[name] = cp.zeros_like(param)
            f32 = param.dtype.type
            _adam_update(
                grads[name], f32(clip / scale), f32(self.beta1), f32(self.beta2), f32(lr_t), f32(self.eps),
                param, self.m[name], self.v[name],
            )
        return grad_norm

    def state_dict(self) -> dict[str, FloatArray]:
        """Everything needed to continue with the exact same updates, as flat checkpoint entries.

        Returns:
            ``optimizer.step_count`` (t, drives the bias correction), the hyperparameters, the
            loss scale, and ``optimizer.m.<param name>`` / ``optimizer.v.<param name>`` per weight.
        """
        state = {
            "optimizer.step_count": cp.asarray(self.step_count, dtype=cp.int64),
            "optimizer.learning_rate": cp.asarray(self.learning_rate),
            "optimizer.beta1": cp.asarray(self.beta1),
            "optimizer.beta2": cp.asarray(self.beta2),
            "optimizer.eps": cp.asarray(self.eps),
            "optimizer.max_grad_norm": cp.asarray(self.max_grad_norm),
            "optimizer.loss_scale": cp.asarray(self.loss_scale),
            "optimizer.good_steps": cp.asarray(self.good_steps, dtype=cp.int64),
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
        if "optimizer.loss_scale" in state:  # absent in checkpoints from float32 training
            self.loss_scale = float(state["optimizer.loss_scale"])
            self.good_steps = int(state["optimizer.good_steps"])
        self.m, self.v = {}, {}
        for name, param in params.items():
            m, v = state.get(f"optimizer.m.{name}"), state.get(f"optimizer.v.{name}")
            if m is None or v is None:
                continue  # weight never updated yet; step() creates zeros for it
            if m.shape != param.shape or v.shape != param.shape:
                raise ValueError(f"optimizer state for {name} has shape {m.shape}, weight is {param.shape}")
            self.m[name] = cp.asarray(m, dtype=param.dtype)
            self.v[name] = cp.asarray(v, dtype=param.dtype)
