"""Training loop: forward -> loss -> backward -> Adam step (documents/encoders/8-end-to-end.md, Stage 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data_preprocessing.batching import CreateTrainingBatch
from src.models.encoder import TransformerEncoder
from src.tensor_types import BoolArray, IntArray
from src.training.loss import cross_entropy_loss
from src.training.optimizer import Adam


def train_step(
    model: TransformerEncoder,
    optimizer: Adam,
    batcher: CreateTrainingBatch,
    token_ids: IntArray,
    attention_mask: BoolArray,
    mlm_probability: float,
) -> tuple[float, float] | None:
    """One full training step on a batch. Returns (loss, grad_norm), or None if nothing was masked.

    Args:
        token_ids:      [B, T] original token IDs.
        attention_mask: [B, T] bool.
    """
    masked_ids, target_mask = batcher.create_mlm_inputs(token_ids, attention_mask, mlm_probability)
    if not target_mask.any():
        return None
    targets = token_ids[target_mask]  # [N] the hidden tokens the model must recover

    _, probabilities = model.forward(masked_ids, attention_mask, output_mask=target_mask)  # [N, V]
    loss, d_logits = cross_entropy_loss(probabilities, targets)
    model.backward(d_logits)
    grad_norm = optimizer.step(model.parameters(), model.gradients())
    return loss, grad_norm


def train_epoch(
    model: TransformerEncoder,
    batcher: CreateTrainingBatch,
    optimizer: Adam,
    data_path: str | Path,
    mlm_probability: float = 0.15,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Stream every batch in ``data_path`` once and train on it."""
    batcher.create_batch_sentences(data_path)
    total_loss, steps = 0.0, 0
    progress = tqdm(desc="train", unit="batch", total=max_steps)

    while (batch := batcher.create_input_tensor()) is not None:
        result = train_step(model, optimizer, batcher, *batch, mlm_probability)
        if result is None:
            continue
        loss, grad_norm = result
        total_loss += loss
        steps += 1
        progress.update()
        progress.set_postfix(loss=f"{loss:.4f}", avg=f"{total_loss / steps:.4f}", grad=f"{grad_norm:.2f}", T=batch[0].shape[1])
        if max_steps is not None and steps >= max_steps:
            break

    progress.close()
    return {"loss": total_loss / max(steps, 1), "steps": steps}


def evaluate_model(
    model: TransformerEncoder,
    batcher: CreateTrainingBatch,
    data_path: str | Path,
    mlm_probability: float = 0.15,
) -> dict[str, float]:
    """Masked-token loss on held-out data, forward pass only (no weight updates)."""
    batcher.create_batch_sentences(data_path)
    total_loss, steps = 0.0, 0
    while (batch := batcher.create_input_tensor()) is not None:
        token_ids, attention_mask = batch
        masked_ids, target_mask = batcher.create_mlm_inputs(token_ids, attention_mask, mlm_probability)
        if not target_mask.any():
            continue
        _, probabilities = model.forward(masked_ids, attention_mask, output_mask=target_mask)
        total_loss += cross_entropy_loss(probabilities, token_ids[target_mask])[0]
        steps += 1
    return {"loss": total_loss / max(steps, 1), "steps": steps}


def save_checkpoint(model: TransformerEncoder, path: str | Path) -> None:
    """Save every weight to a single .npz file, keyed by ``model.parameters()`` names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model.parameters())


def load_checkpoint(model: TransformerEncoder, path: str | Path) -> None:
    """Copy weights from a .npz file into an already-built model with the same config."""
    with np.load(path) as saved:
        for name, param in model.parameters().items():
            param[...] = saved[name]  # in place, so the layers keep the same arrays
