"""Training loop: forward -> loss -> backward -> Adam step (documents/encoders/8-end-to-end.md, Stage 5)."""

from __future__ import annotations

from pathlib import Path

import cupy as cp
from tqdm import tqdm

from src.data_preprocessing.batching import CreateTrainingBatch
from src.models.encoder import TransformerEncoder
from src.tensor_types import BoolArray, IntArray
from src.training.loss import cross_entropy_loss
from src.training.optimizer import Adam


def train_step(
    model: TransformerEncoder,
    optimizer: Adam,
    token_ids: IntArray,
    attention_mask: BoolArray,
    loss_mask: BoolArray,
    targets: IntArray,
) -> tuple[float, float]:
    """One full training step on a batch. Returns (loss, grad_norm).

    Args:
        token_ids:      [B, T] ``<s> article </s> summary </s>`` token IDs.
        attention_mask: [B, T, T] bool prefix-LM mask.
        loss_mask:      [B, T] bool, the N positions that predict a summary token.
        targets:        [N] the next token at each of those positions.
    """
    # Only the summary positions reach the LM head, so the loss covers the summary alone.
    _, probabilities = model.forward(token_ids, attention_mask, output_mask=loss_mask)  # [N, V]
    loss, d_logits = cross_entropy_loss(probabilities, targets)
    model.backward(d_logits)
    grad_norm = optimizer.step(model.parameters(), model.gradients())
    return loss, grad_norm


def train_epoch(
    model: TransformerEncoder,
    batcher: CreateTrainingBatch,
    optimizer: Adam,
    article_dir: str | Path,
    summary_dir: str | Path,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Stream every (article, summary) batch once and train on it."""
    batcher.create_batch_pairs(article_dir, summary_dir)
    total_loss, steps = 0.0, 0
    progress = tqdm(desc="train", unit="batch", total=max_steps)

    while (batch := batcher.create_input_tensor()) is not None:
        loss, grad_norm = train_step(model, optimizer, *batch)
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
    article_dir: str | Path,
    summary_dir: str | Path,
) -> dict[str, float]:
    """Summary-token loss on held-out pairs, forward pass only (no weight updates)."""
    batcher.create_batch_pairs(article_dir, summary_dir)
    total_loss, steps = 0.0, 0
    while (batch := batcher.create_input_tensor()) is not None:
        token_ids, attention_mask, loss_mask, targets = batch
        _, probabilities = model.forward(token_ids, attention_mask, output_mask=loss_mask)
        total_loss += cross_entropy_loss(probabilities, targets)[0]
        steps += 1
    return {"loss": total_loss / max(steps, 1), "steps": steps}


def save_checkpoint(model: TransformerEncoder, path: str | Path) -> None:
    """Save every weight to a single .npz file, keyed by ``model.parameters()`` names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cp.savez(path, **model.parameters())


def load_checkpoint(model: TransformerEncoder, path: str | Path) -> None:
    """Copy weights from a .npz file into an already-built model with the same config."""
    with cp.load(path) as saved:
        for name, param in model.parameters().items():
            param[...] = saved[name]  # in place, so the layers keep the same arrays
