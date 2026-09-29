"""Training loop: forward -> loss -> backward -> Adam step (documents/encoders/8-end-to-end.md, Stage 5)."""

from __future__ import annotations

import math
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

    The loss gradient is multiplied by ``optimizer.loss_scale`` before backward so small fp16
    gradients survive; ``optimizer.step`` divides it back out, or skips the update (grad_norm is
    then inf / NaN) if a gradient overflowed.

    Args:
        token_ids:      [B, T] ``<s> article </s> summary </s>`` token IDs.
        attention_mask: [B, T, T] bool prefix-LM mask.
        loss_mask:      [B, T] bool, the N positions that predict a summary token.
        targets:        [N] the next token at each of those positions.
    """
    # Only the summary positions reach the LM head, so the loss covers the summary alone.
    _, probabilities = model.forward(token_ids, attention_mask, output_mask=loss_mask)  # [N, V]
    loss, d_logits = cross_entropy_loss(probabilities, targets)
    d_logits *= optimizer.loss_scale
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
    skip_steps: int = 0,
    save_every: int | None = None,
    partial_checkpoint_stem: str | Path | None = None,
    epoch: int | None = None,
    loss_sum: float = 0.0,
    loss_steps: int = 0,
) -> dict[str, float]:
    """Stream every (article, summary) batch once and train on it.

    Args:
        skip_steps:              batches to pass over untrained, to continue an epoch that
                                 stopped midway (the data order is the same every epoch).
        save_every:              save a mid-epoch checkpoint every this many batches.
        partial_checkpoint_stem: e.g. ``checkpoints/summarizer_epoch_10``; mid-epoch checkpoints
                                 are written as ``{stem}_step_{step}.npz``, keeping only the newest.
                                 They include the Adam state and the ``state.*`` entries.
        epoch:                   epoch number, stored in mid-epoch checkpoints.
        loss_sum, loss_steps:    running loss restored from a mid-epoch checkpoint, so the
                                 epoch average continues instead of restarting.
    """
    batcher.create_batch_pairs(article_dir, summary_dir)
    batcher.skip_batches(skip_steps)
    total_loss, steps = loss_sum, loss_steps
    step = skip_steps  # batches into the epoch, counting the skipped ones
    last_partial: Path | None = None
    progress = tqdm(desc="train", unit="batch", total=max_steps, initial=skip_steps)

    loss = float("nan")
    while (batch := batcher.create_input_tensor()) is not None:
        loss, grad_norm = train_step(model, optimizer, *batch)
        if math.isfinite(loss):  # an fp16 overflow step (skipped by Adam) stays out of the average
            total_loss += loss
            steps += 1
        step += 1
        progress.update()
        progress.set_postfix(
            loss=f"{loss:.4f}", avg=f"{total_loss / max(steps, 1):.4f}", grad=f"{grad_norm:.2f}",
            scale=f"{optimizer.loss_scale:g}", T=batch[0].shape[1],
        )
        if save_every and partial_checkpoint_stem and step % save_every == 0:
            path = Path(f"{partial_checkpoint_stem}_step_{step}.npz")
            training_state = {
                "epoch": epoch if epoch is not None else 0,
                "step": step,
                "batch_size": batcher.batch_size,
                "epoch_complete": False,
                "loss_sum": total_loss,
                "loss_steps": steps,
                "last_loss": loss,
            }
            save_checkpoint(model, path, optimizer, training_state)
            if last_partial is not None and last_partial != path:
                last_partial.unlink(missing_ok=True)
            last_partial = path
        if max_steps is not None and step >= max_steps:
            break

    progress.close()
    if last_partial is not None:
        last_partial.unlink(missing_ok=True)  # the end-of-epoch checkpoint supersedes it
    return {"loss": total_loss / max(steps, 1), "steps": steps, "step": step, "last_loss": loss}


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


TRAINING_STATE_PREFIX = "state."


def save_checkpoint(
    model: TransformerEncoder,
    path: str | Path,
    optimizer: Adam | None = None,
    training_state: dict[str, float | int | bool] | None = None,
) -> None:
    """Save every weight to a single .npz file, keyed by ``model.parameters()`` names.

    With ``optimizer`` and ``training_state`` the file also holds everything needed to continue
    exactly where training stopped:
        optimizer.*  Adam's step count t and its m / v per weight (see ``Adam.state_dict``)
        state.*      where in the data training was and the loss so far, e.g. ``state.epoch``,
                     ``state.step`` (batches done in that epoch), ``state.epoch_complete``,
                     ``state.loss_sum`` / ``state.loss_steps`` (running average), ``state.last_loss``

    The weight keys are unchanged, so these files still load in ``load_checkpoint`` for inference.
    Layer caches and gradients are not saved: every forward/backward pass recomputes them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(model.parameters())
    if optimizer is not None:
        arrays.update(optimizer.state_dict())
    for key, value in (training_state or {}).items():
        arrays[f"{TRAINING_STATE_PREFIX}{key}"] = cp.asarray(value)
    # Write to a temp file first, so a crash mid-save never leaves a truncated checkpoint.
    tmp_path = path.with_name(f"{path.stem}.tmp.npz")
    cp.savez(tmp_path, **arrays)
    tmp_path.replace(path)


def load_checkpoint(
    model: TransformerEncoder, path: str | Path, optimizer: Adam | None = None
) -> dict[str, float | int | bool]:
    """Copy weights from a .npz file into an already-built model with the same config.

    If ``optimizer`` is given and the file has Adam state, that is restored too.

    Returns:
        The ``state.*`` entries without the prefix, e.g. ``{"epoch": 10, "step": 24000, ...}``;
        empty for checkpoints saved before training state was stored.
    """
    with cp.load(path) as saved:
        keys = getattr(saved, "npz_file", saved).files  # cupy wraps numpy's NpzFile
        for name, param in model.parameters().items():
            param[...] = saved[name]  # in place, so the layers keep the same arrays
        if optimizer is not None and "optimizer.step_count" in keys:
            optimizer_state = {key: saved[key] for key in keys if key.startswith("optimizer.")}
            optimizer.load_state_dict(optimizer_state, model.parameters())
        training_state = {}
        for key in keys:
            if key.startswith(TRAINING_STATE_PREFIX):
                value = saved[key].item()  # 0-d array -> Python int / float / bool
                training_state[key[len(TRAINING_STATE_PREFIX):]] = value
    return training_state
