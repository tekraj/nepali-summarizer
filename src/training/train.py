"""Training skeleton and utility functions."""

from __future__ import annotations

from typing import Dict, Iterable, List


def train_epoch(model, dataloader: Iterable, optimizer=None, device: str = "cpu") -> Dict[str, float]:
    """Placeholder training step that returns summary statistics."""
    total_loss = 0.0
    steps = 0

    for batch_index, _batch in enumerate(dataloader):
        steps += 1
        total_loss += 0.0

    return {"loss": total_loss / max(steps, 1), "steps": steps, "device": device}


def evaluate_model(model, dataloader: Iterable, device: str = "cpu") -> Dict[str, float]:
    """Placeholder evaluation function."""
    return {"loss": 0.0, "steps": 0, "device": device}


def save_checkpoint(model, path: str) -> None:
    """Save a model checkpoint to disk."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("checkpoint placeholder\n")
