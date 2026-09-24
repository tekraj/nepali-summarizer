"""Dataset utilities for sequence training and batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class SequencePair:
    """Simple pair container for source and target sequence data."""

    source: List[int]
    target: List[int]


class TextDataset:
    """Lightweight dataset wrapper for text-to-text training examples."""

    def __init__(self, examples: Sequence[SequencePair], max_length: int = 512):
        self.examples = list(examples)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SequencePair:
        return self.examples[index]

    def collate_batch(self, batch: Iterable[SequencePair]) -> Tuple[List[List[int]], List[List[int]]]:
        """Pad sequences and return source and target arrays for training."""
        source_batch = []
        target_batch = []

        for item in batch:
            source_batch.append(item.source[: self.max_length])
            target_batch.append(item.target[: self.max_length])

        max_len = max(len(seq) for seq in source_batch) if source_batch else 0
        padded_sources = [seq + [0] * (max_len - len(seq)) for seq in source_batch]
        padded_targets = [seq + [0] * (max_len - len(seq)) for seq in target_batch]

        return padded_sources, padded_targets
