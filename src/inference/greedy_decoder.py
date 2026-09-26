"""Greedy decoding of LM-head probabilities (documents/encoders/7-linearization-and-softmax.md, Step 2).

    probabilities [B, T, V] ──argmax──> token IDs [B, T] ──vocab lookup──> tokens
"""

import numpy as np

from src.tensor_types import FloatArray, IntArray


class GreedyDecoder:
    """Turns the probability distribution from LanguageModelingHead into token IDs and strings."""

    def __init__(self, vocab_dict: dict[str, int] | None = None) -> None:
        """
        Args:
            vocab_dict: token -> ID mapping (vocab.json), e.g. {"<s>": 0, "<pad>": 1, ...}.
        """
        self.vocab_dict = vocab_dict
        # Reverse lookup ID -> token.
        self.id_to_token = {v: k for k, v in vocab_dict.items()} if vocab_dict else {}

    def decode_token_ids(self, probabilities: FloatArray) -> IntArray:
        """Pick the most probable token at every position.

        Args:
            probabilities: [B, T, V] softmax output.

        Returns:
            [B, T] predicted token IDs.
        """
        return np.argmax(probabilities, axis=-1)

    def decode_to_text(self, predicted_ids: IntArray) -> list[list[str]]:
        """Map a [B, T] array of IDs to B lists of T token strings (unknown IDs -> "<unk>")."""
        if not self.id_to_token:
            raise ValueError("Vocabulary dictionary is required to decode text.")
        return [[self.id_to_token.get(int(i), "<unk>") for i in row] for row in predicted_ids]
