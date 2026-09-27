"""Greedy decoding of LM-head probabilities (documents/encoders/7-linearization-and-softmax.md, Step 2).

    probabilities [B, T, V] ──argmax──> token IDs [B, T] ──vocab lookup──> tokens

``greedy_summarize`` applies the same argmax one token at a time to generate a summary.
"""

import cupy as cp

from src.data_preprocessing.batching import CreateTrainingBatch, build_attention_mask
from src.models.encoder import TransformerEncoder
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
        return cp.argmax(probabilities, axis=-1)

    def decode_to_text(self, predicted_ids: IntArray) -> list[list[str]]:
        """Map a [B, T] array of IDs to B lists of T token strings (unknown IDs -> "<unk>")."""
        if not self.id_to_token:
            raise ValueError("Vocabulary dictionary is required to decode text.")
        return [[self.id_to_token.get(int(i), "<unk>") for i in row] for row in predicted_ids]


def greedy_summarize(model: TransformerEncoder, batcher: CreateTrainingBatch, article: str) -> str:
    """Autoregressive greedy summary with the prefix-LM encoder, matching the training layout.

        <s> article </s>  ──forward──> P(next) at the last position ──argmax──> append ──> repeat

    The loop stops at </s> or after ``batcher.max_summary_length`` tokens. Each step reruns the
    full forward pass over prefix + summary so far (no KV cache).

    Args:
        model:   a trained TransformerEncoder.
        batcher: supplies the tokenizer, special-token IDs and the article/summary budgets.
        article: raw Nepali article text.

    Returns:
        The decoded summary string.
    """
    prefix = batcher.encode_articles([" ".join(article.split())])[0]  # <s> article[:766] </s>
    token_ids = list(prefix)
    summary_ids: list[int] = []

    for _ in range(batcher.max_summary_length):
        seq_len = len(token_ids)
        attention_mask = build_attention_mask(cp.array([seq_len]), cp.array([len(prefix)]), seq_len)  # [1, T, T]
        last_position = cp.zeros((1, seq_len), dtype=bool)
        last_position[0, -1] = True  # only the newest position predicts the next token
        _, probabilities = model.forward(cp.array([token_ids], dtype=cp.int64), attention_mask, last_position)  # [1, V]
        next_id = int(cp.argmax(probabilities[0]))
        if next_id == batcher.eos_id:
            break
        token_ids.append(next_id)
        summary_ids.append(next_id)

    return batcher.tokenizer.decode(summary_ids)
