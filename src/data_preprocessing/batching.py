"""Batch creation for encoder-only summarization (documents/encoders/2-positional_encoding.md, sections 3-5).

    data/cleaned/{name}.txt ──┐
    data/summary/{name}-summary.txt ──┴──stream──> B (article, summary) pairs ──BPE──>

        <s> article[:766] </s> summary[:255] </s>          (≤ 768 + 256 = 1024 tokens)
        └────── prefix (≤ 768) ──────┘└─ target (≤ 256) ─┘

    token IDs [B, T] + prefix-LM attention mask [B, T, T] + loss mask [B, T] + targets [N]
      └─> TransformerEncoder.forward

The model is an encoder, so without extra masking every summary token could simply look at
itself. The prefix-LM mask stops that: article tokens attend to the whole article, summary
tokens attend to the whole article plus only the summary tokens before them. Position p is
trained to predict token p + 1, and only positions whose next token is part of the summary are
scored, so the loss covers the summary sequence alone.

Only integer token IDs leave this module. The embedding lookup and positional encoding
happen inside the model so that backpropagation can reach the embedding matrix.
"""

import itertools
import json
from pathlib import Path
from typing import Generator

import cupy as cp
from tokenizers import ByteLevelBPETokenizer

from src.tensor_types import BoolArray, IntArray

SUMMARY_SUFFIX = "-summary.txt"


def build_attention_mask(lengths: IntArray, prefix_lengths: IntArray, seq_len: int) -> BoolArray:
    """Prefix-LM mask: which keys each query may attend to.

    Args:
        lengths:        [B] real tokens per sample (the rest is <pad>).
        prefix_lengths: [B] tokens in ``<s> article </s>``.
        seq_len:        T, padded length of the batch.

    Returns:
        [B, T, T] bool (query x key), True = may attend. A key is visible if it is a real token
        and it is either in the article prefix or not after the query (causal over the summary).
    """
    positions = cp.arange(seq_len)
    keys = positions[None, None, :]  # [1, 1, T]
    queries = positions[None, :, None]  # [1, T, 1]
    in_prefix = keys < prefix_lengths[:, None, None]  # [B, 1, T]
    is_real = keys < lengths[:, None, None]  # [B, 1, T]
    return (in_prefix | (keys <= queries)) & is_real  # [B, T, T]


class CreateTrainingBatch:
    """Streams (article, summary) pairs from disk and turns them into padded, masked token-ID batches."""

    def __init__(
        self,
        batch_size: int,
        vocab_json_data: str | Path,
        merges_txt_data: str | Path,
        max_article_length: int = 768,
        max_summary_length: int = 256,
    ) -> None:
        """
        Args:
            batch_size:         B, pairs per batch.
            vocab_json_data:    BPE ``vocab.json`` (token -> ID).
            merges_txt_data:    BPE ``merges.txt``.
            max_article_length: tokens for ``<s> article </s>``; the article's right side is cut.
            max_summary_length: tokens for ``summary </s>``; the summary's right side is cut.
        """
        self.batch_size = batch_size
        self.max_article_length = max_article_length
        self.max_summary_length = max_summary_length
        self.vocab: dict[str, int] = {}
        self.load_vocab(Path(vocab_json_data))

        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<s>"]
        self.eos_id = self.vocab["</s>"]  # ends the article prefix and the summary

        # Truncation and padding are done by hand, because article and summary have separate budgets.
        self.tokenizer = ByteLevelBPETokenizer(str(vocab_json_data), str(merges_txt_data))

        self._pair_stream: Generator[tuple[str, str], None, None] | None = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def load_vocab(self, vocab_json_data: Path) -> None:
        """Load the token -> ID mapping from ``vocab.json``."""
        with open(vocab_json_data, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def create_batch_pairs(self, article_dir: str | Path, summary_dir: str | Path) -> None:
        """(Re)start a lazy stream of (article, summary) texts.

        Every ``{name}-summary.txt`` in ``summary_dir`` is paired with ``{name}.txt`` in
        ``article_dir``; summaries without an article are skipped. Call once per epoch; the
        stream reads one pair at a time, so RAM use stays flat.
        """
        article_dir, summary_dir = Path(article_dir), Path(summary_dir)
        summary_files = sorted(summary_dir.glob(f"*{SUMMARY_SUFFIX}"))

        def generate_pairs() -> Generator[tuple[str, str], None, None]:
            for summary_file in summary_files:
                article_file = article_dir / f"{summary_file.name[: -len(SUMMARY_SUFFIX)]}.txt"
                if not article_file.is_file():
                    continue
                article = " ".join(article_file.read_text(encoding="utf-8").split())
                summary = " ".join(summary_file.read_text(encoding="utf-8").split())
                if article and summary:
                    yield article, summary

        self._pair_stream = generate_pairs()

    def get_next_batch(self) -> list[tuple[str, str]] | None:
        """Up to B raw (article, summary) pairs, or None when the stream is exhausted."""
        if self._pair_stream is None:
            raise RuntimeError("Call create_batch_pairs() before getting batches.")
        batch = list(itertools.islice(self._pair_stream, self.batch_size))
        return batch or None

    def skip_batches(self, num_batches: int) -> None:
        """Drop the next ``num_batches`` batches without tokenizing them (to resume mid-epoch)."""
        for _ in range(num_batches):
            if self.get_next_batch() is None:
                break

    def encode_articles(self, articles: list[str]) -> list[list[int]]:
        """``<s> article </s>`` IDs, the article's right side truncated to fit ``max_article_length``."""
        budget = self.max_article_length - 2  # room for <s> and </s>
        return [
            [self.bos_id, *e.ids[:budget], self.eos_id] for e in self.tokenizer.encode_batch(articles)
        ]

    def encode_summaries(self, summaries: list[str]) -> list[list[int]]:
        """``summary </s>`` IDs, the summary's right side truncated to fit ``max_summary_length``."""
        budget = self.max_summary_length - 1  # room for </s>
        return [[*e.ids[:budget], self.eos_id] for e in self.tokenizer.encode_batch(summaries)]

    def create_input_tensor(self) -> tuple[IntArray, BoolArray, BoolArray, IntArray] | None:
        """Tokenize the next batch of pairs into model inputs and targets.

        Returns:
            token_ids:      [B, T] int64 ``<s> article </s> summary </s>``, padded with <pad>.
            attention_mask: [B, T, T] bool prefix-LM mask (see ``build_attention_mask``).
            loss_mask:      [B, T] bool, True at the N positions whose next token is a summary token.
            targets:        [N] int64, the next token at each of those positions.
            Or None when the training data is exhausted.
        """
        pairs = self.get_next_batch()
        if pairs is None:
            return None

        articles, summaries = zip(*pairs)
        prefixes = self.encode_articles(list(articles))
        targets = self.encode_summaries(list(summaries))

        prefix_lengths = cp.array([len(p) for p in prefixes], dtype=cp.int64)  # [B]
        lengths = prefix_lengths + cp.array([len(t) for t in targets], dtype=cp.int64)  # [B]
        seq_len = int(lengths.max())

        # Pad on the host as Python lists, then copy the whole batch to the GPU in one transfer.
        rows = [prefix + target for prefix, target in zip(prefixes, targets)]
        token_ids = cp.array([row + [self.pad_id] * (seq_len - len(row)) for row in rows], dtype=cp.int64)  # [B, T]

        attention_mask = build_attention_mask(lengths, prefix_lengths, seq_len)  # [B, T, T]

        # Position p predicts token p + 1. Score from the prefix's closing </s> (predicts the
        # first summary token) up to the last summary token (predicts the final </s>).
        positions = cp.arange(seq_len)[None, :]  # [1, T]
        loss_mask = (positions >= prefix_lengths[:, None] - 1) & (positions < lengths[:, None] - 1)  # [B, T]
        next_ids = cp.full_like(token_ids, self.pad_id)
        next_ids[:, :-1] = token_ids[:, 1:]  # [B, T] token IDs shifted left by one
        return token_ids, attention_mask, loss_mask, next_ids[loss_mask]
