"""Batch creation (documents/encoders/2-positional_encoding.md, sections 3-5).

    .txt files ──stream──> B raw documents ──BPE──> token IDs [B, T] + attention mask [B, T]
                                                       └─> TransformerEncoder.forward

Only integer token IDs leave this module. The embedding lookup and positional encoding
happen inside the model so that backpropagation can reach the embedding matrix.
"""

import itertools
import json
from pathlib import Path
from typing import Generator

import numpy as np
from tokenizers import ByteLevelBPETokenizer

from src.tensor_types import BoolArray, IntArray


class CreateTrainingBatch:
    """Streams documents from disk and turns them into padded, masked token-ID batches."""

    def __init__(
        self,
        batch_size: int,
        vocab_json_data: str | Path,
        merges_txt_data: str | Path,
        max_seq_length: int = 512,
    ) -> None:
        """
        Args:
            batch_size:      B, documents per batch.
            vocab_json_data: BPE ``vocab.json`` (token -> ID).
            merges_txt_data: BPE ``merges.txt``.
            max_seq_length:  T cap; longer documents are truncated.
        """
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.vocab: dict[str, int] = {}
        self.load_vocab(Path(vocab_json_data))

        self.pad_id = self.vocab["<pad>"]
        self.mask_id = self.vocab["<mask>"]

        self.tokenizer = ByteLevelBPETokenizer(str(vocab_json_data), str(merges_txt_data))
        # Section 4 — padding: truncate to T <= 512, pad to the longest document in the batch.
        self.tokenizer.enable_truncation(max_length=max_seq_length)
        self.tokenizer.enable_padding(pad_id=self.pad_id, pad_token="<pad>")

        self._sentence_stream: Generator[str, None, None] | None = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def load_vocab(self, vocab_json_data: Path) -> None:
        """Load the token -> ID mapping from ``vocab.json``."""
        with open(vocab_json_data, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def create_batch_sentences(self, training_data_path: str | Path) -> None:
        """(Re)start a lazy stream of non-empty lines from a .txt file or a directory of them.

        Call once per epoch; the stream reads one line at a time, so RAM use stays flat.
        """
        training_data_path = Path(training_data_path)
        text_files = (
            [training_data_path]
            if training_data_path.is_file()
            else sorted(training_data_path.rglob("*.txt"))
        )

        def generate_sentences() -> Generator[str, None, None]:
            for text_file in text_files:
                with text_file.open("r", encoding="utf-8") as file:
                    for line in file:
                        if sentence := line.strip():
                            yield sentence

        self._sentence_stream = generate_sentences()

    def get_next_batch(self) -> list[str] | None:
        """Up to B raw strings, or None when the stream is exhausted."""
        if self._sentence_stream is None:
            raise RuntimeError("Call create_batch_sentences() before getting batches.")
        batch = list(itertools.islice(self._sentence_stream, self.batch_size))
        return batch or None

    def create_input_tensor(self) -> tuple[IntArray, BoolArray] | None:
        """Tokenize the next batch into model inputs.

        Returns:
            token_ids:      [B, T] int64, padded with the <pad> ID.
            attention_mask: [B, T] bool, True = real token, False = [PAD] (section 5).
            Or None when the training data is exhausted.
        """
        sentences = self.get_next_batch()
        if sentences is None:
            return None

        # encode_batch runs in Rust (parallel) and already truncates and pads.
        encoded = self.tokenizer.encode_batch(sentences)
        token_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=bool)
        return token_ids, attention_mask

    def create_mlm_inputs(
        self, token_ids: IntArray, attention_mask: BoolArray, mlm_probability: float = 0.15
    ) -> tuple[IntArray, BoolArray]:
        """Masked-language-modelling corruption: hide a random ~15% of real tokens behind <mask>.

        The encoder sees the whole sentence at once, so predicting the *same* token would be
        trivial. Instead it must reconstruct the hidden tokens from their context.

        Args:
            token_ids:      [B, T] original IDs (these are also the targets).
            attention_mask: [B, T] bool, pads are never selected.

        Returns:
            masked_ids:  [B, T] copy of ``token_ids`` with selected positions set to <mask>.
            target_mask: [B, T] bool, True where the loss is computed (N positions in total).
        """
        target_mask = (np.random.rand(*token_ids.shape) < mlm_probability) & attention_mask
        masked_ids = np.where(target_mask, self.mask_id, token_ids)
        return masked_ids, target_mask
