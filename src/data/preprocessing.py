"""Utilities for cleaning, tokenizing, and saving Nepali text for training."""

from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Dict, Tuple


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to a stable canonical form."""
    return unicodedata.normalize("NFC", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML-like tags from raw text."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Apply the basic Nepali text cleaning pipeline."""
    cleaned = normalize_unicode(text)
    cleaned = remove_html_tags(cleaned)
    # Remove Zero-Width Non-Joiner and Zero-Width Joiner
    cleaned = cleaned.replace("\u200c", "").replace("\u200d", "")
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def collect_source_documents(source_dir: str | Path) -> List[Path]:
    """Collect all source documents in a directory."""
    directory = Path(source_dir)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".json", ".md"}
    )


class BPETokenizer:
    """A pure Python implementation of Byte-Pair Encoding for Nepali text."""
    
    def __init__(self):
        self.bpe_codes: Dict[Tuple[str, str], int] = {}  # Maps a pair of tokens to their merge order
        self.vocab: Dict[str, int] = {}                  # Maps subword string to integer ID
        self.inverse_vocab: Dict[int, str] = {}          # Maps integer ID to subword string

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count the frequency of adjacent symbol pairs in the vocabulary."""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in all words in the vocabulary."""
        v_out = {}
        bigram = re.escape(' '.join(pair))
        # Regex matches the exact pair surrounded by whitespace/boundaries
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, corpus_texts: Iterable[str], num_merges: int = 10000):
        """Train the BPE model on the entire corpus to build the vocabulary."""
        word_freqs = collections.Counter()
        
        # 1. Pre-tokenize into whole words and count frequencies
        print("Pre-tokenizing corpus and counting word frequencies...")
        for text in corpus_texts:
            cleaned = clean_text(text)
            words = re.findall(r"[^\s।,\.?;:()\'\"“”]+", cleaned)
            word_freqs.update(words)

        # 2. Format base vocabulary with spaces between chars and </w> at the end
        # Example: 'नेपाल' -> 'न े प ा ल </w>'
        bpe_vocab = {" ".join(list(word)) + " </w>": freq for word, freq in word_freqs.items()}

        # 3. Iteratively find and merge the most frequent pairs
        print(f"Executing {num_merges} BPE merges...")
        for i in range(num_merges):
            pairs = self._get_stats(bpe_vocab)
            if not pairs:
                break
            
            best = max(pairs, key=pairs.get)
            self.bpe_codes[best] = i
            bpe_vocab = self._merge_vocab(best, bpe_vocab)
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} merges...")

        # 4. Extract final subword tokens and assign Integer IDs
        unique_tokens = set()
        for word in bpe_vocab:
            unique_tokens.update(word.split())
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encode a new text string into a list of integer token IDs."""
        if not self.bpe_codes:
            raise ValueError("Tokenizer has not been trained yet.")

        cleaned = clean_text(text)
        words = re.findall(r"[^\s।,\.?;:()\'\"“”]+", cleaned)
        encoded_tokens = []

        for word in words:
            word_chars = " ".join(list(word)) + " </w>"
            
            # Apply learned merges in order of their rank
            while True:
                symbols = word_chars.split()
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                if not pairs:
                    break
                
                # Find the pair that was learned earliest during training
                pair_to_merge = min(pairs, key=lambda p: self.bpe_codes.get(p, float('inf')))
                
                # If the pair wasn't in our training data, stop merging this word
                if pair_to_merge not in self.bpe_codes:
                    break
                    
                # Merge the pair
                bigram = re.escape(' '.join(pair_to_merge))
                p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
                word_chars = p.sub(''.join(pair_to_merge), word_chars)
            
            # Convert final subword strings to integer IDs
            for token in word_chars.split():
                if token in self.vocab:
                    encoded_tokens.append(self.vocab[token])
                else:
                    # Optional: Handle UNK (Unknown) tokens here if necessary
                    pass
                    
        return encoded_tokens

    def save_model(self, filepath: str | Path):
        """Save the trained BPE model to a JSON file."""
        model_data = {
            "bpe_codes": {f"{p[0]}|||{p[1]}": rank for p, rank in self.bpe_codes.items()},
            "vocab": self.vocab
        }
        Path(filepath).write_text(json.dumps(model_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_model(self, filepath: str | Path):
        """Load a trained BPE model from a JSON file."""
        model_data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        self.bpe_codes = {tuple(k.split("|||")): v for k, v in model_data["bpe_codes"].items()}
        self.vocab = model_data["vocab"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


def save_cleaned_documents(source_files: Iterable[str | Path], output_dir: str | Path) -> List[Path]:
    """Clean raw documents and save the cleaned text to disk for training."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for source_file in source_files:
        file_path = Path(source_file)
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        cleaned = clean_text(content)
        target_file = output_path / f"{file_path.stem}.cleaned{file_path.suffix}"
        target_file.write_text(cleaned, encoding="utf-8")
        saved_files.append(target_file)

    return saved_files


def save_tokenized_documents(source_files: Iterable[str | Path], output_dir: str | Path, tokenizer: BPETokenizer) -> List[Path]:
    """Tokenize raw documents using the trained BPE tokenizer and save as JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for source_file in source_files:
        file_path = Path(source_file)
        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(content)
        
        # Now returns a list of Integer IDs ready for the Transformer Embedding layer
        token_ids = tokenizer.encode(cleaned_text)
        
        payload = {
            "filename": file_path.name,
            "source_path": str(file_path),
            "tokens": token_ids,
            "text": cleaned_text,
        }

        target_file = output_path / f"{file_path.stem}.tokenized.json"
        target_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_files.append(target_file)

    return saved_files


# --- Workflow Example ---
# 1. Collect files: 
#    files = collect_source_documents("./raw_data")
# 2. Extract text for training the tokenizer:
#    corpus = [Path(f).read_text(encoding="utf-8") for f in files]
# 3. Initialize and train:
#    tokenizer = BPETokenizer()
#    tokenizer.train(corpus, num_merges=15000)
#    tokenizer.save_model("./nepali_bpe_model.json")
# 4. Tokenize and save documents:
#    save_tokenized_documents(files, "./tokenized_data", tokenizer)