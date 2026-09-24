"""Local lemmatization utilities for Nepali text."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEMMA_DICT_PATH = PROJECT_ROOT / "data" / "raw" / "nepali_lemma_dict.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "raw" / "nepali_hmm_pipeline.pkl"


def load_lemma_dict(path: str | Path = DEFAULT_LEMMA_DICT_PATH) -> dict:
    """Load the local Nepali lemma dictionary and flatten it into a single map."""
    with open(path, "r", encoding="utf-8") as handle:
        categories = json.load(handle)

    flat: dict[str, str] = {}
    for mapping in categories.values():
        if isinstance(mapping, dict):
            flat.update(mapping)
    return flat


def tokenize_nepali_text(text: str) -> List[str]:
    """Split a Nepali sentence into tokens while preserving script text."""
    return re.findall(r"[^\s।,\.?;:()\'\"“”]+", text)


NEPALI_STOP_WORDS = {
    "म", "मेरो", "मलाई", "हामी", "हाम्रो", "हामीलाई", "तँ", "तँलाई", "तेरो", "तिमी", "तिम्रो", "तिमीलाई",
    "तपाईं", "तपाईंको", "तपाईंलाई", "उ", "उसको", "उसलाई", "उसले", "उनी", "उनको", "उनलाई", "उनले", "उनीहरु",
    "उनीहरुको", "उनीहरुलाई", "यिनीहरु", "तिनीहरु", "आफू", "आफ्नो", "र", "पनि", "तर", "कि", "वा", "अथवा",
    "तथा", "भने", "यदि", "यद्यपि", "तथापि", "किनभने", "किनकि", "बरु", "त्यसैले", "तसर्थ", "अतः", "को",
    "का", "की", "लाई", "ले", "बाट", "द्वारा", "मा", "माथि", "तल", "भित्र", "बाहिर", "सँग", "सित", "बिना",
    "बाहेक", "लागि", "निम्ति", "तर्फ", "तिर", "भन्दा", "छ", "छन्", "छु", "छौं", "छस्", "छौ", "हो", "हुन्",
    "हुँ", "हौं", "होस्", "हौ", "थियो", "थिए", "थिइन्", "थिएँ", "थियौं", "हुनेछ", "भयो", "भए", "भएन",
    "गर्छ", "गर्छन्", "गर्छु", "गर्छौं", "के", "कुन", "किन", "कसरी", "कस्तो", "कहाँ", "कहिले", "कति",
    "कसको", "कसलाई", "कसले", "अझै", "अधिक", "अन्य", "अन्यत्र", "अन्यथा", "अब", "अरु", "अर्को", "अर्थात्",
    "अलग", "आज", "हिजो", "भोलि", "अघि", "पछि", "सधैं", "कहिल्यै", "अक्सर", "धेरै", "थोरै", "कम", "सबै",
    "केही", "कोही", "जहाँ", "त्यहाँ", "यहाँ", "यसो", "त्यसो", "यस्तो", "त्यस्तो", "जस्तो", "उस्तो", "यस",
    "त्यस", "यी", "ती", "जुन", "कुरा", "एकदम", "ज्यादै", "अति", "अनि", "लौ", "पो", "नि", "त", "नै",
    "मात्र", "भरि", "सम्म", "एउटा", "दुइटा", "आदि", "इत्यादि",
}


class NepaliLemmatizer:
    """Local lemmatizer using the bundled dictionary and a heuristic fallback."""

    def __init__(
        self,
        lemma_dict_path: str | Path = DEFAULT_LEMMA_DICT_PATH,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        use_pickle: bool = True,
    ) -> None:
        self.lemma_dict = load_lemma_dict(lemma_dict_path)
        self.model_path = Path(model_path)
        self.model = None
        if use_pickle and self.model_path.exists():
            try:
                with open(self.model_path, "rb") as handle:
                    self.model = pickle.load(handle)
            except Exception:
                self.model = None

        self.irregulars = {
            "छ": "हुनु",
            "छन्": "हुनु",
            "हो": "हुनु",
            "हुन्": "हुनु",
            "भयो": "हुनु",
            "भए": "हुनु",
            "हुन्छ": "हुनु",
            "थियो": "हुनु",
            "गयो": "जानु",
            "गए": "जानु",
            "जान्छ": "जानु",
            "गएको": "जानु",
            "आयो": "आउनु",
            "आए": "आउनु",
            "आउँछ": "आउनु",
        }
        self.noun_suffixes = [
            "हरुलाई", "हरुले", "हरुको", "हरुबाट", "हरुमा", "हरुका", "हरुकी", "हरु", "लाई", "बाट",
            "देखि", "द्वारा", "सम्म", "ले", "को", "का", "की", "मा",
        ]
        self.verb_endings = ["ँदैछ", "ँदैछन्", "न्छन्", "ेका", "ेको", "ेकी", "छन्", "न्छ", "यौं", "यो", "ौं", "नेछ"]
        self.stop_words = set(NEPALI_STOP_WORDS)

    def _apply_pickle_model(self, tokens: Sequence[str]) -> List[str] | None:
        """Try a loaded pickled model using the common sklearn transform contract."""
        if self.model is None:
            return None

        for method_name in ("transform", "predict"):
            method = getattr(self.model, method_name, None)
            if method is None:
                continue
            try:
                result = method(tokens)
            except TypeError:
                try:
                    result = method([tokens])
                except TypeError:
                    continue
            if isinstance(result, list):
                if result and isinstance(result[0], (list, tuple)):
                    return [token for item in result for token in item]
                return [str(item) for item in result]
            if isinstance(result, tuple) and result and isinstance(result[0], list):
                return [str(token) for item in result for token in item]
            return None
        return None

    def lemmatize_token(self, token: str) -> str:
        """Convert a single Nepali token to its lemma using dictionary, irregulars, and suffix heuristics."""
        word = token.strip()
        if not word:
            return ""

        word = word.replace("\u200c", "").replace("\u200d", "")
        if word in self.lemma_dict:
            return self.lemma_dict[word]
        if word in self.irregulars:
            return self.irregulars[word]

        original_length = len(word)
        for suffix in self.noun_suffixes:
            if word.endswith(suffix) and original_length > len(suffix) + 2:
                return word[: -len(suffix)]

        for ending in self.verb_endings:
            if word.endswith(ending) and original_length > len(ending) + 1:
                base = word[: -len(ending)]
                if base.endswith("्"):
                    base = base[:-1]
                return base + "नु"

        return word

    def lemmatize_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Lemmatize a sequence of tokens, with a safe fallback to the pickled model if available."""
        token_list = [str(token) for token in tokens if str(token).strip()]

        if not token_list:
            return []

        model_result = self._apply_pickle_model(token_list)
        if model_result is not None:
            return [str(item) for item in model_result]

        return [self.lemmatize_token(token) for token in token_list]

    def lemmatize_text(self, text: str) -> str:
        """Lemmatize a sentence or paragraph and return a whitespace-joined string."""
        tokens = tokenize_nepali_text(text)
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        return " ".join(lemmatized_tokens)

    def remove_stopwords(self, tokens: Iterable[str]) -> List[str]:
        """Filter out common stop-words after lemmatization."""
        return [token for token in self.lemmatize_tokens(tokens) if token not in self.stop_words]
