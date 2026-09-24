"""Local Nepali lemmatization package."""

from .lemmatizer import NEPALI_STOP_WORDS, NepaliLemmatizer, load_lemma_dict

__all__ = ["NepaliLemmatizer", "load_lemma_dict", "NEPALI_STOP_WORDS"]
