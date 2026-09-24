"""Smoke tests for the local Nepali lemmatizer package."""

from src.nepali_lemmatizer import NepaliLemmatizer


def test_dictionary_lemmas_are_resolved() -> None:
    lemmatizer = NepaliLemmatizer()

    assert lemmatizer.lemmatize_tokens(["गयो", "मैले", "स्कूल"]) == ["जानु", "म", "स्कूल"]
    assert lemmatizer.lemmatize_text("गयो मैले स्कूल") == "जानु म स्कूल"
