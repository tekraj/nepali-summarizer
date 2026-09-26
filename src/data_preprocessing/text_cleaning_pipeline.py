"""
- Fix the Invalid UTF-8 error in the text cleaning pipeline by replacing invalid bytes with the Unicode replacement character (U+FFFD).
- Ensure that the text is in a consistent encoding format for further processing.
- Normalize Unicode text to a stable canonical form (NFC) to make equivalent Unicode encodings consistent, including in Nepali text.
- Remove zero-width non-joiner and zero-width joiner characters from the cleaned text.
- Collapse repeated whitespace to a single space and strip leading/trailing whitespace.
- Remove HTML-like tags from raw text.
- Get a list of all raw text files in the specified directory.
- Save the cleaned text to a specified file path.
- Clean every text file in an input directory into an output directory.
"""
import re
import unicodedata
import warnings
from pathlib import Path


def remove_html_tags(text: str) -> str:
    """Remove HTML-like tags from raw text."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()

"""
Converts the raw text into Unicode, replacing invalid bytes with the Unicode replacement character (U+FFFD).
This ensures that the text is in a consistent encoding format, which is essential for further processing and
"""
def text_to_unicode(file: Path) -> str:
    raw_bytes = file.read_bytes()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        warnings.warn(
            f"{file} contains invalid UTF-8 starting at byte "
            f"{error.start}; replacing invalid bytes.",
            UnicodeWarning,
            stacklevel=2,
        )
        return raw_bytes.decode("utf-8", errors="replace")
    
"""
Normalize text to Unicode NFC (Canonical Composition).
NFC composes canonically equivalent character sequences into their standard
composed representation where one exists. It does not remove diacritics or
change a character's meaning; it makes equivalent Unicode encodings
consistent, including in Nepali text.
"""
def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to a stable canonical form."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """Apply the basic Nepali text cleaning pipeline."""
    cleaned = remove_html_tags(normalize_unicode(text))
    # Remove zero-width non-joiner and zero-width joiner characters.
    cleaned = cleaned.replace("\u200c", "").replace("\u200d", "")
    return normalize_whitespace(cleaned)


def get_all_raw_files(directory: str | Path) -> list[Path]:
    """Get a list of all raw text files in the specified directory."""
    input_dir = Path(directory)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(input_dir.glob("*.txt"))


def save_cleaned_text(cleaned_text: str, output_path: str | Path) -> None:
    """Save the cleaned text to a specified file path."""
    Path(output_path).write_text(cleaned_text, encoding="utf-8")

def process_and_save_cleaned_text(input_path: str | Path, output_path: str | Path) -> None:
    """Clean every text file in an input directory into an output directory."""
    input_files = get_all_raw_files(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        raw_text = text_to_unicode(input_file)
        cleaned_text = clean_text(raw_text)
        save_cleaned_text(cleaned_text, output_dir / input_file.name)