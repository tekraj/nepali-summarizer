"""Clean raw text files into the project's cleaned data directory."""

from pathlib import Path

from src.data_preprocessing.text_cleaning_pipeline import process_and_save_cleaned_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_news"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
def main() -> None:
    """Clean raw text files and save them under the project data directory."""
    process_and_save_cleaned_text(DATA_DIR, CLEANED_DIR)
    


if __name__ == "__main__":
    main()