"""Runner script for Hugging Face native Nepali summarizer."""

from argparse import ArgumentParser
from pathlib import Path
from src.data_preprocessing.summarizer import summarize_files_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
SUMMARY_DIR = PROJECT_ROOT / "data" / "summary"
DEFAULT_MODEL = "GenzNepal/mt5-summarize-nepali"


def main() -> None:
    parser = ArgumentParser(description="Batch generate summaries for cleaned Nepali files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name to use")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--tokenizer-max-len", type=int, default=4096, help="Max tokens for tokenizer/truncation")
    parser.add_argument("--generate-max-len", type=int, default=1024, help="Max tokens to generate for summary")
    parser.add_argument("--summary-dir", default=str(SUMMARY_DIR), help="Directory to write summaries")
    parser.add_argument("--cleaned-dir", default=str(CLEANED_DIR), help="Directory containing cleaned .txt files")
    parser.add_argument("--force", action="store_true", help="Re-generate and overwrite existing summaries")

    args = parser.parse_args()

    cleaned_dir = Path(args.cleaned_dir)
    summary_dir = Path(args.summary_dir)

    files = sorted(cleaned_dir.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {cleaned_dir}")
        return

    # filter out files that already have a summary in SUMMARY_DIR
    # summary files are named as '<original_stem>-summary.txt'
    if args.force:
        unsummarized = files
    else:
        unsummarized = [
            f for f in files if not (summary_dir / f"{f.stem}-summary.txt").exists()
        ]

    if not unsummarized:
        print(f"All files in {cleaned_dir} already summarized in {summary_dir}")
        return

    summarize_files_batch(
        files=unsummarized,
        summary_dir=summary_dir,
        model_name=args.model if args.model else None,
        batch_size=args.batch_size if args.batch_size else None,
        max_length=args.tokenizer_max_len if args.tokenizer_max_len else None,
        generate_max_length=args.generate_max_len if args.generate_max_len else None,
    )


if __name__ == "__main__":
    main()