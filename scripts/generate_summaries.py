"""Runner script for Hugging Face native Nepali summarizer."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union

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

    run_summarizer(
        model=args.model,
        batch_size=args.batch_size,
        tokenizer_max_len=args.tokenizer_max_len,
        generate_max_len=args.generate_max_len,
        cleaned_dir=cleaned_dir,
        summary_dir=summary_dir,
        force=args.force,
    )


def run_summarizer(
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    tokenizer_max_len: Optional[int] = None,
    generate_max_len: Optional[int] = None,
    cleaned_dir: Optional[Union[Path, str]] = None,
    summary_dir: Optional[Union[Path, str]] = None,
    force: bool = False,
):
    """Programmatic entrypoint suitable for notebooks.

    Parameters map directly to the CLI arguments from `main()`; pass None to use
    the summarizer defaults.
    """
    # import summarizer lazily to avoid hard dependency at import time
    try:
        from src.data_preprocessing.summarizer import summarize_files_batch
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Missing runtime dependencies: install `torch` and `transformers` in the environment.\n"
            "Colab example: `!pip install -q torch transformers sentencepiece`"
        ) from exc

    cleaned = Path(cleaned_dir) if cleaned_dir else CLEANED_DIR
    summary = Path(summary_dir) if summary_dir else SUMMARY_DIR

    files = sorted(cleaned.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {cleaned}")
        return

    if force:
        unsummarized = files
    else:
        unsummarized = [
            f for f in files if not (summary / f"{f.stem}-summary.txt").exists()
        ]

    if not unsummarized:
        print(f"All files in {cleaned} already summarized in {summary}")
        return

    summarize_files_batch(
        files=unsummarized,
        summary_dir=summary,
        model_name=model,
        batch_size=batch_size,
        max_length=tokenizer_max_len,
        generate_max_length=generate_max_len,
    )