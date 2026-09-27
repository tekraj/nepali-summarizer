"""Hugging Face native Nepali text summarization logic using mBART.

Uses direct PyTorch model.generate() batching to bypass pipeline task key errors.
"""

import re
from pathlib import Path
import torch
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL = "GenzNepal/mt5-summarize-nepali"
# Colab T4-friendly default batch size
DEFAULT_BATCH_SIZE = 16
# Default maximum tokens for generated summary (will be lowered for T4)
DEFAULT_GENERATE_MAX_LENGTH = 1024


def is_strictly_devnagari(text: str) -> bool:
    """Zero-tolerance check: Rejects text containing any English ASCII letters [a-zA-Z]."""
    if not text or not text.strip():
        return False
    if re.search(r"[a-zA-Z]", text):
        return False
    if not re.search(r"[\u0900-\u097F]", text):
        return False
    return True


def normalize_text(text: str, limit: int = 2500) -> str:
    """Collapse whitespace and clip long text to model-safe length."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "…"
    return cleaned


def summarize_files_batch(
    files: list[Path | str],
    summary_dir: Path | str,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    # max input length (tokens) passed to the tokenizer
    max_length: int = 1024,
    # max tokens to generate for the summary
    generate_max_length: int | None = None,
) -> None:
    """Process a list of files using direct PyTorch batching on GPU."""
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    file_paths = [Path(f) for f in files]

    # Coerce optional parameters to defaults when called from CLI
    if model_name is None:
        model_name = DEFAULT_MODEL
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if max_length is None:
        max_length = 1024

    files_to_process = []
    for filepath in file_paths:
        summary_path = summary_dir / f"{filepath.stem}-summary.txt"
        if not summary_path.exists():
            files_to_process.append(filepath)

    if not files_to_process:
        print("All provided files already have summaries!")
        return

    print(f"Total files provided: {len(file_paths)}")
    print(f"Files remaining to process: {len(files_to_process)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect device name when possible (helps tune defaults for Colab T4)
    device_name = ""
    if device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "cuda"

    print(f"\nLoading model '{model_name}' onto device: {device} ({device_name})...")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    completed = 0
    rejected = 0

    print("\nStarting batch summarization...")

    # Tune runtime-friendly presets for T4 (Google Colab)
    t4_mode = False
    if device.type == "cuda" and device_name:
        if "T4" in device_name.upper() or "TESLA" in device_name.upper():
            t4_mode = True

    effective_batch_size = batch_size
    tokenizer_max_length = max_length
    if generate_max_length is None:
        generate_max_length = DEFAULT_GENERATE_MAX_LENGTH

    if t4_mode:
        effective_batch_size = min(batch_size, DEFAULT_BATCH_SIZE)
        tokenizer_max_length = min(max_length, 2048)
        generate_max_length = min(generate_max_length, DEFAULT_GENERATE_MAX_LENGTH)

    print(f"Using batch_size={effective_batch_size}, tokenizer_max_length={tokenizer_max_length}, generate_max_length={generate_max_length}")

    # Process in chunks of effective_batch_size
    for i in range(0, len(files_to_process), effective_batch_size):
        batch_files = files_to_process[i : i + effective_batch_size]
        
        # Read and prepare batch texts
        raw_texts = []
        for filepath in batch_files:
            text = filepath.read_text(encoding="utf-8")
            clean_text = normalize_text(text)
            if not clean_text:
                clean_text = "खाली"
            raw_texts.append(clean_text)

        # Tokenize batch
        inputs = tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="pt",
        ).to(device)

        # Inference in FP16/mixed precision when on CUDA
        autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
        with torch.no_grad():
            with autocast_ctx():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=generate_max_length,
                    num_beams=2,
                    early_stopping=True,
                )

        # Decode generated IDs to text
        summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Save outputs
        for filepath, summary_text in zip(batch_files, summaries):
            summary_text = re.sub(
                r"^.*?(सारांश|संक्षेप|Summary)\s*[:：\-]*\s*",
                "",
                summary_text,
                flags=re.IGNORECASE,
            ).strip()

            summary_path = summary_dir / f"{filepath.stem}-summary.txt"

            if is_strictly_devnagari(summary_text):
                summary_path.write_text(summary_text + "\n", encoding="utf-8")
                completed += 1
            else:
                print(f"⚠️ Rejected output for {filepath.name} (contained English or invalid text).")
                rejected += 1

        processed_so_far = min(i + batch_size, len(files_to_process))
        if processed_so_far % 100 == 0 or processed_so_far == len(files_to_process) or i == 0:
            print(f"Processed {processed_so_far}/{len(files_to_process)} files")

    print(f"\nBatch processing finished. Successfully saved: {completed}, Rejected: {rejected}")