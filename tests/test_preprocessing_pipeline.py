from pathlib import Path

import json

from src.data.preprocessing import (
    collect_source_documents,
    save_cleaned_documents,
    save_tokenized_documents,
)


def test_collect_and_save_cleaned_and_tokenized_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text("गयो मैले स्कूल", encoding="utf-8")
    (source_dir / "doc2.txt").write_text("तिम्रो घर ठूलो छ", encoding="utf-8")

    cleaned_dir = tmp_path / "cleaned"
    tokenized_dir = tmp_path / "tokenized"

    files = collect_source_documents(source_dir)
    assert len(files) == 2

    cleaned_files = save_cleaned_documents(files, cleaned_dir)
    assert len(cleaned_files) == 2
    assert (cleaned_dir / "doc1.cleaned.txt").exists()
    assert (cleaned_dir / "doc2.cleaned.txt").exists()
    assert "गयो" in (cleaned_dir / "doc1.cleaned.txt").read_text(encoding="utf-8")

    tokenized_files = save_tokenized_documents(files, tokenized_dir)
    assert len(tokenized_files) == 2
    json_path = tokenized_dir / "doc1.tokenized.json"
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["filename"] == "doc1.txt"
    assert payload["tokens"] == ["गयो", "मैले", "स्कूल"]
