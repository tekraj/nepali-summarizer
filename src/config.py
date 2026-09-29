"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class ProjectConfig:
    """Flat view of config/config.yaml; defaults match the documents (D = 512, T = 512)."""

    project_name: str = "nepali-summarizer"

    # training
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    precision: str = "fp16"  # "fp16" = mixed precision (float32 master weights), or "fp32"

    # data
    max_article_length: int = 768
    max_summary_length: int = 256
    max_sequence_length: int = 1024
    cleaned_dir: str = "data/cleaned"
    summary_dir: str = "data/summary"
    vocab_dir: str = "data/nepali_vocab_output"

    # model
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    tie_weights: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ProjectConfig":
        """Read the YAML sections (training, data, model, ...) into one flat config."""
        with open(path, "r", encoding="utf-8") as handle:
            data: Dict[str, Any] = yaml.safe_load(handle) or {}

        flat = {key: value for section in data.values() if isinstance(section, dict) for key, value in section.items()}
        flat["project_name"] = data.get("project", {}).get("name", cls.project_name)
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in flat.items() if key in known})

    @property
    def compute_dtype(self):
        """``cp.float16`` or ``cp.float32``, from ``precision``."""
        import cupy as cp

        dtypes = {"fp16": cp.float16, "fp32": cp.float32}
        if self.precision not in dtypes:
            raise ValueError(f"precision must be one of {sorted(dtypes)}, got {self.precision!r}")
        return dtypes[self.precision]

    def resolve(self, relative: str) -> Path:
        """Turn a config path into an absolute path under the project root."""
        return PROJECT_ROOT / relative
