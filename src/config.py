"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    """Simple configuration container."""

    project_name: str = "nepali-summarizer"
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    max_sequence_length: int = 512
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data: Dict[str, Any] = yaml.safe_load(handle) or {}

        project = data.get("project", {})
        training = data.get("training", {})
        data_cfg = data.get("data", {})
        model_cfg = data.get("model", {})

        return cls(
            project_name=project.get("name", cls.project_name),
            batch_size=training.get("batch_size", cls.batch_size),
            epochs=training.get("epochs", cls.epochs),
            learning_rate=training.get("learning_rate", cls.learning_rate),
            max_sequence_length=data_cfg.get("max_sequence_length", cls.max_sequence_length),
            d_model=model_cfg.get("d_model", cls.d_model),
            num_heads=model_cfg.get("num_heads", cls.num_heads),
            num_layers=model_cfg.get("num_layers", cls.num_layers),
            dropout=model_cfg.get("dropout", cls.dropout),
        )
