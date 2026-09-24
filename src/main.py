"""Main entry point for the Nepali summarization project skeleton."""

from __future__ import annotations

from pathlib import Path

from src.config import ProjectConfig


def main() -> None:
    """Load config and report the project startup state."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config = ProjectConfig.from_yaml(config_path)

    print(f"Project: {config.project_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Model dimension: {config.d_model}")


if __name__ == "__main__":
    main()
