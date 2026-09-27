"""Summarize a Nepali article with a trained encoder checkpoint (greedy, token by token).

    python -m scripts.infer --checkpoint checkpoints/summarizer_epoch_3.npz --article data/cleaned/100001.txt
    python -m scripts.infer --checkpoint checkpoints/summarizer_epoch_3.npz --text "..."
"""

import argparse
from pathlib import Path

from src.config import DEFAULT_CONFIG_PATH, ProjectConfig
from src.data_preprocessing.batching import CreateTrainingBatch
from src.inference.greedy_decoder import greedy_summarize
from src.models.encoder import TransformerEncoder
from src.training.train import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config file")
    parser.add_argument("--checkpoint", required=True, help=".npz weights saved by scripts.train_model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--article", help="path to an article .txt file")
    source.add_argument("--text", help="article text given directly")
    return parser.parse_args()


def main() -> None:
    """Entry point for inference."""
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)

    vocab_dir = config.resolve(config.vocab_dir)
    batcher = CreateTrainingBatch(
        batch_size=1,
        vocab_json_data=vocab_dir / "vocab.json",
        merges_txt_data=vocab_dir / "merges.txt",
        max_article_length=config.max_article_length,
        max_summary_length=config.max_summary_length,
    )
    model = TransformerEncoder(
        vocab_size=batcher.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_length=config.max_sequence_length,
        tie_weights=config.tie_weights,
    )
    load_checkpoint(model, args.checkpoint)

    article = Path(args.article).read_text(encoding="utf-8") if args.article else args.text
    print(greedy_summarize(model, batcher, article))


if __name__ == "__main__":
    main()
