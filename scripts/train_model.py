"""Train the Transformer encoder to summarize: <s> article </s> summary </s>, loss on the summary only.

    python -m scripts.train_model                  # full run, settings from config/config.yaml
    python -m scripts.train_model --max-steps 5    # quick smoke test
"""

import argparse

import cupy as cp

from src.config import DEFAULT_CONFIG_PATH, ProjectConfig
from src.data_preprocessing.batching import CreateTrainingBatch
from src.models.encoder import TransformerEncoder
from src.training.optimizer import Adam
from src.training.train import load_checkpoint, save_checkpoint, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config file")
    parser.add_argument("--data-dir", help="directory of article .txt files (default: config cleaned_dir)")
    parser.add_argument("--summary-dir", help="directory of {name}-summary.txt files (default: config summary_dir)")
    parser.add_argument("--epochs", type=int, help="override config epochs")
    parser.add_argument("--batch-size", type=int, help="override config batch_size")
    parser.add_argument("--max-steps", type=int, help="stop each epoch after this many steps")
    parser.add_argument("--resume", help="load weights from this .npz checkpoint first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    cp.random.seed(config.seed)

    # 1. Data: stream (article, summary) pairs -> token IDs [B, T] + prefix-LM mask [B, T, T] + loss mask [B, T]
    vocab_dir = config.resolve(config.vocab_dir)
    batcher = CreateTrainingBatch(
        batch_size=config.batch_size,
        vocab_json_data=vocab_dir / "vocab.json",
        merges_txt_data=vocab_dir / "merges.txt",
        max_article_length=config.max_article_length,
        max_summary_length=config.max_summary_length,
    )
    data_dir = args.data_dir or config.resolve(config.cleaned_dir)
    summary_dir = args.summary_dir or config.resolve(config.summary_dir)

    # 2. Model: embedding -> PE -> N x EncoderBlock -> LM head
    model = TransformerEncoder(
        vocab_size=batcher.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_length=config.max_sequence_length,
        tie_weights=config.tie_weights,
    )
    if args.resume:
        load_checkpoint(model, args.resume)
    num_params = sum(p.size for p in model.parameters().values())
    print(
        f"Encoder: V={batcher.vocab_size} D={config.d_model} H={config.num_heads} "
        f"N={config.num_layers} D_ff={config.d_ff} T<={config.max_article_length}+{config.max_summary_length} "
        f"B={config.batch_size} | {num_params / 1e6:.1f}M parameters"
    )

    # 3. Optimizer
    optimizer = Adam(learning_rate=config.learning_rate, max_grad_norm=config.max_grad_norm)

    # 4. Train, saving a checkpoint after every epoch
    checkpoint_dir = config.resolve(config.checkpoint_dir)
    for epoch in range(1, config.epochs + 1):
        stats = train_epoch(model, batcher, optimizer, data_dir, summary_dir, args.max_steps)
        path = checkpoint_dir / f"summarizer_epoch_{epoch}.npz"
        save_checkpoint(model, path)
        print(f"epoch {epoch}/{config.epochs}: loss={stats['loss']:.4f} steps={stats['steps']} -> {path}")


if __name__ == "__main__":
    main()
