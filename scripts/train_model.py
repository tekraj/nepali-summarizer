"""Train the Transformer encoder to summarize: <s> article </s> summary </s>, loss on the summary only.

    python -m scripts.train_model                  # full run, settings from config/config.yaml
    python -m scripts.train_model --max-steps 5    # quick smoke test

    # continue after epoch 9 (runs epochs 10..config epochs)
    python -m scripts.train_model --resume checkpoints/summarizer_epoch_9.npz --start-epoch 10

    # continue epoch 10 from a mid-epoch checkpoint saved at batch 25000
    python -m scripts.train_model --resume checkpoints/summarizer_epoch_10_step_25000.npz \\
        --start-epoch 10 --skip-steps 25000

    # checkpoints saved by this version also hold the Adam state, epoch, batch and running loss,
    # so --resume alone continues exactly where training stopped
    python -m scripts.train_model --resume checkpoints/summarizer_epoch_10_step_24000.npz
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
    parser.add_argument(
        "--start-epoch", type=int,
        help="number of the first epoch to run; runs start-epoch..epochs (use N+1 when resuming epoch N). "
        "Default: 1, or read from a --resume checkpoint that stores its training state",
    )
    parser.add_argument(
        "--skip-steps", type=int,
        help="skip this many batches of the first epoch run (to resume from a _step_ checkpoint). "
        "Default: 0, or read from a --resume checkpoint that stores its training state",
    )
    parser.add_argument(
        "--save-every", type=int, default=2000,
        help="also save a mid-epoch checkpoint every N batches, keeping only the newest (0 = off)",
    )
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
    # 3. Optimizer, created before --resume so its Adam state can be restored too
    optimizer = Adam(learning_rate=config.learning_rate, max_grad_norm=config.max_grad_norm)

    training_state = {}
    if args.resume:
        training_state = load_checkpoint(model, args.resume, optimizer)
    num_params = sum(p.size for p in model.parameters().values())
    print(
        f"Encoder: V={batcher.vocab_size} D={config.d_model} H={config.num_heads} "
        f"N={config.num_layers} D_ff={config.d_ff} T<={config.max_article_length}+{config.max_summary_length} "
        f"B={config.batch_size} | {num_params / 1e6:.1f}M parameters"
    )

    # Where to start: explicit flags win; otherwise continue from the checkpoint's training state.
    start_epoch, skip_steps, loss_sum, loss_steps = 1, 0, 0.0, 0
    if "epoch" in training_state:
        if training_state["epoch_complete"]:
            start_epoch = training_state["epoch"] + 1
        else:
            start_epoch, skip_steps = training_state["epoch"], training_state["step"]
            loss_sum, loss_steps = training_state["loss_sum"], training_state["loss_steps"]
            saved_batch_size = training_state.get("batch_size", config.batch_size)
            if args.skip_steps is None and saved_batch_size != config.batch_size:
                raise SystemExit(
                    f"checkpoint was saved with batch_size={saved_batch_size}, now {config.batch_size}; "
                    "skipping by batch count would land on different data. Use the same --batch-size."
                )
        print(
            f"Resuming: epoch {start_epoch}, after batch {skip_steps}, Adam step {optimizer.step_count}"
            + (f", running loss {loss_sum / loss_steps:.4f}" if loss_steps else "")
        )
    elif args.resume:
        print("Resuming weights only: this checkpoint has no Adam or training state")
    if args.start_epoch is not None or args.skip_steps is not None:
        start_epoch = args.start_epoch if args.start_epoch is not None else start_epoch
        skip_steps = args.skip_steps if args.skip_steps is not None else 0
        loss_sum, loss_steps = 0.0, 0  # position chosen by hand, so the saved running loss may not match

    # 4. Train, saving a checkpoint after every epoch
    checkpoint_dir = config.resolve(config.checkpoint_dir)
    if start_epoch > config.epochs:
        raise SystemExit(f"--start-epoch {start_epoch} is past the last epoch ({config.epochs}); raise --epochs")
    for epoch in range(start_epoch, config.epochs + 1):
        first = epoch == start_epoch
        stats = train_epoch(
            model, batcher, optimizer, data_dir, summary_dir, args.max_steps,
            skip_steps=skip_steps if first else 0,
            save_every=args.save_every,
            partial_checkpoint_stem=checkpoint_dir / f"summarizer_epoch_{epoch}",
            epoch=epoch,
            loss_sum=loss_sum if first else 0.0,
            loss_steps=loss_steps if first else 0,
        )
        path = checkpoint_dir / f"summarizer_epoch_{epoch}.npz"
        stopped_early = args.max_steps is not None and stats["step"] >= args.max_steps
        save_checkpoint(model, path, optimizer, {
            "epoch": epoch,
            "step": stats["step"],
            "batch_size": config.batch_size,
            "epoch_complete": not stopped_early,
            "loss_sum": stats["loss"] * stats["steps"],
            "loss_steps": stats["steps"],
            "last_loss": stats["last_loss"],
            "epoch_loss": stats["loss"],
        })
        print(f"epoch {epoch}/{config.epochs}: loss={stats['loss']:.4f} steps={stats['steps']} -> {path}")

if __name__ == "__main__":
    main()
