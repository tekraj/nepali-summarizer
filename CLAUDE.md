# CLAUDE.md

This file gives Claude Code the context it needs to work in this repository.

## What this project is

A Nepali (Devanagari) text summarization Transformer built **from scratch in NumPy**. There is no PyTorch or autograd. Every layer has a hand-written `forward()` and `backward()`, and the loss and Adam optimizer are hand-written too. The project is also a learning project: each layer comes with a long-form explanation in [documents/](documents/), and the code follows those documents step by step.

**Current state:** the **encoder** is complete (forward, backward, training loop) and is trained with **masked language modelling (MLM)** as a pretraining objective. The decoder (masked self-attention, cross-attention), seq2seq summarization training, and autoregressive / beam-search inference **have not been written yet**. [scripts/infer.py](scripts/infer.py) is a placeholder. [src/inference/README.md](src/inference/README.md) describes the planned decoder loop.

## Setup and commands

Python ≥ 3.12. Dependencies are managed with `uv` (`uv.lock`, `.venv/`); `pip install -e .` also works.

```bash
uv sync                                    # or: python -m pip install -e .

# Pipeline, in order
python -m scripts.clean_data               # data/raw -> data/cleaned   (console script: clean-data)
python -m scripts.train_nepali_bpe         # data/cleaned -> data/nepali_vocab_output/{vocab.json,merges.txt}
python -m scripts.train_model              # MLM-train the encoder, settings from config/config.yaml
python -m scripts.train_model --max-steps 5 --epochs 1   # fast smoke test; use this to check changes
python -m scripts.train_model --resume checkpoints/encoder_epoch_1.npz
```

Run modules from the project root with `python -m ...`, because imports are absolute (`from src....`). There is **no test suite and no linter config**. To check a change, run the `--max-steps` smoke test. If you change a `backward()`, also compare it against a numerical finite-difference gradient check.

## Layout

```
config/config.yaml          training / data / model hyperparameters (flattened by src/config.py:ProjectConfig)
scripts/                    CLI entry points (thin wrappers around src/)
src/
  config.py                 ProjectConfig dataclass; .resolve() makes paths absolute from the project root
  tensor_types.py           FloatArray / IntArray / BoolArray + the shape legend (B, T, D, H, Dh, D_ff, V, N)
  data_preprocessing/
    text_cleaning_pipeline  NFC normalize, strip HTML, remove ZWJ/ZWNJ, collapse whitespace
    bpe_tokenization        trains HF `tokenizers` ByteLevelBPE (vocab 30000 target; the current vocab has 6302)
    batching.py             CreateTrainingBatch: streams .txt lines -> [B,T] ids + [B,T] mask; MLM masking
  models/
    encoder.py              EncoderBlock (post-LN) and TransformerEncoder (embed -> PE -> N blocks -> LM head)
    encoder_layers/         one file per doc step: input_embedding, positional_encoding,
                            multi_head_attention, add_and_norm, feed_forward, lm_head, functional
  training/                 loss.py (softmax+CE combined grad), optimizer.py (Adam + global-norm clip), train.py
  inference/greedy_decoder  argmax over LM-head probabilities (MLM-style, not autoregressive)
documents/                  math and explanations; encoders/1..9 map to the encoder_layers files
data/raw, data/cleaned      ~1.3k Nepali news .txt files (both are committed to git)
checkpoints/                .npz weights (gitignored)
```

## Architecture conventions (follow these when adding layers)

- **Layer interface:** every parameterised layer exposes
  - `forward(...)`, which stores whatever backward needs in `self.cache`
  - `backward(d_out) -> d_input`, which fills `self.grads[name]`
  - `parameters() -> {name: ndarray}`, whose keys match `grads`

  `TransformerEncoder._layers()` combines these into dotted names (for example `blocks.0.attention.W_q`). `parameters()`, `gradients()`, the optimizer and the checkpoints all use those names.
- **In-place updates:** Adam changes parameter arrays in place (`param -= ...`) and `load_checkpoint` uses `param[...] = ...`. Never rebind a weight attribute to a new array, because the optimizer state and the tied weights (`W_lm = E.T` is a *view*) would stop pointing at it.
- **Residuals in backward:** the gradient that flows around the sublayer is *added* to the gradient that flows through it (see `EncoderBlock.backward`).
- **dtype:** weights and activations are `float32`, token ids are `int64`, masks are `bool` (True means a real token or a position to keep).
- **Shapes:** NumPy types cannot express shapes, so every function writes its shapes in comments and docstrings using the legend in [src/tensor_types.py](src/tensor_types.py), for example `[B, T, D]` or `[B, H, T, Dh]`. Keep this style. Each module's docstring names the document it implements, for example `(documents/encoders/3-multi-head-attention.md)`.
- **MHA:** all heads are packed into single `[D, D]` matrices (head *i* owns columns `i*Dh:(i+1)*Dh`). Padding uses `MASK_VALUE = -1e9`, not `-inf`.
- **MLM training:** the LM head scores only the masked positions (`output_mask` gives `[N, D] -> [N, V]`). `backward` then scatters the gradient back into `[B, T, D]`.
- **Initialisation:** N(0, 0.02) for the embeddings, FFN and LM head. Xavier for attention. Embeddings are multiplied by √D before the positional encoding is added.
- **Special tokens** (from the BPE vocab): `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`. Look up their ids in `vocab.json`; don't hard-code them.

## Gotchas

- **Memory:** every block caches its activations, and the attention weights alone take `B×H×T×T` floats per block. That is why `batch_size` defaults to 4. Before raising `B`, `T` or `num_layers`, think about how much RAM it needs.
- The batcher yields **one line per sample** (each cleaned file is one normalized line), truncated to `max_sequence_length` and padded to the longest sample in the batch. As a result `T` changes from batch to batch.
- Some documents and READMEs are out of date. [documents/encoders/9-backpropagation.md](documents/encoders/9-backpropagation.md) refers to `src/models/layers/` (the real path is `src/models/encoder_layers/`). README.md lists `scripts.preprocess_data`, which does not exist. [src/data_preprocessing/README.md](src/data_preprocessing/README.md) describes a from-scratch BPE, but the code uses HF `tokenizers`. When the code and the prose disagree, trust the code.
- `scripts/train_nepali_bpe.py` and `scripts/clean_data.py` hard-code their paths instead of reading `config.yaml`.
