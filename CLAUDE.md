# CLAUDE.md

This file gives Claude Code the context it needs to work in this repository.

## What this project is

A Nepali (Devanagari) text summarization Transformer built **from scratch in CuPy** (NumPy-compatible arrays on an NVIDIA GPU, fp16 mixed precision). There is no PyTorch or autograd. Every layer has a hand-written `forward()` and `backward()`, and the loss and Adam optimizer are hand-written too. The project is also a learning project: each layer comes with a long-form explanation in [documents/](documents/), and the code follows those documents step by step.

**Current state:** the **encoder** is complete (forward, backward, training loop) and is trained as an **encoder-only prefix-LM summarizer**. Each sample is `<s> article[:766] </s> summary[:255] </s>` (≤ 768 + 256 = 1024 tokens; the article's right side is truncated). A `[B, T, T]` prefix-LM attention mask lets article tokens attend to the whole article, while summary tokens attend to the article plus earlier summary tokens only. Position p predicts token p+1, and the loss is computed only over summary tokens. [scripts/infer.py](scripts/infer.py) generates greedily, one token at a time. There is no decoder or cross-attention. [src/inference/README.md](src/inference/README.md) describes a decoder loop that was planned but not built.

## Setup and commands

Python ≥ 3.12 and an NVIDIA GPU with a CUDA 12.x driver (`cupy-cuda12x`). Dependencies are managed with `uv` (`uv.lock`, `.venv/`); `pip install -e .` also works.

```bash
uv sync                                    # or: python -m pip install -e .

# Pipeline, in order
python -m scripts.clean_data               # data/raw -> data/cleaned   (console script: clean-data)
python -m scripts.train_nepali_bpe         # data/cleaned -> data/nepali_vocab_output/{vocab.json,merges.txt}
python -m scripts.train_model              # train the summarizer on data/cleaned + data/summary pairs
python -m scripts.train_model --max-steps 5 --epochs 1   # fast smoke test; use this to check changes
python -m scripts.train_model --resume checkpoints/summarizer_epoch_1.npz
python -m scripts.infer --checkpoint checkpoints/summarizer_epoch_1.npz --article data/cleaned/100001.txt
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
    bpe_tokenization        trains HF `tokenizers` ByteLevelBPE (V = 30000)
    batching.py             CreateTrainingBatch: pairs {name}.txt with {name}-summary.txt -> [B,T] ids,
                            [B,T,T] prefix-LM mask, [B,T] loss mask, [N] next-token targets
  models/
    encoder.py              EncoderBlock (post-LN) and TransformerEncoder (embed -> PE -> N blocks -> LM head)
    encoder_layers/         one file per doc step: input_embedding, positional_encoding,
                            multi_head_attention, add_and_norm, feed_forward, lm_head, functional
  training/                 loss.py (softmax+CE combined grad), optimizer.py (Adam + global-norm clip), train.py
  inference/greedy_decoder  argmax over LM-head probabilities; greedy_summarize = autoregressive loop
documents/                  math and explanations; encoders/1..9 map to the encoder_layers files
data/cleaned                ~132K pure-Devanagari Nepali news articles, {name}.txt (committed to git)
data/summary                one target summary per article, {name}-summary.txt (committed to git)
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
- **Arrays:** every model/training module does `import cupy as cp` and keeps arrays on the GPU. Only Python lists from the tokenizer cross from host to device (one `cp.array` per batch).
- **dtype (mixed precision, `precision: fp16` in config.yaml):** weights, gradients and Adam state are `float32` (the master copy the optimizer updates and checkpoints store). Each forward casts weights with `to_compute()` and runs activations and matmuls in `float16`; LayerNorm statistics, softmax sums, the LM-head softmax and the loss are `float32`. Use `functional.matmul` rather than `@` (CuPy's batched fp16 `@` skips the tensor cores). `train_step` multiplies `d_logits` by `optimizer.loss_scale`; `Adam.step` unscales, and skips the step and halves the scale on inf/NaN. `precision: fp32` runs everything in `float32` (use it for finite-difference gradient checks). Token ids are `int64`, masks are `bool` (True means a real token or a position to keep).
- **Shapes:** CuPy types cannot express shapes, so every function writes its shapes in comments and docstrings using the legend in [src/tensor_types.py](src/tensor_types.py), for example `[B, T, D]` or `[B, H, T, Dh]`. Keep this style. Each module's docstring names the document it implements, for example `(documents/encoders/3-multi-head-attention.md)`.
- **MHA:** all heads are packed into single `[D, D]` matrices (head *i* owns columns `i*Dh:(i+1)*Dh`). Padding uses `MASK_VALUE = -1e9`, not `-inf`.
- **Summary-only loss:** the LM head scores only the positions that predict a summary token (`output_mask` gives `[N, D] -> [N, V]`). `backward` then scatters the gradient back into `[B, T, D]`. MHA accepts either a `[B, T]` padding mask or a `[B, T, T]` query x key mask.
- **Initialisation:** N(0, 0.02) for the embeddings, FFN and LM head. Xavier for attention. Embeddings are multiplied by √D before the positional encoding is added.
- **Special tokens** (from the BPE vocab): `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`. Look up their ids in `vocab.json`; don't hard-code them.

## Gotchas

- **Memory:** every block caches its activations, and the attention weights alone take `B×H×T×T` floats per block. In fp16 that is about 2.4 GB peak at B=4, T=1024; `batch_size` is 16 for Colab's 16 GB T4 (use `--batch-size 4` on a 6 GB GPU). Before raising `B`, `T` or `num_layers`, think about how much RAM it needs.
- The batcher yields **one (article, summary) pair per sample**, and only for articles that have a `data/summary/{name}-summary.txt` file. Samples are padded to the longest one in the batch, so `T` changes from batch to batch.
- Checkpoints trained before the 30000-token vocab cannot be loaded, because `E` and `W_lm` have a different shape. This includes the old MLM `checkpoints/encoder_epoch_*.npz` (1088-token vocab) and anything trained with the earlier 6302-token vocab.
- With V = 30000, `E` and `W_lm` hold 15.36M parameters each, and the whole model has about 49.7M (`tie_weights: false`). The loss at step 0 should be about ln 30000 ≈ 10.3.
- The ByteLevel pre-tokenizer (GPT-2 regex) splits Devanagari words at every vowel sign, because matras are Unicode marks (Mn/Mc), not `\p{L}`. So no BPE token spans a matra: "नेपाल" is 5 tokens even with the 30K vocab, and text averages about 1.5 characters per token. Keep this in mind for the 766-token article budget.
- Some documents and READMEs are out of date. [documents/encoders/9-backpropagation.md](documents/encoders/9-backpropagation.md) refers to `src/models/layers/` (the real path is `src/models/encoder_layers/`). README.md lists `scripts.preprocess_data`, which does not exist. [src/data_preprocessing/README.md](src/data_preprocessing/README.md) describes a from-scratch BPE, but the code uses HF `tokenizers`. When the code and the prose disagree, trust the code.
- `scripts/train_nepali_bpe.py` and `scripts/clean_data.py` hard-code their paths instead of reading `config.yaml`.
