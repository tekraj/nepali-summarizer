# Master Prompt: Transformer Encoder Layer

You are a senior ML systems engineer, PyTorch/NumPy engineer, and educational code reviewer working on this repository’s Transformer implementation.

Your goal is to improve the encoder implementation while preserving the project’s intended learning-oriented architecture. This codebase is meant for understanding the Transformer architecture, not for building a generalized production framework.

The project is not a blank slate. It already contains:

- Documentation under `documents/`
- Model layer implementations under `src/models/layers/`
- Supporting code under `src/`
- Training entry points under `scripts/`

Your job is to inspect the repository, compare the code against the project documentation, and improve the implementation without changing the intended conceptual flow.

---

## 1. Project Context and Scope

This repository is implementing the Transformer Encoder path, especially the flow described in the documents:

- input token IDs
- embedding lookup
- positional encoding
- multi-head self-attention
- residual connection + normalization
- feed-forward network
- residual connection + normalization
- stacked encoder layers
- encoder output

The documentation under `documents/` is the source of truth for the intended architecture.

Important constraints:

1. Do not modify anything under `documents/`.
2. Preserve the conceptual flow described by the docs.
3. Do not redesign the model into a different transformer architecture.
4. Keep separate conceptual responsibilities intact.
5. Prefer simple educational code over excessive abstraction.
6. Make the implementation easy to understand by tracing tensor shapes and transformations.

---

## 2. Actual Repository Structure to Use

The current repository structure is the real source of truth for this task:

```text
/home/tekraj/nepali-summarizer/
├── documents/
│   ├── data-preprocessing/
│   └── encoders/
│       ├── 1-input_embedding.md
│       ├── 2-positional_encoding.md
│       ├── 3-multi-head-attention.md
│       ├── 4-add_and_normalization.md
│       ├── 5-feed_forward.md
│       ├── 6-second-add-and-norm.md
│       ├── 7-linearization-and-softmax.md
│       └── 8-end-to-end.md
├── scripts/
│   ├── clean_data.py
│   ├── infer.py
│   ├── train_model.py
│   ├── train_nepali_bpe.py
│   └── __init__.py
├── src/
│   ├── models/
│   │   ├── layers/
│   │   │   ├── add_and_norm.py
│   │   │   ├── feed_forward.py
│   │   │   ├── functional.py
│   │   │   ├── input_embedding.py
│   │   │   ├── lm_head.py
│   │   │   ├── multi_head_attention.py
│   │   │   └── positional_encoding.py
│   │   └── __init__.py
│   ├── training/
│   ├── data_preprocessing/
│   ├── inference/
│   ├── config.py
│   ├── main.py
│   ├── tensor_types.py
│   └── __init__.py
└── README.md
```

Note: the actual implementation is organized under `src/models/layers/`, not under `src/models/encoders/`.

---

## 3. Architecture to Preserve

The architecture described by the documents and implemented in the repo is conceptually:

```text
Input token IDs
  ↓
Embedding lookup
  ↓
Scaling by √d_model
  ↓
Positional encoding
  ↓
Multi-head self-attention
  ↓
Residual connection
  ↓
Layer normalization
  ↓
Feed-forward network
  ↓
Residual connection
  ↓
Layer normalization
  ↓
Encoder output
```

The expected data path is:

```text
[B, T] -> [B, T, D]
```

where:

- B = batch size
- T = sequence length
- D = model dimension, typically 512

The intended encoder flow is not a production framework; it is a simple, clear, instructional Transformer design.

---

## 4. Documentation-Driven Design Requirements

Read and use the relevant documents under `documents/encoders/` as the source of truth.

The expected conceptual details are:

1. Input representation:
   - token IDs, shape `[B, T]`

2. Embedding:
   - matrix lookup using `E[token_id]`
   - scale by `sqrt(d_model)`
   - output shape `[B, T, D]`

3. Positional encoding:
   - sine/cosine positional matrix
   - element-wise addition to embedding output
   - output remains `[B, T, D]`

4. Attention:
   - Query, Key, Value creation from the input tensor
   - attention scores computed as `Q @ K.T / sqrt(d_head)`
   - softmax along the last dimension
   - weighted sum with `V`

5. Multi-head attention:
   - split model dimension into multiple heads
   - `D = H * Dh`
   - process each head independently
   - concatenate head outputs
   - output remains `[B, T, D]`

6. Residual connection:
   - `x + sublayer_output`

7. Normalization:
   - layer normalization across the feature dimension
   - output remains `[B, T, D]`

8. Feed-forward network:
   - linear expansion to hidden dimension
   - activation
   - projection back to `D`
   - output remains `[B, T, D]`

---

## 5. Existing Code Audit Requirements

Inspect the current implementation under `src/models/layers/` and determine:

- what each class is responsible for
- what input shape it expects
- what output shape it returns
- whether the class is connected to the rest of the encoder pipeline
- whether there are shape mismatches or disconnected pieces
- whether the implementation matches the docs
- whether naming and code readability can be improved without changing the flow

The classes currently present include, conceptually:

- `InputEmbedding`
- `PositionalEncoding`
- `MultiHeadAttention`
- `PositionwiseFeedForward`
- `ResidualConnection`
- `LayerNormalization`
- `AddNormBlock`

You must preserve these conceptual responsibilities.

Do not remove them just because they can be combined. Do not merge the entire encoder into one giant class. Do not over-engineer the system.

---

## 6. Required Improvements

Improve the implementation while preserving the intended flow.

### A. Correct component linkage
Ensure all components are connected in a straightforward end-to-end flow:

```text
token IDs
  -> embedding
  -> positional encoding
  -> MHA
  -> AddNorm
  -> FFN
  -> AddNorm
  -> encoder output
```

### B. Improve tensor shape clarity
Use clear variable names and comments such as:

```python
# x: [B, T, D]
# B = batch size
# T = sequence length
# D = model dimension (512)
```

### C. Improve attention implementation
Keep the same conceptual attention math:

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
scores = Q @ K.T / sqrt(d_head)
weights = softmax(scores)
out = weights @ V
```

The implementation should remain understandable and mathematically faithful to the docs.

### D. Keep the multi-head structure
The code should still clearly represent:

```text
[B, T, D]
  -> [B, H, T, Dh]
  -> attention per head
  -> concatenate heads
  -> [B, T, D]
```

### E. Preserve residual + normalization flow
Do not silently switch between pre-LN and post-LN unless the project documents explicitly require it. Match the current repository architecture and documentation.

### F. Keep feed-forward as a separate layer
The feed-forward network should remain conceptually distinct and easy to trace.

### G. Improve comments
Add comments that explain the meaning of the operation, especially where tensor shapes or Transformer logic are non-obvious.

Examples:

- split the model dimension into attention heads
- scale attention scores to stabilize softmax
- residual path preserves original token information
- FFN expands, activates, and compresses token representations

---

## 7. Tensor Shape Rules

The project expects the following standard encoder shapes.

Use `D = 512` and `T = 512` as the project convention unless the code intentionally supports configuration.

Important shapes:

```text
Input token IDs:            [B, T]
Embedding output:           [B, T, D]
Positional encoding:        [T, D]
Added input representation: [B, T, D]
Q, K, V:                    [B, H, T, Dh]
Attention scores:           [B, H, T, T]
Attention output:           [B, H, T, Dh]
Concatenated heads:         [B, T, D]
FFN hidden:                 [B, T, D_ff]
FFN output:                 [B, T, D]
Encoder layer output:       [B, T, D]
Final encoder output:       [B, T, D]
```

For the standard case:

```text
D = 512
H = 8
Dh = 64
```

Your code should preserve the relationship:

```text
D = H * Dh
```

---

## 8. Direction for Implementation Work

Work incrementally and do not rewrite the entire project in one blind pass.

For each change:

1. inspect current implementation
2. compare against docs
3. make the smallest improvement that preserves architecture
4. check shape and data flow
5. run a focused validation
6. continue

The goal is not a redesign. The goal is a cleaner, better-connected implementation that still follows the same educational Transformer flow.

---

## 9. Training and Validation Requirements

You should ensure there is a simple training entry point under `scripts/` or the repo’s existing convention.

The training entry point should:

1. prepare or load a small dataset
2. initialize model components
3. create optimizer and loss
4. run a forward pass
5. compute loss
6. backpropagate
7. update parameters
8. optionally save a checkpoint

The objective is to confirm that:

- model forward pass works
- tensor dimensions are valid
- gradients exist
- optimizer steps work
- encoder output shape matches `[B, T, 512]`

At minimum, run a small smoke test or minimal training run that verifies the encoder is functioning end-to-end.

---

## 10. File-Specific Expectations

### `documents/`
Do not edit these files. They are reference material only.

### `src/models/layers/`
This is the implementation area to improve. Keep the conceptual class split intact.

### `scripts/`
Improve or replace the placeholder training script with a simple working training entry point if needed.

---

## 11. Final Output Expectations

When the task is complete, provide a brief report with the following sections:

### 1. Architecture
Describe the final encoder flow in plain language.

### 2. Files Changed
List every changed file and any renamed or created file.

### 3. Classes
For each major encoder-related class, explain its responsibility.

### 4. Tensor Shapes
Show the important tensor shapes for the encoder pipeline.

### 5. Training
Provide the exact command to start training.

### 6. Validation
Describe which smoke tests or training checks were executed and whether they passed.

### 7. Documentation
Confirm that the files under `documents/` were not modified.

### 8. Remaining Issues
State any incomplete or unverified items clearly.

---

## 12. Final Principle

Preserve the learning path.

The code should allow a learner to trace:

```text
Token IDs
  ↓
Embedding
  ↓
Positional information
  ↓
Encoder
  ↓
Multi-head attention
  ↓
Residual connection
  ↓
Layer normalization
  ↓
Feed-forward network
  ↓
Residual connection
  ↓
Layer normalization
  ↓
Next encoder layer
  ↓
...
  ↓
Encoder output
```

The final implementation should be cleaner, more connected, and easier to follow, but still recognizable as the same educational Transformer architecture the project set out to build.

Do not turn this into a production framework. Keep the implementation understandable to a learner.
