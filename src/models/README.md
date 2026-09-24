### 1. Special Tokens & Batch Matrix Alignment

Before creating matrices, reserve three special indices in your vocabulary:

* `<PAD>` (ID: 0) — Fills shorter sequences so every document in a batch has equal length.
* `<BOS>` (ID: 1) — Signals the "Beginning of Sequence" (essential for the Decoder).
* `<EOS>` (ID: 2) — Signals the "End of Sequence" (tells the model to stop generating text).

Take a batch of variable-length integer lists and truncate or pad them to a fixed sequence length $T$ (e.g., $T = 512$).

This turns your raw lists into a 2D NumPy integer array:


$$\text{Input IDs Shape: } (B, T)$$


where $B$ is the batch size and $T$ is the fixed sequence length.

---

### 2. Token Embedding Lookup

Initialize a learnable weight matrix $W_{emb}$ with shape $(V, d_{model})$, where $V$ is your vocabulary size and $d_{model}$ is your feature dimension (e.g., $512$).

In NumPy, you do not need matrix multiplication for embeddings. You use integer array indexing to look up vector representations:

$$X_{emb} = W_{emb}[\text{input\_ids}] \times \sqrt{d_{model}}$$

* **Input:** Integer array of shape $(B, T)$
* **Output Matrix:** Float array of shape $(B, T, d_{model})$

*Note: Scaling by $\sqrt{d_{model}}$ keeps the magnitude of embeddings balanced when adding positional encodings.*

---

### 3. Adding Positional Encodings

Transformers process all tokens in parallel and have no built-in sense of word order. You must add static positional encodings ($PE$) directly to your embedding matrix.

Generate a 2D matrix $PE$ of shape $(T, d_{model})$ using alternating sine and cosine functions:

* For even indices $2i$: $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
* For odd indices $2i+1$: $PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

Broadcast and add this matrix directly to $X_{emb}$:

$$X = X_{emb} + PE$$

* **Output Shape:** $(B, T, d_{model})$

---

### 4. Creating Attention Masks

Because padded tokens (`<PAD>`) carry no meaning, and decoder tokens cannot look into the future during training, you must construct mask matrices before calling Multi-Head Attention.

#### A. Padding Mask (For Encoder & Decoder)

Identifies where `<PAD>` tokens exist in your input:

* Create a boolean mask of shape $(B, 1, 1, T)$ where value is `1` for real tokens and `0` for `<PAD>`.
* During Scaled Dot-Product Attention ($QK^T / \sqrt{d_k}$), replace `0` positions with $-1e9$ ($-\infty$). When passed into `softmax()`, these positions evaluate to $0$ probability, ignoring the padding completely.

#### B. Look-Ahead / Causal Mask (For Decoder Self-Attention)

Prevents position $i$ from attending to future positions $j > i$ during summary generation:

* Create a lower-triangular matrix of shape $(T, T)$ filled with `1`s on and below the diagonal, and $0$s above.
* Combine this with the padding mask to ensure the decoder only attends to past real tokens.

---

### 5. Passing Data Through the Architecture

With your padded matrices and masks ready, the data flows through two distinct paths:

```
Source Document (Nepali Text)
       │
   [BPE Tokenization] -> Integer IDs: (B, T_src)
       │
   [Embedding + PE]   -> Shape: (B, T_src, d_model)
       │
┌──────┴─────────────────────────────────────────┐
│                 ENCODER                        │
│ Inputs: X + Padding Mask                       │
│ Computes: Self-Attention & Feed-Forward        │
└──────┬─────────────────────────────────────────┘
       │
       │ Output: Memory Matrix (B, T_src, d_model)
       ▼
┌────────────────────────────────────────────────┐
│                 DECODER                        │
│ Inputs: Summary Tokens (Shifted Right)         │
│         Memory Matrix from Encoder             │
│ Computes: Masked Self-Attention +              │
│           Cross-Attention (Queries from Decoder│
│           Keys/Values from Encoder Memory)     │
└──────┬─────────────────────────────────────────┘
       │
       │ Output Matrix: (B, T_tgt, d_model)
       ▼
[Linear Projection (d_model -> V)] + Softmax
       │
       ▼
Logits / Probabilities over Vocabulary (B, T_tgt, V)

```

1. **Encoder Stream:** Takes the source Nepali text embeddings and outputs a refined context matrix of shape $(B, T_{src}, d_{model})$.
2. **Decoder Stream:** Takes the target summary tokens (prepended with `<BOS>`), runs Masked Self-Attention, and then performs **Cross-Attention** where:
* **Queries ($Q$)** come from the Decoder's previous layer.
* **Keys ($K$) and Values ($V$)** come directly from the Encoder's context matrix.


3. **Final Projection:** The decoder output of shape $(B, T_{tgt}, d_{model})$ is multiplied by an Output Linear Weight matrix $W_{out}$ of shape $(d_{model}, V)$ to produce token logits of shape $(B, T_{tgt}, V)$.