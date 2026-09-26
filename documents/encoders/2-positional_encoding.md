# The Complete Positional Encoding & Input Preparation Pipeline

Because the Transformer processes all tokens in a sequence simultaneously, its underlying self-attention mechanism is **permutation-invariant**. It has no inherent concept of word order. To prevent the model from treating sentences as an unordered "bag of words," **Positional Encodings** mathematically inject sequence order directly into the token representations.

The pipeline below details the complete journey from raw text to the final 3D batched tensor passed into the Multi-Head Attention layer.

```text
 Sentences ──BPE──> token IDs ──pad to same length──> IDs (B, T) + mask (B, T)
                                                          │
                                         Embedding × √d_model   (1-input_embedding.md)
                                                          │
                                                   (B, T, d_model)
                                                          │
                                                   + PE[:T]        (this document)
                                                          │
                                         X_0 (B, T, d_model) + mask (B, T) ──► Multi-Head Attention
```

> **Order matters:** padding and the mask are created on the **token IDs** (before embedding). The `[PAD]` IDs then get embedded and position-encoded like any other token; the mask is what makes attention ignore them.
>
> **Code:** `src/models/layers/positional_encoding.py` → `PositionalEncoding`; padding and mask in `src/data_preprocessing/batching.py` → `CreateTrainingBatch.create_input_tensor`.

---

## 1. Mathematical Generation of Positional Encodings

The Transformer generates a static (not learned) Positional Encoding ($PE$) matrix of shape $(T_{\max}, d_{\text{model}})$ — in this project $(512, 512)$ — using sine and cosine functions. For any token at sequence position $pos$ (starting at $0$) and dimension-pair index $i \in \left[0, \frac{d_{\text{model}}}{2} - 1\right]$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

In words: dimensions come in pairs $(2i, 2i+1)$. Both dimensions of a pair use the **same frequency** $\frac{1}{10000^{2i/d_{\text{model}}}}$; the even one takes the sine, the odd one takes the cosine.

* **Frequency Decay:** Lower dimensions (small $i$) oscillate rapidly to capture local, immediate word order. Higher dimensions (large $i$) change very slowly and track coarse, global position across long sequences.
* **Linear Relative Shift:** Because of trigonometric angle-addition identities, the encoding at position $pos + k$ is a fixed rotation of the encoding at position $pos$. This lets attention heads compute relative distances between tokens (e.g., *"three words to my left"*), regardless of absolute position.

### Worked Example: $PE$ for $d_{\text{model}} = 4$

With $d_{\text{model}} = 4$ there are two pairs:

* Pair $i = 0$ (dims 0, 1): frequency $= 1 / 10000^{0/4} = 1$ → angle $= pos$
* Pair $i = 1$ (dims 2, 3): frequency $= 1 / 10000^{2/4} = 1/100$ → angle $= pos / 100$

| $pos$ | dim 0: $\sin(pos)$ | dim 1: $\cos(pos)$ | dim 2: $\sin(pos/100)$ | dim 3: $\cos(pos/100)$ |
| --- | --- | --- | --- | --- |
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |
| 3 | 0.1411 | −0.9900 | 0.0300 | 0.9996 |

Dims 0–1 change a lot from one position to the next (fast "clock hand"); dims 2–3 barely move (slow "clock hand").

---

## 2. Element-Wise Addition for Single Sequences

For a single sentence like **"The cat ate fish"** (4 tokens), the model first retrieves the scaled semantic vectors from the Embedding Layer, resulting in a matrix of shape $4 \times d_{\text{model}}$ (e.g., $4 \times 512$).

It then takes the first 4 rows from the pre-computed $PE$ matrix, which also has a shape of $4 \times 512$. Instead of concatenating them, the model performs direct **element-wise addition**:

* **Row 1:** $\text{Embed("The")} + PE(0)$
* **Row 2:** $\text{Embed("cat")} + PE(1)$
* **Row 3:** $\text{Embed("ate")} + PE(2)$
* **Row 4:** $\text{Embed("fish")} + PE(3)$

Because $512$ is a high-dimensional space, the semantic representation absorbs this positional signal without destroying its core identity. The output remains a $4 \times 512$ matrix containing both the **"what"** (semantics) and the **"where"** (position) of every token.

### Worked Example (continuing from `1-input_embedding.md`)

Scaled embeddings for token IDs $[2, 5, 3]$ plus the first 3 rows of the $PE$ table above:

| $pos$ | Scaled embedding | $+\ PE(pos)$ | $= \mathbf{X}_0$ row |
| --- | --- | --- | --- |
| 0 | $[0.04,\ 0.02,\ -0.02,\ 0.06]$ | $[0,\ 1,\ 0,\ 1]$ | $[0.04,\ 1.02,\ -0.02,\ 1.06]$ |
| 1 | $[0.00,\ -0.02,\ 0.04,\ 0.02]$ | $[0.8415,\ 0.5403,\ 0.01,\ 1.0]$ | $[0.8415,\ 0.5203,\ 0.05,\ 1.02]$ |
| 2 | $[-0.06,\ 0.04,\ 0.02,\ -0.02]$ | $[0.9093,\ -0.4161,\ 0.02,\ 0.9998]$ | $[0.8493,\ -0.3761,\ 0.04,\ 0.9798]$ |

If the same token appeared at positions 0 and 2, its two rows in $\mathbf{X}_0$ would now be **different** — that is exactly how the model can tell word order apart.

---

## 3. Batching Multiple Sequences (3D Tensors)

Transformers process multiple independent sentences in parallel to maximize hardware throughput (on a GPU in large frameworks; in this project NumPy runs the same batched matrix math on the CPU).

If we have two independent sentences:

* `"The cat ate fish"` $\rightarrow \text{Shape: } (4 \times 512)$
* `"The fish ate cat"` $\rightarrow \text{Shape: } (4 \times 512)$

Rather than concatenating them into a single $8 \times 512$ sequence, we stack them along a new axis called the **Batch Dimension**. This creates a 3D tensor of shape:

$$(Batch, Sequence, d_{\text{model}}) = \mathbf{(2, 4, 512)}$$

* **$2$:** Batch Size $B$ (Number of independent sentences)
* **$4$:** Sequence Length $T$ (Tokens per sentence)
* **$512$:** Vector dimension ($d_{\text{model}}$)

The same $PE[:T]$ matrix of shape $(T, d_{\text{model}})$ is added to **every** sentence in the batch (NumPy broadcasting). Inside the attention layer, sentences never mix: each sentence only attends to its own tokens.

---

## 4. Padding Variable-Length Sequences

Matrix operations require uniform shapes and cannot process "ragged" batches where sentences have varying lengths. To enforce uniformity, the pipeline uses **Padding** on the token IDs:

1. **Truncate:** Any sentence longer than $T_{\max}$ (512 in this project) is cut to 512 tokens.
2. **Determine Max Length:** Identify the longest sequence in the current batch (e.g., 6 tokens). That becomes $T$ for this batch.
3. **Append Filler Tokens:** Append the special `<pad>` token to shorter sentences until all sequences match the max length.

* **Sentence 1:** `["The", "cat", "ate", "fish", "[PAD]", "[PAD]"]`
* **Sentence 2:** `["The", "cat", "ate", "[PAD]", "[PAD]", "[PAD]"]`
* **Sentence 3:** `["The", "cat", "ate", "fish", "quickly", "today"]`

This guarantees a perfectly uniform ID tensor of shape $(3, 6)$, which becomes $(3, 6, 512)$ after embedding.

> **Which ID is `[PAD]`?** It depends on the tokenizer. Many tutorials use ID $0$; **our BPE tokenizer uses `<pad>` = ID $1$** (`<s>` = 0, `<pad>` = 1, `</s>` = 2, `<unk>` = 3, `<mask>` = 4). The code always reads it from `vocab.json` instead of hard-coding it.

---

## 5. Applying the Attention Mask

While padding satisfies shape requirements, `[PAD]` tokens are meaningless filler. The attention mechanism must not assign weights to them or allow them to influence real words.

**Step A — binary mask (created with the padding).** A mask of shape $(B, T)$ marks real tokens with $1$ (`True`) and padding with $0$ (`False`):

$$\text{Attention Mask} = \begin{bmatrix} 1 & 1 & 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 & 0 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}$$

**Step B — additive mask (used inside attention).** The binary mask is **not** added directly. It is first converted: $1 \rightarrow 0$ and $0 \rightarrow -\infty$. This converted version is the $\mathbf{M}$ that is added to the attention scores right before softmax:

$$\mathbf{M} = \begin{bmatrix} 0 & 0 & 0 & 0 & -\infty & -\infty \\ \dots \end{bmatrix} \qquad \text{Softmax}(\text{Scores} + \mathbf{M})$$

Because $e^{-\infty} = 0$, the softmax output is exactly $0.0$ for all padded positions.

### Worked Example

Scores of one query against 4 keys, where the 4th key is `[PAD]`:

| | key 1 | key 2 | key 3 | key 4 (`[PAD]`) |
| --- | --- | --- | --- | --- |
| Scores | 2.0 | 1.0 | 0.5 | 0.3 |
| Softmax **without** mask | 0.5638 | 0.2074 | 0.1258 | 0.1030 ❌ |
| Scores $+ \mathbf{M}$ | 2.0 | 1.0 | 0.5 | $-\infty$ |
| Softmax **with** mask | 0.6285 | 0.2312 | 0.1402 | **0.0000** ✅ |

Without the mask, 10% of the attention would be wasted on filler. With it, the weight is redistributed over the real tokens and still sums to $1.0$.

> In code, $-\infty$ is replaced by $-10^9$: $e^{-10^9}$ is also exactly $0.0$ in floating point, but it avoids `NaN` in the corner case where every key in a row is masked.

---

> **Final Output:** The resulting package — a uniform 3D tensor $\mathbf{X}_0$ of shape $(B, T, d_{\text{model}})$ carrying integrated positional signals, together with its $(B, T)$ boolean attention mask — is delivered directly into the first Multi-Head Attention block.
