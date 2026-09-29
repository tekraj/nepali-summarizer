# End-to-End Encoder Pipeline

This document puts documents 1–7 together into one picture, with the exact shapes and numbers used in this project.

One distinction first: **within a single forward pass**, you do not re-fetch or re-add the raw input embeddings at later blocks; **across training steps (rounds)**, the weight values inside the Input Embedding matrix $E$ are continuously updated via backpropagation.

> **Code:** `src/models/encoder.py` → `EncoderBlock` (one block) and `TransformerEncoder` (the whole model)

---

## Complete End-to-End Transformer Encoder Pipeline

```text
Token IDs (B, T) + attention mask (B, T)
        │
        ▼
[ Embedding E × √d_model  +  PE ] ──► X_0 (B, T, d_model)
                                        │
┌───────────────────────────────────────┴──────────────────────────────┐
│ ENCODER BLOCK  (repeated N times; N = 6 here)                        │
│                                                                      │
│  X_in ──┬──► [ Multi-Head Attention ] ──┐   (uses the mask)          │
│         │                               ▼                            │
│         └─────────────────────────────►(+)──► [ LayerNorm 1 ]        │
│                                                     │                │
│                                                  X_norm1             │
│                                                     │                │
│                     ┌───────────────────────────────┤                │
│                     ▼                               │                │
│            [ Feed-Forward Net ]                     │                │
│                     │                               │                │
│                     └─────────────────────────────►(+)──► [ LayerNorm 2 ] ──► X_out
└──────────────────────────────────────────────────────────────────────┘
                                        │
                                  X_N (B, T, d_model)
                                        │
                    [ LM Head: Linear Projection (d_model → V) ]
                                        │
                              [ Softmax ] → Token Probabilities (B, T, V)
```

### Shapes at Every Stage (this project: $B = 4$, $T = 512$, $d_{model} = 512$, $h = 8$, $d_k = 64$, $d_{ff} = 2048$, $V = 30000$)

| Stage | Tensor | Shape | Example |
| --- | --- | --- | --- |
| Input | token IDs, mask | $(B, T)$ | $(4, 512)$ |
| Embedding | $E[\text{ids}] \cdot \sqrt{512}$ | $(B, T, d_{model})$ | $(4, 512, 512)$ |
| Positional encoding | $PE[:T]$ | $(T, d_{model})$ | $(512, 512)$ |
| Block input | $\mathbf{X}_0$ | $(B, T, d_{model})$ | $(4, 512, 512)$ |
| Attention | $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | $(B, h, T, d_k)$ | $(4, 8, 512, 64)$ |
| Attention | scores / weights | $(B, h, T, T)$ | $(4, 8, 512, 512)$ |
| Attention | concatenated heads | $(B, T, d_{model})$ | $(4, 512, 512)$ |
| FFN | hidden | $(B, T, d_{ff})$ | $(4, 512, 2048)$ |
| Block output | $\mathbf{X}_{out}$ | $(B, T, d_{model})$ | $(4, 512, 512)$ |
| Encoder output | $\mathbf{X}_N$ | $(B, T, d_{model})$ | $(4, 512, 512)$ |
| LM head | logits / probabilities | $(B, T, V)$ | $(4, 512, 30000)$ |

---

## Stage 1: Input Embedding & Positional Encoding

### 1. Vocabulary Lookup

Given a batch of tokenized, padded sentences of shape $(B, T)$, each token ID is looked up in the **Input Embedding Matrix** $E \in \mathbb{R}^{V \times d_{model}}$ (here $V = 30000$, $d_{model} = 512$).

* **Scaling:** The retrieved vectors are multiplied by $\sqrt{d_{model}} \approx 22.63$ so that the embeddings are not dominated by the positional encodings.
* **Output Shape:** $(B, T, d_{model})$

### 2. Positional Encoding Addition

Because attention operates on all tokens simultaneously without inherent sequence order, fixed sinusoidal positional encodings ($PE$) are added directly to the embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$\mathbf{X}_0 = (E[\text{tokens}] \cdot \sqrt{d_{model}}) + PE[:T]$$

* **Operation:** Direct element-wise addition (the same $PE$ rows are added to every sentence in the batch).
* **Output Shape:** $\mathbf{X}_0 \in \mathbb{R}^{B \times T \times d_{model}}$

---

## Stage 2: The Stacked Encoder Blocks ($1 \text{ to } N$)

The tensor $\mathbf{X}_0$ enters the first of $N$ identical (in structure, not in weights) Encoder blocks. Each block consists of two main sub-layers: **Multi-Head Attention** and a **Feed-Forward Network**, both wrapped in **Add & Norm** residual connections.

### 1. Multi-Head Attention (MHA)

* **Projection:** $\mathbf{X}_{in}$ is multiplied by $h$ independent sets of learnable weights ($W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d_{model} \times d_k}$, where $d_k = d_{model} / h = 64$) to generate Queries ($\mathbf{Q}_i$), Keys ($\mathbf{K}_i$), and Values ($\mathbf{V}_i$).
* **Scaled Dot-Product Attention:**

$$\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}_i$$

*(where $\mathbf{M}$ is the additive mask: $0$ for real tokens, $-\infty$ for `[PAD]` keys)*

* **Concatenation & Output Projection:**

$$\text{MHA}(\mathbf{X}_{in}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \cdot W_O$$

*(where $W_O \in \mathbb{R}^{d_{model} \times d_{model}}$)*

* **Shape:** Transforms $(B, T, d_{model}) \rightarrow (B, T, d_{model})$.

### 2. First Add & Norm

* **Residual Connection:** Adds the block input $\mathbf{X}_{in}$ directly to the attention output to prevent vanishing gradients and preserve token identity:

$$\mathbf{X}_{res1} = \mathbf{X}_{in} + \text{MHA}(\mathbf{X}_{in})$$

* **Layer Normalization:** Calculates mean $\mu$ and variance $\sigma^2$ across the $d_{model}$ features for each token independently:

$$\mathbf{X}_{norm1} = \gamma_1 \odot \frac{\mathbf{X}_{res1} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_1$$

* **Shape:** $(B, T, d_{model})$

### 3. Position-Wise Feed-Forward Network (FFN)

* **Up-Projection:** Expands each token vector individually from $d_{model}$ to $d_{ff}$ ($4 \times d_{model} = 2048$):

$$\mathbf{H} = \text{ReLU}(\mathbf{X}_{norm1} W_1 + b_1) \quad \text{where } W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$$

* **Down-Projection:** Compresses back to model dimension:

$$\text{FFN}(\mathbf{X}_{norm1}) = \mathbf{H} W_2 + b_2 \quad \text{where } W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$$

* **Shape:** $(B, T, d_{model}) \rightarrow (B, T, d_{ff}) \rightarrow (B, T, d_{model})$

### 4. Second Add & Norm

* **Residual Connection:** Adds the input of the FFN ($\mathbf{X}_{norm1}$) to its output:

$$\mathbf{X}_{res2} = \mathbf{X}_{norm1} + \text{FFN}(\mathbf{X}_{norm1})$$

* **Layer Normalization:** Normalizes features token-by-token using independent parameters $\gamma_2, \beta_2$:

$$\mathbf{X}_{out} = \text{LayerNorm}_2(\mathbf{X}_{res2})$$

* **Shape:** $(B, T, d_{model})$

---

## Stage 3: Sequential Block Passing

The output tensor $\mathbf{X}_{out}$ from Block 1 becomes the direct input $\mathbf{X}_{in}$ for Block 2, and so on up to Block $N$. The attention mask is passed unchanged to every block.

**Note on Forward Pass Isolation:** During this forward pass, raw embeddings $E$ and positional encodings $PE$ are **never re-fetched or re-added**. The residual connections inside each block naturally carry the original positional and semantic signals forward through all $N$ blocks.

---

## Stage 4: Language Modeling Head

After exiting the final block $N$, the tensor $\mathbf{X}_N \in \mathbb{R}^{B \times T \times d_{model}}$ is converted into token predictions across the vocabulary:

1. **Linear Projection (Un-embedding):**

$$\mathbf{Z} = \mathbf{X}_N \cdot W_{LM} + b_{LM} \quad \text{where } W_{LM} \in \mathbb{R}^{d_{model} \times V}$$

* **Shape Output:** $\mathbf{Z} \in \mathbb{R}^{B \times T \times V}$ (raw scores/logits for all $V$ vocabulary tokens at every position).

2. **Softmax:**

$$P(w_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

* **Shape Output:** $\mathbb{R}^{B \times T \times V}$ (probability distributions summing to $1.0$).

*(During training only the $N$ masked positions are sent through the head — see `7-linearization-and-softmax.md`, Section 2.)*

---

## Stage 5: Forward Pass vs. Training Updates (The Distinction)

| Aspect | Forward Pass (Within 1 Sequence Run) | Backward Pass (Across Training Steps / Rounds) |
| --- | --- | --- |
| **Data Flow** | Tensors flow strictly forward from Input $\rightarrow$ Blocks $1\dots N \rightarrow$ LM Head. | Gradients flow backward from Loss $\rightarrow$ LM Head $\rightarrow$ Blocks $N\dots 1 \rightarrow$ Embedding Matrix $E$. |
| **Input Embedding Usage** | Embeddings are looked up **once** at Layer 0. They are not touched or re-added at higher blocks. | The weight values inside matrix $E$ are **updated** by the optimizer (Adam) using calculated gradients. |
| **State of Weights** | Weights ($E, W_Q, W_K, W_V, W_O, \gamma, \beta, W_1, b_1, W_2, b_2, W_{LM}, b_{LM}$) remain constant during computation. | Weights are adjusted after every batch. In training step $k+1$, the model uses the updated weights for new inputs. |
| **Inference Behavior** | Identical to forward pass during training. | Disabled. All weights are frozen and no parameters are updated. |

---

## Project-Specific Training and Target Data Preparation Strategy

This project trains the encoder as a **summarizer**, not as a BERT-style masked language model. There is no decoder, so one encoder stack reads the article *and* writes the summary. This setup is called a **prefix-LM** (prefix language model).

> **Code:** `src/data_preprocessing/batching.py` → `CreateTrainingBatch.create_input_tensor` and `build_attention_mask`

### 1. The data

* **Articles:** about 132,000 pure-Devanagari Nepali news articles, `data/cleaned/{name}.txt`.
* **Targets:** one summary per article, `data/summary/{name}-summary.txt`. The batcher pairs files by `{name}` and skips a summary whose article is missing.
* **One sample = one (article, summary) pair.** Pairs are streamed from disk one at a time, so RAM use stays flat however big the corpus is.

### 2. Three kinds of "masking", and which ones we use

| Kind | What it does | Used here? |
| --- | --- | --- |
| **Token masking (MLM)** | Replace ~15% of input tokens with `<mask>` and predict them (Section 2 of `7-linearization-and-softmax.md`) | ❌ Only in the old pre-training setup. **No token is ever replaced with `<mask>` here.** |
| **Attention masking** | Hide some positions from others inside attention (score → $-10^9$) | ✅ the prefix-LM mask, $(B, T, T)$ |
| **Loss masking** | Compute the loss only at some positions | ✅ loss only on the summary, $(B, T)$ |

When papers or tutorials say "the input should be masked" for summarization, they mean **loss masking**: the model reads the whole article, but is never graded on reproducing it.

### 3. Step A — Build one sequence per sample

The article and summary are tokenized separately, each with its own budget, then joined:

$$\underbrace{\texttt{<s>}\ \ \text{article}[{:}766]\ \ \texttt{</s>}}_{\text{prefix: at most } 768 \text{ tokens}}\ \ \underbrace{\text{summary}[{:}255]\ \ \texttt{</s>}}_{\text{target: at most } 256 \text{ tokens}}\ \ \texttt{<pad>} \dots$$

* **Budgets, not a fixed split.** 768 + 256 = 1024 is the *maximum* length (75% / 25%). A short article is **not** padded up to 768. The summary starts right after the article's real `</s>`, and `<pad>` is only added **at the end** of the row, up to the longest sample in the batch.
* **Truncation keeps the start.** An article longer than 766 tokens loses its right side. News articles put the key facts first, so this keeps the most useful part. The same rule cuts summaries at 255 tokens.
* **How often it happens (2,000 real pairs):** the median article is 1,125 tokens and 66% of articles are truncated. The median summary is 162 tokens and about 6% are truncated.
* **The input row contains the real summary.** This is *teacher forcing*: the true summary tokens go into the model as input, and the attention mask (Step B) stops any position from seeing a token it has to predict.

### 4. Step B — The attention mask: who may look at whom

`build_attention_mask` builds a $(B, T, T)$ boolean mask (query × key). A key is visible to a query when it is a real token **and** either it belongs to the prefix **or** it is not after the query:

$$\text{visible}(q, k) = \big(k < \text{prefix\_len} \ \lor\ k \le q\big) \ \land\ k < \text{len}$$

Toy example: article `a1 a2 a3` and summary `s1 s2` (prefix length 5, real length 8, then one `<pad>`):

```text
              keys →  <s> a1  a2  a3 </s>  s1  s2 </s> <pad>
query <s>              ✓   ✓   ✓   ✓   ✓    ·   ·   ·    ·
      a1               ✓   ✓   ✓   ✓   ✓    ·   ·   ·    ·     article rows: the whole article,
      a2               ✓   ✓   ✓   ✓   ✓    ·   ·   ·    ·     in both directions (like an encoder);
      a3               ✓   ✓   ✓   ✓   ✓    ·   ·   ·    ·     never the summary
      </s>             ✓   ✓   ✓   ✓   ✓    ·   ·   ·    ·
      s1               ✓   ✓   ✓   ✓   ✓    ✓   ·   ·    ·     summary rows: the whole article +
      s2               ✓   ✓   ✓   ✓   ✓    ✓   ✓   ·    ·     earlier summary tokens only
      </s>             ✓   ✓   ✓   ✓   ✓    ✓   ✓   ✓    ·     (causal, like a decoder)
```

Compared with an encoder–decoder Transformer, the top-left block does the encoder's job, the bottom-left block does cross-attention's job, and the bottom-right triangle is the decoder's causal mask. `<pad>` keys are hidden from everyone.

### 5. Step C — Targets and the loss mask

Position $p$ is trained to predict the token at $p + 1$. The loss mask keeps only the positions whose next token is a summary token, from the prefix's closing `</s>` to the last summary token:

$$\text{loss\_mask}[p] = \big(p \ge \text{prefix\_len} - 1\big) \ \land\ \big(p < \text{len} - 1\big)$$

```text
position:    0    1    2    3    4     5    6    7     8
input:      <s>   a1   a2   a3  </s>   s1   s2  </s>  <pad>
next token:  a1   a2   a3  </s>  s1    s2  </s> <pad>   –
loss?        ✗    ✗    ✗    ✗    ✓     ✓    ✓    ✗     ✗
```

* **The article positions get no loss.** This is the "input is masked" part.
* **`targets` has shape $(N)$.** The code gathers only the next tokens at the $N$ loss positions (here `[s1, s2, </s>]`), instead of keeping a $(B, T)$ label row filled with "ignore" values.
* **The final `</s>` is a target.** The model learns when to stop, which is how inference knows the summary is finished.
* The LM head runs only on those $N$ positions: $(N, 512) \rightarrow (N, V)$. The gradient is scattered back into $(B, T, 512)$ in backward (`9-backpropagation.md`, Step 2).

> **Common misconception:** "the input holds 768 article tokens and then a mask, and the target holds 768 masked positions and then 255 summary tokens." That is only true when the article is exactly 766 tokens long. In general, the input holds the **real** article (whatever its length, up to 766) followed by the **real** summary, with nothing masked out. The article/summary boundary sits at `prefix_len`, which is different for every sample. Masking happens in the attention mask and the loss mask, never in the token IDs.

### 6. Worked example: a real batch with $B = 2$

| | `100001.txt` | `100089.txt` |
| --- | --- | --- |
| Article tokens | 2,046 → cut to 766 | 427 (fits) |
| Prefix `<s> article </s>` | 768 | 429 |
| Summary tokens (+ `</s>`) | 213 + 1 = 214 | 123 + 1 = 124 |
| Real length | 982 | 553 |
| `<pad>` added | 0 | 429 |
| Loss positions | 767 … 980 → 214 | 428 … 551 → 124 |

The batch has $T = 982$ (the longest row, not 1024), so `token_ids` is $(2, 982)$, the attention mask is $(2, 982, 982)$ and the loss mask is $(2, 982)$. It has $N = 214 + 124 = 338$ loss positions, so `targets` is $(338)$ and the LM head produces $(338, V)$ probabilities.

### 7. Inference uses the same layout

`src/inference/greedy_decoder.py` → `greedy_summarize` starts from `<s> article[:766] </s>` with no summary. It builds the same prefix-LM mask, runs the LM head on the last position only, appends the argmax token, and repeats until `</s>` or 256 summary tokens. The model sees exactly the pattern it was trained on, except that the summary tokens are its own predictions instead of the ground truth.

---

## One Training Step in This Project

```text
1. batcher.create_input_tensor()      → token IDs (B, T), attention mask (B, T, T),
                                        loss mask (B, T), targets (N)
2. model.forward(..., output_mask)    → probabilities at the N summary positions (N, V)
3. cross_entropy_loss(...)            → loss (a number) and d_logits (N, V)
4. model.backward(d_logits)           → a gradient for every weight and bias
5. optimizer.step(params, grads)      → Adam updates every weight in place
```

(`src/training/train.py` → `train_step`. Start training with `python -m scripts.train_model`.)

## Total Parameter Count

| Component | Count |
| --- | --- |
| Input embedding $E$ ($30000 \times 512$) | 15,360,000 |
| 6 encoder blocks × 3,150,336 | 18,902,016 |
| LM head ($W_{LM}$ + $b_{LM}$) | 15,390,000 |
| Positional encoding | 0 (fixed, not learned) |
| **Total** | **49,652,016** |
