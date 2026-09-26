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

### Shapes at Every Stage (this project: $B = 4$, $T = 512$, $d_{model} = 512$, $h = 8$, $d_k = 64$, $d_{ff} = 2048$, $V = 6302$)

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
| LM head | logits / probabilities | $(B, T, V)$ | $(4, 512, 6302)$ |

---

## Stage 1: Input Embedding & Positional Encoding

### 1. Vocabulary Lookup

Given a batch of tokenized, padded sentences of shape $(B, T)$, each token ID is looked up in the **Input Embedding Matrix** $E \in \mathbb{R}^{V \times d_{model}}$ (here $V = 6302$, $d_{model} = 512$).

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

## One Training Step in This Project

```text
1. batcher.create_input_tensor()      → token IDs (B, T), mask (B, T)
2. batcher.create_mlm_inputs()        → replace ~15% of real tokens with <mask>
3. model.forward(...)                 → probabilities at the N masked positions (N, V)
4. cross_entropy_loss(...)            → loss (a number) and d_logits (N, V)
5. model.backward(d_logits)           → a gradient for every weight and bias
6. optimizer.step(params, grads)      → Adam updates every weight in place
```

(`src/training/train.py` → `train_step`. Start training with `python -m scripts.train_model`.)

## Total Parameter Count

| Component | Count |
| --- | --- |
| Input embedding $E$ ($6302 \times 512$) | 3,226,624 |
| 6 encoder blocks × 3,150,336 | 18,902,016 |
| LM head ($W_{LM}$ + $b_{LM}$) | 3,232,926 |
| Positional encoding | 0 (fixed, not learned) |
| **Total** | **25,361,566** |
