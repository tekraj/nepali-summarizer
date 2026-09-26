# Add & Norm (after Multi-Head Attention)

**In one sentence:** add the attention block's input back to its output ("Add"), then rescale every token vector to mean $0$ and variance $1$ ("Norm").

```text
X_input (B, T, d_model) ─┬──► Multi-Head Attention ──► X_attn ─┐
                         │                                     ▼
                         └──────────────────────────────────► ( + ) ──► LayerNorm ──► (B, T, d_model)
```

This is the **Post-LN** arrangement from the original Transformer paper (normalize *after* adding), which is what this project uses.

> **Code:** `src/models/layers/add_and_norm.py` → `ResidualConnection` (Add), `LayerNormalization` (Norm), `AddNormBlock` (both together)

---

## Phase 1: The "Add" Component (Residual Connections)

The "Add" step is a mathematical shortcut known as a residual connection (or skip connection). It takes the original tensor that entered the Multi-Head Attention block and adds it directly to the tensor that just came out of it.

If we denote the original input tensor as $\mathbf{X}_{input}$ and the output of the attention mechanism as $\text{MHA}(\mathbf{X}_{input})$, the operation is strict element-wise addition:

$$\mathbf{X}_{res} = \mathbf{X}_{input} + \text{MHA}(\mathbf{X}_{input})$$

Both tensors have shape $(B, T, d_{model})$, so the addition is position-by-position, number-by-number. No weights are involved.

### Why Residual Connections are Critical

1. **Solving the Vanishing Gradient Problem:** In deep neural networks, multiplying gradients across dozens of layers during backpropagation causes the numbers to shrink towards zero, freezing the early layers (like the embedding layer) from learning. Residual connections act as an unimpeded highway: the derivative of $a + b$ with respect to $a$ is exactly $1$, so gradients flow backwards through the addition without shrinking (see `9-backpropagation.md`).
2. **Preserving Original Identity:** The Multi-Head Attention block aggressively remixes vectors to inject context. While context is crucial, a word must not completely "forget" what it actually is. Adding the input vector back into the contextualized vector guarantees that the token's core semantic identity and positional location are preserved.

---

## Phase 2: The "Norm" Component (Layer Normalization)

Once the tensors are added together, the resulting values can become large or unstable. To standardize these values, the Transformer applies Layer Normalization.

### Why Layer Normalization Over Batch Normalization?

In computer vision, networks use Batch Normalization, which calculates the mean and variance across an entire batch of images. This fails in NLP because sentences have highly variable sequence lengths and rely heavily on `[PAD]` tokens.

Layer Normalization ignores the batch and sequence length entirely. Instead, it calculates the mean and variance independently for **every single token** across its own vector dimensions ($d_{model}$).

```text
Batch Norm:  statistics taken DOWN the batch (same feature, many examples)
Layer Norm:  statistics taken ACROSS one token's own 512 features   ◄── used here
```

### The Mathematical Formulation

For every single token vector $\mathbf{x} \in \mathbb{R}^{d_{model}}$ inside the sequence, the layer computes:

1. **Mean ($\mu$):** The average value of the $512$ numbers inside that specific token's vector.

$$\mu = \frac{1}{d_{model}} \sum_{j=1}^{d_{model}} x_j$$

2. **Variance ($\sigma^2$):** How spread out those $512$ numbers are from the mean.

$$\sigma^2 = \frac{1}{d_{model}} \sum_{j=1}^{d_{model}} (x_j - \mu)^2$$

3. **Normalization & Scaling:** The vector is normalized and then scaled/shifted using two learnable parameter vectors, $\gamma$ (scale) and $\beta$ (shift), each of shape $(d_{model})$ and optimized during training.

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

*(Notes: $\epsilon$ is a tiny number like $10^{-5}$ that prevents division by zero. $\odot$ means element-wise multiplication. At initialization $\gamma = [1, 1, \dots, 1]$ and $\beta = [0, 0, \dots, 0]$, so the layer starts as a pure normalization; training can later learn to rescale or shift individual features.)*

---

## Worked Example ($d_{model} = 4$, one token)

| | dim 1 | dim 2 | dim 3 | dim 4 |
| --- | --- | --- | --- | --- |
| $\mathbf{X}_{input}$ | 1.0 | 2.0 | 3.0 | 4.0 |
| $\mathbf{X}_{attn}$ (MHA output) | 0.5 | −1.0 | 1.0 | 1.5 |
| **Add:** $\mathbf{X}_{res}$ | **1.5** | **1.0** | **4.0** | **5.5** |

**Norm:**

1. Mean: $\mu = (1.5 + 1.0 + 4.0 + 5.5) / 4 = 12 / 4 = 3.0$
2. Variance: $\sigma^2 = \big((-1.5)^2 + (-2.0)^2 + 1.0^2 + 2.5^2\big) / 4 = (2.25 + 4 + 1 + 6.25) / 4 = 3.375$
3. Standard deviation: $\sqrt{3.375 + 0.00001} \approx 1.8371$
4. Normalize: $(\mathbf{x} - 3.0) / 1.8371$

| | dim 1 | dim 2 | dim 3 | dim 4 |
| --- | --- | --- | --- | --- |
| $\mathbf{x} - \mu$ | −1.5 | −2.0 | 1.0 | 2.5 |
| $\hat{\mathbf{x}} = (\mathbf{x} - \mu) / 1.8371$ | −0.8165 | −1.0887 | 0.5443 | 1.3608 |
| $\gamma \odot \hat{\mathbf{x}} + \beta$ (with $\gamma = 1, \beta = 0$) | −0.8165 | −1.0887 | 0.5443 | 1.3608 |

Check: the output has mean $0$ and variance $1$. The *pattern* (dim 4 largest, dim 2 smallest) is kept; only the scale is standardized.

---

## Phase 3: The Complete Add & Norm Pipeline

When the $(B, T, d_{model})$ tensor exits the Multi-Head Attention block, the Add & Norm sequence executes:

1. **The Inputs:** Two tensors are held in memory:
    * $\mathbf{X}_{input}$: The $(B, T, 512)$ tensor that entered this encoder block. For **block 1** this is Word Embeddings + Positional Encodings ($\mathbf{X}_0$); for **later blocks** it is the output of the previous block.
    * $\mathbf{X}_{attn}$: The new $(B, T, 512)$ tensor produced by the Multi-Head Attention block.

2. **Element-Wise Addition:** They are added index by index. If the 3rd word in the sentence has a value of $0.44$ at dimension $12$ in $\mathbf{X}_{input}$, and a value of $1.12$ in $\mathbf{X}_{attn}$, the new value is $1.56$. The resulting tensor $\mathbf{X}_{res}$ keeps the exact shape $(B, T, 512)$.
3. **Token-by-Token Normalization:** For every token in the batch ($B \times T$ tokens), the algorithm scans its $512$ features, calculates the mean and variance, centers the values around $0$, and applies the learnable $\gamma$ and $\beta$ weights.
4. **Final Output:** The block outputs a stable, contextualized, and normalized 3D tensor of shape **$(B, T, d_{model})$**, called $\mathbf{X}_{norm1}$. It becomes the input of the Feed-Forward Network.

This entire Add & Norm sequence acts as a stabilizing wrapper. In the standard Transformer architecture, this wrapper is used twice per encoder block: once immediately after the Multi-Head Attention (this document), and a second time immediately after the Feed-Forward Network (`6-second-add-and-norm.md`). The two uses have **separate** $\gamma, \beta$ parameters.

**Learnable parameters here:** $\gamma_1$ (512) + $\beta_1$ (512) = 1,024 numbers per block.
