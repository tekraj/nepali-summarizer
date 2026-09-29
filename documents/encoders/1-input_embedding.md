# Transformer Input Embedding Layer

```
[ Input Token IDs ]  ──>  [ Matrix Lookup (Row Index) ]  ──>  [ Vector Scaling (× √d_model) ]  ──>  [ Scaled Input Embeddings ]
 (e.g., [258, 890])             E[258, :], E[890, :]                 Multiply by √d_model              Ready for Positional Encoding
```

**In one sentence:** every token ID is just a row number; the embedding layer fetches that row from a big table of numbers and multiplies it by $\sqrt{d_{\text{model}}}$.

| | Shape | This project |
| --- | --- | --- |
| Input (token IDs) | $(B, T)$ | $(4, 512)$ integers |
| Embedding matrix $E$ | $(V, d_{\text{model}})$ | $(30000, 512)$ |
| Output | $(B, T, d_{\text{model}})$ | $(4, 512, 512)$ |

> **Code:** `src/models/layers/input_embedding.py` → `InputEmbedding`

---

## 1. Matrix Architecture & Dimensions

The embedding layer consists of a lookup table represented by the 2D matrix $E$ with dimensions $V \times d_{\text{model}}$:

* **$V$ (Vocabulary Size):** The total number of unique tokens created during the Byte-Pair Encoding (BPE) process. Large models use e.g. $50,000$; **our Nepali BPE vocabulary has $V = 30,000$**.
* **$d_{\text{model}}$ (Embedding Dimension):** The width of the Transformer's hidden vector space (e.g., $512$ or $4096$). **We use $d_{\text{model}} = 512$.**

Row $r$ of the matrix is the vector for token ID $r$. Rows are numbered from $0$ to $V-1$ (matching token IDs), columns from $1$ to $d_{\text{model}}$:

$$\text{Embedding Matrix } E = \begin{bmatrix} e_{0,1} & e_{0,2} & \dots & e_{0,d_{\text{model}}} \\ e_{1,1} & e_{1,2} & \dots & e_{1,d_{\text{model}}} \\ \vdots & \vdots & \ddots & \vdots \\ e_{V-1,1} & e_{V-1,2} & \dots & e_{V-1,d_{\text{model}}} \end{bmatrix}$$

Size in this project: $30000 \times 512 = 15,360,000$ learnable numbers.

---

## 2. Initialization & Theoretical Foundations

When the Transformer is instantiated, matrix $E$ is populated with small random numbers drawn from a normal distribution with **mean $0$ and standard deviation $0.02$**:

$$E_{i, j} \sim \mathcal{N}(\mu = 0,\ \sigma = 0.02)$$

> **Notation note:** here the second number is the *standard deviation* $\sigma$ (not the variance $\sigma^2 = 0.0004$). In NumPy: `np.random.normal(0, 0.02, size=(V, d_model))`.
> GPT and BERT use a *truncated* normal (values beyond $\pm 2\sigma$ are re-drawn). Our code uses a plain normal; with $\sigma = 0.02$ the difference is negligible.

### Why $\sigma = 0.02$ is Used

* **Preventing Exploding Gradients:** Deep neural networks multiply these weights across dozens of layers. If starting values are drawn from $\mathcal{N}(0, 1)$, numbers grow exponentially during matrix multiplications, leading to exploding gradients and training instability.
* **Avoiding Softmax Saturation:** Large initial values push output predictions to extreme probabilities (approaching $0$ or $1$). This saturates the softmax activation function, dropping loss gradients to near zero and freezing the learning process. With $\sigma = 0.02$, about $99.7\%$ of values fall within $\pm 3\sigma = [-0.06, 0.06]$, which keeps all predictions neutral and highly responsive to updates.

---

## 3. The Step-by-Step Pipeline

### Step 1: Matrix Lookup

For an input sequence of $T$ token IDs $\mathbf{x} = [x_1, x_2, \dots, x_T]$ where each $x_i \in [0, V-1]$, the layer extracts the corresponding row vectors from $E$:

$$\mathbf{v}_i = E[x_i, :] \in \mathbb{R}^{d_{\text{model}}}$$

This yields an unscaled sequence matrix $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{T \times d_{\text{model}}}$. No multiplication happens here — it is pure indexing (`E[token_ids]` in NumPy), which also works for a whole batch: $(B, T) \rightarrow (B, T, d_{\text{model}})$.

---

### Step 2: Vector Scaling by $\sqrt{d_{\text{model}}}$

Each extracted vector is multiplied by the square root of the embedding dimension:

$$\mathbf{e}_i = \mathbf{v}_i \cdot \sqrt{d_{\text{model}}}$$

> **Why Scaling is Mandatory:**
> The next component in the Transformer pipeline is adding Positional Encodings generated via sine and cosine functions, which strictly output values in the range $[-1, 1]$.
>
> Because raw initial embedding values are tiny (around $\pm 0.02$), adding positional encodings directly would completely drown out the token's content. For $d_{\text{model}} = 512$, $\sqrt{512} \approx 22.63$, so a typical value of $0.02$ becomes $0.02 \times 22.63 \approx 0.45$ — now on the same scale as the positional signal. Positional information can then shift word order without washing out the word's base representation.

---

### Worked Example (tiny numbers)

Use a toy vocabulary of $V = 6$ tokens and $d_{\text{model}} = 4$, so the scale factor is $\sqrt{4} = 2$.

$$E = \begin{bmatrix}
\phantom{-}0.01 & -0.02 & \phantom{-}0.03 & \phantom{-}0.00 \\
\phantom{-}0.00 & \phantom{-}0.00 & \phantom{-}0.00 & \phantom{-}0.00 \\
\phantom{-}0.02 & \phantom{-}0.01 & -0.01 & \phantom{-}0.03 \\
-0.03 & \phantom{-}0.02 & \phantom{-}0.01 & -0.01 \\
\phantom{-}0.01 & \phantom{-}0.01 & \phantom{-}0.02 & -0.02 \\
\phantom{-}0.00 & -0.01 & \phantom{-}0.02 & \phantom{-}0.01
\end{bmatrix}
\begin{matrix} \leftarrow \text{ID } 0 \\ \leftarrow \text{ID } 1 \\ \leftarrow \text{ID } 2 \\ \leftarrow \text{ID } 3 \\ \leftarrow \text{ID } 4 \\ \leftarrow \text{ID } 5 \end{matrix}$$

Input sentence with token IDs $[2, 5, 3]$ ($T = 3$):

| Token ID | Step 1: row $E[x_i]$ | Step 2: $\times 2$ |
| --- | --- | --- |
| 2 | $[0.02,\ 0.01,\ -0.01,\ 0.03]$ | $[0.04,\ 0.02,\ -0.02,\ 0.06]$ |
| 5 | $[0.00,\ -0.01,\ 0.02,\ 0.01]$ | $[0.00,\ -0.02,\ 0.04,\ 0.02]$ |
| 3 | $[-0.03,\ 0.02,\ 0.01,\ -0.01]$ | $[-0.06,\ 0.04,\ 0.02,\ -0.02]$ |

Output shape: $(T, d_{\text{model}}) = (3, 4)$; with a batch dimension, $(1, 3, 4)$. This matrix goes straight into the Positional Encoding step (the same numbers are continued in `2-positional_encoding.md`).

---

### Step 3: Acquiring Meaning via Backpropagation

At step zero, these vectors are merely arbitrary coordinates. Real semantic structure is carved out dynamically during training:

1. **Forward Pass:** The scaled tensor $\mathbf{X}_{\text{scaled}} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ (plus positional encodings) enters the self-attention blocks.
2. **Loss Calculation:** The model predicts target tokens and computes loss $\mathcal{L}$.
3. **Gradient Update:** Backpropagation calculates partial derivatives $\frac{\partial \mathcal{L}}{\partial E}$, and an optimizer (we use Adam) updates the matrix values. In plain gradient descent this would be:

$$E[x_i, :] \leftarrow E[x_i, :] - \eta \cdot \frac{\partial \mathcal{L}}{\partial E[x_i, :]}$$

Only the rows of tokens that appeared in the batch get a non-zero gradient. The exact formula (including the $\sqrt{d_{\text{model}}}$ factor and repeated tokens) is derived in `9-backpropagation.md`.

Over many iterations, tokens appearing in similar contexts receive similar gradient updates. Their vectors align in $d_{\text{model}}$-dimensional space, transforming random numbers into a structured semantic map.
