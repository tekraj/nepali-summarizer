# Multi-Head Self-Attention

**In one sentence:** every token asks a question (Query), every token offers a label (Key) and a payload (Value); each token's new vector is a weighted average of all Values, weighted by how well its Query matches each Key.

```text
X (B, T, d_model) ──► Q, K, V ──► softmax(QKᵀ/√d_k + M) ──► × V ──► concat heads ──► × W_O ──► (B, T, d_model)
```

> **Code:** `src/models/layers/multi_head_attention.py` → `MultiHeadAttention`

---

## Phase 1: Data Ingestion (The Starting State)

The attention mechanism receives its input directly from the Positional Encoding pipeline.

At this exact moment, the input is a 3D tensor $\mathbf{X}$ with the shape **$(B, T, d_{model})$**, where:

* **$B$ (Batch Size):** The number of independent sentences being processed simultaneously.
* **$T$ (Sequence Length):** The maximum number of tokens in the sequence (including padded tokens).
* **$d_{model}$ (Embedding Dimension):** The vector size for each token (e.g., $512$).

It also receives the $(B, T)$ attention mask that marks which tokens are `[PAD]`.

Every 512-dimensional vector inside this tensor contains the combined semantic meaning (from the embedding matrix) and physical location (from the sinusoidal positional encodings) of a single word. However, the words remain completely isolated; they have no context about the other words in their sequence.

(For blocks 2…N, $\mathbf{X}$ is simply the output of the previous encoder block — same shape.)

---

## Phase 2: Single-Head Attention Pipeline

To build context, the Transformer uses the **Scaled Dot-Product Attention** mechanism. This allows every word to mathematically query the sequence, evaluate how much attention to pay to every other word, and update its own representation based on the results.

### Step 1: Initialization of Weight Matrices

The mechanism relies on three learnable linear projection matrices. When the model is built, these are initialized with random values (we use Xavier/Glorot initialization: normal with $\sigma = \sqrt{2 / (\text{fan}_{in} + \text{fan}_{out})}$):

* **$W_Q$ (Query Weights):** Shape $(d_{model}, d_k)$
* **$W_K$ (Key Weights):** Shape $(d_{model}, d_k)$
* **$W_V$ (Value Weights):** Shape $(d_{model}, d_v)$

*(Note: $d_v$ is practically always equal to $d_k$. With a single head, $d_k$ could be the full $512$. With $h$ heads, each head gets $d_k = d_{model}/h$ — that is where the value $64$ comes from; see Phase 3.)*

### Step 2: Creating Q, K, and V

The input tensor $\mathbf{X}$ is multiplied by these three weight matrices to create three distinct contextual components for every token:

1. **Query ($\mathbf{Q} = \mathbf{X} W_Q$):** What the token is looking for in the sequence.
2. **Key ($\mathbf{K} = \mathbf{X} W_K$):** What the token advertises about its own meaning/grammar.
3. **Value ($\mathbf{V} = \mathbf{X} W_V$):** The actual payload of semantic information the token will share if selected.

Each resulting tensor now has the shape $(B, T, d_k)$.

### Step 3: The Scaled Dot-Product Math

The single-head attention formula is executed on these tensors:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

1. **Dot Product (Similarity Scores):** The model calculates $\mathbf{Q}\mathbf{K}^T$. This produces a $(B, T, T)$ grid of raw scores: entry $[t_1, t_2]$ says how strongly the Query of token $t_1$ matches the Key of token $t_2$.
2. **Scaling:** The raw scores are divided by $\sqrt{d_k}$ (e.g., $\sqrt{64} = 8$). Dot products of long vectors produce large numbers that would push softmax into regions with near-zero gradients. Scaling keeps their variance around $1$.
3. **Masking:** The **additive** mask $\mathbf{M}$ is added to the score grid: $0$ for real tokens, $-\infty$ for padded keys (it is built from the binary $1/0$ mask — see `2-positional_encoding.md`, Section 5).
4. **Softmax:** The softmax function is applied across the last dimension (over keys) of the grid. Each row becomes weights that sum to $1.0$. Because $e^{-\infty} = 0$, padded tokens get exactly $0.0$ attention.
5. **Weighted Sum:** The $(B, T, T)$ weight grid is multiplied by the Value tensor $\mathbf{V}$ of shape $(B, T, d_k)$.

**Single-Head Output:** A context-aware tensor of shape $(B, T, d_k)$. Token representations are no longer isolated; they are now weighted averages of the Values of all relevant words in the sequence.

### Worked Example (one head, $T = 3$, $d_k = 2$, token 3 is `[PAD]`)

Suppose the projections have already produced (one row per token):

$$\mathbf{Q} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad
\mathbf{K} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{bmatrix}, \quad
\mathbf{V} = \begin{bmatrix} 1 & 2 \\ 3 & 0 \\ 9 & 9 \end{bmatrix}$$

**1. Scores $\mathbf{Q}\mathbf{K}^T$** (row = query token, column = key token):

$$\mathbf{Q}\mathbf{K}^T = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 2 \\ 1 & 2 & 2 \end{bmatrix}$$

**2. Scale by $\sqrt{2} \approx 1.4142$:**

$$\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{2}} = \begin{bmatrix} 0.7071 & 0.7071 & 0 \\ 0 & 0.7071 & 1.4142 \\ 0.7071 & 1.4142 & 1.4142 \end{bmatrix}$$

**3–4. Mask column 3, then softmax each row:**

$$\text{weights} = \begin{bmatrix} 0.5000 & 0.5000 & 0 \\ 0.3302 & 0.6698 & 0 \\ 0.3302 & 0.6698 & 0 \end{bmatrix}$$

Row 2 would have preferred the `[PAD]` key (score 1.4142), but the mask forces that weight to 0.

**5. Weighted sum with $\mathbf{V}$:**

$$\text{weights} \cdot \mathbf{V} = \begin{bmatrix} 0.5 \cdot [1,2] + 0.5 \cdot [3,0] \\ 0.3302 \cdot [1,2] + 0.6698 \cdot [3,0] \\ \dots \end{bmatrix} = \begin{bmatrix} 2.0000 & 1.0000 \\ 2.3395 & 0.6605 \\ 2.3395 & 0.6605 \end{bmatrix}$$

Notice the `[PAD]` Value $[9, 9]$ never leaks into any output.

---

## Phase 3: Multi-Head Attention (MHA) Pipeline

Language is complex. A word often needs to track multiple relationships simultaneously (e.g., subject-verb agreement, emotional tone, and direct objects). A single head forces the network to average all these distinct needs into one Query. Multi-Head Attention solves this by splitting the vector space so the model can track multiple independent relationships in parallel.

### Step 1: The Dimension Split

Instead of using one massive attention operation, the Transformer configures $h$ independent "heads" (e.g., $h = 8$). To keep the computational cost identical to single-head attention, the model divides the embedding dimension by the number of heads:

$$d_k = \frac{d_{model}}{h} = \frac{512}{8} = 64$$

### Step 2: Unique Weight Initialization

The model initializes 8 entirely distinct sets of weight matrices, each of shape $(512, 64)$:

* **$W_Q^1 \dots W_Q^8$**
* **$W_K^1 \dots W_K^8$**
* **$W_V^1 \dots W_V^8$**

Because each of these 24 matrices is initialized with completely different random values, symmetry is broken. As the network trains, Head 1 might learn to track grammatical structure, while Head 2 explores rhyming or semantic groupings.

> **How the code stores them:** the 8 matrices $W_Q^1 \dots W_Q^8$ (each $512 \times 64$) are placed side by side into one $512 \times 512$ matrix $W_Q$. Columns $0$–$63$ belong to head 1, columns $64$–$127$ to head 2, and so on. The math is identical, but one big matrix multiply replaces 24 small ones.

### Step 3: Parallel Execution

The input tensor $\mathbf{X}$ $(B, T, 512)$ is projected into 8 independent sets of $\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i$ matrices. In code this is done by one projection followed by a reshape:

| Step | Shape | Example ($B = 4$, $T = 512$) |
| --- | --- | --- |
| $\mathbf{X} W_Q$ | $(B, T, 512)$ | $(4, 512, 512)$ |
| split last axis into 8 heads of 64 | $(B, T, 8, 64)$ | $(4, 512, 8, 64)$ |
| move the head axis forward → $\mathbf{Q}$ | $(B, h, T, d_k)$ | $(4, 8, 512, 64)$ |
| scores $\mathbf{Q}\mathbf{K}^T$ | $(B, h, T, T)$ | $(4, 8, 512, 512)$ |
| weights $\times \mathbf{V}$ | $(B, h, T, d_k)$ | $(4, 8, 512, 64)$ |

The Scaled Dot-Product math detailed in Phase 2 runs entirely independently for each of the 8 heads at the same time (the head axis behaves just like an extra batch axis).

**Output of the Parallel Execution:** 8 distinct tensors, each with the shape $(B, T, 64)$ — stored together as one $(B, 8, T, 64)$ tensor.

### Step 4: Concatenation

To synthesize the parallel insights back into a single vector per token, the model concatenates the 8 output tensors side-by-side along the vector feature dimension:

$$\text{Concat}(\text{head}_1, \dots, \text{head}_8)$$

Gluing eight $64$-dimensional vectors together restores the sequence to its original $512$-dimensional width. The resulting tensor shape is $(B, T, 512)$. (In code: the reverse of the reshape above, $(B, 8, T, 64) \rightarrow (B, T, 8, 64) \rightarrow (B, T, 512)$.)

### Step 5: The Final Linear Projection

Simply concatenating the vectors leaves the information segregated (e.g., dimensions 0-63 only contain Head 1's data). To fuse these separated subspaces into a cohesive representation, the concatenated tensor is multiplied by a final, learnable output weight matrix, $W_O$.

* **$W_O$ Shape:** $(h \cdot d_k, d_{model}) = (512, 512)$

$$\mathbf{Output} = \text{Concat}(\text{head}_1, \dots, \text{head}_8) \cdot W_O$$

**Final Block Output:** The Multi-Head Attention mechanism outputs a single tensor of shape **$(B, T, d_{model})$**. This shape matches the tensor that originally entered the block, but its contents are transformed. It is now ready to be passed directly into the Add & Norm residual connection.

### Parameter Count (per encoder block)

| Weight | Shape | Numbers |
| --- | --- | --- |
| $W_Q$ (8 heads combined) | $512 \times 512$ | 262,144 |
| $W_K$ | $512 \times 512$ | 262,144 |
| $W_V$ | $512 \times 512$ | 262,144 |
| $W_O$ | $512 \times 512$ | 262,144 |
| **Total** | | **1,048,576** |

There are no bias vectors in our attention layer.
