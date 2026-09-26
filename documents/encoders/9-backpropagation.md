# Backpropagation: How Every Weight and Bias Gets Updated

Documents 1–8 describe the **forward pass**: token IDs go in, probabilities come out. This document describes the **backward pass**: how the loss tells every single weight and bias in the encoder which way to move, and by how much.

**In one sentence:** starting from the loss, we walk the forward pass in reverse, and at every layer we use the chain rule to compute (a) the gradient for that layer's own parameters and (b) the gradient to hand to the layer below.

> **Code:** every layer has a `backward()` method next to its `forward()` (`src/models/layers/*.py`); `TransformerEncoder.backward` in `src/models/encoder.py` calls them in reverse order; `src/training/loss.py` and `src/training/optimizer.py` hold the loss and Adam.

---

## 0. The Big Picture

### Notation

For any tensor $\mathbf{A}$ in the network, we write

$$d\mathbf{A} \;=\; \frac{\partial \mathcal{L}}{\partial \mathbf{A}}$$

"how much the loss $\mathcal{L}$ changes when each number in $\mathbf{A}$ is nudged up a little". **$d\mathbf{A}$ always has exactly the same shape as $\mathbf{A}$.** That single rule catches most mistakes: $dW_1$ must be $(512, 2048)$ because $W_1$ is $(512, 2048)$.

- A **positive** gradient means "increasing this number increases the loss" → move it **down**.
- A **negative** gradient means "increasing this number decreases the loss" → move it **up**.

### Forward saves, backward uses

During the forward pass every layer stores ("caches") the inputs it will need later — e.g. the FFN keeps its input $\mathbf{x}$ and its ReLU output. The backward pass reads those caches. This is why training uses much more memory than inference.

### The order

```text
FORWARD:   ids → Embedding → +PE → [Block 1] → … → [Block 6] → LM head → softmax → loss

BACKWARD:  loss → softmax+CE → LM head → [Block 6] → … → [Block 1] → PE → Embedding
                                             │
             inside each block (reverse):    ▼
             Add&Norm 2 → FFN → Add&Norm 1 → Multi-Head Attention
```

### Every learnable parameter in the encoder

| # | Parameter | Where | Shape | Count |
| --- | --- | --- | --- | --- |
| 1 | $E$ | Input embedding | $(6302, 512)$ | 3,226,624 |
| 2–5 | $W_Q, W_K, W_V, W_O$ | Attention (×6 blocks) | $(512, 512)$ each | 262,144 each |
| 6–7 | $\gamma_1, \beta_1$ | Add & Norm 1 (×6) | $(512)$ each | 512 each |
| 8–9 | $W_1, b_1$ | FFN (×6) | $(512, 2048)$, $(2048)$ | 1,048,576 + 2,048 |
| 10–11 | $W_2, b_2$ | FFN (×6) | $(2048, 512)$, $(512)$ | 1,048,576 + 512 |
| 12–13 | $\gamma_2, \beta_2$ | Add & Norm 2 (×6) | $(512)$ each | 512 each |
| 14–15 | $W_{LM}, b_{LM}$ | LM head | $(512, 6302)$, $(6302)$ | 3,226,624 + 6,302 |
| | **Total** | | | **25,361,566** |

The positional encoding $PE$ is fixed, so it has **no** gradient to compute. Each of the 6 blocks has its own copy of parameters 2–13, and each copy gets its own gradient.

---

## 1. Three Rules That Cover Everything

### Rule 1 — The chain rule

If $\mathbf{y} = f(\mathbf{x})$ and the loss depends on $\mathbf{y}$, then

$$d\mathbf{x} = d\mathbf{y} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

Each layer only needs to know its **own** local derivative $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$; the incoming $d\mathbf{y}$ already contains everything that happens above it.

### Rule 2 — The linear layer (used 9 times per block + the LM head)

Almost every weight in the Transformer sits in a layer of the form

$$\mathbf{Y} = \mathbf{X} W + b \qquad \mathbf{X}: (\dots, n_{in}),\ W: (n_{in}, n_{out}),\ b: (n_{out}),\ \mathbf{Y}: (\dots, n_{out})$$

Given the incoming gradient $d\mathbf{Y}$:

$$\boxed{\;dW = \mathbf{X}^T d\mathbf{Y} \qquad db = \sum_{\text{rows}} d\mathbf{Y} \qquad d\mathbf{X} = d\mathbf{Y}\, W^T\;}$$

- **Why $\mathbf{X}^T d\mathbf{Y}$?** Weight $W_{jk}$ connects input feature $j$ to output feature $k$. Its effect on the loss is (input $j$) × (gradient at output $k$), added up over every token that used it.
- **Why sum for $db$?** The same bias is added to every token, so its gradient is the sum of all tokens' output gradients.
- **Why $d\mathbf{Y} W^T$?** Each input feature fed into every output through $W$; we send the gradient back along the same connections.

**The weights are shared by every token in every sentence**, so for 3D tensors $(B, T, n)$ we flatten to $(B \cdot T, n)$ first: the weight gradient is summed over all $B \times T$ token positions. (In code: `matmul_weight_grad` in `src/models/layers/functional.py`.)

**Example** — 2 tokens, $n_{in} = n_{out} = 2$:

$$\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & -1 \end{bmatrix},\quad
W = \begin{bmatrix} 0.5 & -1 \\ 2 & 0 \end{bmatrix},\quad
b = [0.1,\ 0] \quad\Rightarrow\quad
\mathbf{Y} = \begin{bmatrix} 4.6 & -1 \\ -0.4 & -3 \end{bmatrix}$$

Suppose the layer above sends back $d\mathbf{Y} = \begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix}$. Then

$$dW = \mathbf{X}^T d\mathbf{Y} = \begin{bmatrix} 1 & 3 \\ 2 & -1 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} -2 & 6 \\ 3 & -2 \end{bmatrix}$$

$$db = [1 + (-1),\ 0 + 2] = [0,\ 2] \qquad
d\mathbf{X} = d\mathbf{Y} W^T = \begin{bmatrix} 0.5 & 2 \\ -2.5 & -2 \end{bmatrix}$$

### Rule 3 — Addition copies, re-use adds up

- **Addition** $\mathbf{c} = \mathbf{a} + \mathbf{b}$: $\frac{\partial \mathbf{c}}{\partial \mathbf{a}} = \frac{\partial \mathbf{c}}{\partial \mathbf{b}} = 1$, so both inputs receive the **same** gradient: $d\mathbf{a} = d\mathbf{b} = d\mathbf{c}$. This is why residual connections are a "gradient highway" — nothing shrinks the gradient on the skip path.
- **Re-use:** if a tensor is used in two places (e.g. $\mathbf{X}_{in}$ goes into attention **and** into the residual), its gradient is the **sum** of the gradients coming back from both places.

---

## 2. Step 1 — Loss and Softmax: $d\mathbf{Z}$

Forward (for the $N$ masked positions, see `7-linearization-and-softmax.md`):

$$\mathbf{P} = \text{softmax}(\mathbf{Z}), \qquad \mathcal{L} = -\frac{1}{N}\sum_{n=1}^{N} \log P_{n,\,\text{target}_n}$$

Differentiating softmax and cross-entropy **together** gives a famously simple result:

$$\boxed{\;d\mathbf{Z} = \frac{1}{N}\big(\mathbf{P} - \mathbf{Y}_{\text{one-hot}}\big)\;} \qquad \text{shape } (N, V)$$

i.e. "predicted probability minus the truth": subtract $1$ at the correct token, leave the others as they are, divide by $N$.

<details>
<summary>Why it simplifies (for one position)</summary>

$\mathcal{L} = -\log P_c = -z_c + \log\sum_j e^{z_j}$ where $c$ is the correct token.
$\frac{\partial \mathcal{L}}{\partial z_i} = -[i = c] + \frac{e^{z_i}}{\sum_j e^{z_j}} = P_i - [i = c]$.
</details>

**Example** (the toy vocabulary from document 7, correct token = "नेपाल", $N = 1$):

| token | $P$ | one-hot $\mathbf{Y}$ | $d\mathbf{Z} = \mathbf{P} - \mathbf{Y}$ |
| --- | --- | --- | --- |
| "नेपाल" ✅ | 0.6381 | 1 | **−0.3619** → push this logit **up** |
| "सुन्दर" | 0.2347 | 0 | 0.2347 → push down |
| "देश" | 0.0954 | 0 | 0.0954 → push down |
| "हो" | 0.0318 | 0 | 0.0318 → push down |

The gradients always sum to $0$: probability taken from wrong tokens is given to the right one. Wrong tokens that got more probability get pushed down harder.

---

## 3. Step 2 — LM Head: $W_{LM}$, $b_{LM}$

Forward: $\mathbf{Z} = \mathbf{X}_N W_{LM} + b_{LM}$ with $\mathbf{X}_N$ the $(N, 512)$ encoder vectors at the masked positions. This is Rule 2:

$$dW_{LM} = \mathbf{X}_N^T\, d\mathbf{Z} \quad (512, V) \qquad
db_{LM} = \sum_{n} d\mathbf{Z}_n \quad (V) \qquad
d\mathbf{X}_N = d\mathbf{Z}\, W_{LM}^T \quad (N, 512)$$

**Example** ($d_{model} = 2$, $V = 4$, one position): $\mathbf{x} = [1,\ -2]$, $d\mathbf{z}$ from Step 1, and
$W_{LM} = \begin{bmatrix} 0.1 & 0.2 & 0 & -0.1 \\ 0.3 & -0.2 & 0.1 & 0 \end{bmatrix}$.

$$dW_{LM} = \mathbf{x}^T d\mathbf{z} = \begin{bmatrix} 1 \\ -2 \end{bmatrix} [-0.3619,\ 0.2347,\ 0.0954,\ 0.0318] = \begin{bmatrix} -0.3619 & 0.2347 & 0.0954 & 0.0318 \\ 0.7239 & -0.4695 & -0.1909 & -0.0635 \end{bmatrix}$$

$$db_{LM} = [-0.3619,\ 0.2347,\ 0.0954,\ 0.0318] \qquad d\mathbf{x} = d\mathbf{z}\, W_{LM}^T = [0.0076,\ -0.1460]$$

**Back to $(B, T, 512)$:** only the $N$ masked positions went through the head, so the code creates a zero tensor of shape $(B, T, 512)$ and writes the $N$ rows of $d\mathbf{X}_N$ into their original positions. Unmasked positions start with gradient $0$ — but they will still receive gradient later through attention, because the masked tokens attended to them.

**Weight tying** (if enabled): $W_{LM} = E^T$, so $dW_{LM}^T$ is added to $dE$ (Step 8) instead of updating a separate matrix.

---

## 4. Step 3 — Second Add & Norm: $\gamma_2$, $\beta_2$

Forward: $\mathbf{X}_{out} = \text{LayerNorm}_2(\mathbf{X}_{norm1} + \mathbf{X}_{ffn\_out})$. We go through it in reverse: first LayerNorm, then the addition.

### LayerNorm backward

Forward, for each token vector $\mathbf{x}$ (length $D = 512$):

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad \mathbf{y} = \gamma \odot \hat{\mathbf{x}} + \beta$$

**Parameter gradients** (summed over all $B \times T$ tokens, since $\gamma, \beta$ are shared):

$$\boxed{\;d\gamma = \sum_{b,t} d\mathbf{y} \odot \hat{\mathbf{x}} \qquad d\beta = \sum_{b,t} d\mathbf{y}\;} \qquad \text{both shape } (512)$$

(Same idea as Rule 2: $\gamma_j$ multiplies $\hat{x}_j$, and $\beta_j$ is a bias.)

**Input gradient.** Let $d\hat{\mathbf{x}} = d\mathbf{y} \odot \gamma$. Because $\mu$ and $\sigma^2$ are computed from all $D$ numbers of the same token, every input affects every output, and the result is:

$$\boxed{\;d\mathbf{x} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}\Big(d\hat{\mathbf{x}} \;-\; \text{mean}(d\hat{\mathbf{x}}) \;-\; \hat{\mathbf{x}} \odot \text{mean}(d\hat{\mathbf{x}} \odot \hat{\mathbf{x}})\Big)\;}$$

Intuition for the two subtracted terms: normalization throws away the vector's **average** and its **overall scale**. So any part of the gradient that would only shift the average (first term) or only stretch the scale (second term) has no effect after normalization and is removed. A consequence you can check: the entries of $d\mathbf{x}$ always sum to $0$.

**Example** — the token from `4-add_and_normalization.md`: $\mathbf{x} = [1.5,\ 1.0,\ 4.0,\ 5.5]$, $\sqrt{\sigma^2 + \epsilon} = 1.8371$, $\hat{\mathbf{x}} = [-0.8165,\ -1.0887,\ 0.5443,\ 1.3608]$, $\gamma = 1$, $\beta = 0$. Incoming gradient $d\mathbf{y} = [1,\ 0,\ 0,\ -1]$.

1. $d\gamma = d\mathbf{y} \odot \hat{\mathbf{x}} = [-0.8165,\ 0,\ 0,\ -1.3608]$
2. $d\beta = d\mathbf{y} = [1,\ 0,\ 0,\ -1]$
3. $d\hat{\mathbf{x}} = d\mathbf{y} \odot \gamma = [1,\ 0,\ 0,\ -1]$; $\ \text{mean}(d\hat{\mathbf{x}}) = 0$
4. $d\hat{\mathbf{x}} \odot \hat{\mathbf{x}} = [-0.8165,\ 0,\ 0,\ -1.3608]$; $\ \text{mean} = -0.5443$
5. $d\mathbf{x} = \frac{1}{1.8371}\big([1, 0, 0, -1] - 0 + 0.5443 \cdot \hat{\mathbf{x}}\big) = \frac{1}{1.8371}[0.5556,\ -0.5926,\ 0.2963,\ -0.2593]$

$$d\mathbf{x} = [0.3024,\ -0.3226,\ 0.1613,\ -0.1411] \qquad (\text{sum} = 0\ ✓)$$

### The addition

$\mathbf{X}_{res2} = \mathbf{X}_{norm1} + \mathbf{X}_{ffn\_out}$, so by Rule 3 **both** $\mathbf{X}_{norm1}$ (skip path) and $\mathbf{X}_{ffn\_out}$ (FFN path) receive the same gradient $d\mathbf{X}_{res2}$.

---

## 5. Step 4 — Feed-Forward Network: $W_2$, $b_2$, $W_1$, $b_1$

Forward: $\mathbf{h} = \mathbf{x} W_1 + b_1$, $\ \mathbf{a} = \text{ReLU}(\mathbf{h})$, $\ \mathbf{y} = \mathbf{a} W_2 + b_2$. Backward, top to bottom:

| Step | Formula | Shape |
| --- | --- | --- |
| Linear 2 (Rule 2) | $dW_2 = \mathbf{a}^T d\mathbf{y}$, $\ db_2 = \sum d\mathbf{y}$, $\ d\mathbf{a} = d\mathbf{y}\, W_2^T$ | $(2048, 512)$, $(512)$, $(B,T,2048)$ |
| ReLU | $d\mathbf{h} = d\mathbf{a} \odot [\mathbf{h} > 0]$ | $(B, T, 2048)$ |
| Linear 1 (Rule 2) | $dW_1 = \mathbf{x}^T d\mathbf{h}$, $\ db_1 = \sum d\mathbf{h}$, $\ d\mathbf{x} = d\mathbf{h}\, W_1^T$ | $(512, 2048)$, $(2048)$, $(B,T,512)$ |

**ReLU's derivative** is $1$ where the input was positive and $0$ elsewhere: units that were switched off in the forward pass pass **no** gradient back and their incoming weights are not updated for that token.

**Example** — the FFN from `5-feed_forward.md` ($\mathbf{x} = [1, -1]$, $\mathbf{h} = [1, -2, 1, 0.5]$, $\mathbf{a} = [1, 0, 1, 0.5]$), with incoming gradient $d\mathbf{y} = [1,\ -1]$:

1. $dW_2 = \mathbf{a}^T d\mathbf{y} = \begin{bmatrix} 1 \\ 0 \\ 1 \\ 0.5 \end{bmatrix}[1,\ -1] = \begin{bmatrix} 1 & -1 \\ 0 & 0 \\ 1 & -1 \\ 0.5 & -0.5 \end{bmatrix}$, $\quad db_2 = [1,\ -1]$
2. $d\mathbf{a} = d\mathbf{y}\, W_2^T = [1,\ -1,\ 0,\ -3]$
3. ReLU mask $[\mathbf{h} > 0] = [1, 0, 1, 1]$ → $d\mathbf{h} = [1,\ 0,\ 0,\ -3]$ (unit 2 was off, so its gradient is blocked)
4. $dW_1 = \mathbf{x}^T d\mathbf{h} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}[1,\ 0,\ 0,\ -3] = \begin{bmatrix} 1 & 0 & 0 & -3 \\ -1 & 0 & 0 & 3 \end{bmatrix}$, $\quad db_1 = [1,\ 0,\ 0,\ -3]$
5. $d\mathbf{x} = d\mathbf{h}\, W_1^T = [1\cdot 1 + (-3)\cdot 2,\ \ 1 \cdot 0 + (-3) \cdot 1] = [-5,\ -3]$

---

## 6. Step 5 — First Add & Norm: $\gamma_1$, $\beta_1$

$\mathbf{X}_{norm1}$ was used **twice** in the forward pass: as the FFN input and on the skip path into Add & Norm 2. By Rule 3 its total gradient is the sum:

$$d\mathbf{X}_{norm1} = \underbrace{d\mathbf{X}_{res2}}_{\text{skip path}} + \underbrace{d\mathbf{x}_{\text{FFN}}}_{\text{through the FFN}}$$

Then exactly the same LayerNorm backward as Step 3 gives $d\gamma_1$, $d\beta_1$ and $d\mathbf{X}_{res1}$, and the addition $\mathbf{X}_{res1} = \mathbf{X}_{in} + \mathbf{X}_{attn}$ copies $d\mathbf{X}_{res1}$ to both $\mathbf{X}_{in}$ (skip) and $\mathbf{X}_{attn}$ (attention output).

---

## 7. Step 6 — Multi-Head Attention: $W_O$, $W_Q$, $W_K$, $W_V$

Forward recap (per head, with $\mathbf{P}$ the attention weights):

$$\mathbf{Q} = \mathbf{X}W_Q,\ \mathbf{K} = \mathbf{X}W_K,\ \mathbf{V} = \mathbf{X}W_V,\quad
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M},\quad
\mathbf{P} = \text{softmax}(\mathbf{S}),\quad
\mathbf{A} = \mathbf{P}\mathbf{V},\quad
\text{out} = \text{Concat}(\mathbf{A})\, W_O$$

Backward, one line per forward operation, in reverse:

| # | Undo | Gradient | Shape |
| --- | --- | --- | --- |
| a | output projection (Rule 2) | $dW_O = \text{Concat}^T d\text{out}$, $\quad d\text{Concat} = d\text{out}\, W_O^T$ | $(512, 512)$, $(B,T,512)$ |
| b | concatenation | split $d\text{Concat}$ back into heads → $d\mathbf{A}$ | $(B, h, T, d_k)$ |
| c | $\mathbf{A} = \mathbf{P}\mathbf{V}$ | $d\mathbf{P} = d\mathbf{A}\, \mathbf{V}^T$, $\quad d\mathbf{V} = \mathbf{P}^T d\mathbf{A}$ | $(B,h,T,T)$, $(B,h,T,d_k)$ |
| d | softmax (per row) | $d\mathbf{S} = \mathbf{P} \odot \big(d\mathbf{P} - \textstyle\sum_j (d\mathbf{P} \odot \mathbf{P})_j\big)$ | $(B, h, T, T)$ |
| e | scaling by $\frac{1}{\sqrt{d_k}}$ | $d\mathbf{S} \leftarrow d\mathbf{S} / \sqrt{d_k}$ | $(B, h, T, T)$ |
| f | $\mathbf{Q}\mathbf{K}^T$ | $d\mathbf{Q} = d\mathbf{S}\, \mathbf{K}$, $\quad d\mathbf{K} = d\mathbf{S}^T \mathbf{Q}$ | $(B, h, T, d_k)$ each |
| g | split into heads | merge $d\mathbf{Q}, d\mathbf{K}, d\mathbf{V}$ back to $(B, T, 512)$ | $(B, T, 512)$ |
| h | input projections (Rule 2) | $dW_Q = \mathbf{X}^T d\mathbf{Q}$, $\ dW_K = \mathbf{X}^T d\mathbf{K}$, $\ dW_V = \mathbf{X}^T d\mathbf{V}$ | $(512, 512)$ each |
| i | $\mathbf{X}$ used 3 times (Rule 3) | $d\mathbf{X} = d\mathbf{Q} W_Q^T + d\mathbf{K} W_K^T + d\mathbf{V} W_V^T$ | $(B, T, 512)$ |

**The mask needs no special backward code.** Masked positions have $P = 0$, and row (d) multiplies by $\mathbf{P}$, so their $d\mathbf{S}$ is exactly $0$: `[PAD]` keys receive no gradient.

**Softmax backward, tiny example:** $\mathbf{P} = [0.5,\ 0.5,\ 0]$, $d\mathbf{P} = [1,\ 0,\ 2]$. Then $\sum_j P_j\, dP_j = 0.5$ and

$$d\mathbf{S} = [0.5 \cdot (1 - 0.5),\ \ 0.5 \cdot (0 - 0.5),\ \ 0 \cdot (2 - 0.5)] = [0.25,\ -0.25,\ 0]$$

**Full example** — the attention head from `3-multi-head-attention.md` ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$ and weights $\mathbf{P}$ as computed there; token 3 is `[PAD]`). Suppose only token 1's output gets a gradient, $d\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}$:

1. $d\mathbf{V} = \mathbf{P}^T d\mathbf{A}$: token 1 used $50\%$ of $\mathbf{V}_1$ and $50\%$ of $\mathbf{V}_2$ →
   $d\mathbf{V} = \begin{bmatrix} 0.5 & 0 \\ 0.5 & 0 \\ 0 & 0 \end{bmatrix}$ (the `[PAD]` Value gets nothing)
2. $d\mathbf{P}$ (row 1) $= [1, 0] \cdot \mathbf{V}^T = [1,\ 3,\ 9]$
3. Softmax backward: $\sum_j P_j\, dP_j = 0.5 \cdot 1 + 0.5 \cdot 3 + 0 \cdot 9 = 2$ →
   $d\mathbf{S}$ (row 1) $= [0.5(1 - 2),\ 0.5(3 - 2),\ 0 \cdot (9 - 2)] = [-0.5,\ 0.5,\ 0]$
4. Divide by $\sqrt{2}$: $[-0.3536,\ 0.3536,\ 0]$
5. $d\mathbf{Q}$ (row 1) $= -0.3536 \cdot \mathbf{K}_1 + 0.3536 \cdot \mathbf{K}_2 = -0.3536 \cdot [1, 0] + 0.3536 \cdot [1, 1] = [0,\ 0.3536]$
6. $d\mathbf{K} = d\mathbf{S}^T \mathbf{Q}$: $\ d\mathbf{K}_1 = [-0.3536,\ 0]$, $\ d\mathbf{K}_2 = [0.3536,\ 0]$, $\ d\mathbf{K}_3 = [0,\ 0]$

Reading it: $d\mathbf{A} = [1, 0]$ means "the loss goes up when the first number of token 1's output goes up". Key 2's Value has a larger first number ($3$) than key 1's ($1$), so gradient descent — which moves *against* $d\mathbf{S} = [-0.35,\ +0.35]$ — raises the score for key 1 and lowers it for key 2, shifting token 1's attention toward the smaller Value. Even though the `[PAD]` Value $[9, 9]$ has the largest $dP$, it gets **zero** gradient because of the mask.

---

## 8. Step 7 — Through the Stack of Blocks

At the bottom of each block, $\mathbf{X}_{in}$ was used twice (attention input and skip path), so:

$$d\mathbf{X}_{in} = \underbrace{d\mathbf{X}_{res1}}_{\text{skip path}} + \underbrace{d\mathbf{X}_{\text{attention}}}_{\text{Step 6 (i)}}$$

This $d\mathbf{X}_{in}$ of Block $k$ **is** the $d\mathbf{X}_{out}$ of Block $k - 1$. The code simply loops over the blocks in reverse:

```python
for block in reversed(self.blocks):
    d_hidden = block.backward(d_hidden)
```

Thanks to the skip paths, part of the gradient reaches Block 1 having passed through only additions and LayerNorms — never through a long chain of weight matrices. This is why the Transformer's early layers and embeddings keep learning even when the network is deep.

---

## 9. Step 8 — Positional Encoding and Input Embedding: $E$

**Positional encoding:** $\mathbf{X}_0 = \mathbf{X}_{emb} + PE$. $PE$ is a constant, so by Rule 3 the gradient passes through unchanged: $d\mathbf{X}_{emb} = d\mathbf{X}_0$. Nothing to update.

**Embedding:** forward was $\mathbf{X}_{emb}[b, t] = E[\text{id}_{b,t}] \cdot \sqrt{d_{model}}$ — just picking rows and scaling. So each position sends its gradient (times $\sqrt{d_{model}}$) back to **the row it came from**:

$$\boxed{\;dE[r] = \sqrt{d_{model}} \sum_{(b,t)\,:\ \text{id}_{b,t} = r} d\mathbf{X}_0[b, t]\;}$$

- Tokens **not** in the batch: gradient row is all zeros.
- A token that appears **several times**: its gradients are **added** (the code uses `np.add.at`, which accumulates repeats correctly).

**Example** ($d_{model} = 4$, so $\sqrt{4} = 2$; vocabulary of 6; sentence IDs $[2, 5, 2]$ — token 2 appears twice):

| position | ID | $d\mathbf{X}_0$ |
| --- | --- | --- |
| 0 | 2 | $[0.1,\ 0,\ -0.2,\ 0.3]$ |
| 1 | 5 | $[0,\ 0.5,\ 0,\ 0]$ |
| 2 | 2 | $[0.2,\ 0.1,\ 0,\ -0.1]$ |

$$dE[2] = 2 \times \big([0.1, 0, -0.2, 0.3] + [0.2, 0.1, 0, -0.1]\big) = [0.6,\ 0.2,\ -0.4,\ 0.4]$$
$$dE[5] = 2 \times [0, 0.5, 0, 0] = [0,\ 1,\ 0,\ 0]$$
$$dE[0] = dE[1] = dE[3] = dE[4] = [0,\ 0,\ 0,\ 0]$$

**Weight tying:** if $W_{LM} = E^T$, then add $dW_{LM}^T$ (from Step 2) to $dE$ — the same matrix gets gradient from both ends of the network.

---

## 10. Updating the Parameters

After `model.backward()` every parameter $W$ has a gradient $dW$ of the same shape.

### Plain gradient descent (the idea)

$$W \leftarrow W - \eta \cdot dW \qquad (\eta = \text{learning rate}, \text{ e.g. } 10^{-4})$$

### Adam (what this project uses)

Plain gradient descent uses one step size for every weight. **Adam** keeps two running averages **for each individual number** in every parameter tensor (so $m$ and $v$ have the same shape as $W$):

$$m \leftarrow \beta_1 m + (1 - \beta_1)\, dW \qquad \text{(average gradient — "momentum", } \beta_1 = 0.9\text{)}$$
$$v \leftarrow \beta_2 v + (1 - \beta_2)\, dW^2 \qquad \text{(average squared gradient, } \beta_2 = 0.999\text{)}$$

Because $m$ and $v$ start at $0$, they are too small in the first steps; **bias correction** fixes that ($t$ = step number):

$$\hat{m} = \frac{m}{1 - \beta_1^t} \qquad \hat{v} = \frac{v}{1 - \beta_2^t}$$

$$\boxed{\;W \leftarrow W - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}\;} \qquad (\epsilon = 10^{-8})$$

$\hat{m} / \sqrt{\hat{v}}$ is roughly "average direction ÷ typical size", so each weight moves by about $\eta$ per step regardless of whether its raw gradients are tiny or huge. Weights whose gradient keeps flipping sign get smaller steps.

**Example** — one weight $w = 0.5$, $\eta = 0.001$:

| step $t$ | gradient $dw$ | $m$ | $v$ | $\hat{m}$ | $\hat{v}$ | step $= \eta\, \hat{m} / \sqrt{\hat{v}}$ | new $w$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2 | 0.02 | 0.00004 | 0.2 | 0.04 | 0.001 | **0.499** |
| 2 | −0.1 | 0.008 | 0.00004996 | 0.042105 | 0.024992 | 0.000266 | **0.498734** |

Step 1: $m = 0.1 \cdot 0.2 = 0.02$, $v = 0.001 \cdot 0.04 = 0.00004$; corrected $\hat{m} = 0.02 / 0.1 = 0.2$, $\hat{v} = 0.00004 / 0.001 = 0.04$; step $= 0.001 \cdot 0.2 / 0.2 = 0.001$.
Step 2: the gradient flipped sign, so momentum partly cancels and the step shrinks to $0.000266$ — Adam is cautious when the direction is uncertain.

### Gradient clipping (safety belt)

Before the Adam update, the code measures the size of **all** gradients together (global norm):

$$\|g\| = \sqrt{\sum_{\text{all parameters}} \sum_{\text{all entries}} dW^2}$$

If $\|g\| > 1.0$ (`max_grad_norm`), every gradient is multiplied by $1.0 / \|g\|$. Example: two gradients $[3]$ and $[4]$ have norm $\sqrt{9 + 16} = 5$, so they become $[0.6]$ and $[0.8]$ — same direction, safe size. This prevents one bad batch from wrecking the weights.

---

## 11. Summary: Every Gradient in One Table

| Parameter | Gradient formula | Code |
| --- | --- | --- |
| $W_{LM}$, $b_{LM}$ | $\mathbf{X}_N^T d\mathbf{Z}$, $\ \sum d\mathbf{Z}$, with $d\mathbf{Z} = (\mathbf{P} - \mathbf{Y}) / N$ | `LanguageModelingHead.backward` |
| $\gamma_2$, $\beta_2$ | $\sum d\mathbf{y} \odot \hat{\mathbf{x}}$, $\ \sum d\mathbf{y}$ | `LayerNormalization.backward` |
| $W_2$, $b_2$ | $\mathbf{a}^T d\mathbf{y}$, $\ \sum d\mathbf{y}$ | `PositionwiseFeedForward.backward` |
| $W_1$, $b_1$ | $\mathbf{x}^T d\mathbf{h}$, $\ \sum d\mathbf{h}$, with $d\mathbf{h} = d\mathbf{a} \odot [\mathbf{h} > 0]$ | `PositionwiseFeedForward.backward` |
| $\gamma_1$, $\beta_1$ | $\sum d\mathbf{y} \odot \hat{\mathbf{x}}$, $\ \sum d\mathbf{y}$ | `LayerNormalization.backward` |
| $W_O$ | $\text{Concat}^T d\text{out}$ | `MultiHeadAttention.backward` |
| $W_Q$, $W_K$, $W_V$ | $\mathbf{X}^T d\mathbf{Q}$, $\ \mathbf{X}^T d\mathbf{K}$, $\ \mathbf{X}^T d\mathbf{V}$ | `MultiHeadAttention.backward` |
| $E$ | $\sqrt{d_{model}} \cdot$ (sum of $d\mathbf{X}_0$ rows per token ID) | `InputEmbedding.backward` |
| $PE$ | — (fixed, no gradient) | `PositionalEncoding.backward` |

**Sums** ($\sum$) run over every token position in the batch, because the same parameter is used for every token.

---

## 12. How We Know the Formulas Are Right: Gradient Checking

Every formula above can be tested numerically. Pick one weight $w$, nudge it by a tiny $h$ (e.g. $10^{-5}$), and measure the loss on both sides:

$$\frac{\partial \mathcal{L}}{\partial w} \approx \frac{\mathcal{L}(w + h) - \mathcal{L}(w - h)}{2h}$$

If the backward pass is correct, this number matches the computed $dw$. This check was run on a tiny version of this encoder (every parameter tensor, with and without weight tying); the largest relative difference was about $2 \times 10^{-7}$, and every worked example in this document was verified the same way.
