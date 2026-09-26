# Position-wise Feed-Forward Network (FFN)

The Position-wise Feed-Forward Network (FFN) sits immediately after the first Add & Norm layer in the Transformer block. While the Multi-Head Attention layer is responsible for routing information *between* different words in a sentence, the FFN is responsible for processing each word's new contextualized vector *individually* to extract deeper semantic features.

In the attention layer, tokens "talk" to each other. In the feed-forward layer, tokens process what they just learned in complete isolation.

```text
X_norm1 (B, T, 512) ──× W1 + b1──► (B, T, 2048) ──ReLU──► (B, T, 2048) ──× W2 + b2──► (B, T, 512)
                        expand                      gate                    compress
```

> **Code:** `src/models/layers/feed_forward.py` → `PositionwiseFeedForward`

## 1. The Core Architecture

The FFN consists of two standard linear transformations (dense layers) with a non-linear activation function placed between them. It is applied to each position (each token) separately and identically — the **same** $W_1, b_1, W_2, b_2$ are used for every token in every sentence. That is what "position-wise" means.

The mathematical formulation for a single token vector $\mathbf{x}$ (a row of length $d_{model}$) is:

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + b_1)W_2 + b_2$$

* **$W_1$ and $b_1$:** The weights and biases for the first linear layer.
* **$\max(0, z)$:** The ReLU (Rectified Linear Unit) activation function, applied to each number separately. Modern models often swap this for GELU, but the architectural principle remains identical.
* **$W_2$ and $b_2$:** The weights and biases for the second linear layer.

**Initialization in this project:** $W_1, W_2 \sim \mathcal{N}(0,\ \sigma = 0.02)$; $b_1, b_2$ start at zero.

## 2. The Expansion and Compression Strategy

The Transformer does not keep the vector dimension static through this network. It uses an expansion-then-compression strategy to give the network mathematical space to learn complex feature representations.

If the input vector dimension is $d_{model}$ (e.g., 512), the inner layer dimension, denoted as $d_{ff}$, is typically scaled up by a factor of 4 (e.g., $d_{ff} = 2048$).

### Step 1: The Up-Projection ($W_1$)

The contextualized $(B, T, d_{model})$ tensor enters the FFN. Every single word vector is multiplied by $W_1$ and $b_1$ is added.

* **$W_1$ Shape:** $(d_{model}, d_{ff}) \rightarrow (512, 2048)$; **$b_1$ Shape:** $(2048)$
* **Tensor Shape Output:** $(B, T, 2048)$
* **Purpose:** The network projects the 512-dimensional vector into a much larger 2048-dimensional space. This allows the model to disentangle complex combinations of features that were mixed together during the attention phase.

### Step 2: The Non-Linearity (ReLU)

The network applies the ReLU function to the $(B, T, 2048)$ tensor, converting any negative number to $0$ and leaving positive numbers untouched.

* **Purpose:** Without this non-linear function, the two linear layers ($W_1$ and $W_2$) would mathematically collapse into a single linear layer ($\mathbf{x} W_1 W_2 = \mathbf{x} W'$), stripping the network of its deep-learning capabilities. ReLU acts as a gate, discarding irrelevant features (turning them to zero) and allowing strong signals to pass through.

### Step 3: The Down-Projection ($W_2$)

The activated $(B, T, 2048)$ tensor is multiplied by $W_2$ and $b_2$ is added.

* **$W_2$ Shape:** $(d_{ff}, d_{model}) \rightarrow (2048, 512)$; **$b_2$ Shape:** $(512)$
* **Tensor Shape Output:** $(B, T, 512)$
* **Purpose:** The network takes the filtered information from the 2048-dimensional space and compresses it back down into the standard 512-dimensional space required by the rest of the Transformer.

## 3. Worked Example (one token, $d_{model} = 2$, $d_{ff} = 4$)

(Here the expansion is ×2 instead of ×4 just to keep the numbers small.)

$$\mathbf{x} = [1,\ -1], \quad
W_1 = \begin{bmatrix} 1 & -1 & 0.5 & 2 \\ 0 & 1 & -0.5 & 1 \end{bmatrix}, \quad
b_1 = [0,\ 0,\ 0,\ -0.5]$$

$$W_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ -1 & 2 \end{bmatrix}, \quad
b_2 = [0.1,\ 0.1]$$

**Step 1 — expand:** each hidden unit $k$ is $x_1 W_1[1,k] + x_2 W_1[2,k] + b_1[k]$:

| unit | calculation | hidden |
| --- | --- | --- |
| 1 | $1 \cdot 1 + (-1) \cdot 0 + 0$ | $1.0$ |
| 2 | $1 \cdot (-1) + (-1) \cdot 1 + 0$ | $-2.0$ |
| 3 | $1 \cdot 0.5 + (-1) \cdot (-0.5) + 0$ | $1.0$ |
| 4 | $1 \cdot 2 + (-1) \cdot 1 - 0.5$ | $0.5$ |

**Step 2 — ReLU:** $[1.0,\ -2.0,\ 1.0,\ 0.5] \rightarrow [1.0,\ \mathbf{0.0},\ 1.0,\ 0.5]$ (unit 2 is switched off)

**Step 3 — compress:** $[1, 0, 1, 0.5] \cdot W_2 + b_2$

* output 1: $1 \cdot 1 + 0 \cdot 0 + 1 \cdot 1 + 0.5 \cdot (-1) + 0.1 = 1.6$
* output 2: $1 \cdot 0 + 0 \cdot 1 + 1 \cdot 1 + 0.5 \cdot 2 + 0.1 = 2.1$

$$\text{FFN}([1, -1]) = [1.6,\ 2.1]$$

Same width in and out (2 → 4 → 2), just like $512 \rightarrow 2048 \rightarrow 512$ in the real model.

## 4. The Final Tensor Flow

Tracking a batched tensor through this specific block:

1. **Input:** A stable, contextualized tensor arrives from the first Add & Norm layer: **$(B, T, 512)$**.
2. **Linear 1 ($W_1$, $b_1$):** Expands the tensor to **$(B, T, 2048)$**.
3. **Activation:** Applies ReLU independently to every number in the tensor. The shape remains **$(B, T, 2048)$**.
4. **Linear 2 ($W_2$, $b_2$):** Compresses the tensor back to **$(B, T, 512)$**.
5. **Output:** The FFN outputs a transformed tensor of shape **$(B, T, 512)$**.

Once the tensor exits the Feed-Forward Network, it is immediately routed into a **second Add & Norm layer**. The original $(B, T, 512)$ input to the FFN is added to the FFN's output (Residual Connection), and the result is normalized. This completes one entire Encoder block. This block is then repeated sequentially to build deeper representations — 12 times in BERT-base, **6 times in this project**.

### Parameter Count (per encoder block)

| Parameter | Shape | Numbers |
| --- | --- | --- |
| $W_1$ | $512 \times 2048$ | 1,048,576 |
| $b_1$ | $2048$ | 2,048 |
| $W_2$ | $2048 \times 512$ | 1,048,576 |
| $b_2$ | $512$ | 512 |
| **Total** | | **2,099,712** |

The FFN holds about two-thirds of each block's parameters.
