# Second Add & Norm (after the Feed-Forward Network)

The second **Add & Norm** layer sits directly after the Position-wise Feed-Forward Network (FFN), forming the final component of a complete Transformer Encoder block.

It uses the exact same mathematical formulation and principles as the first Add & Norm layer (`4-add_and_normalization.md`), acting as a stabilizing wrapper for the transformations executed by the FFN. The only differences are **what** gets added and that it has its **own** $\gamma_2, \beta_2$.

```text
X_norm1 (B, T, d_model) ─┬──► FFN ──► X_ffn_out ─┐
   (= X_ffn_in)          │                       ▼
                         └────────────────────► ( + ) ──► LayerNorm₂ ──► X_out (B, T, d_model)
```

> **Code:** the second `AddNormBlock` inside `EncoderBlock` (`src/models/encoder.py`, attribute `add_norm_2`)

---

## 1. The "Add" Component (Second Residual Connection)

The input tensor that was fed into the FFN — which we call $\mathbf{X}_{ffn\_in}$, shape $(B, T, d_{model})$ — is the output of the **first** Add & Norm ($\mathbf{X}_{norm1}$). It is kept in memory. After the FFN processes it through its $d_{model} \rightarrow d_{ff} \rightarrow d_{model}$ expansion and compression, the output $\text{FFN}(\mathbf{X}_{ffn\_in})$ is added directly back to it:

$$\mathbf{X}_{res2} = \mathbf{X}_{ffn\_in} + \text{FFN}(\mathbf{X}_{ffn\_in})$$

### Why it is applied here

* **Information Highway:** The FFN applies heavy non-linear filtering ($\text{ReLU}$) which discards information in the 2048-dimensional space. The skip connection ensures that the contextualized attention information from the earlier stage is not accidentally destroyed.
* **Gradient Preservation:** During backpropagation, the loss gradient can flow directly through this addition back into the attention layers and embedding matrix without being diminished by the FFN's weights ($W_1, W_2$).

---

## 2. The "Norm" Component (Second Layer Normalization)

The tensor $\mathbf{X}_{res2}$ is immediately passed into a second, independent Layer Normalization.

This block maintains its own distinct set of learnable parameters: $\gamma_2$ (scale) and $\beta_2$ (shift), both of shape $(d_{model})$. They start as $\gamma_2 = 1$, $\beta_2 = 0$ and are trained separately from $\gamma_1, \beta_1$.

For every single token vector $\mathbf{x} \in \mathbb{R}^{d_{model}}$ across the batch and sequence:

1. **Calculate Token Mean ($\mu$):**

$$\mu = \frac{1}{d_{model}} \sum_{j=1}^{d_{model}} x_j$$

2. **Calculate Token Variance ($\sigma^2$):**

$$\sigma^2 = \frac{1}{d_{model}} \sum_{j=1}^{d_{model}} (x_j - \mu)^2$$

3. **Normalize, Scale, and Shift:**

$$\text{LayerNorm}_2(\mathbf{x}) = \gamma_2 \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_2$$

---

## 3. Worked Example (continuing the FFN example, $d_{model} = 2$)

From `5-feed_forward.md`: $\mathbf{X}_{ffn\_in} = [1,\ -1]$ and $\text{FFN}(\mathbf{X}_{ffn\_in}) = [1.6,\ 2.1]$.

1. **Add:** $\mathbf{X}_{res2} = [1 + 1.6,\ -1 + 2.1] = [2.6,\ 1.1]$
2. **Mean:** $\mu = (2.6 + 1.1)/2 = 1.85$
3. **Variance:** $\sigma^2 = \big(0.75^2 + (-0.75)^2\big)/2 = 0.5625$, so $\sqrt{\sigma^2 + \epsilon} \approx 0.75$
4. **Normalize:** $[(2.6 - 1.85)/0.75,\ (1.1 - 1.85)/0.75] = [1.0,\ -1.0]$
5. **Scale & shift** ($\gamma_2 = 1$, $\beta_2 = 0$): $\mathbf{X}_{out} = [1.0,\ -1.0]$

*(With only 2 features the normalized values are always $\pm 1$; with 512 features you get a full spread of values, as in the example in `4-add_and_normalization.md`.)*

---

## 4. Top-to-Bottom Flow of the Second Add & Norm Layer

1. **Input Tensors in Memory:**
    * $\mathbf{X}_{ffn\_in}$: The normalized tensor coming out of the *first* Add & Norm block — shape $(B, T, 512)$.
    * $\mathbf{X}_{ffn\_out}$: The tensor produced by the Feed-Forward Network ($W_1 \rightarrow \text{ReLU} \rightarrow W_2$) — shape $(B, T, 512)$.

2. **Element-Wise Addition:**
    * Compute $\mathbf{X}_{res2} = \mathbf{X}_{ffn\_in} + \mathbf{X}_{ffn\_out}$.
    * Output shape remains **$(B, T, 512)$**.

3. **Layer Normalization:**
    * For every token $t \in [1, T]$ in every sentence $b \in [1, B]$, the $512$ feature values are normalized to have zero mean and unit variance, then rescaled via $\gamma_2$ and shifted via $\beta_2$.

4. **Final Block Output:**
    * The layer outputs a clean, stable 3D tensor of shape **$(B, T, d_{model})$**.

**Learnable parameters here:** $\gamma_2$ (512) + $\beta_2$ (512) = 1,024 numbers per block.

---

## Summary of the Entire Encoder Block Cycle

With the completion of the second Add & Norm layer, the data tensor has traveled through one complete Transformer Encoder block:

```text
[ Input Tensor X_in (B, T, d_model) ]
              │
              ├───► [ Multi-Head Attention ] ───┐
              │                                 │
              └─────────────────────────────────┴───► [ Add & Norm (1) ]  = X_norm1
                                                              │
                                                              ├───► [ Feed-Forward Net ] ───┐
                                                              │                             │
                                                              └─────────────────────────────┴───► [ Add & Norm (2) ]
                                                                                                        │
                                                                                         [ Output Tensor X_out (B, T, d_model) ]
```

| Sub-layer | Learnable parameters | Count |
| --- | --- | --- |
| Multi-Head Attention | $W_Q, W_K, W_V, W_O$ | 1,048,576 |
| Add & Norm 1 | $\gamma_1, \beta_1$ | 1,024 |
| Feed-Forward | $W_1, b_1, W_2, b_2$ | 2,099,712 |
| Add & Norm 2 | $\gamma_2, \beta_2$ | 1,024 |
| **One encoder block** | | **3,150,336** |

The output tensor has the exact same dimensions **$(B, T, d_{model})$** as the tensor that entered the block. Because the shape is preserved, this output can be fed directly into the **next** Encoder block in the stack (Block 2 of 6 in this project) to build deeper abstract representations. Each block has its own, separately learned weights.
