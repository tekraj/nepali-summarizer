The output from the Multi-Head Attention block doesn't go straight to the next layer. It passes through a sequence of addition, normalization, and an expansion-compression neural network. This combination stabilizes training and allows the model to process the patterns it just discovered during attention.

### 1. Residual Connections (The "Add" in Add & Norm)

Deep neural networks suffer from vanishing gradients, meaning the error signal gets too small to update the early layers during backpropagation. Transformers solve this by adding the original input of a sub-layer directly to its output.

If $X$ is the input matrix before Multi-Head Attention (MHA), and $MHA(X)$ is the output:


$$X_{res} = X + MHA(X)$$

* **NumPy Operation:** `np.add(X, MHA_Output)`
* **Dimensionality:** Both matrices must be exactly $(B, T, d_{model})$.
* **Intuition:** If the attention mechanism fails to learn anything useful, the model can simply bypass it and pass the original embeddings forward.

### 2. Layer Normalization

While Batch Normalization normalizes across the batch dimension, Layer Normalization normalizes across the feature dimension ($d_{model}$) for each token independently. This ensures that the vector representation of every single word has a mean of $0$ and a variance of $1$, keeping the network numerically stable.

For every token vector $x_i$ of size $d_{model}$:

1. **Calculate Mean:** $\mu = \frac{1}{d_{model}} \sum_{j=1}^{d_{model}} x_{ij}$
2. **Calculate Variance:** $\sigma^2 = \frac{1}{d_{model}} \sum_{j=1}^{d_{model}} (x_{ij} - \mu)^2$
3. **Normalize:** $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ (where $\epsilon$ is a tiny number like `1e-5` to prevent division by zero).
4. **Scale and Shift:** Apply learnable parameters $\gamma$ (scale) and $\beta$ (shift), both initialized as arrays of size $(d_{model})$.

$$Norm(x_i) = \gamma \odot \hat{x}_i + \beta$$



* **NumPy Implementation:**
* `mean = X_res.mean(axis=-1, keepdims=True)`
* `var = X_res.var(axis=-1, keepdims=True)`
* `X_norm = (X_res - mean) / np.sqrt(var + 1e-5)`
* `Output = gamma * X_norm + beta`



### 3. Position-wise Feed-Forward Network (FFN)

The Attention mechanism mixes information *between* different tokens. The FFN processes the information *within* each token individually.

It is a two-layer Multi-Layer Perceptron (MLP) applied to every token position identically and independently. It temporarily expands the dimensionality (usually by a factor of 4) to create a richer feature space, applies a non-linearity, and projects it back down.

Let $d_{ff} = 4 \times d_{model}$ (e.g., if $d_{model}=512$, $d_{ff}=2048$).
You need two learnable weight matrices and biases:

* $W_1$ shape: $(d_{model}, d_{ff})$, $b_1$ shape: $(d_{ff})$
* $W_2$ shape: $(d_{ff}, d_{model})$, $b_2$ shape: $(d_{model})$

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

* **NumPy Implementation:**
* **Linear 1 (Expansion):** `hidden = np.matmul(X_norm, W1) + b1`
* **Activation (ReLU):** `activated = np.maximum(0, hidden)`
* **Linear 2 (Compression):** `ffn_out = np.matmul(activated, W2) + b2`



**Output Shape:** Back to $(B, T, d_{model})$.

After the FFN, the tensor passes through a second identical Add & Norm block:
`Final_Block_Output = LayerNorm(X_norm + ffn_out)`

This exact block (MHA $\rightarrow$ Add & Norm $\rightarrow$ FFN $\rightarrow$ Add & Norm) is repeated $N$ times (e.g., 6 layers) in both the Encoder and the Decoder.