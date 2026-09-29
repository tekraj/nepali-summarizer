# Linearization (LM Head) and Softmax

**In one sentence:** a final linear layer turns each 512-number token vector into one score per vocabulary word, and softmax turns those scores into probabilities.

> **Code:** `src/models/layers/lm_head.py` → `LanguageModelingHead`; loss in `src/training/loss.py`

## 1. The Linearization and Softmax Head

After passing through $N$ repeated Encoder blocks (6 in this project; 12 or 24 in larger models), the tensor exiting the final block has the shape **$(B, T, d_{model})$**.

To convert these hidden representations back into predictions over real words (vocabulary tokens), the model routes the tensor through a **Language Modeling Head** (or Classification Head).

```text
[ Output Tensor from Final Encoder Block ]  (B, T, d_model)
                         │
                         ▼
             [ Final Linear Projection Layer ]     Weights: (d_model, V), bias: (V)
                         │
                         ▼
                     [ Logits ]                    (B, T, V)
                         │
                         ▼
                     [ Softmax ]                   (B, T, V)  (Probabilities)
```

### Step 1: The Linear Projection (Un-embedding)

The model passes the $(B, T, d_{model})$ tensor through a standard Dense (Linear) layer that projects the vector size from $d_{model}$ (512) directly to the size of the entire vocabulary $V$.

$$\mathbf{Z} = \mathbf{X}_{final} \cdot W_{LM} + b_{LM}$$

| | General example | This project |
| --- | --- | --- |
| Input $\mathbf{X}_{final}$ | $(B, T, 512)$ | $(4, 512, 512)$ |
| $W_{LM}$ | $(512, 50000)$ | $(512, 30000)$ |
| $b_{LM}$ | $(50000)$ | $(30000)$ |
| Output $\mathbf{Z}$ (logits) | $(B, T, 50000)$ | $(4, 512, 30000)$ |

The output values inside $\mathbf{Z}$ are called **Logits**. Every token position $t$ in every sentence now has an unnormalized score for every possible token in the vocabulary. Logits can be any real number (negative, zero, positive).

*(Note on Weight Tying: In many architectures, $W_{LM}$ is the **transposed Input Embedding matrix** $E^T$ — shape $(512, V)$ — instead of a separate matrix. Reusing the input embedding weights cuts the parameter count and can improve stability. The code supports it via `tie_weights`; this project's config has it switched **off**.)*

### Step 2: The Softmax Layer

To convert these raw logit scores into probabilities, the model applies the Softmax function along the vocabulary dimension ($V$), separately for every position:

$$P(w_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

* **Input Shape:** $(B, T, V)$
* **Output Shape:** $(B, T, V)$

For every position in the sequence, the $V$ values now sum to $1.0$ ($100\%$). The token with the highest probability is the model's prediction.

> **Numerical safety:** $e^{z}$ overflows for large $z$. The code subtracts the row maximum first, $\text{softmax}(\mathbf{z}) = \text{softmax}(\mathbf{z} - \max \mathbf{z})$, which gives the identical result without overflow.

### Worked Example (one position, toy vocabulary of $V = 4$)

| token | logit $z_i$ | $e^{z_i}$ | probability $P(w_i)$ |
| --- | --- | --- | --- |
| "नेपाल" | 2.0 | 7.389 | **0.6381** |
| "सुन्दर" | 1.0 | 2.718 | 0.2347 |
| "देश" | 0.1 | 1.105 | 0.0954 |
| "हो" | −1.0 | 0.368 | 0.0318 |
| **sum** | | 11.580 | **1.0000** |

Each probability is $e^{z_i} / 11.580$. The model predicts "नेपाल".

---

## 2. What Is the Target? (Masked Language Modelling)

> **This project's summarizer does not use MLM.** It trains the encoder as a prefix-LM: the input is `<s> article </s> summary </s>`, and the target at each summary position is the next summary token. See **"Project-Specific Training and Target Data Preparation Strategy"** in `8-end-to-end.md`. MLM is explained below because it is the classic encoder objective and was this project's earlier pre-training setup. The "score only $N$ positions" idea carries over unchanged.

The loss compares the predicted probabilities with a **ground-truth token**. But an encoder sees the whole sentence at once — if we asked it to predict the token at position $t$ while that token is visible, it could simply copy its input and learn nothing.

The standard solution for encoders (used by BERT) is **Masked Language Modelling (MLM)**:

1. Pick a random ~15% of the real (non-`[PAD]`) tokens in the batch.
2. Replace them in the input with the special `<mask>` token (ID 4 in our vocabulary).
3. Run the encoder. Because of self-attention, the vector at a masked position can gather clues from the surrounding words.
4. Compute the loss **only at the masked positions**, with the **original** token as the target.

```text
Original:   नेपाल  सुन्दर  देश  हो
Input:      नेपाल  <mask>  देश  हो
Target:       –    सुन्दर    –    –     ← loss only here
```

Because only masked positions matter, the code runs the LM head on just those $N$ vectors: $(N, 512) \rightarrow (N, V)$ instead of $(B, T, 512) \rightarrow (B, T, V)$. With $B = 4$, $T = 512$ and 15% masking, $N \approx 307$ instead of $2048$ positions — about 85% less work, same learning signal. In the summarizer, the $N$ positions are the summary positions instead, at most $B \times 256 = 1024$ per batch.

### Cross-Entropy Loss

For each scored position (masked token in MLM, summary token in the summarizer), the loss is the negative log of the probability the model gave to the correct token; the batch loss is the average over the $N$ scored positions:

$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \log P(\text{target}_n)$$

With the toy example above:

* If the true token is "नेपाल" ($P = 0.6381$): $\mathcal{L} = -\ln 0.6381 = 0.449$ (good prediction, small loss)
* If the true token is "देश" ($P = 0.0954$): $\mathcal{L} = -\ln 0.0954 = 2.349$ (bad prediction, large loss)

**Sanity check for this project:** an untrained model spreads probability evenly, $P \approx 1/30000$, so the first loss should be close to $\ln 30000 \approx 10.31$.

---

## 3. Initial Guesses vs. Actual Training (Updating Embeddings)

### Where Does Actual Training Happen?

Training happens through **End-to-End Backpropagation**:

1. **Forward Pass:** The input token IDs pass through the initial Embeddings $\rightarrow$ Positional Encodings $\rightarrow$ Attention Blocks $\rightarrow$ FFNs $\rightarrow$ Linear Projection $\rightarrow$ Softmax.
2. **Loss Calculation:** The model compares its predicted probability distribution against the true (masked) tokens using **Cross-Entropy Loss**.
3. **Backward Pass (Backpropagation):** The loss produces an error gradient ($\nabla \mathcal{L}$). This gradient travels **all the way back** through every single layer from the top down:

$$\text{Loss} \rightarrow \text{Linear Layer} \rightarrow \text{Block } N \rightarrow \dots \rightarrow \text{Block 1} \rightarrow \text{Positional Addition} \rightarrow \text{Input Embeddings}$$

4. **Optimizer Step:** Every weight matrix in the entire pipeline is updated using its gradient. The simplest rule is plain gradient descent:

$$W_{new} = W_{old} - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}$$

This project uses **Adam**, which adapts the step size per weight (AdamW is a variant with decoupled weight decay). Every formula — for every weight and bias — is worked out in `9-backpropagation.md`.

---

### Do Input Embeddings Update After the First Round?

**Yes! Input embeddings are continuously updated throughout the entire training process.**

1. **Initial State (Step 0 of Training):**
    * The Input Embedding matrix $E$ of shape $(V, d_{model})$ is initialized with random numbers (from $\mathcal{N}(0, \sigma = 0.02)$).
    * At step 0, the word `"cat"` and the word `"dog"` have completely random, meaningless vectors.

2. **During Training (Many Iterations):**
    * In every training step, gradients flow directly into the rows of the embedding matrix $E$ corresponding to the tokens in that batch.
    * If `"cat"` and `"dog"` repeatedly appear in similar contextual positions across many sentences, backpropagation pushes their 512-dimensional vectors closer together in coordinate space.

3. **Inference State (Deployment / Post-Training):**
    * Once training is complete, all weights—including the Input Embedding matrix, $W_Q, W_K, W_V, W_O, W_1, W_2$, and the Linear Head—are **frozen**.
    * When you run a trained model, the embeddings are static, fully optimized lookup vectors that no longer change.

### Parameter Count

| Parameter | Shape | Numbers |
| --- | --- | --- |
| $W_{LM}$ | $512 \times 30000$ | 15,360,000 |
| $b_{LM}$ | $30000$ | 30,000 |
| **Total** | | **15,390,000** (only $b_{LM}$ if weights are tied) |
