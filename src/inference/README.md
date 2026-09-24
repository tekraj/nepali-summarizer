During training, the Transformer processes the entire summary at once using a causal mask (Teacher Forcing). During inference, the model does not have the target summary. It must generate it completely blind, feeding its own predictions back into itself. This is called **autoregressive generation**.

Here is the exact algorithmic loop for generating a Nepali summary using your NumPy matrices.

### Phase 1: The One-Time Encoder Pass

Before the loop starts, you process the source Nepali document exactly once.

1. Tokenize the long source document into IDs: Shape `(1, T_src)`. *(Batch size is 1 for a single inference generation).*
2. Pass it through the Encoder blocks (Self-Attention $\rightarrow$ Add & Norm $\rightarrow$ FFN $\rightarrow$ Add & Norm).
3. The Encoder outputs the **Memory Matrix**: $M$.
* **Shape of $M$:** `(1, T_src, d_model)`.
* This matrix contains the rich, contextualized understanding of the original document. It will remain static for the rest of the generation process.



### Phase 2: The Autoregressive Loop

You must initialize the generation process by giving the Decoder a starting signal.

1. **Initialize Decoder Input:** Create a NumPy array containing only the Begin-of-Sequence token ID.
`decoder_input = np.array([[BOS_ID]])`  *(Shape: `(1, 1)`)*
2. **Start the `while` Loop:** This loop runs until the model predicts the `<EOS>` token, or you hit a `max_length` limit (e.g., 100 summary tokens).
**Inside the Loop (Step $t$):**
* **A. Embed & Encode Position:** Convert `decoder_input` (current length $t$) into embeddings and add positional encodings. Shape becomes `(1, t, d_model)`.
* **B. Masked Self-Attention:** The Decoder looks at the tokens it has generated *so far*. Because you are generating step-by-step, you don't strictly need the triangular look-ahead mask here (since future tokens literally don't exist in the array yet), but the attention math remains the same.
* **C. Cross-Attention (The Crucial Step):** The Decoder bridges to the original document.
* **Queries ($Q$):** Come from the Decoder's current state (Shape: `(1, t, d_model)`).
* **Keys ($K$) and Values ($V$):** Come directly from the Encoder's static Memory Matrix $M$ (Shape: `(1, T_src, d_model)`).
* The matrix multiplication $Q \times K^T$ calculates how much the currently generated summary needs to focus on specific words in the original long document to figure out what to say next.


* **D. Feed-Forward & Projection:** The output passes through the FFN. You then slice out ONLY the vector for the very last token in the sequence (the one you just processed):
`last_token_vector = output_matrix[:, -1, :]` *(Shape: `(1, d_model)`)*
* **E. Compute Probabilities:** Multiply this single vector by your vocabulary output weight matrix $W_{out}$ (Shape: `(d_model, V)`).
`logits = np.matmul(last_token_vector, W_out)` *(Shape: `(1, V)`)*
Apply softmax to get the probability distribution over all Nepali subwords in your vocabulary.



### Phase 3: Token Selection (Greedy vs. Beam Search)

Once you have the probabilities for step $t$, you must pick a token.

**The Greedy Approach (Simplest in NumPy):**
Simply take the index of the highest probability:
`next_token_id = np.argmax(logits, axis=-1)`
You then append this integer to your sequence:
`decoder_input = np.append(decoder_input, [[next_token_id]], axis=1)`

*(If `next_token_id == EOS_ID`, you break the loop. The summary is complete.)*

**The Problem with Greedy Search:**
If the model chooses a slightly suboptimal token at step 3, it cannot undo it. The error compounds. For high-quality Nepali summarization, you would implement **Beam Search**:
Instead of keeping 1 sequence, you keep the top $k$ (e.g., 3) most likely sequences simultaneously. At each step, you predict the next token for all 3 beams, calculate the combined sequence probabilities, and prune back down to the top 3.

### The Optimization: KV Caching

In the naive loop above, at step $t=10$, you are recalculating the self-attention for tokens $1$ through $9$ all over again, even though they haven't changed.

To make your NumPy inference drastically faster, you implement **KV Caching**. During the Decoder's Self-Attention phase, you save the computed $K$ and $V$ matrices for step $t$. At step $t+1$, you only compute $Q$, $K$, and $V$ for the *new single token*, and append its $K$ and $V$ to your cached matrices. This reduces the time complexity of generation from $O(N^2)$ to $O(N)$ per token.