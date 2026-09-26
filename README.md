# Building a Nepali Summarization Transformer from Scratch (NumPy)

## 1. Data Preparation & Tokenization

Building an NLP pipeline for Nepali requires handling specific Devanagari script nuances before any matrix math begins.

*   **Corpus Cleaning:** Standardize the 200,000 documents by removing HTML tags, English boilerplate, and normalizing Devanagari characters. You must handle Zero-Width Joiners (ZWJ) and Zero-Width Non-Joiners (ZWNJ) consistently, as well as normalize composite characters (e.g., halants joining consonants) into their canonical Unicode forms.
*   **Subword Tokenization (BPE):** Word-level tokenization struggles with Nepali's rich morphology and compounding. Implement a Byte-Pair Encoding (BPE) algorithm to build a vocabulary. 
    1. Initialize a vocabulary with all distinct Unicode characters in the corpus.
    2. Iteratively find the most frequent adjacent pair of tokens and merge them.
    3. Map the final subwords to integer IDs (e.g., $0$ to $V-1$).
*   **Sequence Formatting:** For summarization, structure your training pairs as `[Encoder_Input, Decoder_Input, Target_Output]`. Pad sequences to a fixed length and create attention masks (to prevent attending to padding) and subsequent masks (to prevent the decoder from looking ahead).

## 2. Matrix Architecture & Forward Pass

Since you are bypassing frameworks, your core task is managing multidimensional arrays (tensors) and executing the mathematical graph sequentially.

*   **Embeddings & Positional Encoding:** Initialize a weight matrix of shape $(V, d_{model})$ using normal distribution scaling. Because NumPy arrays have no inherent sequence order, generate a fixed Positional Encoding matrix using sine and cosine functions for odd and even indices:
    $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
    $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$
    Add this to your token embeddings.
*   **Multi-Head Attention (MHA):** This is the engine of the Transformer. For each head, initialize weight matrices $W^Q, W^K, W^V$. 
    1. Project inputs into Queries ($Q$), Keys ($K$), and Values ($V$) via matrix multiplication (`np.dot` or `np.matmul`).
    2. Compute scaled dot-product attention: 
       $$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    3. Concatenate the heads and multiply by an output weight matrix $W^O$.
*   **Feed-Forward & Layer Norm:** Implement a two-layer multi-layer perceptron (MLP) with a non-linear activation (like ReLU or GELU) applied element-wise in between. Wrap both the MHA and MLP blocks in residual connections ($Output = Input + Sublayer(Input)$) followed by standard Layer Normalization computing mean and variance across the feature dimension.

## 3. Backpropagation (The Custom Autograd Challenge)

Without PyTorch's Autograd, you must manually calculate the chain rule derivatives for every operation from the loss function back to the embeddings. 

*   **Loss Calculation:** Implement Cross-Entropy Loss comparing the final Softmax probabilities against the one-hot encoded target tokens.
*   **Gradient Derivation:** Work backward through the network. You must code the exact mathematical derivative for:
    *   The Softmax and Cross-Entropy combined (this simplifies cleanly).
    *   Linear layers (gradients with respect to weights, biases, and inputs).
    *   Activation functions (e.g., $ReLU'(x) = 1$ if $x > 0$ else $0$).
    *   Layer Normalization (calculating gradients for the scaling $\gamma$ and shifting $\beta$ parameters, and the inputs based on the mean/variance derivations).
    *   The Multi-Head Attention block, which requires carefully routing gradients through the $QK^T$ multiplication and the internal softmax.
*   **Optimizer:** Implement the Adam optimizer algorithm, requiring you to maintain running states of the first (momentum) and second (RMSprop) moments for every single weight matrix in the model to update them after each backward pass.

## 4. Training Loop & Inference

*   **Batching & Memory:** NumPy executes on the CPU by default. Processing 200,000 documents requires writing a custom data loader that yields mini-batches (e.g., 16 or 32 sequences at a time) to avoid RAM exhaustion.
*   **Autoregressive Generation:** For inference, the model cannot output the summary in one pass. Write a decoding loop that feeds the source document into the encoder once, then starts the decoder with a `<BOS>` (Begin of Sequence) token.
*   **Search Strategy:** At each step, take the NumPy array of output probabilities. Implement either Greedy Search (taking the `np.argmax` of the highest probability token) or a Beam Search algorithm to maintain the top $k$ most likely sequences until the model outputs an `<EOS>` (End of Sequence) token.

## Project Setup and Commands

Install the project and its dependencies in editable mode:

```bash
python -m pip install -e .
```

Then run the installed cleaning command:

```bash
clean-data
```

From the project root, modules can also be run without installation:

```bash
python -m scripts.clean_data
python -m scripts.preprocess_data
python -m scripts.train_model
python -m scripts.infer
python -m src.main
```