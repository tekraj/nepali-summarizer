# Building a Nepali Summarization Transformer from Scratch (CuPy)

Every layer, its backward pass, the loss and the Adam optimizer are hand-written with [CuPy](https://cupy.dev/), the NumPy-compatible array library that runs on NVIDIA GPUs. There is no PyTorch and no autograd. Training uses fp16 mixed precision: `float16` activations and matmuls (GPU tensor cores), with `float32` master weights, gradients and Adam state, and dynamic loss scaling.

## 1. Data Preparation & Tokenization

Building an NLP pipeline for Nepali requires handling specific Devanagari script nuances before any matrix math begins.

*   **Corpus:** about 132,000 pure-Devanagari Nepali news articles (`data/cleaned/{name}.txt`), each paired with a target summary (`data/summary/{name}-summary.txt`).
*   **Corpus Cleaning:** Standardize the documents by removing HTML tags, English boilerplate, and normalizing Devanagari characters. You must handle Zero-Width Joiners (ZWJ) and Zero-Width Non-Joiners (ZWNJ) consistently, as well as normalize composite characters (e.g., halants joining consonants) into their canonical Unicode forms.
*   **Subword Tokenization (BPE):** Word-level tokenization struggles with Nepali's rich morphology and compounding. Implement a Byte-Pair Encoding (BPE) algorithm to build a vocabulary. 
    1. Initialize a vocabulary with all distinct Unicode characters in the corpus.
    2. Iteratively find the most frequent adjacent pair of tokens and merge them.
    3. Map the final subwords to integer IDs ($0$ to $V-1$; this project uses $V = 30,000$).
*   **Sequence Formatting (encoder-only, prefix-LM):** Each article `data/cleaned/{name}.txt` is paired with its summary `data/summary/{name}-summary.txt`. One training sample is `<s> article </s> summary </s>`. The article part is at most 768 tokens, and anything past that is cut from the end of the article. The summary part is at most 256 tokens. Samples are padded to the longest one in the batch, up to 1024 tokens.
    *   **Attention mask `[B, T, T]`:** article tokens see the whole article. Summary tokens see the article plus only the summary tokens before them, so the model cannot look ahead.
    *   **Loss mask:** position $p$ predicts token $p+1$, and only positions whose next token is part of the summary are scored. The loss covers the summary sequence alone.

## 2. Matrix Architecture & Forward Pass

Since you are bypassing frameworks, your core task is managing multidimensional arrays (tensors) and executing the mathematical graph sequentially.

*   **Embeddings & Positional Encoding:** Initialize a weight matrix of shape $(V, d_{model})$ using normal distribution scaling. Because the arrays have no inherent sequence order, generate a fixed Positional Encoding matrix using sine and cosine functions for odd and even indices:
    $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
    $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$
    Add this to your token embeddings.
*   **Multi-Head Attention (MHA):** This is the engine of the Transformer. For each head, initialize weight matrices $W^Q, W^K, W^V$. 
    1. Project inputs into Queries ($Q$), Keys ($K$), and Values ($V$) via matrix multiplication (`cp.matmul`, or `@`).
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

*   **Batching & Memory:** CuPy runs on the GPU, so every weight, activation and gradient lives in VRAM. The data loader streams (article, summary) pairs from disk and yields mini-batches of `batch_size` pairs (default 4). Each block caches its activations for backprop, and the attention weights alone take `B × H × T × T` floats per block. At B=4 and T=1024, one training step needs about 3 GB, which fits on a 6 GB GPU.
*   **Autoregressive Generation:** For inference, the model cannot output the summary in one pass. The loop starts from `<s> article </s>`, runs the encoder with the same prefix-LM mask used in training, and appends the predicted next token each step. It stops at `</s>` or after 256 summary tokens. There is no KV cache, so each step reruns the full forward pass.
*   **Search Strategy:** Greedy Search: at each step take `cp.argmax` of the output probabilities. Beam Search, which keeps the top $k$ sequences, is not implemented.

## Project Setup and Commands

**Requirements:** an NVIDIA GPU with a CUDA 12.x driver. The `cupy-cuda12x` wheel bundles the CUDA runtime libraries, so the full CUDA toolkit is not needed. On WSL2, install the NVIDIA Windows driver with WSL support, and check that `nvidia-smi` works inside WSL. CuPy does not support macOS.

Install the project and its dependencies in editable mode:

```bash
uv sync                        # or: python -m pip install -e .
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"   # should print >= 1
```

Then run the installed cleaning command:

```bash
clean-data
```

From the project root, modules can also be run without installation:

```bash
python -m scripts.clean_data          # data/raw -> data/cleaned
python -m scripts.train_nepali_bpe    # data/cleaned -> data/nepali_vocab_output
python -m scripts.train_model         # train on data/cleaned + data/summary pairs
python -m scripts.train_model --max-steps 5 --epochs 1                  # quick smoke test
python -m scripts.infer --checkpoint checkpoints/summarizer_epoch_3.npz --article data/cleaned/100001.txt
python -m src.main
```

Training settings (article/summary lengths, batch size, epochs, model size) are in [config/config.yaml](config/config.yaml). A checkpoint is saved after every epoch as `checkpoints/summarizer_epoch_N.npz`.