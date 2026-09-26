### 1. Why Subwords are Perfect for Nepali (Ka vs. Ke)

In Devanagari, "Ka" is **क** and "Ke" is **के** (which is structurally **क** + **े**).

If you used whole-word tokenization, your model would have to learn entirely separate mathematical representations for "नेपाल", "नेपालको", "नेपालमा", and "नेपाललाई".

With our BPE approach, the algorithm initially splits everything into base characters and matras (vowel signs). "के" starts as `['क', 'े']`. Because "के" is highly frequent in Nepali, the algorithm quickly merges it into a single subword token: `'के'`.

* For common words, it merges all the way up to the full word (e.g., `'नेपाल'`).
* For common suffixes, it creates chunks (e.g., `'लाई'`).
* For rare words, it leaves them as base characters (e.g., `'क'`, `'्'`, `'ष'`).

This allows the model to understand the root of a word and its suffix as separate building blocks.

### 2. How We Convert Text to Numbers

We do not use a search-engine style "inverted index" (which maps words to document locations). Instead, we created a **direct lookup dictionary**, starting exactly from $0$.

If you look at this line in the `train` method:
`self.vocab = {token: idx for idx, token in enumerate(sorted(unique_tokens))}`

Here is exactly what that does:

1. It takes every unique token we ended up with (base characters + merged subwords).
2. It sorts them alphabetically.
3. It loops through them (`enumerate`), assigning $0$ to the first, $1$ to the second, all the way up to $V-1$ (where $V$ is your vocabulary size).

The resulting `self.vocab` dictionary looks like this:

```json
{
  "</w>": 0,
  "अ": 1,
  "आ": 2,
  ...
  "क": 45,
  "के": 46,
  ...
  "नेपाल</w>": 5012
}

```

When you call `encode()`, it simply looks up the string in this dictionary and replaces it with the integer. `['नेपाल</w>', 'मा</w>']` becomes `[5012, 1045]`.

**Why start from 0?**
Because in NumPy, these integers will be used as **row indices** for your Embedding Matrix. If your token ID is `45`, your forward pass will simply grab `EmbeddingMatrix[45, :]` to get the vector representation for that specific subword.

The `self.inverse_vocab` in the code is just the exact reverse of this dictionary (`{45: "क"}`). We need it so that when your model eventually generates an array of numbers, you can translate them back into readable Nepali text.