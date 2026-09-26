# Byte-Pair Encoding (BPE) Tokenization for Devanagari Text

This document explains how modern Large Language Model (LLM) tokenizers handle multi-byte scripts like Devanagari using **Byte-Level Byte-Pair Encoding (BPE)**, with real numbers from this project's tokenizer.

**In one sentence:** text is turned into bytes, and the tokenizer has learned which byte sequences occur together often enough to deserve their own token ID.

> **Code:** training in `src/data_preprocessing/bpe_tokenization.py` (`python -m scripts.train_nepali_bpe`); output files in `data/nepali_vocab_output/` (`vocab.json`, `merges.txt`); used by `src/data_preprocessing/batching.py`.

---

## Architecture Overview

```
[ Human Text: नेपाल ]
         │
         ▼
[ Phase 1: Training ] ──► Builds Vocabulary (`vocab.json`) & Merge Rules (`merges.txt`)   (done once)
         │
         ▼
[ Phase 2: Encoding ] ──► Raw Bytes ──► Applies Rules ──► Integer Token IDs            (every input)
         │
         ▼
[ Phase 3: Decoding ] ──► Token IDs ──► Byte Sequence ──► Rendered Text                (model output)
```

---

## Phase 1: Training the Tokenizer (Building the Dictionary)

### 1. Base Initialization (Special Tokens + The 256 Bytes)

The tokenizer starts with a fixed, unbreakable foundation of exactly **256 base tokens**. These represent every possible raw 8-bit computer byte:

$$\text{Range: } 00000000_2 \text{ to } 11111111_2 \quad (\text{Decimal } 0 \text{ to } 255)$$

Because every text is made of bytes, **any** input can be encoded — there is never an "unknown character".

In this project, 5 **special tokens** are added *before* the bytes, so they take the first IDs:

| ID | Token | Purpose |
| --- | --- | --- |
| 0 | `<s>` | beginning of sequence |
| 1 | `<pad>` | padding (see `encoders/2-positional_encoding.md`) |
| 2 | `</s>` | end of sequence |
| 3 | `<unk>` | unknown (rarely needed with byte-level BPE) |
| 4 | `<mask>` | hidden token for masked language modelling |
| 5 – 260 | the 256 bytes | base alphabet |
| 261 – 6301 | learned merges | built during training |

### 2. Raw Byte Breakdown

The entire Devanagari training corpus is broken down into a stream of single bytes.

UTF-8 is a variable-width encoding: English letters take 1 byte, but **every Devanagari character takes 3 bytes**. For example:

$$\text{Character 'न' (U+0928)} \longrightarrow [224, 164, 168]$$

Almost every Devanagari character starts with byte $224$ followed by $164$ or $165$ — which is why those pairs get merged first (see below).

### 3. Frequency Counting

The algorithm scans the byte sequences across the dataset and counts how often every adjacent pair of tokens occurs.

> The tokenizer first splits text at spaces (a "pre-tokenization" step), so merges never cross word boundaries. A space is kept attached to the start of the next word.

### 4. Statistical Merging

The single most frequent pair in the text is merged to create a brand-new token, assigned the next available ID.

* **Real example from this project:** the first merge learned was byte $224$ + byte $164$ — the shared prefix of most Devanagari characters. It became **ID 261** (the first ID after the 5 special tokens and 256 bytes).

### 5. Dictionary Mapping

The tokenizer records each new token in two files:

* **Vocabulary File (`vocab.json`):** token → ID, e.g. the token for bytes $[224, 164]$ → $261$.
* **Rulebook (`merges.txt`):** the ordered list of merges, one per line (first line = first merge).

> **Why do the files look like `à¤¨` instead of Nepali?** Byte-level BPE stores each byte as a printable stand-in character so the files are valid text: byte $224$ → `à`, $164$ → `¤`, $165$ → `¥`, a space → `Ġ`, and so on. So `à¤¨` in `vocab.json` is simply bytes $[224, 164, 168]$ = **न**. The first line of our `merges.txt` is `à ¤`, i.e. "merge byte 224 with byte 164".

### 6. Iteration and Expansion

The tokenizer replaces every occurrence of the merged pair with the new single token, recounts adjacent pairs, and repeats:

1. Merges `[224, 164]` (ID 261) with `[168]` to create the full character **न** (ID 267 in our vocabulary).
2. Continues merging characters into common syllables and word pieces.
3. Merges pieces into frequent words.

This loop repeats until the vocabulary reaches the target size, or until no pair is frequent enough (our training requires a pair to appear at least twice, `min_frequency=2`).

> **This project:** the target was $30,000$, but on our corpus the loop stopped at **$6,302$ tokens** ($V = 6302$). Large models use $32,000$–$50,000$ or more, trained on far more text.

---

## Phase 2: Tokenization (Encoding New Input)

When processing a new piece of text, the trained tokenizer performs three steps. Real output for **"नेपाल"** (5 Unicode characters: न + े + प + ा + ल):

```
Input Text: "नेपाल"
      │
      ▼
1. Byte Splitting   ──► [224,164,168, 224,165,135, 224,164,170, 224,164,190, 224,164,178]   (5 chars × 3 = 15 bytes)
                          └── न ──┘   └── े ──┘   └── प ──┘   └── ा ──┘   └── ल ──┘
      │
      ▼
2. Merge Rules      ──► Applies merges from `merges.txt` in the order they were learned
      │
      ▼
3. Final IDs        ──► [267, 270, 282, 264, 272]
                          न    े    प    ा    ल
```

1. **Byte Splitting:** The input text is split into its raw UTF-8 bytes.
2. **Applying Merge Rules:** The tokenizer executes learned merges in the exact order they were created during training.
3. **Final ID Output:** Merged chunks are replaced with their integer IDs from `vocab.json`. These IDs are passed into the model's **Embedding Layer** (`encoders/1-input_embedding.md`).

With our small vocabulary, "नेपाल" becomes one token per character. More frequent sequences do become single tokens — e.g. in **"नेपाल सरकार"** the piece " सरक" (space + स + र + क) is a single token (ID 405):

| Text | Token IDs | Count |
| --- | --- | --- |
| न | `[267]` | 1 |
| नेपाल | `[267, 270, 282, 264, 272]` | 5 |
| नेपाल सरकार | `[267, 270, 282, 264, 272, 405, 264, 266]` | 8 (31 bytes) |
| काठमाडौं | `[268, 264, 463, 264, 314, 397]` | 6 (24 bytes) |

A bigger vocabulary (more training text) would merge whole common words like नेपाल into a single token, making sequences shorter.

---

## Phase 3: Decoding (Translating Back to Human Text)

When the model outputs token IDs, the process runs in reverse to render readable text:

```
Model Output: [267]
      │
      ▼
1. Dictionary Lookup ──► `vocab.json` maps ID 267 ──► stored as "à¤¨" ──► Byte Sequence: [224, 164, 168]
      │
      ▼
2. Join the bytes of all output tokens in order
      │
      ▼
3. UTF-8 decode      ──► Renders: 'न'
```

1. **Dictionary Lookup:** The tokenizer takes an integer ID (e.g., `267`) and finds its token in `vocab.json`.
2. **Byte Retrieval:** It converts the stored stand-in characters back to the raw bytes (`[224, 164, 168]`) and joins the bytes of all tokens.
3. **UTF-8 Decoding:** The byte sequence is decoded as UTF-8 and shown as the glyph **न**.

> **Important:** a single token can be *part* of a character (e.g. ID 261 = bytes $[224, 164]$ only). Always decode the **whole** list of IDs together (`tokenizer.decode(ids)`), not one token at a time — otherwise you may get broken characters or `à¤`-style strings.
