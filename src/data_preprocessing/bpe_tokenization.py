import os
import glob
from tokenizers import ByteLevelBPETokenizer

def train_nepali_bpe(input_directory: str, output_directory: str, target_vocab_size: int = 30000) -> None:
    # 1. Gather all text files in the target directory
    file_paths = glob.glob(os.path.join(input_directory, "*.txt"))
    
    if not file_paths:
        print(f"No text files found in {input_directory}.")
        return

    print(f"Found {len(file_paths)} files. Initializing BPE tokenizer...")

    # 2. Initialize the Byte-Level BPE Tokenizer
    # Byte-level tokenizers map Unicode safely to a base vocabulary of 256 bytes, 
    # preventing unknown token (<unk>) issues with complex scripts.
    tokenizer = ByteLevelBPETokenizer()

    # 3. Train the tokenizer on the corpus
    tokenizer.train(
        files=file_paths,
        vocab_size=target_vocab_size,
        min_frequency=2,
        special_tokens=[
            "<s>",    # Beginning of sequence
            "<pad>",  # Padding token
            "</s>",   # End of sequence
            "<unk>",  # Unknown token
            "<mask>"  # Masking token for masked language modeling
        ]
    )

    # 4. Save the vocabulary and merge rules to the specified output folder
    os.makedirs(output_directory, exist_ok=True)
    
    # This generates two files: 'vocab.json' (the vocabulary) and 'merges.txt'
    tokenizer.save_model(output_directory)
    print(f"Tokenizer vocabulary successfully saved to: {output_directory}")

