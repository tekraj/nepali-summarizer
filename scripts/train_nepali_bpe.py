    # Directory containing your Nepali Unicode .txt files
from src.data_preprocessing.bpe_tokenization import train_nepali_bpe


INPUT_DIR = "./data/cleaned"  
OUTPUT_DIR = "./data/nepali_vocab_output"
def main():
    train_nepali_bpe(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()