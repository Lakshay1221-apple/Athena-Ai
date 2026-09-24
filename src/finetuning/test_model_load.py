"""Test loading base model and tokenizer."""

from src.finetuning.model_loader import load_model_and_tokenizer


def main():
    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded successfully")
    print("Vocab size:", tokenizer.vocab_size)


if __name__ == "__main__":
    main()