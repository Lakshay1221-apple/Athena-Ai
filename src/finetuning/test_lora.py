from model_loader import load_model_and_tokenizer
from lora_config import apply_lora


def main():

    model, tokenizer = load_model_and_tokenizer()

    model = apply_lora(model)

    print("\nLoRA attached successfully!\n")

    model.print_trainable_parameters()


if __name__ == "__main__":
    main()