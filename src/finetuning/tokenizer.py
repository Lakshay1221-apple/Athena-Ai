try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

from transformers import AutoTokenizer


MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True


def load_model_and_tokenizer():

    if FastLanguageModel is not None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )
    else:
        model = None
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return model, tokenizer


if __name__ == "__main__":

    model, tokenizer = load_model_and_tokenizer()

    print("\nTokenizer Loaded Successfully")
    print(f"Model Name      : {MODEL_NAME}")
    print(f"Vocabulary Size : {tokenizer.vocab_size}")
    print(f"Pad Token       : {tokenizer.pad_token}")
    print(f"EOS Token       : {tokenizer.eos_token}")
    print(f"Max Length      : {MAX_SEQ_LENGTH}")