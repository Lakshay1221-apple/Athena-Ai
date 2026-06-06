from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True


def load_model_and_tokenizer():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    return model, tokenizer