# test_model_load.py

import unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("Model loaded successfully")
print("Vocab size:", tokenizer.vocab_size)