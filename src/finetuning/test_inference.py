from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/final_model",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": "What is regularization?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)

print(response)