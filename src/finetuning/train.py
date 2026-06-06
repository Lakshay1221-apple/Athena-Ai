import json 
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import AutoTokenizer 

with open("Data/datasets/v1/merged/athena_dataset.json","r", encoding = 'utf-8',) as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)
def formatting_func(example):

    messages = [
        {
            "role": "user",
            "content": example["instruction"],
        },
        {
            "role": "assistant",
            "content": example["response"],
        },
    ]

    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,)