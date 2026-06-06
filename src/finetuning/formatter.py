from datasets import Dataset
from transformers import AutoTokenizer

def format_dataset(dataset, model_name : str = "unsloth/Llama-3.2-1B-Instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def format_example(example):

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

        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False,)

        return {"text": text}

    formatted_dataset = dataset.map(format_example)
    formatted_dataset.to_json(
    "Data/datasets/v1/merged/formatted_dataset.jsonl")


    print("Formatted dataset saved.")

    return formatted_dataset
    
if __name__ == "__main__":

    import json

    with open("Data/datasets/v1/merged/athena_dataset.json","r",encoding="utf-8",) as f:

        data = json.load(f)

    dataset = Dataset.from_list(data)

    formatted_dataset = format_dataset(dataset)

    print("\nFormatted Example:\n")
    print(formatted_dataset[0]["text"])
