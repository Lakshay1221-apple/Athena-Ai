from dataset_loader import load_formatted_dataset
from model_loader import load_model_and_tokenizer
from lora_config import apply_lora
from trainer import create_trainer


def main():

    train_dataset, val_dataset = load_formatted_dataset(
        "Data/datasets/v1/merged/formatted_dataset.jsonl"
    )

    model, tokenizer = load_model_and_tokenizer()

    model = apply_lora(model)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("\nTrainer Created Successfully!\n")
    print(trainer)

    trainer.train()

    model.save_pretrained("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")


if __name__ == "__main__":
    main()