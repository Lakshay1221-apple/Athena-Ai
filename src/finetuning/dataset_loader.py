from datasets import load_dataset


def load_formatted_dataset(
    dataset_path: str,
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Load pre-formatted dataset containing a 'text' column.
    """

    dataset = load_dataset("json",data_files=dataset_path,)["train"]

    if "text" not in dataset.column_names:
        raise ValueError("Formatted dataset must contain a 'text' column.")

    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    split_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=True,
    )

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Loaded dataset with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print(f"Total examples      : {len(dataset)}")
    print(f"Training examples   : {len(train_dataset)}")
    print(f"Validation examples : {len(val_dataset)}")

    return train_dataset, val_dataset


if __name__ == "__main__":

    train_dataset, val_dataset = load_formatted_dataset(
        "Data/datasets/v1/merged/formatted_dataset.jsonl"
    )

    print(train_dataset[0])