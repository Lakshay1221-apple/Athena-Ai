from datasets import load_dataset

def load_athena_dataset(dataset_path :str, test_size : float = 0.2,seed : int = 42):

    """
    Load Athena AI dataset and create train/validation splits.

    Args:
        dataset_path: Path to JSON dataset
        test_size: Validation split size
        seed: Random seed

    Returns:
        train_dataset, val_dataset
    """

    dataset = load_dataset("json", data_files = dataset_path,)['train']

    required_columns = {"instruction", "response"}

    if not required_columns.issubset(set(dataset.column_names)):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}"
    )
    
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["instruction", "response"]])

    split_dataset = dataset.train_test_split (test_size = test_size, seed = seed, shuffle = True)  

    train_dataset = split_dataset['train'] 
    val_dataset = split_dataset['test'] 

    print(f"Loaded dataset with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    print(f"Total examples      : {len(dataset)}")
    print(f"Training examples   : {len(train_dataset)}")
    print(f"Validation examples : {len(val_dataset)}")

    return train_dataset, val_dataset


if __name__ == "__main__":

    train_dataset, val_dataset = load_athena_dataset(
        dataset_path="Data/datasets/v1/ml_teacher_dataset.jsonl"

    )

    print("\nSample Training Example:\n")
    print(train_dataset[0])





