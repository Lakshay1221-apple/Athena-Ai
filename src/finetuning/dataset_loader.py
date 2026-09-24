"""Dataset loader module for Athena AI.

Loads pre-formatted dataset and creates deterministic train/validation splits.
"""

from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, load_dataset

from src.common.config import FINETUNING_CONFIG, PATHS
from src.common.logger import get_logger

logger = get_logger("dataset_loader")


def load_formatted_dataset(
    dataset_path: Optional[Path] = None,
    validation_split: float = FINETUNING_CONFIG.validation_split,
    seed: int = FINETUNING_CONFIG.seed,
) -> Tuple[Dataset, Dataset]:
    """
    Load pre-formatted dataset containing a canonical 'text' column,
    and split into train and validation datasets deterministically.
    """
    path = Path(dataset_path) if dataset_path else PATHS.v1_formatted_jsonl
    if not path.exists():
        raise FileNotFoundError(f"Formatted dataset file not found at: {path}")

    logger.info(f"Loading formatted dataset from {path}...")
    dataset = load_dataset("json", data_files=str(path))["train"]

    if "text" not in dataset.column_names:
        raise ValueError(
            f"Formatted dataset must contain a 'text' column. Found columns: {dataset.column_names}"
        )

    # Keep only the text column for clean SFT training
    columns_to_remove = [col for col in dataset.column_names if col != "text"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    # Deterministic Train / Validation Split
    split_dataset = dataset.train_test_split(
        test_size=validation_split,
        seed=seed,
        shuffle=True,
    )

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    logger.info(
        f"Dataset Split (seed={seed}, val_ratio={validation_split}): "
        f"Total={len(dataset)}, Train={len(train_dataset)}, Validation={len(val_dataset)}"
    )

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_ds, val_ds = load_formatted_dataset()
    logger.info(f"Sample train record text preview:\n{train_ds[0]['text'][:200]}...")