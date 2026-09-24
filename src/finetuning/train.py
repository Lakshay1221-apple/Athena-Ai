"""Main training orchestration script for Athena AI fine-tuning."""

import argparse
import datetime
import platform
import sys
from pathlib import Path

import torch
import transformers
import trl
from peft import __version__ as peft_version

from src.common.config import FINETUNING_CONFIG, PATHS
from src.common.logger import get_logger
from src.common.utils import calculate_file_hash, ensure_directory, save_json
from src.finetuning.dataset_loader import load_formatted_dataset
from src.finetuning.dataset_validator import DatasetValidator
from src.finetuning.lora_config import apply_lora
from src.finetuning.model_loader import load_model_and_tokenizer
from src.finetuning.trainer import create_trainer

logger = get_logger("train")


def train(
    dataset_path: Path = PATHS.v1_formatted_jsonl,
    resume_from_checkpoint: bool = False,
    checkpoint_dir: str = None,
):
    """
    Run full SFT training pipeline with isolated run artifact directories and reproducibility metadata.
    """
    # 1. Validate dataset before training
    logger.info("Validating dataset integrity before training...")
    validator = DatasetValidator()
    validation_report = validator.validate_dataset(dataset_path)
    if validation_report["status"] != "PASS":
        raise ValueError(
            f"Dataset validation failed with {validation_report['invalid_count']} invalid records. "
            "Please fix the dataset before fine-tuning."
        )

    # 2. Setup run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = PATHS.runs_dir / run_id
    adapter_dir = run_dir / "adapter"
    checkpoints_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluation"
    logs_dir = run_dir / "logs"

    for d in [adapter_dir, checkpoints_dir, evaluation_dir, logs_dir]:
        ensure_directory(d)

    logger.info(f"Initialized training run: {run_id}")
    logger.info(f"Artifacts will be stored at: {run_dir}")

    # 3. Load dataset
    train_dataset, val_dataset = load_formatted_dataset(
        dataset_path=dataset_path,
        validation_split=FINETUNING_CONFIG.validation_split,
        seed=FINETUNING_CONFIG.seed,
    )

    # 4. Load base model & tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 5. Apply LoRA
    model = apply_lora(model)
    model.print_trainable_parameters()

    # 6. Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=checkpoints_dir,
    )

    # 7. Execute training
    logger.info("Starting SFT Training...")
    if resume_from_checkpoint:
        resume_target = checkpoint_dir if checkpoint_dir else True
        logger.info(f"Resuming from checkpoint: {resume_target}")
        train_result = trainer.train(resume_from_checkpoint=resume_target)
    else:
        train_result = trainer.train()

    logger.info("Training completed successfully!")

    # 8. Save adapter & tokenizer to run adapter directory
    logger.info(f"Saving final LoRA adapter to {adapter_dir}...")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Also maintain latest pointer under outputs/final_model for convenience
    latest_adapter_dir = ensure_directory(PATHS.outputs_dir / "final_model")
    model.save_pretrained(str(latest_adapter_dir))
    tokenizer.save_pretrained(str(latest_adapter_dir))

    # 9. Compute & save run metadata for reproducibility
    dataset_hash = calculate_file_hash(dataset_path)
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "base_model": FINETUNING_CONFIG.base_model,
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "total_dataset_size": len(train_dataset) + len(val_dataset),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "seed": FINETUNING_CONFIG.seed,
        "python_version": platform.python_version(),
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft_version,
        },
        "lora_config": {
            "r": FINETUNING_CONFIG.lora_r,
            "alpha": FINETUNING_CONFIG.lora_alpha,
            "dropout": FINETUNING_CONFIG.lora_dropout,
            "target_modules": FINETUNING_CONFIG.lora_target_modules,
        },
        "training_params": {
            "epochs": FINETUNING_CONFIG.num_train_epochs,
            "learning_rate": FINETUNING_CONFIG.learning_rate,
            "batch_size": FINETUNING_CONFIG.per_device_train_batch_size,
            "grad_accum": FINETUNING_CONFIG.gradient_accumulation_steps,
            "max_seq_length": FINETUNING_CONFIG.max_seq_length,
        },
        "metrics": {
            "train_runtime": getattr(train_result, "metrics", {}).get("train_runtime", None),
            "train_loss": getattr(train_result, "metrics", {}).get("train_loss", None),
        },
    }

    save_json(metadata, run_dir / "metadata.json")
    logger.info(f"Run metadata saved to {run_dir / 'metadata.json'}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train Athena AI teaching assistant model")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(PATHS.v1_formatted_jsonl),
        help="Path to formatted dataset JSONL",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Specific checkpoint path to resume from",
    )
    args = parser.parse_args()

    train(
        dataset_path=Path(args.dataset_path),
        resume_from_checkpoint=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()