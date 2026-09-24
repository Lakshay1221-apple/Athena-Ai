"""Trainer configuration and factory for Athena AI SFT training."""

from pathlib import Path
from typing import Optional


from datasets import Dataset
from transformers import PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer

from src.common.config import FINETUNING_CONFIG, PATHS
from src.common.logger import get_logger

logger = get_logger("trainer")


def create_trainer(
    model,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Optional[Path] = None,
    config: Optional[object] = None,
) -> SFTTrainer:
    """
    Create a configured SFTTrainer instance with evaluation and checkpointing.
    """
    cfg = config if config is not None else FINETUNING_CONFIG
    checkpoints_dir = Path(output_dir) if output_dir else PATHS.outputs_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Compute step intervals dynamically if dataset is small
    num_samples = len(train_dataset)
    effective_batch_size = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, num_samples // effective_batch_size)

    # Set evaluation and save intervals to roughly 0.5 - 1.0 epochs
    eval_steps = max(5, min(cfg.eval_steps, steps_per_epoch))
    save_steps = max(10, min(cfg.save_steps, steps_per_epoch * 2))

    logger.info(
        f"Configuring Training: epochs={cfg.num_train_epochs}, "
        f"effective_batch={effective_batch_size}, steps_per_epoch={steps_per_epoch}, "
        f"eval_steps={eval_steps}, save_steps={save_steps}"
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field=cfg.dataset_text_field,
        max_seq_length=cfg.max_seq_length,
        packing=False,
        args=training_args,
    )

    return trainer