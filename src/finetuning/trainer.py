from trl import SFTTrainer
from transformers import TrainingArguments


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
):

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        args = TrainingArguments(
                    output_dir="outputs",
                    num_train_epochs=3,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=8,
                    learning_rate=2e-4,
                    logging_steps=1,
                    save_strategy="no",
                    save_total_limit=2,
                    eval_strategy="no",
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    warmup_steps=10,
                    fp16=True,
                    bf16=False,
                    report_to="none",
        ),

    )

    return trainer