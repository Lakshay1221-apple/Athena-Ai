from trl import SFTTrainer 
from transformers import TrainingArguments


def create_trainer(
        model,
        tokenizer,
        dataset,
        formatting_func,):
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        formatting_func = formatting_func,
        max_seq_length = 2048,

        args = TrainingArguments(
            output_dir="outputs",
            num_train_epochs=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            fp16=True,
        ),
    )

    return trainer

    
    
