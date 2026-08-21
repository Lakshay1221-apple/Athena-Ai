# 1. BitsAndBytesConfig (Quantization)

```python
bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,            # Load model weights in 4-bit instead of 16/32-bit
                                 # Reduces VRAM usage dramatically

    bnb_4bit_quant_type="nf4",    # Quantization method
                                 # NF4 is recommended for QLoRA

    bnb_4bit_compute_dtype=torch.float16,
                                 # Computations happen in float16
                                 # Faster and memory efficient

    bnb_4bit_use_double_quant=False
                                 # Compress already compressed weights again
                                 # Saves more VRAM but may slow training
)
```

---

# 2. LoRA / PEFT Config

```python
peft_config = LoraConfig(

    r=16,                        # Rank of LoRA matrices
                                 # Higher = more trainable parameters

    lora_alpha=32,               # Scaling factor
                                 # Usually 2 × r

    lora_dropout=0.05,           # Dropout for LoRA layers
                                 # Helps prevent overfitting

    bias="none",                 # Whether bias parameters are trained
                                 # Usually keep as "none"

    task_type="CAUSAL_LM"        # Model type
                                 # Use for Llama, Mistral, Qwen etc.
)
```

---

# 3. TrainingArguments

```python
training_args = TrainingArguments(

    output_dir="./results",      # Where checkpoints are saved

    num_train_epochs=3,          # Number of full passes through dataset

    per_device_train_batch_size=4,
                                 # Samples processed simultaneously

    gradient_accumulation_steps=4,
                                 # Accumulate gradients
                                 # Effective batch = 4 × 4 = 16

    learning_rate=2e-4,          # Most important hyperparameter
                                 # Controls update size

    weight_decay=0.001,          # Regularization
                                 # Helps reduce overfitting

    warmup_ratio=0.03,           # Slowly increase LR at start

    max_grad_norm=0.3,           # Gradient clipping
                                 # Prevents exploding gradients

    fp16=True,                   # Use float16 training
                                 # Saves memory and speeds training

    save_steps=100,              # Save checkpoint every 100 steps

    logging_steps=25,            # Print logs every 25 steps

    lr_scheduler_type="cosine",  # Learning rate schedule

    optim="paged_adamw_32bit",   # Optimizer

    report_to="none"             # Disable WandB/TensorBoard
)
```

---

# 4. GenerationConfig

Used only after training.

```python
generation_config = GenerationConfig(

    max_new_tokens=256,          # Maximum response length

    temperature=0.7,            # Creativity
                                # Lower = deterministic
                                # Higher = creative

    top_p=0.9,                  # Nucleus sampling
                                # Consider top 90% probability mass

    top_k=50,                   # Consider only top 50 candidate tokens

    repetition_penalty=1.1,     # Reduce repeated words/sentences

    do_sample=True              # Enable sampling
                                # Required for temperature/top_p
)
```

---

# 5. SFTTrainer

```python
trainer = SFTTrainer(

    model=model,                # Base model + LoRA

    train_dataset=dataset,      # Dataset used for training

    peft_config=peft_config,    # LoRA configuration

    tokenizer=tokenizer,        # Tokenizer

    dataset_text_field="text",  # Column containing training text

    args=training_args,         # Training hyperparameters

    packing=False               # Combine multiple examples into one sequence
                                # Usually False for beginners
)
```

---

# The 8 Hyperparameters You Should Know First

If you were preparing for projects or interviews, I'd focus on these first:

| Hyperparameter   | What it Controls           |
| ---------------- | -------------------------- |
| `learning_rate`  | How fast model learns      |
| `batch_size`     | Samples processed together |
| `epochs`         | Number of passes over data |
| `r`              | LoRA adapter size          |
| `lora_alpha`     | LoRA strength              |
| `temperature`    | Creativity                 |
| `top_p`          | Response diversity         |
| `max_new_tokens` | Response length            |

These eight account for most of the tuning decisions you'll make when fine-tuning LLMs.

As you continue learning ML engineering, you'll notice that 90% of experiments are essentially changing:

```text
learning_rate
batch_size
epochs
LoRA rank (r)
```

and seeing how loss and model quality change. The other parameters are usually adjusted much less frequently.

**more realistic ML Engineer style pipeline** that includes:

✅ Dataset Loading
✅ Dataset Cleaning
✅ Dataset Formatting
✅ Train/Validation Split
✅ Tokenizer
✅ Quantization (QLoRA)
✅ LoRA Config
✅ TrainingArguments
✅ Evaluation Dataset
✅ Early Stopping
✅ TensorBoard Logging
✅ Checkpoint Saving
✅ Resume Training
✅ Save Adapter
✅ Merge LoRA
✅ Inference

This is much closer to what you'd build in a real project.

```python
# =====================================================
# IMPORTS
# =====================================================

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
    EarlyStoppingCallback
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

from trl import SFTTrainer


# =====================================================
# DATASET
# =====================================================

raw_dataset = load_dataset(
    "mlabonne/guanaco-llama2-1k",
    split="train"
)


# =====================================================
# DATA CLEANING
# =====================================================

def clean_dataset(example):

    question = example["text"].strip()

    return {
        "text": question
    }

raw_dataset = raw_dataset.map(clean_dataset)


# =====================================================
# DATA FORMATTING
# =====================================================

def format_prompt(example):

    return {
        "text": example["text"]
    }

dataset = raw_dataset.map(format_prompt)


# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]


# =====================================================
# MODEL NAME
# =====================================================

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# =====================================================
# QUANTIZATION CONFIG
# =====================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=False
)


# =====================================================
# LOAD MODEL
# =====================================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False


# =====================================================
# LORA CONFIG
# =====================================================

peft_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# =====================================================
# WRAP MODEL WITH LORA
# =====================================================

model = get_peft_model(
    model,
    peft_config
)

model.print_trainable_parameters()


# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=3,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    weight_decay=0.001,

    warmup_ratio=0.03,

    max_grad_norm=0.3,

    fp16=True,

    save_strategy="steps",

    save_steps=100,

    save_total_limit=3,

    eval_strategy="steps",

    eval_steps=100,

    logging_steps=25,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    lr_scheduler_type="cosine",

    report_to="tensorboard"
)


# =====================================================
# TRAINER
# =====================================================

trainer = SFTTrainer(

    model=model,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    peft_config=peft_config,

    tokenizer=tokenizer,

    dataset_text_field="text",

    args=training_args,

    packing=False,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ]
)


# =====================================================
# TRAIN
# =====================================================

trainer.train()


# =====================================================
# RESUME TRAINING LATER
# =====================================================

# trainer.train(
#     resume_from_checkpoint=True
# )


# =====================================================
# SAVE ADAPTER
# =====================================================

adapter_path = "./my_lora_adapter"

trainer.model.save_pretrained(
    adapter_path
)

tokenizer.save_pretrained(
    adapter_path
)


# =====================================================
# LOAD BASE MODEL
# =====================================================

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  
)


# =====================================================
# LOAD ADAPTER
# =====================================================

model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)


# =====================================================
# MERGE LORA
# =====================================================

merged_model = model.merge_and_unload()

merged_model.save_pretrained(
    "./merged_model"
)

tokenizer.save_pretrained(
    "./merged_model"
)


# =====================================================
# GENERATION CONFIG
# =====================================================

generation_config = GenerationConfig(

    max_new_tokens=256,

    temperature=0.7,

    top_p=0.9,

    top_k=50,

    repetition_penalty=1.1,

    do_sample=True
)


# =====================================================
# INFERENCE
# =====================================================

prompt = "Explain neural networks."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    generation_config=generation_config
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)
```

One thing to note: in **actual production training**, the code is usually split into multiple files:

```text
project/
│
├── train.py
├── inference.py
├── config.py
├── dataset.py
├── formatting.py
├── trainer.py
├── utils.py
│
├── data/
├── checkpoints/
├── logs/
└── outputs/
```

