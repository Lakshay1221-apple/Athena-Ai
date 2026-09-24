"""Centralized configuration for Athena AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Find project root directory (directory containing pyproject.toml or src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "Data"
    books_dir: Path = PROJECT_ROOT / "Data" / "books"
    datasets_dir: Path = PROJECT_ROOT / "Data" / "datasets"
    evaluation_dir: Path = PROJECT_ROOT / "Data" / "evaluation"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    runs_dir: Path = PROJECT_ROOT / "outputs" / "runs"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Default v1 dataset paths
    v1_dataset_dir: Path = PROJECT_ROOT / "Data" / "datasets" / "v1"
    v1_gemma_jsonl: Path = PROJECT_ROOT / "Data" / "datasets" / "v1" / "ml_teacher_dataset.jsonl"
    v1_notebooklm_dir: Path = PROJECT_ROOT / "Data" / "datasets" / "v1" / "notebooklm"
    v1_merged_json: Path = PROJECT_ROOT / "Data" / "datasets" / "v1" / "merged" / "athena_dataset.json"
    v1_formatted_jsonl: Path = PROJECT_ROOT / "Data" / "datasets" / "v1" / "merged" / "formatted_dataset.jsonl"


@dataclass
class DatasetGenConfig:
    source_book_default: str = "HandsonMachine-Learning.pdf"
    chunk_size: int = 750
    chunk_overlap: int = 75
    margin_top: float = 50.0
    margin_bottom: float = 55.0
    min_response_length: int = 200
    output_version: str = "v1"


@dataclass
class GenerationModelConfig:
    ollama_model: str = "gemma2:2b"
    temperature: float = 0.3
    num_predict: int = 2048
    num_ctx: int = 4096
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class FineTuningConfig:
    base_model: str = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # SFT Training hyperparameters
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 5
    eval_steps: int = 10
    save_steps: int = 20
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataset_text_field: str = "text"
    validation_split: float = 0.1
    seed: int = 42


@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


# Global configuration instances
PATHS = PathConfig()
DATASET_GEN_CONFIG = DatasetGenConfig()
GEN_MODEL_CONFIG = GenerationModelConfig()
FINETUNING_CONFIG = FineTuningConfig()
INFERENCE_CONFIG = InferenceConfig()
