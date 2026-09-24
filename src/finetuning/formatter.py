"""Dataset formatter for Athena AI.

Formats instructional pairs into standard ChatML conversation format with canonical 'text' field.
"""

from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer

from src.common.config import FINETUNING_CONFIG, PATHS
from src.common.logger import get_logger
from src.common.utils import ensure_directory, load_json, save_jsonl

logger = get_logger("formatter")

SYSTEM_PROMPT = (
    "You are Athena, an expert Machine Learning Teaching Assistant. "
    "Provide clear, pedagogically sound, and mathematically rigorous explanations "
    "accompanied by code snippets and practical intuitions where appropriate."
)


def format_dataset(
    dataset_input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    model_name: str = FINETUNING_CONFIG.base_model,
) -> Path:
    """
    Format dataset examples into ChatML text using model's tokenizer.
    """
    in_path = Path(dataset_input_path) if dataset_input_path else PATHS.v1_merged_json
    out_path = Path(output_path) if output_path else PATHS.v1_formatted_jsonl

    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {in_path}")

    logger.info(f"Loading raw dataset from {in_path}...")
    data = load_json(in_path)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of examples.")

    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    formatted_records = []
    for item in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["instruction"].strip()},
            {"role": "assistant", "content": item["response"].strip()},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        formatted_records.append({
            "text": text,
            "concept": item.get("concept", ""),
            "difficulty": item.get("difficulty", "Intermediate"),
            "example_hash": item.get("example_hash", ""),
        })

    ensure_directory(out_path.parent)
    save_jsonl(formatted_records, out_path)
    logger.info(f"Formatted dataset saved to: {out_path} ({len(formatted_records)} examples)")
    return out_path


if __name__ == "__main__":
    format_dataset()
