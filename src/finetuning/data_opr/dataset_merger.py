"""Dataset merger for Athena AI.

Merges generated Gemma outputs and NotebookLM batches, validates schemas,
cleans conversational filler, deduplicates deterministically, and saves the final dataset.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.config import PATHS
from src.common.logger import get_logger
from src.common.utils import calculate_text_hash, ensure_directory, load_json, load_jsonl, save_json

logger = get_logger("dataset_merger")


class DatasetMerger:
    REQUIRED_FIELDS = [
        "instruction",
        "response",
        "concept",
        "difficulty",
        "chapter",
        "source_book",
    ]

    GREETING_PATTERNS = [
        r"^Hello!?",
        r"^Hi!?",
        r"^Certainly!?",
        r"^Great question!?",
        r"^I'd be happy to help\.?",
        r"^I would be glad to help\.?",
        r"^I'd be glad to explain\.?",
        r"^Let's dive in\.?",
        r"^Sure,?\s*",
    ]

    def __init__(
        self,
        gemma_path: Optional[Path] = None,
        notebooklm_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ):
        self.gemma_path = Path(gemma_path) if gemma_path else PATHS.v1_gemma_jsonl
        self.notebooklm_dir = Path(notebooklm_dir) if notebooklm_dir else PATHS.v1_notebooklm_dir
        self.output_path = Path(output_path) if output_path else PATHS.v1_merged_json

    def load_gemma_dataset(self) -> List[Dict[str, Any]]:
        examples = []
        if not self.gemma_path.exists():
            logger.warning(f"Gemma dataset file not found at: {self.gemma_path}")
            return examples

        examples = load_jsonl(self.gemma_path)
        logger.info(f"Loaded {len(examples)} Gemma examples from {self.gemma_path.name}")
        return examples

    def load_notebooklm_dataset(self) -> List[Dict[str, Any]]:
        examples = []
        if not self.notebooklm_dir.exists():
            logger.warning(f"NotebookLM directory not found at: {self.notebooklm_dir}")
            return examples

        batch_files = sorted(self.notebooklm_dir.glob("*.json"))
        for file in batch_files:
            try:
                batch = load_json(file)
                if isinstance(batch, list):
                    examples.extend(batch)
                    logger.info(f"Loaded {len(batch)} examples from {file.name}")
                else:
                    logger.warning(f"File {file.name} is not a list of examples.")
            except Exception as e:
                logger.error(f"Failed to load {file.name}: {e}")

        logger.info(f"Loaded total {len(examples)} NotebookLM examples")
        return examples

    def validate_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            is_valid = True
            for field in self.REQUIRED_FIELDS:
                if field not in example or not str(example[field]).strip():
                    is_valid = False
                    break
            if is_valid:
                valid.append(example)

        logger.info(f"Valid examples: {len(valid)} / {len(examples)}")
        return valid

    def clean_response(self, response: str) -> str:
        cleaned = response.strip()
        for pattern in self.GREETING_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def clean_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for example in examples:
            example["instruction"] = example["instruction"].strip()
            example["response"] = self.clean_response(example["response"])
            example["concept"] = example["concept"].strip()
            # Normalize difficulty
            diff = str(example.get("difficulty", "")).strip().capitalize()
            if diff in ["Beginner", "Intermediate", "Advanced"]:
                example["difficulty"] = diff
            else:
                example["difficulty"] = "Intermediate"
        return examples

    def deduplicate(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_hashes = set()
        unique_examples = []

        for example in examples:
            # Deterministic hash of instruction + response
            comb_text = f"{example['instruction']} || {example['response']}"
            ex_hash = calculate_text_hash(comb_text)

            if ex_hash not in seen_hashes:
                seen_hashes.add(ex_hash)
                # Store example hash for traceability
                example["example_hash"] = ex_hash
                unique_examples.append(example)

        logger.info(f"Unique examples after deduplication: {len(unique_examples)} (filtered {len(examples) - len(unique_examples)} duplicates)")
        return unique_examples

    def save(self, examples: List[Dict[str, Any]]) -> Path:
        ensure_directory(self.output_path.parent)
        save_json(examples, self.output_path)
        logger.info(f"Saved merged dataset to: {self.output_path}")
        return self.output_path

    def run(self) -> List[Dict[str, Any]]:
        logger.info("Starting Dataset Merge Process...")
        gemma_examples = self.load_gemma_dataset()
        notebooklm_examples = self.load_notebooklm_dataset()

        all_examples = gemma_examples + notebooklm_examples
        logger.info(f"Total Raw Combined Examples: {len(all_examples)}")

        all_examples = self.validate_examples(all_examples)
        all_examples = self.clean_examples(all_examples)
        all_examples = self.deduplicate(all_examples)
        self.save(all_examples)

        logger.info(f"Dataset Merge Complete. Final Dataset Size: {len(all_examples)}")
        return all_examples


if __name__ == "__main__":
    merger = DatasetMerger()
    merger.run()