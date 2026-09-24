"""Dataset validation module for Athena AI.

Validates datasets before fine-tuning to ensure quality, schema compliance,
and absence of duplicates.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.config import PATHS
from src.common.logger import get_logger
from src.common.utils import calculate_text_hash, load_json, load_jsonl

logger = get_logger("dataset_validator")


class DatasetValidator:
    REQUIRED_CORE_FIELDS = ["instruction", "response"]

    def __init__(self, min_response_length: int = 50, max_token_approx: int = 4096):
        self.min_response_length = min_response_length
        self.max_token_approx = max_token_approx

    def validate_example(
        self, example: Dict[str, Any], seen_hashes: set
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a single example (supports both formatted datasets with 'text'
        and raw datasets with 'instruction'/'response').
        Returns: (is_valid, failure_reason, example_hash)
        """
        if not isinstance(example, dict):
            return False, "Record is not a valid JSON dictionary", None

        # Case 1: Formatted dataset (contains canonical 'text' field)
        if "text" in example:
            text_val = example["text"]
            if not isinstance(text_val, str) or not text_val.strip():
                return False, "Field 'text' is empty or not a string", None

            if len(text_val.strip()) < self.min_response_length:
                return False, f"Formatted text too short ({len(text_val.strip())} chars < {self.min_response_length})", None

            ex_hash = calculate_text_hash(text_val)
            if ex_hash in seen_hashes:
                return False, "Duplicate example found", ex_hash

            if len(text_val) > self.max_token_approx * 4:
                return False, f"Excessive sequence length (~{len(text_val) // 4} tokens)", ex_hash

            return True, None, ex_hash

        # Case 2: Raw instruction-response dataset
        for field in self.REQUIRED_CORE_FIELDS:
            if field not in example:
                return False, f"Missing required field: '{field}' (or 'text' for formatted datasets)", None
            val = example[field]
            if not isinstance(val, str) or not val.strip():
                return False, f"Field '{field}' is empty or not a string", None

        # Check response length
        if len(example["response"].strip()) < self.min_response_length:
            return (
                False,
                f"Response too short ({len(example['response'].strip())} chars < {self.min_response_length})",
                None,
            )

        # Check duplicates
        comb_text = f"{example['instruction'].strip()} || {example['response'].strip()}"
        ex_hash = calculate_text_hash(comb_text)
        if ex_hash in seen_hashes:
            return False, "Duplicate example found", ex_hash

        # Approximate token count (1 token ~= 4 chars)
        total_len = len(example["instruction"]) + len(example["response"])
        if total_len > self.max_token_approx * 4:
            return False, f"Excessive sequence length (~{total_len // 4} tokens)", ex_hash

        return True, None, ex_hash

    def validate_dataset(
        self, dataset_path: Path
    ) -> Dict[str, Any]:
        """
        Validate entire dataset file (JSON or JSONL).
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

        if dataset_path.suffix == ".jsonl":
            records = load_jsonl(dataset_path)
        elif dataset_path.suffix == ".json":
            data = load_json(dataset_path)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "examples" in data:
                records = data["examples"]
            else:
                raise ValueError("JSON file must be a list of records or dict containing 'examples'")
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

        total_examples = len(records)
        valid_records = []
        invalid_records = []
        duplicate_count = 0
        seen_hashes = set()

        for idx, record in enumerate(records):
            is_valid, reason, ex_hash = self.validate_example(record, seen_hashes)
            if is_valid:
                valid_records.append(record)
                if ex_hash:
                    seen_hashes.add(ex_hash)
            else:
                invalid_records.append({
                    "index": idx,
                    "reason": reason,
                    "record_sample": str(record)[:150]
                })
                if reason == "Duplicate example found":
                    duplicate_count += 1

        is_passed = len(invalid_records) == 0 and total_examples > 0

        report = {
            "dataset_path": str(dataset_path),
            "total_examples": total_examples,
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records),
            "duplicate_count": duplicate_count,
            "status": "PASS" if is_passed else "FAIL",
            "issues": invalid_records[:20],  # show first 20 issues
        }

        self.print_report(report)
        return report

    def print_report(self, report: Dict[str, Any]):
        status_color = "PASS" if report["status"] == "PASS" else "FAIL"
        print("\n" + "=" * 50)
        print(" Athena AI — Dataset Validation Report")
        print("=" * 50)
        print(f" Dataset File    : {report['dataset_path']}")
        print(f" Total Examples  : {report['total_examples']}")
        print(f" Valid           : {report['valid_count']}")
        print(f" Invalid         : {report['invalid_count']}")
        print(f" Duplicates      : {report['duplicate_count']}")
        print(f" Status          : {status_color}")
        print("=" * 50)
        if report["issues"]:
            print("\nFirst Issues Detected:")
            for issue in report["issues"][:5]:
                print(f" - [Record {issue['index']}]: {issue['reason']}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for Athena AI fine-tuning")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(PATHS.v1_merged_json),
        help="Path to JSON/JSONL dataset file",
    )
    args = parser.parse_args()

    validator = DatasetValidator()
    report = validator.validate_dataset(Path(args.dataset_path))

    if report["status"] != "PASS":
        logger.warning(f"Dataset validation FAILED for {args.dataset_path}")
        exit(1)
    else:
        logger.info(f"Dataset validation PASSED ({report['valid_count']} valid examples)")


if __name__ == "__main__":
    main()
