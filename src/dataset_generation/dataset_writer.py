"""Dataset writer module with atomic saves and checkpointing for Athena AI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.config import PATHS
from src.common.logger import get_logger
from src.common.utils import ensure_directory

logger = get_logger("dataset_writer")


class DatasetWriter:
    def __init__(self, output_base_dir: Optional[Path] = None):
        self.output_base_dir = Path(output_base_dir) if output_base_dir else PATHS.datasets_dir

    def _atomic_write_json(self, data: Any, target_path: Path, indent: int = 2):
        """Write to temp file and atomically replace target."""
        ensure_directory(target_path.parent)
        temp_path = target_path.with_suffix(f".tmp_{datetime.now().strftime('%f')}")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
        temp_path.replace(target_path)

    def write_dataset(self, examples: List[Dict[str, Any]], version: str) -> Dict[str, Path]:
        version_dir = self.output_base_dir / version
        ensure_directory(version_dir)

        json_path = version_dir / "ml_teacher_dataset.json"
        jsonl_path = version_dir / "ml_teacher_dataset.jsonl"

        # 1. Atomic write standard JSON format
        self._atomic_write_json(examples, json_path)

        # 2. Atomic write JSONL format
        temp_jsonl = jsonl_path.with_suffix(f".tmp_{datetime.now().strftime('%f')}")
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            f.flush()
        temp_jsonl.replace(jsonl_path)

        logger.info(f"Dataset successfully saved to {json_path} and {jsonl_path}")
        return {
            "json": json_path,
            "jsonl": jsonl_path,
        }

    def append_example(self, example: Dict[str, Any], version: str) -> Path:
        version_dir = self.output_base_dir / version
        ensure_directory(version_dir)
        jsonl_path = version_dir / "ml_teacher_dataset.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            f.flush()
        return jsonl_path

    def write_checkpoint(self, completed_chunks: List[str], version: str) -> Path:
        version_dir = self.output_base_dir / version
        checkpoint_path = version_dir / "checkpoint.json"
        data = {
            "completed_chunks": completed_chunks,
            "generated_examples": len(completed_chunks),
            "timestamp": datetime.now().isoformat(),
        }
        self._atomic_write_json(data, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, version: str) -> List[str]:
        checkpoint_path = self.output_base_dir / version / "checkpoint.json"
        if not checkpoint_path.exists():
            return []
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("completed_chunks", [])
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return []

    def write_failed_chunk(self, chunk_id: str, reason: str, version: str) -> Path:
        version_dir = self.output_base_dir / version
        failed_path = version_dir / "failed_chunks.json"

        failures = []
        if failed_path.exists():
            try:
                with open(failed_path, "r", encoding="utf-8") as f:
                    failures = json.load(f)
                    if not isinstance(failures, list):
                        failures = []
            except Exception:
                failures = []

        if not any(f.get("chunk_id") == chunk_id for f in failures):
            failures.append({
                "chunk_id": chunk_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })
            self._atomic_write_json(failures, failed_path)
        return failed_path

    def load_failed_chunks(self, version: str) -> List[Dict[str, Any]]:
        failed_path = self.output_base_dir / version / "failed_chunks.json"
        if not failed_path.exists():
            return []
        try:
            with open(failed_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def write_statistics(self, examples: List[Dict[str, Any]], version: str) -> Path:
        version_dir = self.output_base_dir / version
        stats_path = version_dir / "dataset_statistics.json"

        total_examples = len(examples)
        beginner_count = sum(1 for e in examples if e.get("difficulty") == "Beginner")
        intermediate_count = sum(1 for e in examples if e.get("difficulty") == "Intermediate")
        advanced_count = sum(1 for e in examples if e.get("difficulty") == "Advanced")

        avg_len = 0
        if total_examples > 0:
            avg_len = sum(len(e.get("response", "")) for e in examples) / total_examples

        unique_concepts = len(set(e.get("concept", "").strip().lower() for e in examples if e.get("concept")))

        stats = {
            "total_examples": total_examples,
            "beginner_count": beginner_count,
            "intermediate_count": intermediate_count,
            "advanced_count": advanced_count,
            "average_response_length": round(avg_len, 2),
            "unique_concepts": unique_concepts,
            "timestamp": datetime.now().isoformat(),
        }

        self._atomic_write_json(stats, stats_path)
        return stats_path

    def write_report(self, report_data: Dict[str, Any], version: str) -> Path:
        version_dir = self.output_base_dir / version
        report_path = version_dir / "generation_report.json"

        if "timestamp" not in report_data:
            report_data["timestamp"] = datetime.now().isoformat()

        self._atomic_write_json(report_data, report_path)
        logger.info(f"Generation report successfully saved to {report_path}")
        return report_path
