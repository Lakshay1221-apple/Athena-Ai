import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class DatasetWriter:
    def __init__(self, output_base_dir: str = "Data/datasets"):
        """
        Initialize DatasetWriter.

        Args:
            output_base_dir (str): Base directory where all datasets and versions are stored.
        """
        self.output_base_dir = Path(output_base_dir)

    def write_dataset(self, examples: List[Dict[str, Any]], version: str) -> Dict[str, Path]:
        """
        Save the dataset to JSON and JSONL formats under output_base_dir/{version}/.

        Args:
            examples (List[Dict[str, Any]]): The list of generated examples to save.
            version (str): The dataset version (e.g., "v1", "v2").

        Returns:
            Dict[str, Path]: Paths to the written JSON and JSONL files.
        """
        version_dir = self.output_base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        json_path = version_dir / "ml_teacher_dataset.json"
        jsonl_path = version_dir / "ml_teacher_dataset.jsonl"

        # 1. Write standard JSON format
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        # 2. Write JSON Lines (JSONL) format
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"Dataset successfully saved to {json_path} and {jsonl_path}")
        return {
            "json": json_path,
            "jsonl": jsonl_path
        }

    def append_example(self, example: Dict[str, Any], version: str) -> Path:
        """
        Append a single record to output_base_dir/{version}/ml_teacher_dataset.jsonl immediately.

        Args:
            example (Dict[str, Any]): Example dict to append.
            version (str): Dataset version.

        Returns:
            Path: Path to the JSONL file.
        """
        version_dir = self.output_base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = version_dir / "ml_teacher_dataset.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
        return jsonl_path

    def write_checkpoint(self, completed_chunks: List[str], version: str) -> Path:
        """
        Write list of completed chunk IDs to output_base_dir/{version}/checkpoint.json.
        """
        version_dir = self.output_base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = version_dir / "checkpoint.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "completed_chunks": completed_chunks,
                "generated_examples": len(completed_chunks),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        return checkpoint_path

    def load_checkpoint(self, version: str) -> List[str]:
        """
        Load completed chunk IDs from output_base_dir/{version}/checkpoint.json.
        """
        checkpoint_path = self.output_base_dir / version / "checkpoint.json"
        if not checkpoint_path.exists():
            return []
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("completed_chunks", [])
        except Exception:
            return []

    def write_failed_chunk(self, chunk_id: str, reason: str, version: str) -> Path:
        """
        Write failure details for a chunk to output_base_dir/{version}/failed_chunks.json.
        """
        version_dir = self.output_base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
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
                
        # Check if already added
        if not any(f.get("chunk_id") == chunk_id for f in failures):
            failures.append({
                "chunk_id": chunk_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })
            
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
        return failed_path

    def load_failed_chunks(self, version: str) -> List[Dict[str, Any]]:
        """
        Load failure details from output_base_dir/{version}/failed_chunks.json.
        """
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
        """
        Write summary statistics of the dataset to output_base_dir/{version}/dataset_statistics.json.
        """
        version_dir = self.output_base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
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
            "timestamp": datetime.now().isoformat()
        }
        
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        return stats_path

    def write_report(
        self,
        report_data: Dict[str, Any],
        version: str
    ) -> Path:
        """
        Save the generation report to output_base_dir/{version}/generation_report.json.

        Args:
            report_data (Dict[str, Any]): Dictionary of statistics and metadata.
            version (str): The dataset version.

        Returns:
            Path: Path to the written report.
        """
        version_dir = self.output_base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        report_path = version_dir / "generation_report.json"
        
        # Inject standard timestamp if not present
        if "timestamp" not in report_data:
            report_data["timestamp"] = datetime.now().isoformat()

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"Generation report successfully saved to {report_path}")
        return report_path
