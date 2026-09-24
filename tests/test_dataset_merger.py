"""Unit tests for DatasetMerger module."""

import tempfile
import unittest
from pathlib import Path
from src.finetuning.dataset_merger import DatasetMerger
from src.common.utils import save_json, save_jsonl


class TestDatasetMerger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        self.gemma_file = self.dir_path / "gemma.jsonl"
        self.notebooklm_dir = self.dir_path / "notebooklm"
        self.notebooklm_dir.mkdir()
        self.output_file = self.dir_path / "merged.json"

        # Mock gemma records
        gemma_data = [
            {
                "instruction": "What is SGD?",
                "response": "Hello! Stochastic Gradient Descent computes gradients on batches.",
                "concept": "SGD",
                "difficulty": "Intermediate",
                "chapter": "Ch 4",
                "source_book": "Book A",
            },
            {
                "instruction": "What is SGD?",  # duplicate
                "response": "Certainly! Stochastic Gradient Descent computes gradients on batches.",
                "concept": "SGD",
                "difficulty": "Intermediate",
                "chapter": "Ch 4",
                "source_book": "Book A",
            },
        ]
        save_jsonl(gemma_data, self.gemma_file)

        # Mock NotebookLM batch
        notebook_data = [
            {
                "instruction": "Explain Linear Regression.",
                "response": "Great question! Linear regression models linear relationships.",
                "concept": "Linear Regression",
                "difficulty": "Beginner",
                "chapter": "Ch 1",
                "source_book": "Book B",
            }
        ]
        save_json(notebook_data, self.notebooklm_dir / "batch_01.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clean_response(self):
        merger = DatasetMerger(
            gemma_path=self.gemma_file,
            notebooklm_dir=self.notebooklm_dir,
            output_path=self.output_file,
        )
        cleaned = merger.clean_response("Hello! Here is the explanation.")
        self.assertEqual(cleaned, "Here is the explanation.")

    def test_merge_and_deduplicate(self):
        merger = DatasetMerger(
            gemma_path=self.gemma_file,
            notebooklm_dir=self.notebooklm_dir,
            output_path=self.output_file,
        )
        merged = merger.run()

        # Total 3 raw -> 2 unique after deduplication
        self.assertEqual(len(merged), 2)
        self.assertTrue(self.output_file.exists())
        # Check greetings removed
        instructions = [m["instruction"] for m in merged]
        self.assertIn("What is SGD?", instructions)
        self.assertIn("Explain Linear Regression.", instructions)


if __name__ == "__main__":
    unittest.main()
