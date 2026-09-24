"""End-to-End Integration Test for Athena AI Pipeline."""

import tempfile
import unittest
from pathlib import Path

from src.dataset_generation.chunker import Chunker
from src.dataset_generation.text_cleaner import TextCleaner
from src.finetuning.dataset_loader import load_formatted_dataset
from src.finetuning.dataset_merger import DatasetMerger
from src.finetuning.dataset_validator import DatasetValidator
from src.finetuning.formatter import format_dataset
from src.common.utils import save_json, save_jsonl


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_e2e_dataset_pipeline(self):
        # 1. Text Cleaning & Chunking
        cleaner = TextCleaner()
        raw_text = "Gradient descent is an opti-\nmization algorithm used to find optimal weights."
        cleaned = cleaner.clean_segment(raw_text)
        self.assertIn("optimization algorithm", cleaned)

        chunker = Chunker(chunk_size=15, chunk_overlap=5)
        segments = [
            {"text": cleaned, "chapter": "Chapter 4", "page": 10, "source_book": "Hands-On ML"}
        ]
        chunks = chunker.create_chunks(segments)
        self.assertEqual(len(chunks), 1)

        # 2. Mock Generated Example
        gemma_file = self.dir_path / "gemma_gen.jsonl"
        mock_gemma_records = [
            {
                "instruction": "Explain how gradient descent finds optimal weights.",
                "response": "Gradient descent computes the gradient of the loss function with respect to model parameters and takes steps proportional to the negative gradient.",
                "concept": "Gradient Descent",
                "difficulty": "Intermediate",
                "chapter": "Chapter 4",
                "source_book": "Hands-On ML",
            }
        ]
        save_jsonl(mock_gemma_records, gemma_file)

        # 3. Merge & Deduplicate
        merged_file = self.dir_path / "merged.json"
        notebooklm_dir = self.dir_path / "notebooklm"
        notebooklm_dir.mkdir()
        save_json([
            {
                "instruction": "What is overfitting?",
                "response": "Overfitting occurs when a model learns patterns specific to the training data that do not generalize.",
                "concept": "Overfitting",
                "difficulty": "Intermediate",
                "chapter": "Chapter 1",
                "source_book": "Hands-On ML",
            }
        ], notebooklm_dir / "batch_01.json")

        merger = DatasetMerger(
            gemma_path=gemma_file,
            notebooklm_dir=notebooklm_dir,
            output_path=merged_file,
        )
        merged_dataset = merger.run()
        self.assertEqual(len(merged_dataset), 2)

        # 4. Validate Dataset
        validator = DatasetValidator(min_response_length=20)
        val_report = validator.validate_dataset(merged_file)
        self.assertEqual(val_report["status"], "PASS")

        # 5. Format Dataset
        formatted_file = self.dir_path / "formatted.jsonl"
        format_dataset(
            dataset_input_path=merged_file,
            output_path=formatted_file,
            model_name="unsloth/Llama-3.2-1B-Instruct",
        )
        self.assertTrue(formatted_file.exists())

        # 6. Train/Val Split
        train_ds, val_ds = load_formatted_dataset(
            dataset_path=formatted_file,
            validation_split=0.5,
            seed=42,
        )
        self.assertEqual(len(train_ds), 1)
        self.assertEqual(len(val_ds), 1)
        self.assertIn("text", train_ds.column_names)


if __name__ == "__main__":
    unittest.main()
