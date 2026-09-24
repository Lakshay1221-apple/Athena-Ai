"""Unit tests for Formatter module."""

import tempfile
import unittest
from pathlib import Path
from src.finetuning.data_opr.formatter import format_dataset
from src.common.utils import load_jsonl, save_json


class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        self.input_file = self.dir_path / "athena_dataset.json"
        self.output_file = self.dir_path / "formatted.jsonl"

        data = [
            {
                "instruction": "What is overfitting?",
                "response": "Overfitting happens when a model learns noise from the training data.",
                "concept": "Overfitting",
                "difficulty": "Intermediate",
            }
        ]
        save_json(data, self.input_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_format_dataset_produces_text_field(self):
        format_dataset(
            dataset_input_path=self.input_file,
            output_path=self.output_file,
            model_name="unsloth/Llama-3.2-1B-Instruct",
        )

        self.assertTrue(self.output_file.exists())
        records = load_jsonl(self.output_file)
        self.assertEqual(len(records), 1)
        self.assertIn("text", records[0])
        self.assertIn("What is overfitting?", records[0]["text"])
        self.assertIn("Overfitting happens", records[0]["text"])
        self.assertIn("You are Athena", records[0]["text"])


if __name__ == "__main__":
    unittest.main()
