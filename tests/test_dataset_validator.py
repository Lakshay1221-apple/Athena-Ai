"""Unit tests for DatasetValidator module."""

import tempfile
import unittest
from pathlib import Path
from src.finetuning.data_opr.dataset_validator import DatasetValidator
from src.common.utils import save_json


class TestDatasetValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.validator = DatasetValidator(min_response_length=20)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_dataset_passes(self):
        dataset = [
            {
                "instruction": "What is PCA?",
                "response": "Principal Component Analysis is an unsupervised dimensionality reduction method.",
            },
            {
                "instruction": "What is Ridge Regression?",
                "response": "Ridge regression adds an L2 regularization penalty term to the MSE loss function.",
            },
        ]
        file_path = self.dir_path / "valid.json"
        save_json(dataset, file_path)

        report = self.validator.validate_dataset(file_path)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["valid_count"], 2)
        self.assertEqual(report["invalid_count"], 0)

    def test_invalid_records_detected(self):
        dataset = [
            {
                "instruction": "",  # Empty
                "response": "Valid response that is long enough.",
            },
            {
                "instruction": "Valid instruction?",
                "response": "Short",  # Too short (< 20 chars)
            },
            {
                "instruction": "Missing response key",
            },
        ]
        file_path = self.dir_path / "invalid.json"
        save_json(dataset, file_path)

        report = self.validator.validate_dataset(file_path)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["invalid_count"], 3)
        self.assertEqual(report["valid_count"], 0)

    def test_duplicate_records_detected(self):
        dataset = [
            {
                "instruction": "Explain SGD",
                "response": "Stochastic Gradient Descent calculates gradients on random sample subsets.",
            },
            {
                "instruction": "Explain SGD",
                "response": "Stochastic Gradient Descent calculates gradients on random sample subsets.",
            },
        ]
        file_path = self.dir_path / "duplicates.json"
        save_json(dataset, file_path)

        report = self.validator.validate_dataset(file_path)
        self.assertEqual(report["duplicate_count"], 1)
        self.assertEqual(report["valid_count"], 1)


if __name__ == "__main__":
    unittest.main()
