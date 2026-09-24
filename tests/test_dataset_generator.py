"""Unit tests for GemmaGenerator, JSONValidator, and Deduplicator."""

import unittest
from src.dataset_generation.deduplicator import Deduplicator
from src.dataset_generation.gemma_generator import GemmaGenerator
from src.dataset_generation.json_validator import JSONValidator


class TestJSONValidator(unittest.TestCase):
    def setUp(self):
        self.validator = JSONValidator(min_response_len=50)

    def test_validation_success_and_normalization(self):
        data = {
            "instruction": "What is Gradient Descent?",
            "response": "Gradient Descent is an optimization algorithm used to minimize cost. It has mathematical derivations, equations, and code examples.",
            "concept": "Gradient Descent",
            "difficulty": "intermediate",
        }
        res = self.validator.validate_and_normalize(data)
        self.assertIsNotNone(res)
        self.assertEqual(res["difficulty"], "Intermediate")
        self.assertEqual(res["concept"], "Gradient Descent")
        self.assertEqual(res["instruction"], "What is Gradient Descent?")

    def test_validation_invalid_difficulty_mapping(self):
        data = {
            "instruction": "What is SGD?",
            "response": "Stochastic Gradient Descent description is long enough to pass character count check.",
            "concept": "SGD",
            "difficulty": "unknown_tier",
        }
        res = self.validator.validate_and_normalize(data)
        self.assertIsNone(res)

    def test_validation_response_too_short(self):
        data = {
            "instruction": "Concept?",
            "response": "Too short.",
            "concept": "Concept",
            "difficulty": "Beginner",
        }
        res = self.validator.validate_and_normalize(data)
        self.assertIsNone(res)

    def test_validation_missing_keys(self):
        data = {
            "instruction": "What is PCA?",
            "concept": "PCA",
            "difficulty": "Advanced",
        }
        res = self.validator.validate_and_normalize(data)
        self.assertIsNone(res)


class TestDeduplicator(unittest.TestCase):
    def setUp(self):
        self.deduplicator = Deduplicator()

    def test_duplicate_matching(self):
        self.assertFalse(self.deduplicator.is_duplicate("Explain Gradient Descent"))
        # Same words, diff case
        self.assertTrue(self.deduplicator.is_duplicate("explain gradient descent"))
        # Same words, extra spacing
        self.assertTrue(self.deduplicator.is_duplicate("  Explain   Gradient   Descent  "))
        # Unique instruction
        self.assertFalse(self.deduplicator.is_duplicate("Explain Linear Regression"))


class TestGemmaGeneratorHelpers(unittest.TestCase):
    def setUp(self):
        self.generator = GemmaGenerator(model_name="gemma2:2b")

    def test_json_extraction_from_markdown(self):
        raw = """```json
{
  "concept": "Overfitting",
  "difficulty": "Intermediate",
  "instruction": "Explain overfitting",
  "response": "Overfitting happens when a model learns noise."
}
```"""
        parsed = self.generator._extract_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["concept"], "Overfitting")

    def test_json_extraction_with_surrounding_text(self):
        raw = """Here is the generated output:
{
  "concept": "Bias-Variance",
  "difficulty": "Intermediate",
  "instruction": "What is bias-variance?",
  "response": "Bias-variance is a fundamental tradeoff."
}
Hope this helps!"""
        parsed = self.generator._extract_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["concept"], "Bias-Variance")


if __name__ == "__main__":
    unittest.main()
