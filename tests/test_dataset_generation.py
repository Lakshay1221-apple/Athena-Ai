import unittest
from src.dataset_generation.text_cleaner import TextCleaner
from src.dataset_generation.chunker import Chunker
from src.dataset_generation.json_validator import JSONValidator
from src.dataset_generation.deduplicator import Deduplicator

class TestTextCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()

    def test_unicode_normalization(self):
        # Test ligature normalization (fi -> f + i)
        raw_text = "The efﬁcient algorithm ﬁnds the solution."
        cleaned = self.cleaner.clean_segment(raw_text)
        self.assertEqual(cleaned, "The efficient algorithm finds the solution.")

    def test_ocr_hyphen_fix(self):
        # Test word reconstruction across line boundaries
        raw_text = "This is an opti-\nmization problem with hyper-\nparameters."
        cleaned = self.cleaner.clean_segment(raw_text)
        self.assertEqual(cleaned, "This is an optimization problem with hyperparameters.")

    def test_code_preservation(self):
        # Test that text containing python indicators keeps its line structures
        code_text = "def train_model(X, y):\n    model = LinearRegression()\n    model.fit(X, y)\n    return model"
        cleaned = self.cleaner.clean_segment(code_text)
        self.assertEqual(cleaned, code_text)

    def test_normal_text_line_flattening(self):
        # Test that standard text merges single newlines into spaces
        normal_text = "Gradient Descent is an optimization\nalgorithm used to minimize a cost\nfunction."
        cleaned = self.cleaner.clean_segment(normal_text)
        self.assertEqual(cleaned, "Gradient Descent is an optimization algorithm used to minimize a cost function.")


class TestChunker(unittest.TestCase):
    def setUp(self):
        # Setup chunker with small size for easier testing
        self.chunker = Chunker(chunk_size=10, chunk_overlap=3)

    def test_chunking_and_overlap(self):
        # Create segments (each has 4 words)
        segments = [
            {"text": "one two three four", "chapter": "Ch 1", "page": 1, "source_book": "BookA"},
            {"text": "five six seven eight", "chapter": "Ch 1", "page": 1, "source_book": "BookA"},
            {"text": "nine ten eleven twelve", "chapter": "Ch 2", "page": 2, "source_book": "BookA"},
            {"text": "thirteen fourteen fifteen sixteen", "chapter": "Ch 2", "page": 3, "source_book": "BookA"},
        ]
        chunks = self.chunker.create_chunks(segments)
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "chunk_001")
        self.assertEqual(chunks[0]["source_chunk"], "one two three four\n\nfive six seven eight\n\nnine ten eleven twelve")
        self.assertEqual(chunks[0]["chapter"], "Ch 2")
        self.assertEqual(chunks[0]["start_page"], 1)
        self.assertEqual(chunks[0]["end_page"], 2)

        self.assertEqual(chunks[1]["chunk_id"], "chunk_002")
        self.assertEqual(chunks[1]["source_chunk"], "nine ten eleven twelve\n\nthirteen fourteen fifteen sixteen")
        self.assertEqual(chunks[1]["chapter"], "Ch 2")
        self.assertEqual(chunks[1]["start_page"], 2)
        self.assertEqual(chunks[1]["end_page"], 3)


class TestJSONValidator(unittest.TestCase):
    def setUp(self):
        self.validator = JSONValidator(min_response_len=50)

    def test_validation_success_and_normalization(self):
        data = {
            "instruction": "What is Gradient Descent?",
            "response": "Gradient Descent is an optimization algorithm used to minimize cost. It has definitions, derivations, code, and common mistakes.",
            "concept": "Gradient Descent",
            "difficulty": "intermediate"  # lowercase, will be normalized
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
            "difficulty": "unicorn"  # invalid, cannot map
        }
        res = self.validator.validate_and_normalize(data)
        self.assertIsNone(res)

    def test_validation_response_too_short(self):
        data = {
            "instruction": "Concept?",
            "response": "Too short.",
            "concept": "Concept",
            "difficulty": "Beginner"
        }
        res = self.validator.validate_and_normalize(data)
        self.assertIsNone(res)

    def test_validation_missing_keys(self):
        data = {
            "instruction": "What is PCA?",
            "concept": "PCA",
            "difficulty": "Advanced"
            # Missing response
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


if __name__ == "__main__":
    unittest.main()
