"""Unit tests for PDFProcessor and TextCleaner."""

import unittest
from pathlib import Path
from src.dataset_generation.pdf_processor import PDFProcessor
from src.dataset_generation.text_cleaner import TextCleaner


class TestTextCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()

    def test_unicode_normalization(self):
        # Ligature normalization (fi, fl)
        raw_text = "The efﬁcient algorithm ﬁnds the solution inﬂating accuracy."
        cleaned = self.cleaner.clean_segment(raw_text)
        self.assertEqual(cleaned, "The efficient algorithm finds the solution inflating accuracy.")

    def test_ocr_hyphen_fix(self):
        # Word reconstruction across line boundaries
        raw_text = "This is an opti-\nmization problem with hyper-\nparameters."
        cleaned = self.cleaner.clean_segment(raw_text)
        self.assertEqual(cleaned, "This is an optimization problem with hyperparameters.")

    def test_code_preservation(self):
        # Python code line structure preservation
        code_text = "def train_model(X, y):\n    model = LinearRegression()\n    model.fit(X, y)\n    return model"
        cleaned = self.cleaner.clean_segment(code_text)
        self.assertEqual(cleaned, code_text)

    def test_normal_text_line_flattening(self):
        normal_text = "Gradient Descent is an optimization\nalgorithm used to minimize a cost\nfunction."
        cleaned = self.cleaner.clean_segment(normal_text)
        self.assertEqual(cleaned, "Gradient Descent is an optimization algorithm used to minimize a cost function.")


class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PDFProcessor(margin_top=50.0, margin_bottom=55.0)

    def test_margin_filtering_logic(self):
        # Verify processor initializes with custom margins
        self.assertEqual(self.processor.margin_top, 50.0)
        self.assertEqual(self.processor.margin_bottom, 55.0)


if __name__ == "__main__":
    unittest.main()
