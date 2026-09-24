"""Unit tests for Chunker module."""

import unittest
from src.dataset_generation.chunker import Chunker


class TestChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = Chunker(chunk_size=10, chunk_overlap=3)

    def test_chunking_and_overlap(self):
        segments = [
            {"text": "one two three four", "chapter": "Ch 1", "page": 1, "source_book": "BookA"},
            {"text": "five six seven eight", "chapter": "Ch 1", "page": 1, "source_book": "BookA"},
            {"text": "nine ten eleven twelve", "chapter": "Ch 2", "page": 2, "source_book": "BookA"},
            {"text": "thirteen fourteen fifteen sixteen", "chapter": "Ch 2", "page": 3, "source_book": "BookA"},
        ]
        chunks = self.chunker.create_chunks(segments)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "chunk_001")
        self.assertEqual(
            chunks[0]["source_chunk"],
            "one two three four\n\nfive six seven eight\n\nnine ten eleven twelve"
        )
        self.assertEqual(chunks[0]["chapter"], "Ch 2")
        self.assertEqual(chunks[0]["start_page"], 1)
        self.assertEqual(chunks[0]["end_page"], 2)

        self.assertEqual(chunks[1]["chunk_id"], "chunk_002")
        self.assertEqual(
            chunks[1]["source_chunk"],
            "nine ten eleven twelve\n\nthirteen fourteen fifteen sixteen"
        )
        self.assertEqual(chunks[1]["chapter"], "Ch 2")
        self.assertEqual(chunks[1]["start_page"], 2)
        self.assertEqual(chunks[1]["end_page"], 3)

    def test_empty_segments(self):
        chunks = self.chunker.create_chunks([])
        self.assertEqual(chunks, [])

    def test_short_document(self):
        segments = [
            {"text": "short text segment", "chapter": "Ch 1", "page": 1, "source_book": "BookA"}
        ]
        chunks = self.chunker.create_chunks(segments)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source_chunk"], "short text segment")


if __name__ == "__main__":
    unittest.main()
