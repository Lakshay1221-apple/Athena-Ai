import fitz  # PyMuPDF
from pathlib import Path
import re
from typing import List, Dict, Any

class PDFProcessor:
    def __init__(self, margin_top: float = 50.0, margin_bottom: float = 55.0):
        """
        Initialize PDFProcessor.

        Args:
            margin_top (float): Top margin coordinate. Blocks with y1 < margin_top are ignored (headers).
            margin_bottom (float): Bottom margin from page boundary. Blocks with y0 > page_height - margin_bottom are ignored (footers/page numbers).
        """
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        # List of regexes or strings for watermarks to remove
        self.watermark_patterns = [
            re.compile(r"OceanofPDF\.com", re.IGNORECASE),
            re.compile(r"www\.allitebooks\.com", re.IGNORECASE),
            re.compile(r"www\.it-ebooks\.info", re.IGNORECASE),
        ]
        # Regexes to identify chapter headings or major parts
        self.chapter_patterns = [
            re.compile(r"^\s*Chapter\s+\d+[:.]?", re.IGNORECASE),
            re.compile(r"^\s*Part\s+[IVXLCDM]+[:.]?", re.IGNORECASE),
            re.compile(r"^\s*\d+\.\s+[A-Z][a-zA-Z\s]+$"), # e.g. "1. The Machine Learning Landscape"
        ]

    def _is_chapter_heading(self, text: str) -> bool:
        """Check if a block's text matches a chapter/section heading pattern."""
        cleaned = text.strip()
        # Ensure it's not too long (headings are usually short)
        if len(cleaned) > 120:
            return False
        for pattern in self.chapter_patterns:
            if pattern.match(cleaned):
                return True
        return False

    def _clean_block_text(self, text: str) -> str:
        """Remove watermarks from block text."""
        cleaned = text
        for pattern in self.watermark_patterns:
            cleaned = pattern.sub("", cleaned)
        return cleaned.strip()

    def extract_segments(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract blocks of text from a PDF, filtering out headers, footers, and watermarks.
        Tracks the active chapter name and associates it with each segment.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[Dict[str, Any]]: List of segment dicts with keys:
                                  - text: cleaned segment text
                                  - chapter: detected active chapter title
                                  - page: 1-indexed page number
                                  - source_book: name of the PDF file (source book)
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        source_book = path.stem.replace("_", " ").replace("-", " ")
        document = fitz.open(pdf_path)
        
        segments = []
        current_chapter = "Preface"  # Default initial chapter name
        
        for page_idx in range(len(document)):
            page = document[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            height = rect.height
            
            # Extract blocks: list of (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            
            # Sort blocks top-to-bottom
            blocks.sort(key=lambda b: b[1])
            
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                
                # Check margins
                # Ignore headers (too high)
                if y1 < self.margin_top:
                    continue
                # Ignore footers/page numbers (too low)
                if y0 > (height - self.margin_bottom):
                    continue
                
                cleaned_text = self._clean_block_text(text)
                if not cleaned_text:
                    continue
                
                # Check for chapter headings
                if self._is_chapter_heading(cleaned_text):
                    current_chapter = cleaned_text
                    
                segments.append({
                    "text": cleaned_text,
                    "chapter": current_chapter,
                    "page": page_num,
                    "source_book": source_book
                })
                
        document.close()
        return segments

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file as a flat string (for backwards compatibility).

        Args:
            pdf_path (str): Path to the PDF.

        Returns:
            str: Extracted text joined by double newlines.
        """
        segments = self.extract_segments(pdf_path)
        return "\n\n".join(seg["text"] for seg in segments)

    def save_text(self, text: str, output_path: str) -> None:
        """Save text to a file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Text saved to {output_path}")