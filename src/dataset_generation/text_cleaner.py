import unicodedata
import re

class TextCleaner:
    def __init__(self):
        # Pattern to find hyphens at line boundaries (e.g., "optim-\nization" -> "optimization")
        self.hyphen_pattern = re.compile(r"(\w+)-\s*\n\s*(\w+)")
        # Pattern for multiple spaces
        self.multi_space_pattern = re.compile(r"[ \t]+")

    def _is_code_like(self, text: str) -> bool:
        """Heuristics to determine if a block of text contains code snippets."""
        lines = text.split("\n")
        code_markers = [
            r"^\s*>>>",        # Python REPL prompt
            r"^\s*\.\.\.",      # Python REPL continuation
            r"^\s*import\s+",   # Python imports
            r"^\s*from\s+\S+\s+import",
            r"^\s*def\s+\w+\(.*\):", # function definitions
            r"^\s*class\s+\w+[:\(]",  # class definitions
            r"^\s*#",           # comment line
        ]
        
        # If there are code blocks using backticks
        if "```" in text or "`" in text:
            return True
            
        for line in lines:
            for marker in code_markers:
                if re.match(marker, line):
                    return True
                    
        # Check if it has a high concentration of pythonic symbols
        symbol_indicators = [" = ", " == ", " += ", "()", "[]", "{}"]
        score = sum(1 for sym in symbol_indicators if sym in text)
        if score >= 2 and len(lines) > 1:
            return True
            
        return False

    def clean_segment(self, text: str) -> str:
        """
        Clean and normalize a single text segment/paragraph.

        Args:
            text (str): Raw segment text.

        Returns:
            str: Cleaned segment text.
        """
        if not text:
            return ""

        # 1. Normalize Unicode (NFKC handles ligatures like fi, fl)
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Fix word hyphenation at line breaks
        normalized = self.hyphen_pattern.sub(r"\1\2", normalized)

        # If it looks like code, don't merge lines into a single paragraph!
        if self._is_code_like(normalized):
            # Clean each line's trailing whitespace but preserve structure
            lines = normalized.split("\n")
            cleaned_lines = []
            for line in lines:
                # Extract leading indentation
                indent_match = re.match(r"^([ \t]*)", line)
                indent = indent_match.group(1) if indent_match else ""
                # Clean the rest of the line
                rest = line[len(indent):].rstrip()
                cleaned_rest = self.multi_space_pattern.sub(" ", rest)
                cleaned_lines.append(indent + cleaned_rest)
            return "\n".join(cleaned_lines).strip()
        else:
            # 3. For normal text, replace single newlines with spaces to form a continuous paragraph
            normalized = normalized.replace("\n", " ")
            # 4. Collapse multiple spaces into a single space
            normalized = self.multi_space_pattern.sub(" ", normalized)
            return normalized.strip()

    def clean_text(self, text: str) -> str:
        """
        Clean a long text by splitting it into double-newline paragraphs,
        cleaning each paragraph, and rejoining them.
        """
        paragraphs = text.split("\n\n")
        cleaned_paragraphs = [self.clean_segment(p) for p in paragraphs if p.strip()]
        return "\n\n".join(cleaned_paragraphs)
