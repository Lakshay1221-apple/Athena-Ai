class Deduplicator:
    def __init__(self):
        """Initialize Deduplicator."""
        self.seen_hashes = set()

    def _normalize_text(self, text: str) -> str:
        """Lowercase and strip all whitespaces to create a normalized signature."""
        if not text:
            return ""
        return "".join(text.lower().split())

    def is_duplicate(self, instruction: str) -> bool:
        """
        Check if an instruction has been seen before.
        If it hasn't, store its signature and return False.
        If it has, return True.

        Args:
            instruction (str): The instruction text to check.

        Returns:
            bool: True if duplicate, False otherwise.
        """
        signature = self._normalize_text(instruction)
        if not signature:
            return True  # Empty instructions are treated as duplicates/invalid
            
        if signature in self.seen_hashes:
            return True
            
        self.seen_hashes.add(signature)
        return False

    def clear(self):
        """Reset the seen signatures cache."""
        self.seen_hashes.clear()
