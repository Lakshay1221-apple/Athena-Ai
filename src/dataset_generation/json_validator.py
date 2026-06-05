from typing import Dict, Any, Optional

class JSONValidator:
    def __init__(self, min_response_len: int = 100):
        """
        Initialize JSONValidator.

        Args:
            min_response_len (int): Minimum character length for the response field.
        """
        self.min_response_len = min_response_len
        self.valid_difficulties = {"Beginner", "Intermediate", "Advanced"}

    def validate_and_normalize(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Validate that the dictionary contains the correct fields and schema.
        Normalizes the difficulty field to Title Case.

        Args:
            data (Any): Dict parsed from LLM response.

        Returns:
            Optional[Dict[str, Any]]: Validated and normalized dict, or None if invalid.
        """
        if not isinstance(data, dict):
            return None

        # Check required fields
        required_fields = ["instruction", "response", "concept", "difficulty"]
        for field in required_fields:
            if field not in data or not isinstance(data[field], str) or not data[field].strip():
                return None

        # Clean fields
        instruction = data["instruction"].strip()
        response = data["response"].strip()
        concept = data["concept"].strip()
        difficulty = data["difficulty"].strip()

        # Validate response length
        if len(response) < self.min_response_len:
            return None

        # Normalize and validate difficulty
        # Map case-insensitive values
        normalized_difficulty = difficulty.title()
        if normalized_difficulty not in self.valid_difficulties:
            # Try to map common variations
            lower_diff = difficulty.lower()
            if "begin" in lower_diff or "easy" in lower_diff:
                normalized_difficulty = "Beginner"
            elif "inter" in lower_diff or "med" in lower_diff:
                normalized_difficulty = "Intermediate"
            elif "adv" in lower_diff or "hard" in lower_diff or "expert" in lower_diff:
                normalized_difficulty = "Advanced"
            else:
                return None

        return {
            "instruction": instruction,
            "response": response,
            "concept": concept,
            "difficulty": normalized_difficulty
        }
