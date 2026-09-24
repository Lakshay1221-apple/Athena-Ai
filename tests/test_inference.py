"""Unit tests for inference prompt building and configuration."""

import unittest
from src.common.config import INFERENCE_CONFIG
from src.finetuning.core_finetuning.inference import SYSTEM_PROMPT


class TestInferenceConfig(unittest.TestCase):
    def test_system_prompt_presence(self):
        self.assertIn("Athena", SYSTEM_PROMPT)
        self.assertIn("Machine Learning", SYSTEM_PROMPT)

    def test_inference_config_values(self):
        self.assertGreater(INFERENCE_CONFIG.max_new_tokens, 0)
        self.assertTrue(0.0 <= INFERENCE_CONFIG.temperature <= 1.0)
        self.assertTrue(0.0 < INFERENCE_CONFIG.top_p <= 1.0)
        self.assertGreater(INFERENCE_CONFIG.repetition_penalty, 1.0)


if __name__ == "__main__":
    unittest.main()
