"""Unit tests for ModelLoader and LoRA configuration structures."""

import unittest
from src.common.config import FINETUNING_CONFIG


class TestModelLoaderConfig(unittest.TestCase):
    def test_model_config(self):
        self.assertEqual(FINETUNING_CONFIG.base_model, "unsloth/Llama-3.2-1B-Instruct")
        self.assertEqual(FINETUNING_CONFIG.max_seq_length, 2048)
        self.assertTrue(FINETUNING_CONFIG.load_in_4bit)

    def test_lora_targets(self):
        self.assertIn("q_proj", FINETUNING_CONFIG.lora_target_modules)
        self.assertIn("v_proj", FINETUNING_CONFIG.lora_target_modules)
        self.assertIn("down_proj", FINETUNING_CONFIG.lora_target_modules)


if __name__ == "__main__":
    unittest.main()
