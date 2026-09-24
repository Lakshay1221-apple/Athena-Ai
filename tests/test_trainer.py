"""Unit tests for Trainer configuration logic."""

import unittest
from src.common.config import FINETUNING_CONFIG


class TestTrainerConfig(unittest.TestCase):
    def test_finetuning_config_parameters(self):
        self.assertEqual(FINETUNING_CONFIG.dataset_text_field, "text")
        self.assertEqual(FINETUNING_CONFIG.eval_strategy, "steps")
        self.assertEqual(FINETUNING_CONFIG.save_strategy, "steps")
        self.assertTrue(FINETUNING_CONFIG.load_best_model_at_end)
        self.assertEqual(FINETUNING_CONFIG.metric_for_best_model, "eval_loss")
        self.assertEqual(FINETUNING_CONFIG.lora_r, 8)
        self.assertEqual(FINETUNING_CONFIG.lora_alpha, 16)
        self.assertIn("q_proj", FINETUNING_CONFIG.lora_target_modules)
        self.assertIn("gate_proj", FINETUNING_CONFIG.lora_target_modules)


if __name__ == "__main__":
    unittest.main()
