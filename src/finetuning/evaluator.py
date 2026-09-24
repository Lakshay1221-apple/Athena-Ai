"""Evaluation module for Athena AI fine-tuned models."""

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from unsloth import FastLanguageModel

from src.common.config import FINETUNING_CONFIG, INFERENCE_CONFIG, PATHS
from src.common.logger import get_logger
from src.common.utils import calculate_text_hash, ensure_directory, load_jsonl, save_json

logger = get_logger("evaluator")

SYSTEM_PROMPT = (
    "You are Athena, an expert Machine Learning Teaching Assistant. "
    "Provide clear, pedagogically sound, and mathematically rigorous explanations "
    "accompanied by code snippets and practical intuitions where appropriate."
)


class Evaluator:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        eval_dataset_path: Optional[Path] = None,
        train_dataset_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else PATHS.outputs_dir / "final_model"
        self.eval_dataset_path = (
            Path(eval_dataset_path)
            if eval_dataset_path
            else PATHS.evaluation_dir / "ml_teaching_eval.jsonl"
        )
        self.train_dataset_path = (
            Path(train_dataset_path)
            if train_dataset_path
            else PATHS.v1_formatted_jsonl
        )

    def check_evaluation_leakage(self) -> int:
        """
        Verify that evaluation questions have not leaked into the training set.
        """
        if not self.eval_dataset_path.exists() or not self.train_dataset_path.exists():
            logger.warning("Skipping leakage check: evaluation or training dataset not found.")
            return 0

        eval_records = load_jsonl(self.eval_dataset_path)
        train_records = load_jsonl(self.train_dataset_path)

        train_hashes = set()
        for r in train_records:
            if "text" in r:
                train_hashes.add(calculate_text_hash(r["text"]))
            if "instruction" in r:
                train_hashes.add(calculate_text_hash(r["instruction"]))

        leak_count = 0
        for r in eval_records:
            instr = r.get("instruction", "")
            if calculate_text_hash(instr) in train_hashes:
                logger.error(f"Data Leakage Detected! Prompt: '{instr}' exists in training set.")
                leak_count += 1

        if leak_count == 0:
            logger.info("Evaluation dataset is cleanly isolated (0 leakage detected).")
        return leak_count

    def evaluate_model(
        self,
        output_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run inference evaluation over fixed evaluation dataset and save responses + statistics.
        """
        leak_count = self.check_evaluation_leakage()
        if leak_count > 0:
            logger.warning(f"Proceeding with evaluation despite {leak_count} potential leakages.")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path} for evaluation...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=FINETUNING_CONFIG.max_seq_length,
            load_in_4bit=FINETUNING_CONFIG.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)

        eval_records = load_jsonl(self.eval_dataset_path)
        if max_samples:
            eval_records = eval_records[:max_samples]

        logger.info(f"Evaluating model on {len(eval_records)} fixed questions...")

        results = []
        total_chars = 0
        valid_answers = 0
        start_time = time.time()

        for idx, rec in enumerate(eval_records):
            instruction = rec["instruction"]
            concept = rec.get("concept", "")
            difficulty = rec.get("difficulty", "Intermediate")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=INFERENCE_CONFIG.max_new_tokens,
                temperature=INFERENCE_CONFIG.temperature,
                do_sample=INFERENCE_CONFIG.do_sample,
                top_p=INFERENCE_CONFIG.top_p,
                top_k=INFERENCE_CONFIG.top_k,
                repetition_penalty=INFERENCE_CONFIG.repetition_penalty,
            )

            # Decode only the generated assistant response
            generated_tokens = outputs[0][inputs.shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            is_valid = len(response) > 20
            if is_valid:
                valid_answers += 1
            total_chars += len(response)

            results.append({
                "index": idx + 1,
                "concept": concept,
                "difficulty": difficulty,
                "instruction": instruction,
                "generated_response": response,
                "response_length_chars": len(response),
            })

            logger.info(f"[{idx+1}/{len(eval_records)}] Evaluated '{concept}' ({len(response)} chars)")

        elapsed = time.time() - start_time
        avg_len = total_chars / max(1, len(eval_records))

        eval_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": str(self.model_path),
            "eval_dataset_path": str(self.eval_dataset_path),
            "total_questions": len(eval_records),
            "valid_responses": valid_answers,
            "average_response_chars": round(avg_len, 2),
            "elapsed_seconds": round(elapsed, 2),
            "leak_count": leak_count,
            "results": results,
        }

        save_dir = Path(output_dir) if output_dir else PATHS.outputs_dir / "evaluation"
        ensure_directory(save_dir)
        save_path = save_dir / "inference_results.json"
        save_json(eval_summary, save_path)

        logger.info(f"Evaluation complete! Results saved to {save_path}")
        return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Athena AI model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PATHS.outputs_dir / "final_model"),
        help="Path to adapter or model",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=str(PATHS.evaluation_dir / "ml_teaching_eval.jsonl"),
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PATHS.outputs_dir / "evaluation"),
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    evaluator = Evaluator(
        model_path=Path(args.model_path),
        eval_dataset_path=Path(args.eval_dataset),
    )
    evaluator.evaluate_model(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
