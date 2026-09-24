"""Inference interface for Athena AI teaching assistant."""

import argparse
from pathlib import Path
from typing import Optional

import torch
from unsloth import FastLanguageModel

from src.common.config import FINETUNING_CONFIG, INFERENCE_CONFIG, PATHS
from src.common.logger import get_logger

logger = get_logger("inference")

SYSTEM_PROMPT = (
    "You are Athena, an expert Machine Learning Teaching Assistant. "
    "Provide clear, pedagogically sound, and mathematically rigorous explanations "
    "accompanied by code snippets and practical intuitions where appropriate."
)


class AthenaTeacher:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else PATHS.outputs_dir / "final_model"
        logger.info(f"Loading Athena Teacher from {self.model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=FINETUNING_CONFIG.max_seq_length,
            load_in_4bit=FINETUNING_CONFIG.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        logger.info("Athena Teacher model loaded and ready for inference.")

    def ask(
        self,
        question: str,
        temperature: float = INFERENCE_CONFIG.temperature,
        max_new_tokens: int = INFERENCE_CONFIG.max_new_tokens,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=INFERENCE_CONFIG.do_sample,
            top_p=INFERENCE_CONFIG.top_p,
            top_k=INFERENCE_CONFIG.top_k,
            repetition_penalty=INFERENCE_CONFIG.repetition_penalty,
        )

        generated_tokens = outputs[0][inputs.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response


def interactive_chat(teacher: AthenaTeacher):
    print("\n" + "=" * 60)
    print(" Athena AI — ML Teaching Assistant (Interactive Session)")
    print(" Type 'exit' or 'quit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("\nStudent: ").strip()
            if prompt.lower() in ["exit", "quit"]:
                print("Athena: Goodbye! Happy learning!")
                break
            if not prompt:
                continue

            print("\nAthena: Generating response...\n")
            response = teacher.ask(prompt)
            print(f"Athena:\n{response}\n")
        except KeyboardInterrupt:
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Query Athena ML Teaching Assistant")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PATHS.outputs_dir / "final_model"),
        help="Path to fine-tuned model or adapter",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt question to ask",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive Q&A session",
    )
    args = parser.parse_args()

    teacher = AthenaTeacher(model_path=Path(args.model_path))

    if args.prompt:
        response = teacher.ask(args.prompt)
        print(f"\nPrompt: {args.prompt}\n")
        print(f"Athena:\n{response}\n")
    else:
        interactive_chat(teacher)


if __name__ == "__main__":
    main()
