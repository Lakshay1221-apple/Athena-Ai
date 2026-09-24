"""Athena AI — Unified Command Line Interface.

Provides a single CLI entrypoint for dataset generation, validation, merging, formatting,
training, evaluation, model merging, and interactive inference.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="athena",
        description="Athena AI — Machine Learning Teaching Assistant Pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Dataset Generation
    gen_parser = subparsers.add_parser("dataset-generate", help="Generate synthetic Q&A dataset from PDF chunks")
    gen_parser.add_argument("--pdf-path", type=str, default=None, help="Path to textbook PDF")
    gen_parser.add_argument("--model-name", type=str, default="gemma2:2b", help="Ollama model tag")
    gen_parser.add_argument("--limit-chunks", type=int, default=None, help="Limit number of chunks")
    gen_parser.add_argument("--output-version", type=str, default="v1", help="Dataset version")
    gen_parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    gen_parser.add_argument("--dry-run", action="store_true", help="Validate settings without running LLM")

    # 2. Dataset Merge
    merge_parser = subparsers.add_parser("dataset-merge", help="Merge generated Gemma outputs with NotebookLM batches")
    merge_parser.add_argument("--gemma-path", type=str, default=None, help="Path to Gemma JSONL")
    merge_parser.add_argument("--notebooklm-dir", type=str, default=None, help="Path to NotebookLM JSON folder")
    merge_parser.add_argument("--output-path", type=str, default=None, help="Output merged JSON path")

    # 3. Dataset Validation
    val_parser = subparsers.add_parser("dataset-validate", help="Validate dataset schema, quality, and duplicate absence")
    val_parser.add_argument("--dataset-path", type=str, default=None, help="Path to JSON/JSONL dataset file")

    # 4. Dataset Format
    fmt_parser = subparsers.add_parser("dataset-format", help="Format merged dataset to ChatML JSONL with canonical 'text' field")
    fmt_parser.add_argument("--input-path", type=str, default=None, help="Path to input merged JSON")
    fmt_parser.add_argument("--output-path", type=str, default=None, help="Path to output formatted JSONL")

    # 5. Train
    train_parser = subparsers.add_parser("train", help="Run SFT / QLoRA fine-tuning on formatted dataset")
    train_parser.add_argument("--dataset-path", type=str, default=None, help="Path to formatted JSONL")
    train_parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    train_parser.add_argument("--checkpoint-dir", type=str, default=None, help="Specific checkpoint path")

    # 6. Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run benchmark evaluation over fixed ML evaluation set")
    eval_parser.add_argument("--model-path", type=str, default=None, help="Path to adapter or merged model")
    eval_parser.add_argument("--eval-dataset", type=str, default=None, help="Path to evaluation dataset")
    eval_parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation results")

    # 7. Merge Model
    model_merge_parser = subparsers.add_parser("merge-model", help="Merge trained LoRA adapter into base model")
    model_merge_parser.add_argument("--adapter-path", type=str, default=None, help="Path to adapter directory")
    model_merge_parser.add_argument("--output-path", type=str, default=None, help="Output directory for merged model")
    model_merge_parser.add_argument("--format", type=str, choices=["16bit", "4bit", "lora"], default="16bit")

    # 8. Inference
    infer_parser = subparsers.add_parser("inference", help="Query Athena ML Teacher via single prompt or interactive session")
    infer_parser.add_argument("--model-path", type=str, default=None, help="Path to model or adapter")
    infer_parser.add_argument("--prompt", type=str, default=None, help="Single question to ask")
    infer_parser.add_argument("--interactive", action="store_true", help="Start interactive terminal session")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "dataset-generate":
        from src.dataset_generation.main import main as gen_main
        sys.argv = [sys.argv[0]]
        if args.pdf_path: sys.argv.extend(["--pdf-path", args.pdf_path])
        if args.model_name: sys.argv.extend(["--model-name", args.model_name])
        if args.limit_chunks: sys.argv.extend(["--limit-chunks", str(args.limit_chunks)])
        if args.output_version: sys.argv.extend(["--output-version", args.output_version])
        if args.resume: sys.argv.append("--resume")
        if args.dry_run: sys.argv.append("--dry-run")
        gen_main()

    elif args.command == "dataset-merge":
        from src.finetuning.data_opr.dataset_merger import DatasetMerger
        merger = DatasetMerger(
            gemma_path=Path(args.gemma_path) if args.gemma_path else None,
            notebooklm_dir=Path(args.notebooklm_dir) if args.notebooklm_dir else None,
            output_path=Path(args.output_path) if args.output_path else None,
        )
        merger.run()

    elif args.command == "dataset-validate":
        from src.common.config import PATHS
        from src.finetuning.data_opr.dataset_validator import DatasetValidator
        p = Path(args.dataset_path) if args.dataset_path else PATHS.v1_merged_json
        validator = DatasetValidator()
        report = validator.validate_dataset(p)
        if report["status"] != "PASS":
            sys.exit(1)

    elif args.command == "dataset-format":
        from src.finetuning.data_opr.formatter import format_dataset
        format_dataset(
            dataset_input_path=Path(args.input_path) if args.input_path else None,
            output_path=Path(args.output_path) if args.output_path else None,
        )

    elif args.command == "train":
        from src.finetuning.core_finetuning.train import train as run_train
        from src.common.config import PATHS
        p = Path(args.dataset_path) if args.dataset_path else PATHS.v1_formatted_jsonl
        run_train(
            dataset_path=p,
            resume_from_checkpoint=args.resume,
            checkpoint_dir=args.checkpoint_dir,
        )

    elif args.command == "evaluate":
        from src.finetuning.core_finetuning.evaluator import Evaluator
        evaluator = Evaluator(
            model_path=Path(args.model_path) if args.model_path else None,
            eval_dataset_path=Path(args.eval_dataset) if args.eval_dataset else None,
        )
        evaluator.evaluate_model(
            output_dir=Path(args.output_dir) if args.output_dir else None
        )

    elif args.command == "merge-model":
        from src.finetuning.core_finetuning.merge_model import merge_and_export
        merge_and_export(
            adapter_path=Path(args.adapter_path) if args.adapter_path else None,
            output_path=Path(args.output_path) if args.output_path else None,
            export_format=args.format,
        )

    elif args.command == "inference":
        from src.finetuning.core_finetuning.inference import AthenaTeacher, interactive_chat
        teacher = AthenaTeacher(
            model_path=Path(args.model_path) if args.model_path else None
        )
        if args.prompt:
            res = teacher.ask(args.prompt)
            print(f"\nPrompt: {args.prompt}\n")
            print(f"Athena:\n{res}\n")
        else:
            interactive_chat(teacher)


if __name__ == "__main__":
    main()
