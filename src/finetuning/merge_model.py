"""LoRA Model Merger for Athena AI.

Merges trained LoRA adapters back into the base model weights to produce a standalone model.
"""

import argparse
from pathlib import Path
from typing import Optional

from unsloth import FastLanguageModel

from src.common.config import FINETUNING_CONFIG, PATHS
from src.common.logger import get_logger
from src.common.utils import ensure_directory

logger = get_logger("merge_model")


def merge_and_export(
    adapter_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    export_format: str = "16bit",
):
    """
    Load base model with LoRA adapter and merge into standalone weights.

    Args:
        adapter_path: Path to saved LoRA adapter.
        output_path: Path where merged model should be saved.
        export_format: Format to save merged model ('16bit', '4bit', 'lora').
    """
    in_adapter = Path(adapter_path) if adapter_path else PATHS.outputs_dir / "final_model"
    out_dir = Path(output_path) if output_path else PATHS.outputs_dir / "merged" / "athena-v1"

    if not in_adapter.exists():
        raise FileNotFoundError(f"Adapter directory not found at: {in_adapter}")

    logger.info(f"Loading adapter from {in_adapter} for merging...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(in_adapter),
        max_seq_length=FINETUNING_CONFIG.max_seq_length,
        load_in_4bit=FINETUNING_CONFIG.load_in_4bit,
    )

    ensure_directory(out_dir)
    logger.info(f"Merging LoRA adapter into base model (export_format={export_format})...")

    # Unsloth merge and save
    if export_format == "16bit":
        model.save_pretrained_merged(
            str(out_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    elif export_format == "4bit":
        model.save_pretrained_merged(
            str(out_dir),
            tokenizer,
            save_method="merged_4bit",
        )
    else:
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    logger.info(f"Merged model successfully exported to: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Merge Athena LoRA adapter into base model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(PATHS.outputs_dir / "final_model"),
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PATHS.outputs_dir / "merged" / "athena-v1"),
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["16bit", "4bit", "lora"],
        default="16bit",
        help="Export format for merged model",
    )
    args = parser.parse_args()

    merge_and_export(
        adapter_path=Path(args.adapter_path),
        output_path=Path(args.output_path),
        export_format=args.format,
    )


if __name__ == "__main__":
    main()
