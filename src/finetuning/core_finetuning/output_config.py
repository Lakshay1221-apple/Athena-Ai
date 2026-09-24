"""Output configuration adapter pointing to src.common.config."""

from src.common.config import PATHS

OUTPUT_DIR = str(PATHS.outputs_dir)
LORA_SAVE_DIR = str(PATHS.outputs_dir / "final_model")
MERGED_MODEL_DIR = str(PATHS.outputs_dir / "merged" / "athena-v1")