"""Model and tokenizer loader for Athena AI using Unsloth."""

from typing import Optional, Tuple
from transformers import PreTrainedTokenizer
from unsloth import FastLanguageModel

from src.common.config import FINETUNING_CONFIG
from src.common.logger import get_logger

logger = get_logger("model_loader")


def load_model_and_tokenizer(
    model_name: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    load_in_4bit: Optional[bool] = None,
) -> Tuple[FastLanguageModel, PreTrainedTokenizer]:
    """
    Load base causal language model and tokenizer with 4-bit quantization.
    """
    name = model_name if model_name is not None else FINETUNING_CONFIG.base_model
    seq_len = max_seq_length if max_seq_length is not None else FINETUNING_CONFIG.max_seq_length
    in_4bit = load_in_4bit if load_in_4bit is not None else FINETUNING_CONFIG.load_in_4bit

    logger.info(f"Loading base model '{name}' (max_seq_length={seq_len}, load_in_4bit={in_4bit})...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = seq_len

    logger.info(f"Successfully loaded model '{name}' (vocab_size={tokenizer.vocab_size})")
    return model, tokenizer