"""LoRA PEFT configuration for Athena AI using Unsloth.

Target Modules Rationale:
- q_proj, k_proj, v_proj, o_proj: Attention projections capture cross-token query-key-value interactions and representation routing.
- gate_proj, up_proj, down_proj: MLP/FFN layers store parametric knowledge and domain vocabulary representations.
Applying LoRA to all linear projection layers achieves the highest adaptation capacity with minimal trainable parameter overhead.
"""

from typing import Optional
from unsloth import FastLanguageModel

from src.common.config import FINETUNING_CONFIG
from src.common.logger import get_logger

logger = get_logger("lora_config")


def apply_lora(
    model,
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    target_modules: Optional[list] = None,
):
    """
    Wrap FastLanguageModel with Parameter-Efficient Fine-Tuning (LoRA).
    """
    rank = r if r is not None else FINETUNING_CONFIG.lora_r
    alpha = lora_alpha if lora_alpha is not None else FINETUNING_CONFIG.lora_alpha
    dropout = lora_dropout if lora_dropout is not None else FINETUNING_CONFIG.lora_dropout
    targets = target_modules if target_modules is not None else FINETUNING_CONFIG.lora_target_modules

    logger.info(
        f"Applying LoRA: rank={rank}, alpha={alpha}, dropout={dropout}, targets={targets}"
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=FINETUNING_CONFIG.lora_bias,
        target_modules=targets,
        use_gradient_checkpointing="unsloth",
        random_state=FINETUNING_CONFIG.seed,
    )

    return model