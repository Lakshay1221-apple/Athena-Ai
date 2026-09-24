"""Generation configuration adapter pointing to src.common.config."""

from transformers import GenerationConfig
from src.common.config import INFERENCE_CONFIG


def get_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=INFERENCE_CONFIG.max_new_tokens,
        temperature=INFERENCE_CONFIG.temperature,
        top_p=INFERENCE_CONFIG.top_p,
        top_k=INFERENCE_CONFIG.top_k,
        do_sample=INFERENCE_CONFIG.do_sample,
        repetition_penalty=INFERENCE_CONFIG.repetition_penalty,
    )