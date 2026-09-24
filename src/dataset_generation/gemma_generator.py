"""Ollama / Gemma dataset generation engine for Athena AI."""

import json
import time
from typing import Any, Dict, Optional

import ollama

from src.common.config import GEN_MODEL_CONFIG
from src.common.logger import get_logger

logger = get_logger("gemma_generator")


class GemmaGenerator:
    def __init__(
        self,
        model_name: Optional[str] = None,
        num_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ):
        self.model_name = model_name or GEN_MODEL_CONFIG.ollama_model
        self.num_retries = num_retries or GEN_MODEL_CONFIG.retry_count
        self.retry_delay = retry_delay or GEN_MODEL_CONFIG.retry_delay

    @staticmethod
    def check_ollama_availability(model_name: str) -> bool:
        """
        Check if Ollama daemon is reachable and if the requested model is pulled.
        """
        try:
            models_response = ollama.list()
            # Handle dictionary or object response from ollama.list()
            models_list = getattr(models_response, "models", []) or models_response.get("models", [])
            installed_names = []
            for m in models_list:
                name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else str(m))
                if name:
                    installed_names.append(name)

            # Match base name or exact tag
            has_model = any(
                model_name in name or name.startswith(model_name.split(":")[0])
                for name in installed_names
            )
            if not has_model:
                logger.warning(
                    f"Ollama model '{model_name}' was not found in installed models: {installed_names}. "
                    f"Run `ollama pull {model_name}` or select an installed model from `ollama list`."
                )
                return False
            return True
        except Exception as e:
            logger.error(
                f"Cannot connect to Ollama daemon ({e}). "
                "Ensure Ollama is installed and running (`ollama serve`)."
            )
            return False

    def _build_prompt(self, chunk_text: str) -> str:
        return r"""You are an expert Machine Learning Educator.
Analyze the following text chunk and generate a single, high-quality instruction-response dataset record in JSON format:

---
CONTEXT CHUNK:
""" + chunk_text + r"""
---

Instructions:
1. Identify the single most important machine learning concept, algorithm, or technique discussed in this chunk.
2. Write a clear, natural question asking to explain this concept for the "instruction" field (e.g., "Explain how Gradient Descent works").
3. Write a comprehensive, detailed markdown explanation of the concept for the "response" field. The response must cover:
   - Definition
   - Mathematical Intuition (using LaTeX/markdown equations where appropriate)
   - A short python example or pseudocode (if applicable)
   - Common pitfalls or interview questions.
4. Assign a strict difficulty level to the concept for the "difficulty" field: either "Beginner", "Intermediate", or "Advanced". Choose:
   - "Beginner": Foundational concepts.
   - "Intermediate": Standard algorithms, implementations, or optimization concepts.
   - "Advanced": Complex mathematics, custom derivations, or deep learning architectures.

You must output exactly a JSON object matching this schema:
{
  "concept": "Name of the machine learning concept identified",
  "difficulty": "Beginner" | "Intermediate" | "Advanced",
  "instruction": "A natural question or prompt",
  "response": "A detailed educational markdown explanation containing definitions, equations, python examples, and tips."
}

Ensure that your output is valid JSON and nothing else."""

    def _extract_json(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Helper to extract and parse JSON from the raw response."""
        text = raw_response.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = text[first_brace:last_brace + 1]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    pass
            return None

    def generate_example(self, chunk_text: str) -> Optional[Dict[str, Any]]:
        prompt = self._build_prompt(chunk_text)

        for attempt in range(1, self.num_retries + 1):
            try:
                res = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    format="json",
                    options={
                        "temperature": GEN_MODEL_CONFIG.temperature,
                        "num_predict": GEN_MODEL_CONFIG.num_predict,
                        "num_ctx": GEN_MODEL_CONFIG.num_ctx,
                    },
                )

                raw_text = res.get("response", "")
                if not raw_text:
                    logger.warning(f"Attempt {attempt} returned empty response.")
                    time.sleep(self.retry_delay)
                    continue

                parsed_json = self._extract_json(raw_text)
                if parsed_json is not None:
                    return parsed_json

                logger.warning(f"Attempt {attempt} failed to parse JSON: {repr(raw_text[:120])}...")
            except Exception as e:
                logger.warning(f"Attempt {attempt} encountered Ollama error: {str(e)}")

            if attempt < self.num_retries:
                time.sleep(self.retry_delay)

        logger.error(f"Failed to generate valid example after {self.num_retries} attempts.")
        return None
