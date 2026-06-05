import ollama
import json
import time
from typing import Dict, Any, Optional

class GemmaGenerator:
    def __init__(self, model_name: str = "gemma4:e4b", num_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize GemmaGenerator.

        Args:
            model_name (str): The name of the Ollama model to use.
            num_retries (int): Number of retries on API failure or JSON parse failure.
            retry_delay (float): Delay in seconds between retries.
        """
        self.model_name = model_name
        self.num_retries = num_retries
        self.retry_delay = retry_delay

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
   - Mathematical Intuition (using LaTeX/markdown equations like $E = mc^2$ or $$w = w - \eta \nabla L$$ where appropriate)
   - A short python example or pseudocode (if applicable)
   - Common pitfalls or interview questions.
4. Assign a strict difficulty level to the concept for the "difficulty" field: either "Beginner", "Intermediate", or "Advanced". Choose:
   - "Beginner": Foundational concepts (e.g., supervised learning definition, simple linear regression concepts).
   - "Intermediate": Standard algorithms, implementations, or optimization concepts (e.g., gradient descent mechanics, random forests, regularizations).
   - "Advanced": Complex mathematics, custom derivations, or deep learning architectures (e.g., self-attention formulation, custom loss calculations).

You must output exactly a JSON object matching this schema:
{
  "concept": "Name of the machine learning concept identified",
  "difficulty": "Beginner" | "Intermediate" | "Advanced",
  "instruction": "A natural question or prompt (e.g., 'Explain the concept of Gradient Descent')",
  "response": "A detailed educational markdown explanation containing definitions, equations, python examples, and tips."
}

Ensure that your output is valid JSON and nothing else."""

    def _extract_json(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Helper to extract and parse JSON from the raw response."""
        text = raw_response.strip()
        
        # Handle cases where JSON is wrapped in markdown code blocks
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
            # Try to find the first '{' and last '}' if there is surrounding conversational text
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
        """
        Query Gemma via Ollama to generate an instruction-response record from a chunk.
        Retries up to self.num_retries times.

        Args:
            chunk_text (str): The input text chunk.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON record with keys: concept, instruction, response, difficulty.
                                      Returns None if all retries fail.
        """
        prompt = self._build_prompt(chunk_text)
        
        for attempt in range(1, self.num_retries + 1):
            try:
                # Call Ollama API with format="json" to force JSON output
                res = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    format="json",
                    options={
                        "temperature": 0.3,  # lower temperature for more deterministic/structured output
                        "num_predict": 2048,  # prevent truncated response
                        "num_ctx": 4096,  # increase context size to prevent cutoff
                    }
                )
                
                raw_text = res.get("response", "")
                print(f"\n--- [RAW GEMMA RESPONSE] ---")
                print(raw_text)
                print(f"-----------------------------\n")
                if not raw_text:
                    print(f"[Warning] Attempt {attempt} returned empty response.")
                    time.sleep(self.retry_delay)
                    continue
                    
                parsed_json = self._extract_json(raw_text)
                if parsed_json is not None:
                    # Successful parse
                    return parsed_json
                    
                print(f"[Warning] Attempt {attempt} failed to parse JSON: {repr(raw_text[:200])}...")
            except Exception as e:
                print(f"[Warning] Attempt {attempt} encountered Ollama error: {str(e)}")
                
            if attempt < self.num_retries:
                time.sleep(self.retry_delay)
                
        print(f"[Error] Failed to generate valid example after {self.num_retries} attempts.")
        return None
