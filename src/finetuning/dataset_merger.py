import json
import re
from pathlib import Path


class DatasetMerger:

    REQUIRED_FIELDS = [
        "instruction",
        "response",
        "concept",
        "difficulty",
        "chapter",
        "source_book"
    ]

    def __init__(self,gemma_path: str,notebooklm_dir: str,output_path: str,):

        self.gemma_path = Path(gemma_path)
        self.notebooklm_dir = Path(notebooklm_dir)
        self.output_path = Path(output_path)

    def load_gemma_dataset(self):
        examples = []

        with open(self.gemma_path,"r",encoding="utf-8") as f:

            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        print(f"Loaded Gemma examples: {len(examples)}")

        return examples

    def load_notebooklm_dataset(self):

        examples = []

        batch_files = sorted(self.notebooklm_dir.glob("*.json"))

        for file in batch_files:

            try:
                with open(file,"r",encoding="utf-8") as f:
                    batch = json.load(f)

                    if isinstance(batch, list):
                        examples.extend(batch)

                    print(f"Loaded {len(batch)} examples from {file.name}")

            except Exception as e:

                print(f"Failed to load {file.name}: {e}")

        print(f"Loaded NotebookLM examples: {len(examples)}")

        return examples

    def validate_examples(self, examples):

        valid = []

        for example in examples:

            if all(
                field in example
                and str(example[field]).strip()
                for field in self.REQUIRED_FIELDS
            ):
                valid.append(example)

        print(f"Valid examples: {len(valid)} / {len(examples)}")
        

        return valid

    def clean_response(self, response):

        greetings = [
            r"^Hello!?",
            r"^Hi!?",
            r"^Certainly!?",
            r"^Great question!?",
            r"^I'd be happy to help\.?",
            r"^I would be glad to help\.?",
            r"^I'd be glad to explain\.?",
            r"^Let's dive in\.?"
        ]

        cleaned = response.strip()

        for pattern in greetings:
            cleaned = re.sub(pattern,"",cleaned,flags=re.IGNORECASE)

        return cleaned.strip()

    def clean_examples(self, examples):

        for example in examples:

            example["instruction"] = (example["instruction"].strip())            

            example["response"] = self.clean_response(example["response"])            

            example["concept"] = (example["concept"].strip())
            

        return examples

    def deduplicate(self, examples):

        seen_instructions = set()

        unique_examples = []

        for example in examples:

            key = (example["instruction"].strip().lower())

            if key not in seen_instructions:

                seen_instructions.add(key)
                unique_examples.append(example)

        print(f"Unique examples after deduplication: {len(unique_examples)}")

        return unique_examples

    def save(self, examples):

        self.output_path.parent.mkdir(parents=True,exist_ok=True)

        with open(self.output_path,"w",encoding="utf-8") as f:

            json.dump(examples,f,indent=2,ensure_ascii=False)

        print(f"Saved merged dataset: {self.output_path}")

    def run(self):

        gemma_examples = (self.load_gemma_dataset())

        notebooklm_examples = (self.load_notebooklm_dataset())

        all_examples = (gemma_examples + notebooklm_examples)

        print(f"\nTotal Raw Examples: {len(all_examples)}")

        all_examples = self.validate_examples(all_examples)

        all_examples = self.clean_examples(all_examples)

        all_examples = self.deduplicate(all_examples)

        self.save(all_examples)

        print("\nDataset Merge Complete")
        print(f"Final Dataset Size: {len(all_examples)}")


if __name__ == "__main__":

    merger = DatasetMerger(

        gemma_path=
        "Data/datasets/v1/ml_teacher_dataset.jsonl",

        notebooklm_dir="Data/datasets/v1/notebooklm",

        output_path="Data/datasets/v1/merged/athena_dataset.json"
    )

    merger.run()