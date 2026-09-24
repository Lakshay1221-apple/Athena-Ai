"""Quick smoke test inference script for Athena AI."""

from src.finetuning.core_finetuning.inference import AthenaTeacher


def test_quick_inference():
    teacher = AthenaTeacher()
    test_questions = [
        "What is regularization in machine learning?",
        "Explain the difference between L1 and L2 regularization.",
    ]

    for q in test_questions:
        print(f"\n[Test Question]: {q}")
        ans = teacher.ask(q, max_new_tokens=256)
        print(f"[Athena Response]:\n{ans}\n{'-'*40}")


if __name__ == "__main__":
    test_quick_inference()