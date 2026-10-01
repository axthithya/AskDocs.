"""Basic generation checks (these DO call Nova).

1. Saves answers for each eval question to eval/answer_results.jsonl (for manual reading).
2. No-answer behavioral test: questions not covered by the documents should get the refusal
   phrase. This is a basic behavioral check, NOT a rigorous hallucination metric.

Usage: python -m eval.run_generation
"""
import json
from pathlib import Path

from app.rag import RAG
from eval.run_eval import QUESTIONS_PATH, load_jsonl

NO_ANSWER_PATH = Path(__file__).parent / "no_answer_questions.jsonl"
ANSWERS_PATH = Path(__file__).parent / "answer_results.jsonl"
REFUSAL = "couldn't find that in the provided documents"


def refuses(answer: str) -> bool:
    return REFUSAL in answer.lower().replace("’", "'")


def main() -> None:
    rag = RAG()

    with ANSWERS_PATH.open("w", encoding="utf-8") as f:
        for item in load_jsonl(QUESTIONS_PATH):
            result = rag.ask(item["question"])
            record = {
                "question": item["question"],
                "answer": result["answer"],
                "expected_source": item["source"],
                "retrieved_sources": [f"{s['source']} p.{s['page']}" for s in result["sources"]],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {ANSWERS_PATH}")

    no_answer = load_jsonl(NO_ANSWER_PATH)
    passed = 0
    for item in no_answer:
        answer = rag.ask(item["question"])["answer"]
        ok = refuses(answer)
        passed += ok
        print(f"{'REFUSED' if ok else 'ANSWERED'} | {item['question']}")
        if not ok:
            print(f"  Answer: {answer}")
    print(f"\nNo-answer behavioral check: {passed}/{len(no_answer)} refused (basic check only)")


if __name__ == "__main__":
    main()
