"""Retrieval-only evaluation (hit@k). Never calls the LLM.

Usage: python -m eval.run_eval [k] [index_dir]
"""
import json
import sys
from pathlib import Path

from app import config
from app.retriever import Retriever

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def is_hit(item: dict, results: list[dict]) -> bool:
    """A hit = some retrieved chunk is from the expected file AND contains the keyword."""
    keyword = item["keyword"].lower()
    return any(r["source"] == item["source"] and keyword in r["text"].lower() for r in results)


def evaluate(retriever: Retriever, questions: list[dict], k: int, verbose: bool = True) -> int:
    """Run every question, return number of hits."""
    hits = 0
    for item in questions:
        results = retriever.search(item["question"], k)
        hit = is_hit(item, results)
        hits += hit
        if verbose:
            print(f"{'PASS' if hit else 'FAIL'} | {item['question']}")
            if not hit:
                print(f"  Expected: {item['source']} / {item['keyword']}")
                print("  Retrieved:")
                for r in results:
                    print(f"    - {r['source']} p.{r['page']} score={r['score']:.2f}")
    return hits


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else config.TOP_K
    index_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else config.INDEX_DIR
    questions = load_jsonl(QUESTIONS_PATH)
    if not questions:
        raise SystemExit("eval/questions.jsonl is empty. See eval/README.md to create questions.")
    hits = evaluate(Retriever(index_dir), questions, k)
    total = len(questions)
    print(f"\nhit@{k}: {hits}/{total} = {hits / total:.0%}")


if __name__ == "__main__":
    main()
