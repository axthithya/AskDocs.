"""Chunk-size and top-k experiments. Writes eval/results.csv.

Usage: python -m eval.run_experiments [--rebuild]

Cost: each chunk configuration re-embeds every chunk (one Bedrock call per chunk).
Indexes are cached in data/experiments/<name>/ and reused unless --rebuild is passed.
The production index in data/index/ is never touched.
"""
import csv
import sys
from pathlib import Path

from app import config
from app.ingest import build_index
from app.retriever import Retriever
from eval.run_eval import QUESTIONS_PATH, evaluate, load_jsonl

EXPERIMENT_DIR = config.DATA_DIR / "experiments"
RESULTS_PATH = Path(__file__).parent / "results.csv"
CHUNK_CONFIGS = [("chunk500", 500, 100), ("chunk1000", 1000, 150), ("chunk1500", 1500, 200)]
TOP_KS = [2, 4, 6]


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    questions = load_jsonl(QUESTIONS_PATH)
    if not questions:
        raise SystemExit("eval/questions.jsonl is empty. See eval/README.md to create questions.")
    total = len(questions)
    rows = []

    # Part 1: compare chunk sizes at top_k=4
    retrievers = {}
    for name, size, overlap in CHUNK_CONFIGS:
        index_dir = EXPERIMENT_DIR / name
        if rebuild or not (index_dir / "faiss.index").exists():
            print(f"=== Building {name} (size={size}, overlap={overlap}) ===")
            build_index(size, overlap, index_dir)
        retrievers[name] = Retriever(index_dir)
        hits = evaluate(retrievers[name], questions, 4, verbose=False)
        rows.append((name, size, overlap, 4, hits, total, round(hits / total, 4)))
        print(f"{name}: hit@4 = {hits}/{total}")

    # Part 2: vary top_k on the config with the highest hit@4 (ties -> the default 1000 config)
    best = max(rows, key=lambda r: (r[6], r[0] == "chunk1000"))
    name, size, overlap = best[0], best[1], best[2]
    print(f"=== top-k experiment on {name} ===")
    for k in TOP_KS:
        hits = evaluate(retrievers[name], questions, k, verbose=False)
        rows.append((f"{name}_top{k}", size, overlap, k, hits, total, round(hits / total, 4)))
        print(f"{name}: hit@{k} = {hits}/{total}")

    with RESULTS_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["experiment", "chunk_size", "chunk_overlap", "top_k", "hits", "total", "hit_rate"]
        )
        writer.writerows(rows)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
