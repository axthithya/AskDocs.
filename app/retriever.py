"""Semantic search over the FAISS index."""
import json
from pathlib import Path

import faiss
import numpy as np

from app import bedrock, config


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR) -> None:
        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index not found in {index_dir}. "
                "Add PDFs to data/pdfs and run: python -m app.ingest"
            )
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if self.index.ntotal == 0 or self.index.ntotal != len(self.chunks):
            raise RuntimeError("Index is empty or out of sync with chunks.json; re-run ingestion.")

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = min(k or config.TOP_K, self.index.ntotal)
        vec = np.array([bedrock.embed(query)], dtype="float32")
        scores, ids = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": round(float(score), 4)})
        return results
