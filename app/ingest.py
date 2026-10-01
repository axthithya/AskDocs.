"""Build the FAISS index from PDFs in data/pdfs. Run: python -m app.ingest"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader

from app import bedrock, config

MIN_CHUNK_CHARS = 30


def normalize(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into chunks of at most chunk_size chars, overlapping by `overlap` chars."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    text = normalize(text)
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def load_chunks(
    chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP
) -> tuple[list[dict], int, int]:
    """Read all PDFs; return (chunks, pdf_count, page_count)."""
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    chunks: list[dict] = []
    pages = 0
    for pdf in pdfs:
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            pages += 1
            for text in chunk_text(page.extract_text() or "", chunk_size, overlap):
                chunks.append({"source": pdf.name, "page": page_no, "text": text})
    return chunks, len(pdfs), pages


def build_index(
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    index_dir: Path = config.INDEX_DIR,
) -> None:
    chunks, n_pdfs, n_pages = load_chunks(chunk_size, overlap)
    print(f"PDFs: {n_pdfs}  Pages: {n_pages}  Chunks: {len(chunks)}")
    if not chunks:
        raise SystemExit(
            f"No text found. Put text-based PDFs in {config.PDF_DIR}/ "
            "(scanned/image-only PDFs have no extractable text)."
        )

    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        vectors.append(bedrock.embed(chunk["text"]))
        if i % 10 == 0 or i == len(chunks):
            print(f"Embedded {i}/{len(chunks)}")

    matrix = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(config.EMBED_DIM)  # normalized vectors: inner product = cosine
    index.add(matrix)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Index saved: {index.ntotal} vectors -> {index_dir / 'faiss.index'}")


if __name__ == "__main__":
    build_index()
