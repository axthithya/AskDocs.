"""Central configuration, read from environment variables with defaults."""
import os
from pathlib import Path

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "amazon.nova-lite-v1:0")
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
PDF_DIR = DATA_DIR / "pdfs"
INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
