# AskDocs

AskDocs is a small Retrieval-Augmented Generation (RAG) app for asking questions about your own PDF documents. PDFs are split into chunks, embedded with Amazon Titan, stored in a local FAISS index, and at question time the most relevant chunks are given to Amazon Nova Lite, which answers using only that context.

## Architecture

```
PDF → Chunking → Titan Embeddings → FAISS → Retriever → Nova → FastAPI → Streamlit
```

| File | Role |
|---|---|
| `app/config.py` | Settings from environment variables |
| `app/bedrock.py` | Only place that calls Bedrock (`embed`, `generate`) |
| `app/ingest.py` | PDF → chunks → embeddings → FAISS index |
| `app/retriever.py` | Query → embedding → top-k chunks |
| `app/rag.py` | Retrieval + prompt + Nova answer |
| `app/main.py` | FastAPI: `/health`, `/ask`, `/metrics` |
| `ui/streamlit_app.py` | UI that calls the API |

## Technology

Python, pypdf, FAISS (`IndexFlatIP`), boto3, Amazon Bedrock (Titan Text Embeddings V2, Nova Lite), FastAPI, slowapi, prometheus-client, Streamlit, pytest, ruff.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## AWS credentials

No credentials are stored in the code. Use the standard AWS chain, e.g. `aws configure`, or set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (and `AWS_SESSION_TOKEN`) as environment variables. Your identity needs `bedrock:InvokeModel` and access to the Titan Embeddings V2 and Nova Lite models enabled in the Bedrock console (region `us-east-1` by default).

Optional environment variables (defaults shown): `AWS_REGION=us-east-1`, `EMBED_MODEL_ID=amazon.titan-embed-text-v2:0`, `LLM_MODEL_ID=amazon.nova-lite-v1:0`, `EMBED_DIM=512`, `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, `TOP_K=4`, `DATA_DIR=data`, `INDEX_DIR=data/index`.

## Add PDFs and build the index

Copy text-based PDFs into `data/pdfs/`, then:

```bash
python -m app.ingest
```

This writes `data/index/faiss.index` and `data/index/chunks.json`. Re-run it whenever the PDFs change.

## Start the API

```bash
uvicorn app.main:app --reload
```

Check `http://localhost:8000/health`. Metrics are at `/metrics`. `/ask` is limited to 10 requests/minute per IP.

## Start the UI

```bash
streamlit run ui/streamlit_app.py
```

Set `API_URL` if the API is not at `http://localhost:8000`.

## Example question

> What is AWS Well-Architected Framework?

Or via the API:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is AWS Well-Architected Framework?", "top_k": 4}'
```

## Tests

```bash
pytest -q
ruff check .
```

Tests cover chunking only and need no AWS credentials.

## Evaluation

**hit@k** = the share of evaluation questions for which at least one of the top-k retrieved chunks comes from the expected PDF and contains an expected keyword. Retrieval is evaluated separately from generation: if the right chunk is never retrieved the LLM cannot answer correctly, and measuring retrieval alone (no LLM calls, deterministic, cheap) shows whether a failure is a search problem or a generation problem.

**Dataset:** `eval/questions.jsonl` must be written by hand from the PDFs in `data/pdfs/`, verifying each answer exists in the text (see [eval/README.md](eval/README.md)). **Current status: 0 questions**, because no PDFs had been added when this phase was built, so **there is no baseline result and no experiment table yet**.

Once you have questions and PDFs:

```bash
python -m app.ingest
python -m eval.run_eval 4
python -m eval.run_experiments   # writes eval/results.csv
python -m eval.run_generation    # optional: answers + basic no-answer behavior check
```

Results table (fill from `eval/results.csv` after running; do not fill by hand):

| Chunk Size | Overlap | Top-k | Hit Rate |
| ---------- | ------- | ----- | -------- |
| _not yet measured_ | | | |

Observations will be written here only after real results exist.

## Troubleshooting

- **`AccessDeniedException`** — Enable model access for Titan Text Embeddings V2 and Nova Lite in the Bedrock console (correct region), and make sure your IAM identity allows `bedrock:InvokeModel`.
- **Nova model / inference profile error** — If Bedrock says the model needs an inference profile, set `LLM_MODEL_ID` to the profile ID, e.g. `us.amazon.nova-lite-v1:0`. The error message printed by the app names this.
- **"Index not found"** (API returns 503) — Run `python -m app.ingest` after adding PDFs to `data/pdfs/`.
- **"No text found" during ingestion** — The PDFs are empty or scanned images; pypdf only extracts embedded text. Use text-based PDFs (or OCR them first).
