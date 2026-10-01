"""FastAPI app: /health, /ask, /metrics."""
import time

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.rag import RAG

app = FastAPI(title="AskDocs")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/metrics", make_asgi_app())

REQUESTS = Counter("askdocs_requests_total", "Total /ask requests", ["status"])
LATENCY = Histogram("askdocs_request_seconds", "Latency of /ask requests in seconds")

_rag: RAG | None = None


def get_rag() -> RAG:
    """Load the index on first use so the server can start before ingestion."""
    global _rag
    if _rag is None:
        _rag = RAG()
    return _rag


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, body: AskRequest) -> dict:
    start = time.perf_counter()
    try:
        result = get_rag().ask(body.question, body.top_k)
        REQUESTS.labels(status="success").inc()
        return result
    except FileNotFoundError as e:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
    finally:
        LATENCY.observe(time.perf_counter() - start)
