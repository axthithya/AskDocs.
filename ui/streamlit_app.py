"""Streamlit chat UI. Talks to the FastAPI backend only (never to Bedrock)."""
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PDF_DIR = Path(os.getenv("DATA_DIR", "data")) / "pdfs"
NO_ANSWER_TEXT = "couldn't find that in the provided documents"

st.set_page_config(page_title="AskDocs", page_icon="📄")
st.title("AskDocs")
st.caption("Ask questions about your documents using AI")

# ---------- Sidebar: documents + top-k ----------
with st.sidebar:
    st.header("Documents")
    pdfs = sorted(p.name for p in PDF_DIR.glob("*.pdf"))
    if pdfs:
        for name in pdfs:
            st.write(f"• {name}")
    else:
        st.write("No PDFs found in data/pdfs/")
    top_k = st.slider("Top-K (chunks retrieved)", 1, 8, 4)


def unique_sources(sources: list[dict]) -> list[str]:
    """Turn API sources into 'file.pdf — Page N' lines, without duplicates."""
    lines = []
    for s in sources:
        line = f"{s['source']} — Page {s['page']}"
        if line not in lines:
            lines.append(line)
    return lines


def show_message(msg: dict) -> None:
    """Render one chat message (and its sources, if any)."""
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources**")
            for line in msg["sources"]:
                st.markdown(f"- {line}")


def ask_api(question: str) -> dict:
    """Call POST /ask. Returns {'content': ..., 'sources': [...]}; never raises."""
    try:
        resp = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=60,
        )
    except requests.ConnectionError:
        return {"content": "⚠️ Can't reach the AskDocs API. Start it with: uvicorn app.main:app --reload"}
    except requests.Timeout:
        return {"content": "⚠️ The request timed out. Please try again."}
    except requests.RequestException:
        return {"content": "⚠️ Network error while contacting the API."}

    if resp.status_code == 429:
        return {"content": "⚠️ Too many questions (limit is 10 per minute). Wait a moment and try again."}
    if resp.status_code == 503:
        return {"content": "⚠️ The document index isn't built yet. Run: python -m app.ingest"}
    if resp.status_code == 422:
        return {"content": "⚠️ Questions must be between 3 and 500 characters."}
    if not resp.ok:
        return {"content": f"⚠️ The API returned an error (HTTP {resp.status_code}). Please try again."}

    data = resp.json()
    answer = data.get("answer", "").strip()
    if not answer:
        return {"content": "No answer available."}
    # If the model refused, the retrieved chunks weren't used, so don't list them as sources.
    if NO_ANSWER_TEXT in answer.lower():
        return {"content": answer}
    return {"content": answer, "sources": unique_sources(data.get("sources", []))}


# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    show_message(message)

question = st.chat_input("Ask a question about your documents...")
if question is not None:
    question = question.strip()
    if len(question) < 3:
        st.warning("Please type a question (at least 3 characters).")
    else:
        user_msg = {"role": "user", "content": question}
        st.session_state.messages.append(user_msg)
        show_message(user_msg)

        with st.spinner("Thinking..."):
            result = ask_api(question)
        assistant_msg = {"role": "assistant", **result}
        st.session_state.messages.append(assistant_msg)
        show_message(assistant_msg)
