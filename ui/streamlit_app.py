"""Streamlit front end. Talks to the FastAPI backend only."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("AskDocs")
st.caption("Ask questions about your technical documents.")

question = st.text_input("Your question")
top_k = st.slider("Number of chunks to retrieve (top-k)", 1, 8, 4)

if st.button("Ask"):
    if len(question.strip()) < 3:
        st.warning("Please enter a question (at least 3 characters).")
    else:
        try:
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question.strip(), "top_k": top_k},
                    timeout=60,
                )
            if resp.ok:
                data = resp.json()
                st.subheader("Answer")
                st.write(data["answer"])
                st.subheader("Sources")
                for s in data["sources"]:
                    st.write(f"**{s['source']}** — page {s['page']} — score {s['score']:.3f}")
            else:
                st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {API_URL}: {e}")
