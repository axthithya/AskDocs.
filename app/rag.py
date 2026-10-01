"""Retrieve-then-generate: answer a question from the indexed documents."""
from app import bedrock
from app.retriever import Retriever

PROMPT = """You are a technical documentation assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I couldn't find that in the provided documents."

Do not invent facts.

Be concise and technically accurate.

Context:
{context}

Question: {question}

Answer:"""


class RAG:
    def __init__(self) -> None:
        self.retriever = Retriever()

    def ask(self, question: str, top_k: int | None = None) -> dict:
        hits = self.retriever.search(question, top_k)
        context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)
        answer = bedrock.generate(PROMPT.format(context=context, question=question))
        sources = [{"source": h["source"], "page": h["page"], "score": h["score"]} for h in hits]
        return {"answer": answer, "sources": sources}
