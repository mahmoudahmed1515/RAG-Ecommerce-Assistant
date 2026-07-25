"""
07_prompting.py

Step 7 of the RAG pipeline: build a grounded prompt from retrieved
context and call an LLM (via OpenRouter) to generate the final answer.

This is the module streamlit_app.py imports (as `rag`) and reads/sets
OPENROUTER_API_KEY / OPENROUTER_MODEL on, per the Streamlit secrets
pattern in the project instructions.

IMPORTANT: never hard-code a real API key here. Locally, set it via a
.env file (see .env.example) which is loaded below with python-dotenv.
On Streamlit Cloud, streamlit_app.py overwrites these values from
st.secrets at startup -- see that file for details.
"""

import importlib
import os

import requests
from dotenv import load_dotenv

_retrieve_mod = importlib.import_module("06_retrieve_context")

load_dotenv()  # loads variables from a local .env file, if present (not committed)

# These are read from environment / .env locally, and overwritten from
# st.secrets by streamlit_app.py when deployed.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a helpful e-commerce shopping assistant. Answer the user's "
    "question using ONLY the product context provided below. If the "
    "context does not contain the answer, say so honestly instead of "
    "guessing. Always cite which source(s) (e.g. [Source 1]) you used "
    "for each claim you make."
)


def build_prompt(question: str, contexts: list[dict]) -> str:
    """Combine retrieved chunks into a single numbered context block."""
    context_lines = []
    for i, c in enumerate(contexts, 1):
        context_lines.append(
            f"[Source {i}] (category: {c['category']}, product id: {c['doc_id']})\n{c['text']}"
        )
    context_block = "\n\n".join(context_lines)

    prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the context above. Cite sources like "
        "[Source 1], [Source 2] next to the claims they support."
    )
    return prompt


def generate_answer(question: str, top_k: int = 4) -> dict:
    """
    Full retrieval + generation call: retrieves context, builds the
    prompt, calls OpenRouter, and returns the answer plus the sources
    used so the UI can display citations.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it in a local .env file "
            "or in Streamlit Cloud secrets."
        )

    contexts = _retrieve_mod.retrieve_context(question, top_k=top_k)
    user_prompt = build_prompt(question, contexts)

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return {"answer": answer, "sources": contexts}


if __name__ == "__main__":
    result = generate_answer("What headphones would you recommend?")
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES USED:")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] {s['category']} (doc {s['doc_id']})")
