"""
02_preprocessing.py

Step 2 of the RAG pipeline: clean the raw documents.

- Normalizes whitespace.
- Strips stray quote/control characters.
- Drops empty / near-empty documents.
- Drops exact duplicate texts (the raw dataset has ~45% duplicate rows,
  which would otherwise bias retrieval toward whichever text repeats most).
"""

import importlib
import re

_documents_mod = importlib.import_module("01_documents")


def clean_text(text: str) -> str:
    """Collapse whitespace and strip odd characters from one document's text."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def preprocess_documents(documents: list[dict], min_chars: int = 10) -> list[dict]:
    """
    Clean text, drop near-empty documents, and de-duplicate on exact text match.
    Returns a new list of documents (does not mutate the input).
    """
    cleaned = []
    seen_texts = set()

    for doc in documents:
        text = clean_text(doc["text"])

        if len(text) < min_chars:
            continue  # too short to be useful context
        if text in seen_texts:
            continue  # exact duplicate, skip
        seen_texts.add(text)

        cleaned.append(
            {
                "id": doc["id"],
                "text": text,
                "category": doc["category"],
            }
        )

    return cleaned


if __name__ == "__main__":
    raw_docs = _documents_mod.load_documents()
    clean_docs = preprocess_documents(raw_docs)
    print(f"Raw documents:     {len(raw_docs)}")
    print(f"After preprocessing: {len(clean_docs)}")
    print("Sample cleaned document:")
    print(clean_docs[0])
