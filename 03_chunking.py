"""
03_chunking.py

Step 3 of the RAG pipeline: split documents into retrieval-sized chunks.

Most product descriptions in this dataset are short (median ~490 chars),
but some run to thousands of characters. We chunk by word count with a
small overlap so no chunk loses context at its boundary, and so very
long descriptions don't dominate a single vector.

Each chunk keeps a reference back to its parent document id and category
so retrieved chunks can always be cited to a source product.
"""

import importlib

_documents_mod = importlib.import_module("01_documents")
_preprocessing_mod = importlib.import_module("02_preprocessing")

CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 20


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward, keeping overlap

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Turn a list of documents into a list of chunk dicts:
        {
          "chunk_id": "doc_12_chunk_0",
          "doc_id": "doc_12",
          "category": "Electronics",
          "text": "..."
        }
    """
    chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_chunk_{idx}",
                    "doc_id": doc["id"],
                    "category": doc["category"],
                    "text": piece,
                }
            )
    return chunks


if __name__ == "__main__":
    raw_docs = _documents_mod.load_documents()
    clean_docs = _preprocessing_mod.preprocess_documents(raw_docs)
    chunks = chunk_documents(clean_docs)
    print(f"Documents: {len(clean_docs)}")
    print(f"Chunks:    {len(chunks)}")
    print("Sample chunk:")
    print(chunks[0])
