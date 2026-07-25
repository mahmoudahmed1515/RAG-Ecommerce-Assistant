"""
06_retrieve_context.py

Step 6 of the RAG pipeline: retrieve the most relevant chunks for a
user question from the Chroma store built in 05_create_chroma_store.py.
"""

import importlib

import chromadb

_vectors_mod = importlib.import_module("04_vector_representation")
_store_mod = importlib.import_module("05_create_chroma_store")

CHROMA_DIR = _store_mod.CHROMA_DIR
COLLECTION_NAME = _store_mod.COLLECTION_NAME


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve_context(question: str, top_k: int = 4) -> list[dict]:
    """
    Embed the question and pull the top_k most similar chunks.

    Returns a list of dicts:
        {"text": "...", "doc_id": "doc_12", "category": "Electronics", "distance": 0.23}
    ready to be cited as sources.
    """
    collection = get_collection()
    query_embedding = _vectors_mod.embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    contexts = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for text, meta, dist in zip(docs, metas, dists):
        contexts.append(
            {
                "text": text,
                "doc_id": meta.get("doc_id"),
                "category": meta.get("category"),
                "distance": dist,
            }
        )
    return contexts


if __name__ == "__main__":
    question = "Suggest a good waterproof bluetooth speaker"
    results = retrieve_context(question, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"[{i}] ({r['category']}, doc={r['doc_id']}, dist={r['distance']:.4f})")
        print(r["text"][:200], "...\n")
