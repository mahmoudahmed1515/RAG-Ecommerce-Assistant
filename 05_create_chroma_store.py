"""
05_create_chroma_store.py

Step 5 of the RAG pipeline: build the persistent Chroma vector store.

Run this file once (python 05_create_chroma_store.py) to build the store
on disk under chroma_store/. The Streamlit app and 06_retrieve_context.py
both just open this store read-only at query time -- they do NOT
re-embed the whole dataset on every run.

For speed / demo purposes this script embeds a configurable SAMPLE_SIZE
of chunks rather than the full ~50k-row dataset. Increase or remove the
sample cap for a full production build.
"""

import importlib

import chromadb

_documents_mod = importlib.import_module("01_documents")
_preprocessing_mod = importlib.import_module("02_preprocessing")
_chunking_mod = importlib.import_module("03_chunking")
_vectors_mod = importlib.import_module("04_vector_representation")

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "ecommerce_products"

# Cap the number of chunks embedded for a fast local build. Set to None
# to embed everything (slower, needs more disk/RAM).
SAMPLE_SIZE = 3000


def build_store(sample_size: int | None = SAMPLE_SIZE) -> None:
    print("Loading documents...")
    raw_docs = _documents_mod.load_documents()

    print("Preprocessing...")
    clean_docs = _preprocessing_mod.preprocess_documents(raw_docs)

    print("Chunking...")
    chunks = _chunking_mod.chunk_documents(clean_docs)

    if sample_size is not None and len(chunks) > sample_size:
        chunks = chunks[:sample_size]

    print(f"Embedding {len(chunks)} chunks (this can take a while the first time)...")
    chunks = _vectors_mod.embed_chunks(chunks)

    print("Writing to Chroma store...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh collection each build, so re-running this script doesn't
    # duplicate records.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids = [c["chunk_id"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], "category": c["category"]} for c in chunks]

    # Chroma has a per-call insert limit; add in batches to be safe.
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Done. Stored {collection.count()} chunks in '{COLLECTION_NAME}' at ./{CHROMA_DIR}")


if __name__ == "__main__":
    build_store()
