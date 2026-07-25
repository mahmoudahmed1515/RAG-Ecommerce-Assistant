"""
04_vector_representation.py

Step 4 of the RAG pipeline: turn text chunks into vector embeddings.

We use a local, free sentence-transformers model (all-MiniLM-L6-v2) so
that embedding does NOT require any paid API key. The OpenRouter key
(see 07_prompting.py) is only needed for the final answer-generation
call, not for embeddings.
"""

from functools import lru_cache

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load (and cache) the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, returning a list of float vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an "embedding" field to each chunk dict (in batches, since the
    dataset can have tens of thousands of chunks).
    """
    batch_size = 128
    texts = [c["text"] for c in chunks]

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_embeddings.extend(embed_texts(batch))

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    return chunks


if __name__ == "__main__":
    sample = ["A wireless bluetooth speaker with deep bass.", "A hardcover novel about space travel."]
    vectors = embed_texts(sample)
    print(f"Embedded {len(vectors)} texts.")
    print(f"Vector dimension: {len(vectors[0])}")
