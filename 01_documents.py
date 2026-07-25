"""
01_documents.py

Step 1 of the RAG pipeline: load raw documents.

Source data: data/ecommerceDataset.csv
The CSV has NO header row and two columns:
    column 0 -> category   (Household, Books, Electronics, Clothing & Accessories)
    column 1 -> text       (product title + description)

Each row becomes one "document" with an id, the raw text, and metadata
(the product category) that we carry through the whole pipeline so we
can cite sources later.
"""

import pandas as pd

DATA_PATH = "data/ecommerceDataset.csv"


def load_documents(path: str = DATA_PATH) -> list[dict]:
    """
    Load the raw CSV into a list of document dicts:
        {"id": "doc_0", "text": "...", "category": "Household"}
    """
    df = pd.read_csv(path, header=None, names=["category", "text"])

    # Drop rows with no text at all (there is 1 in this dataset).
    df = df.dropna(subset=["text"])

    documents = []
    for i, row in df.reset_index(drop=True).iterrows():
        documents.append(
            {
                "id": f"doc_{i}",
                "text": str(row["text"]),
                "category": str(row["category"]),
            }
        )
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    print("Sample document:")
    print(docs[0])
