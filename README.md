# E-commerce RAG Assistant

A simple retrieval-augmented generation (RAG) project that answers
questions about a product catalog, built as a sequence of plain Python
scripts (no notebooks) and deployed as a Streamlit app.

## Pipeline

```
documents -> preprocessing -> chunking -> vector representation -> vector store
          -> context retrieval -> prompting -> Streamlit UI
```

| File | Purpose |
|---|---|
| `01_documents.py` | Loads raw product rows from `data/ecommerceDataset.csv` |
| `02_preprocessing.py` | Cleans whitespace, drops empty/duplicate rows |
| `03_chunking.py` | Splits long descriptions into overlapping word chunks |
| `04_vector_representation.py` | Embeds chunks with a local `sentence-transformers` model (free, no API key) |
| `05_create_chroma_store.py` | Builds a persistent Chroma vector store on disk |
| `06_retrieve_context.py` | Retrieves the top-k most relevant chunks for a question |
| `07_prompting.py` | Builds the grounded prompt and calls an LLM via OpenRouter |
| `streamlit_app.py` | Streamlit UI tying the pipeline together |

## Dataset

`data/ecommerceDataset.csv` — ~50k product listings across 4 categories
(Household, Books, Electronics, Clothing & Accessories), each row is a
`category,text` pair with no header.

## Local setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env           # then edit .env and add your real key
```

Build the vector store once (this samples the first 3000 chunks by
default for a fast demo build — see `SAMPLE_SIZE` in
`05_create_chroma_store.py` to change that):

```bash
python 05_create_chroma_store.py
```

Run the app:

```bash
streamlit run streamlit_app.py
```

## API key rules

- Never write your real API key inside any `.py` file.
- Never commit your real `.env` file (`.gitignore` already excludes it).
- Locally, keys are loaded from `.env` via `python-dotenv`.
- On Streamlit Cloud, keys are read from **Streamlit Secrets** instead.

## Deploying to Streamlit Cloud

1. Push this project to a GitHub repository (make sure `.env` and
   `chroma_store/` are **not** included — check `.gitignore`).
2. On [share.streamlit.io](https://share.streamlit.io), create a new
   app pointing at `streamlit_app.py` in your repo.
3. In the app, click **Manage app -> Secrets**, and paste:

   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```

4. Because `chroma_store/` is git-ignored, you'll need the deployed app
   to build its own store on first load, or commit a pre-built store
   via Git LFS / cloud storage. For a class project, the simplest fix
   is to remove `chroma_store/` from `.gitignore` and commit the built
   store folder directly (it's just local files, no secrets inside it).

## Final checklist

- [x] All required Python files exist (`01_documents.py` … `07_prompting.py`, `streamlit_app.py`)
- [x] `requirements.txt` exists
- [x] No real API key committed anywhere in the code
- [x] Streamlit secrets template provided in valid TOML format
- [ ] Confirm the deployed Streamlit app runs successfully (deploy and test)
- [x] Answers are generated only from retrieved context (see `SYSTEM_PROMPT` in `07_prompting.py`)
- [x] Answers cite sources (`[Source N]` tags + expandable source panel in the UI)
