"""
streamlit_app.py

Final UI for the RAG project. Lets a user ask a question about the
product catalog and shows the generated answer plus the retrieved
sources it was grounded in.

Run locally with:
    streamlit run streamlit_app.py

Before running, build the vector store once with:
    python 05_create_chroma_store.py
"""

import importlib

import streamlit as st

# Numbered filenames aren't valid Python identifiers, so we load them
# with importlib instead of a normal `import` statement.
rag = importlib.import_module("07_prompting")

# --- Streamlit secrets wiring -------------------------------------------
# Locally, 07_prompting.py already loaded OPENROUTER_API_KEY / MODEL from
# a .env file. On Streamlit Cloud there is no .env file, so we pull the
# values from st.secrets instead and patch them onto the rag module.
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    # st.secrets raises if no secrets.toml exists (e.g. pure local run
    # without Streamlit secrets configured) -- fall back to .env values.
    pass
# --------------------------------------------------------------------------

st.set_page_config(page_title="E-commerce RAG Assistant", page_icon="🛍️")

st.title("🛍️ E-commerce Product Assistant")
st.caption(
    "Ask a question about the product catalog. Answers are grounded in "
    "retrieved product listings and cite their sources."
)

if not rag.OPENROUTER_API_KEY:
    st.warning(
        "No OPENROUTER_API_KEY found. Add it to a local .env file, or, "
        "if deployed, to your Streamlit app's Secrets (see README)."
    )

top_k = st.sidebar.slider("Number of sources to retrieve", min_value=2, max_value=8, value=4)

question = st.text_input("Your question", placeholder="e.g. Suggest a good bluetooth speaker")
ask_clicked = st.button("Ask")

if ask_clicked and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        try:
            result = rag.generate_answer(question, top_k=top_k)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            result = None

    if result:
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources used")
        for i, source in enumerate(result["sources"], 1):
            with st.expander(f"[Source {i}] {source['category']} — product {source['doc_id']} (distance {source['distance']:.3f})"):
                st.write(source["text"])
elif ask_clicked:
    st.info("Please enter a question first.")
