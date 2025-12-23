# app.py
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIG
# =============================================================================

DB_PATH = "rag_db.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5
MAX_TOP_K = 20


# =============================================================================
# LOAD DB (CACHED, FAST)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_rag_db(db_path: str):
    """
    Loads prebuilt FAISS index + chunks from disk.
    Runs ONCE. Instant on reload.
    """
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    chunks = db["chunks"]
    index = db["index"]

    model = SentenceTransformer(EMBED_MODEL)

    return model, index, chunks


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve(query: str, model, index, chunks, top_k: int):
    if not query.strip():
        return []

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for rank, idx in enumerate(ids[0]):
        results.append({
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            "text": chunks[idx],
        })

    return results


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Deep Learning RAG (No-G)",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Deep Learning Retrieval System")
    st.caption("Retrieval only. No generation. No hallucination.")

    try:
        with st.spinner("Loading retrieval database…"):
            model, index, chunks = load_rag_db(DB_PATH)
        st.success(f"Loaded {len(chunks):,} chunks")
    except Exception as e:
        st.error(f"Failed to load RAG DB: {e}")
        return

    st.markdown("---")

    query = st.text_input(
        "Search query",
        placeholder="e.g. backpropagation intuition, attention mechanism, SGD convergence"
    )

    top_k = st.slider(
        "Top results",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K
    )

    if st.button("Retrieve", type="primary"):
        results = retrieve(query, model, index, chunks, top_k)

        if not results:
            st.info("No results found.")
            return

        for r in results:
            st.markdown(
                f"### #{r['rank']} — score `{r['score']:.4f}`"
            )
            st.code(r["text"])
            st.markdown("---")


if __name__ == "__main__":
    main()
