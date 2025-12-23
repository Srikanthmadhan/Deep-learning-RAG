from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CORPUS_PATH = Path(r"D:\Deep learning rag\corpus.txt")
DB_PATH = Path(r"D:\Deep learning rag\rag_db.pkl")

CHUNK_SIZE = 500      # words
CHUNK_OVERLAP = 100   # words

def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = words[i:i + CHUNK_SIZE]
        if len(chunk) > 50:
            chunks.append(" ".join(chunk))
    return chunks


print("Loading corpus...")
text = CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")

print("Chunking...")
chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding chunks...")
embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("Saving DB...")
with open(DB_PATH, "wb") as f:
    pickle.dump(
        {
            "chunks": chunks,
            "index": index
        },
        f
    )

print("RAG index built and saved.")