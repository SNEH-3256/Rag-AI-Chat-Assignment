import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_FILE = "vector_store.npz"
META_FILE = "vector_store_meta.json"
SAMPLE_DIR = "sample_data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_model = None
_embeddings = None
_meta = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= size:
        yield text
        return
    start = 0
    while start < len(text):
        end = start + size
        yield text[start:end]
        start = max(0, end - overlap)

def build_index():
    global _embeddings, _meta
    docs = []
    meta = []
    for fn in os.listdir(SAMPLE_DIR):
        if not fn.lower().endswith(".txt"):
            continue
        path = os.path.join(SAMPLE_DIR, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_text(text):
            docs.append(chunk)
            meta.append({"source": fn})
    if not docs:
        raise RuntimeError("No docs found in sample_data.")
    model = _get_model()
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    np.savez_compressed(VECTOR_FILE, embeddings=embeddings)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "docs": docs}, f, ensure_ascii=False, indent=2)
    _embeddings = embeddings
    _meta = {"meta": meta, "docs": docs}
    print(f"Built index with {len(docs)} chunks.")

def ensure_index():
    if not os.path.exists(VECTOR_FILE) or not os.path.exists(META_FILE):
        print("Vector store not found. Building index...")
        build_index()

def _load_index():
    global _embeddings, _meta
    if _embeddings is None or _meta is None:
        if not os.path.exists(VECTOR_FILE) or not os.path.exists(META_FILE):
            raise RuntimeError("Vector store files missing. Run build_index.py first.")
        arr = np.load(VECTOR_FILE)
        _embeddings = arr["embeddings"]
        with open(META_FILE, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    return _embeddings, _meta

def _summarize_by_query(texts, query, max_sentences=4):
    text = "\n".join(texts)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    q_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for s in sentences:
        tokens = set(re.findall(r"\w+", s.lower()))
        score = len(q_tokens & tokens)
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [s for sc, s in scored if sc > 0][:max_sentences]
    if not top:
        top = sentences[:max_sentences]
    return " ".join([s.strip() for s in top])

def retrieve(query: str, k: int = 3):
    embeddings, meta = _load_index()
    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    idx = sims.argsort()[::-1][:k]
    results = []
    retrieved_texts = []
    for i in idx:
        results.append({
            "score": float(sims[i]),
            "text": meta["docs"][i],
            "source": meta["meta"][i]["source"]
        })
        retrieved_texts.append(meta["docs"][i])
    answer = _summarize_by_query(retrieved_texts, query)
    return {"answer": answer, "passages": results}
