# RAG + Flow Chatbot — Local Project
This project demonstrates a simple Flow-based guided chatbot and a Retrieval-Augmented-Generation (RAG) pipeline, served by a small FastAPI backend and a minimal single-page frontend.

## What this does
- **Flow Mode:** Guided conversation where the bot asks questions, validates answers, and shows a final summary.
- **RAG Mode:** Indexes local text documents, performs vector search (SentenceTransformers embeddings) and returns the most relevant passages plus a short extractive summary.
- **Chat UI:** A single HTML page to switch between Flow and RAG modes.

## Prerequisites
- **Python 3.9+** (3.10 recommended)
- **pip** (Python package installer)
- **VS Code** (you said you have it — perfect)
- Internet connection for the first run (to download the sentence-transformers model)

## Quick setup (step-by-step)
1. Open VS Code and create a new folder `rag-flow-chatbot`.
2. Inside the folder, create files exactly as in the project structure.
3. Open a terminal in VS Code (Terminal → New Terminal).
4. Create and activate a Python virtual environment:
   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows (Powershell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Build the vector index:
   ```bash
   python build_index.py
   ```
7. Run the app:
   ```bash
   uvicorn main:app --reload
   ```
8. Open your browser and go to `http://127.0.0.1:8000`.

## Files & what they do
- `main.py` — FastAPI main server.
- `flow.py` — Implements the guided flow logic.
- `rag.py` — Indexing and retrieval functions.
- `build_index.py` — Helper script to index `sample_data/*.txt`.
- `templates/index.html` + `static/js/app.js` — Front-end single page app.
- `sample_data/` — small sample dataset.

## Design notes & limitations
- Uses `sentence-transformers` embeddings and simple cosine similarity retrieval.
- Does not call an external LLM. Only extractive summary from retrieved passages.

## Troubleshooting
- If `sentence-transformers` install fails due to `torch`, install `torch` separately.
- Ensure dependencies installed correctly in the venv.
