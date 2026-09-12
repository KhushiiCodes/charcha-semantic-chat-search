# Charcha — Search a Group Chat Properly

Semantic search for a synthetic, messy Hinglish group-chat archive.

## What it demonstrates

- 4,200 messages across 6 months
- 8 participants
- Hinglish, typos, forwarded/media-like messages and short replies
- Semantic retrieval using multilingual sentence embeddings
- Sender filtering
- Date filtering
- Conversation context around every hit
- 40-query evaluation set, including hard zero-overlap queries

## Architecture

React/Vite frontend → FastAPI API → embedding-based cosine retrieval → metadata filters → contextual results.

The search engine deliberately does not use Pinecone, FAISS, Chroma, sklearn nearest-neighbor search, or a hosted vector database.

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The first run downloads the sentence-transformer model and builds embeddings in memory.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://localhost:5173).

If the backend is not on localhost:8000, create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

## Evaluation

From the repository root:

```bash
python evaluation/evaluate.py
```

This reports top-5 retrieval accuracy overall and separately for hard zero-overlap queries.

## Demo queries

Try:

- `When did we decide on the trip?`
- `Which destination became the final choice?`
- `What did Priya say about the application deadline?`
- `What did we discuss about dinner expenses?`

## Important evaluation note

The benchmark labels are stored in `evaluation/queries.json`. The project reports measured results rather than claiming perfection.
