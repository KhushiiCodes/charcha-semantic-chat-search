from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from search import ChatSearch


app = FastAPI(
    title="Charcha Semantic Chat Search",
    version="2.0.0",
    description="Semantic search over a noisy multilingual group chat.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = ChatSearch()


class SearchRequest(BaseModel):
    query: str
    sender: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    top_k: int = 8


@app.get("/health")
def health():
    return {
        "status": "ok",
        "messages": len(search_engine.messages),
        "participants": len(search_engine.senders),
    }


@app.get("/stats")
def stats():
    senders = search_engine.senders
    threads = len(search_engine.thread_messages)

    dates = [m["timestamp"][:10] for m in search_engine.messages]

    return {
        "messages": len(search_engine.messages),
        "participants": senders,
        "participant_count": len(senders),
        "threads": threads,
        "date_from": min(dates),
        "date_to": max(dates),
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
    }


@app.get("/participants")
def participants():
    return {"participants": search_engine.senders}


@app.post("/search")
def search(req: SearchRequest):
    results = search_engine.search(
        req.query,
        req.sender,
        req.date_from,
        req.date_to,
        max(1, min(req.top_k, 20)),
    )

    return {
        "query": req.query,
        "count": len(results),
        "results": results,
    }


# GET endpoint is useful for quick browser/manual testing.
@app.get("/search")
def search_get(
    q: str = Query(..., min_length=1),
    sender: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_k: int = 8,
):
    results = search_engine.search(
        q,
        sender,
        date_from,
        date_to,
        max(1, min(top_k, 20)),
    )

    return {
        "query": q,
        "count": len(results),
        "results": results,
    }
