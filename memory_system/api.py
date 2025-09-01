from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from memory_system.core import MemoryEntry, MemoryManager
from memory_system.adapters import FAISSAdapter
from memory_system.store.sqlite_store import SQLiteStore
from memory_system.embeddings import init_tfidf, embed_text

app = FastAPI(title="Memory System Final", version="0.3.0")

_vec = FAISSAdapter if hasattr(FAISSAdapter, '__call__') else None
_vector = None
_store = SQLiteStore()
_manager = MemoryManager(vector_index=None, store=_store)

class StoreRequest(BaseModel):
    id: str; text: str; vector: Optional[List[float]] = None; metadata: Optional[dict] = {}; tier: Optional[str] = 'wm'

@app.post('/store')
def store(req: StoreRequest):
    e = MemoryEntry(id=req.id, text=req.text, vector=req.vector, metadata=req.metadata or {}, tier=req.tier)
    _manager.store_entry(e)
    return {'ok': True, 'id': req.id}

@app.post('/recall')
def recall(query: Optional[str] = None, vector: Optional[List[float]] = None, top_k: int = 5):
    res = _manager.recall(query=query, vector=vector, top_k=top_k)
    return {'results': [ {'id': r.id, 'text': r.text, 'tier': r.tier} for r in res ]}

@app.post('/start_consolidation')
def start_cons(interval: int = 60):
    _manager.start_consolidation_worker(interval_seconds=interval)
    return {'started': True}

@app.post('/init_tfidf')
def init_tfidf_endpoint(corpus: Optional[List[str]] = None, max_features: int = 512):
    init_tfidf(corpus=corpus or [], max_features=max_features)
    return {'ok': True}
