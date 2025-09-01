from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from memory_system.core import MemoryEntry, MemoryManager, analyze_memory_trends, generate_insights_report
from memory_system.adapters import FAISSAdapter
from memory_system.store.sqlite_store import SQLiteStore
from memory_system.embeddings import init_tfidf, embed_text
import asyncio
from datetime import datetime, timedelta

app = FastAPI(title="Memory System Final", version="0.4.0")

_vec = FAISSAdapter if hasattr(FAISSAdapter, '__call__') else None
_vector = None
_store = SQLiteStore()
_manager = MemoryManager(vector_index=None, store=_store)
_memory_stats: List[tuple[float, float]] = []  # [(timestamp, usage)]
_last_cleanup = datetime.now()

class StoreRequest(BaseModel):
    id: str
    text: str
    vector: Optional[List[float]] = None
    metadata: Optional[dict] = {}
    tier: Optional[str] = 'wm'

class BatchStoreRequest(BaseModel):
    entries: List[StoreRequest]

async def _async_store_batch(entries: List[MemoryEntry]):
    """Asynchronously store multiple entries."""
    for entry in entries:
        _manager.store_entry(entry)
        await asyncio.sleep(0)  # Allow other tasks to run

@app.post('/store')
async def store(req: StoreRequest, background_tasks: BackgroundTasks):
    e = MemoryEntry(id=req.id, text=req.text, vector=req.vector, metadata=req.metadata or {}, tier=req.tier)
    background_tasks.add_task(_manager.store_entry, e)
    return {'ok': True, 'id': req.id}

@app.post('/store_batch')
async def store_batch(req: BatchStoreRequest):
    entries = [
        MemoryEntry(
            id=e.id, text=e.text, vector=e.vector,
            metadata=e.metadata or {}, tier=e.tier
        )
        for e in req.entries
    ]
    await _async_store_batch(entries)
    return {'ok': True, 'count': len(entries)}

@app.post('/recall')
async def recall(query: Optional[str] = None, vector: Optional[List[float]] = None, top_k: int = 5):
    start = datetime.now()
    res = _manager.recall(query=query, vector=vector, top_k=top_k)
    latency = (datetime.now() - start).total_seconds()
    
    # Track memory stats
    _memory_stats.append((datetime.now().timestamp(), len(_manager.tiers.get("wm", []))))
    if len(_memory_stats) > 1000:  # Keep last 1000 points
        _memory_stats.pop(0)
    
    return {
        'results': [{'id': r.id, 'text': r.text, 'tier': r.tier} for r in res],
        'latency': latency
    }

@app.post('/start_consolidation')
async def start_cons(interval: int = 60):
    _manager.start_consolidation_worker(interval_seconds=interval)
    return {'started': True}

@app.post('/init_tfidf')
async def init_tfidf_endpoint(corpus: Optional[List[str]] = None, max_features: int = 512):
    init_tfidf(corpus=corpus or [], max_features=max_features)
    return {'ok': True}

@app.get('/memory_stats')
async def get_memory_stats():
    """Get memory usage statistics and trends."""
    if len(_memory_stats) < 2:
        return {'error': 'Insufficient data points'}
    
    trends = analyze_memory_trends(_memory_stats)
    insights = generate_insights_report(trends)
    
    stats = {
        'current_entries': {
            tier: len(entries) for tier, entries in _manager.tiers.items()
        },
        'trends': trends,
        'insights': insights,
        'quarantine_count': len([
            e for t in _manager.tiers.values() 
            for e in t 
            if e.metadata.get('quarantined')
        ])
    }
    return stats

@app.post('/quarantine/{entry_id}')
async def quarantine_entry(entry_id: str, reason: str):
    """Mark an entry as quarantined."""
    entry = _manager.update(entry_id, metadata={'quarantined': True, 'quarantine_reason': reason})
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {'ok': True, 'entry': entry}
