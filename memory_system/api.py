from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import jwt
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest
from memory_system.core import MemoryEntry, MemoryManager, analyze_memory_trends, generate_insights_report
from memory_system.adapters import FAISSAdapter
from memory_system.store.sqlite_store import SQLiteStore
from memory_system.embeddings import init_tfidf, embed_text
import asyncio
from datetime import datetime, timedelta

# Security and Rate Limiting Configuration
SECRET_KEY = "your-secret-key-here"  # In production, load from environment
ALGORITHM = "HS256"
security = HTTPBearer()

# ML Models for Validation
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
request_history = []  # [(timestamp, endpoint, user)]
rate_windows = {}  # {user: [(timestamp, count)]}

app = FastAPI(title="Memory System Final", version="0.4.0")

def create_access_token(data: dict):
    """Create JWT token with expiry."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate JWT token and extract user info."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def check_rate_limit(user: str, endpoint: str):
    """ML-based adaptive rate limiting."""
    now = datetime.now()
    window_start = now - timedelta(minutes=1)
    
    # Update request history
    request_history.append((now, endpoint, user))
    request_history[:] = [r for r in request_history if r[0] > window_start]
    
    # Update user windows
    if user not in rate_windows:
        rate_windows[user] = []
    rate_windows[user].append((now, 1))
    rate_windows[user][:] = [w for w in rate_windows[user] if w[0] > window_start]
    
    # Calculate features for anomaly detection
    user_req_count = sum(w[1] for w in rate_windows[user])
    total_req_count = len(request_history)
    user_endpoints = len(set(r[1] for r in request_history if r[2] == user))
    
    features = np.array([[user_req_count, total_req_count, user_endpoints]])
    
    # Detect anomalous request patterns
    if len(request_history) > 100:  # Train model when enough data
        train_features = np.array([[r[0].timestamp(), len([x for x in request_history if x[2] == r[2]]), 
                                len(set(x[1] for x in request_history if x[2] == r[2]))]
                                for r in request_history])
        isolation_forest.fit(train_features)
        if isolation_forest.predict(features)[0] == -1:
            raise HTTPException(status_code=429, detail="Anomalous request pattern detected")
    
    # Basic rate limit as fallback
    if user_req_count > 100:  # 100 requests per minute
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

async def validate_input_data(data: Any) -> bool:
    """AI-powered input validation."""
    if isinstance(data, str):
        # Check for common injection patterns
        suspicious_patterns = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", 
            "<script>", "javascript:", "data:text/html"
        ]
        if any(pattern.lower() in data.lower() for pattern in suspicious_patterns):
            return False
        
        # Check for anomalous text patterns
        if len(data) > 1000:  # Basic length check
            return False
            
    elif isinstance(data, (list, tuple)):
        # Validate numeric sequences
        if all(isinstance(x, (int, float)) for x in data):
            values = np.array(data)
            if len(values) > 0:
                mean, std = values.mean(), values.std()
                z_scores = np.abs((values - mean) / (std + 1e-10))
                if np.any(z_scores > 5):  # Z-score threshold
                    return False
    
    return True

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

class LoginRequest(BaseModel):
    username: str
    password: str = Field(..., min_length=8)

@app.post('/login')
async def login(req: LoginRequest):
    """Simple login endpoint - replace with proper auth in production."""
    # Demo credentials - implement proper auth in production
    if req.username == "demo" and req.password == "demo12345":
        token = create_access_token({"sub": req.username})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post('/store')
async def store(
    req: StoreRequest, 
    background_tasks: BackgroundTasks,
    token: dict = Depends(validate_token)
):
    """Store a memory entry with validation."""
    await check_rate_limit(token["sub"], "store")
    
    # Validate input data
    if not await validate_input_data(req.text):
        raise HTTPException(status_code=400, detail="Invalid input data detected")
    if req.vector and not await validate_input_data(req.vector):
        raise HTTPException(status_code=400, detail="Invalid vector data detected")
    e = MemoryEntry(id=req.id, text=req.text, vector=req.vector, metadata=req.metadata or {}, tier=req.tier)
    background_tasks.add_task(_manager.store_entry, e)
    return {'ok': True, 'id': req.id}

@app.post('/store_batch')
async def store_batch(
    req: BatchStoreRequest,
    token: dict = Depends(validate_token)
):
    """Store multiple entries with validation."""
    await check_rate_limit(token["sub"], "store_batch")
    
    # Validate all entries
    for entry in req.entries:
        if not await validate_input_data(entry.text):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid input data detected in entry {entry.id}"
            )
        if entry.vector and not await validate_input_data(entry.vector):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vector data detected in entry {entry.id}"
            )
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
