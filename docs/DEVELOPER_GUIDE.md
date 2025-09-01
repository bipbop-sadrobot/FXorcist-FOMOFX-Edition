# FXorcist Developer Guide

## Project Structure

```
forex_ai_dashboard/
├── pipeline/           # Core pipeline components
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation_metrics.py
├── models/            # Model implementations
│   ├── model_hierarchy.py
│   ├── lstm_model.py
│   └── gnn_model.py
├── memory_system/     # Memory management
│   ├── api.py        # REST endpoints
│   ├── core.py       # Core implementation
│   ├── adapters.py   # Storage adapters
│   └── store/        # Persistence layer
└── dashboard/        # UI components
    ├── components/
    └── utils/
```

## API Reference

### Memory System API

#### Store Operations
```python
POST /store
{
    "id": str,
    "text": str,
    "vector": Optional[List[float]],
    "metadata": Optional[dict],
    "tier": Optional[str]
}

POST /store_batch
{
    "entries": List[StoreRequest]
}
```

#### Recall Operations
```python
POST /recall
{
    "query": Optional[str],
    "vector": Optional[List[float]],
    "top_k": int = 5
}
```

#### Memory Management
```python
GET /memory_stats
Returns:
{
    "current_entries": Dict[str, int],
    "trends": Dict[str, Any],
    "insights": str,
    "quarantine_count": int
}

POST /quarantine/{entry_id}
{
    "reason": str
}
```

### Core Classes

#### MemoryManager
```python
class MemoryManager:
    def store_entry(self, entry: MemoryEntry) -> None
    def recall(self, query: str = None, vector: List[float] = None, top_k: int = 5) -> List[MemoryEntry]
    def update(self, id: str, **changes) -> Optional[MemoryEntry]
    def forget(self, id: str = None, tier: str = None) -> int
```

#### Memory Analysis
```python
def analyze_memory_trends(
    memory_data: List[tuple[float, float]],
    leak_threshold: float = 0.01,
    window: int = 5
) -> Dict[str, Any]

def generate_insights_report(
    trend_report: Dict[str, Any]
) -> str
```

## Extension Points

### Adding New Memory Adapters
1. Implement the base adapter interface
2. Add vector storage capabilities
3. Register with MemoryManager

Example:
```python
class CustomAdapter:
    def add(self, id: str, vector: List[float], metadata: dict) -> None
    def search(self, vector: List[float], top_k: int = 5) -> List[dict]
    def remove(self, id: str) -> None
```

### Custom Memory Analysis
1. Add analysis function to core.py
2. Expose via API endpoint
3. Add visualization to dashboard

### New Model Integration
1. Implement model interface
2. Add memory hooks
3. Register with pipeline

## Testing Guidelines

### Unit Tests
```bash
pytest tests/test_memory.py
pytest tests/test_memory_integration.py
```

Key test areas:
- Memory operations (store/recall)
- Data integrity
- Performance benchmarks
- Integration points

### Integration Tests
```bash
pytest tests/test_integration_pipeline.py
```

Validates:
- Pipeline flow
- Memory system integration
- Data reliability
- Training integration

### Performance Testing
```bash
pytest tests/benchmark_memory.py
```

Metrics:
- Recall latency
- Memory usage
- Batch operation throughput

## Memory System Internals

### Memory Tiers
- Working Memory (WM): Active predictions
- Long-Term Memory (LTM): Historical patterns
- Episodic Memory (EM): Significant events

### Storage Layer
- FAISS for vector storage
- SQLite for metadata
- Async operations for performance

### Memory Lifecycle
1. Entry creation
2. Tier assignment
3. Consolidation
4. Pruning/archival

### Monitoring Hooks
- Usage metrics
- Latency tracking
- Health checks

## Best Practices

### Memory Management
- Use batch operations for bulk inserts
- Monitor memory usage trends
- Regular maintenance tasks

### Data Reliability
- Validate inputs
- Use quarantine for suspect data
- Monitor data quality metrics

### Performance Optimization
- Index critical queries
- Cache frequent lookups
- Use async operations

## Debugging

### Common Issues
1. High memory usage
   - Check consolidation settings
   - Review pruning thresholds

2. Slow recalls
   - Monitor vector index
   - Check batch sizes

3. Data corruption
   - Verify integrity checks
   - Review quarantine logs

### Logging
```python
import logging
logger = logging.getLogger(__name__)
```

Key log files:
- `logs/memory.log`
- `logs/pipeline.log`
- `logs/health.log`

## Development Workflow

1. Feature Development
   - Create feature branch
   - Add tests
   - Update documentation

2. Testing
   - Run unit tests
   - Integration tests
   - Performance benchmarks

3. Deployment
   - Version bump
   - Update changelog
   - Deploy services