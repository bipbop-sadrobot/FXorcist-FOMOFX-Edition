# Roadmap Final

Completed in this bundle:
- Embedding hook (TF-IDF) with API init endpoint
- Background consolidation worker
- Vector adapter with FAISS fallback
- aiokafka async EventBus stub (if aiokafka installed)
- Federated DP aggregator with simple Gaussian noise + simple DP accountant
- Dockerfile + docker-compose for local Kafka + app

Next high-priority items (prod):
- Implement RDP accountant (moments accountant)
- Replace TF-IDF with production embedding model + deterministic seeding
- Add authentication & RBAC on API
- Add comprehensive integration tests (kafka+app+vector scaling)
