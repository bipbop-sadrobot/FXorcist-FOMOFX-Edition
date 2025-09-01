# Project Context and Current Development State

This document provides context for the Forex AI Dashboard project, including its goals, scope, current development state, and next steps.

## Project Goals

The project aims to build an AI-driven system that:

*   Learns progressively while remaining adaptable.
*   Balances efficiency, accuracy, and usability.
*   Creates measurable value for end-users.
*   Follows ethical and transparent development practices.

## Project Scope

The project scope is currently focused on:

*   Memory + anomaly detection + reporting (MVP).
*   Embeddings upgrade (MiniLM instead of TF-IDF).
*   Simple local dashboard.

## Current Development State

The project has a solid skeleton with clear separation of concerns:

*   `core.py`: Contains core logic for memory management and analysis.
*   `adapters.py`: Provides adapters for different memory storage backends.
*   `embeddings.py`: Handles embedding generation and storage.
*   `anomaly.py`: Implements anomaly detection algorithms.
*   `api.py`: Provides an API for interacting with the memory system.

### Code Structure Details

*   `core.py`: Defines the `MemoryManager` class, which is responsible for storing, recalling, updating, and forgetting memory entries. It also includes logic for consolidating memory and summarizing memory trends.
*   `adapters.py`: Defines the `VectorAdapterInterface` and implements a `FAISSAdapter` for vector storage and retrieval using FAISS. If FAISS is not available, it falls back to a simple vector index.
*   `embeddings.py`: Provides functions for embedding text using TF-IDF. It includes functions for initializing the TF-IDF vectorizer and embedding text.
*   `anomaly.py`: Defines the `AnomalyDetector` class, which uses an Isolation Forest model to detect anomalies in memory entries.

The project also includes:

*   Pluggable embeddings (currently TF-IDF, but room to swap in sentence transformers later).
*   Memory lifecycle management (STM → LTM demotion + deduplication).
*   Adapter pattern (FAISS wrapper with fallback).
*   Federated DP (Gaussian mechanism + accountant scaffold, but currently de-prioritized).
*   Async EventBus stub (Kafka optional, but the abstraction is ready, currently de-prioritized).
*   Expert reviews / roadmap docs.

## Next Steps

*   Implement analyze_memory_trends() in core.py.
*   Flesh out generate_insights_report() so it outputs clear text/JSON with key metrics.
*   Write end-to-end integration test: add records → consolidation → anomaly detection → report.
*   Swap TF-IDF for a small local embedding model (sentence-transformers MiniLM runs fine on M1).
*   Add explainability hooks (e.g. SHAP/LIME) for anomalies and trends.
*   Integrate lightweight visualization (Streamlit or Dash).

## File Structure

(The file structure is provided in the environment details.)

## De-prioritized Features

*   Enterprise infra (Kafka, docker-compose with ZooKeeper, federation).
*   Harden DP accountant if you want stronger privacy guarantees.
