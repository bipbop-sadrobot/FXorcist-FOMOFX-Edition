# Expert Review Summary

This document aggregates perspectives from multiple domains to inform design decisions:

## ML Researcher
- Prefer strong privacy accounting (RDP) instead of naive accountant; implement moments accountant for production.
- Use per-example gradient clipping when possible.
- Evaluate utility-privacy tradeoffs experimentally and provide reproducible notebooks.

## Systems Engineer
- Kafka should be deployed via docker-compose for local testing; use zookeeper/kafka bitnami images.
- Monitor topic lag and set retention policies; use compacted topics for model registry events.
- Consider scaling vector index to FAISS on GPU instances for heavy workloads.

## Security Engineer
- HMAC keys must be rotated and stored securely (use Vault/OS keyring). Use TLS for Kafka and client auth in prod.
- DP parameters should be centrally governed; maintain audit logs for all federated rounds.
- Ensure API has authentication (JWT/OAuth2) and RBAC for memory access and deletion endpoints.

## Product/UX
- Provide a simple "explain" endpoint to surface why a memory was recalled (show matching tokens, similarity scores).
- Allow users to pin/lock memories and to request deletion (compliance with GDPR right-to-be-forgotten).

## QA/Testing
- Create reproducible seeds for embedding/vector operations.
- Fuzz tests on API inputs, large payloads, and malformed vectors.
- Load tests for recall latency with thousands of vectors.

Recommendations: Prioritize implementing robust DP accounting, embedding generation with deterministic seed, and explicit authentication for the API prior to sharing across machines.
