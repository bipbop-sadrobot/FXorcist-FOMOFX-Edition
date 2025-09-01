from typing import List, Dict, Any, Optional
try:
    import faiss
    _have_faiss = True
except Exception:
    _have_faiss = False

class VectorAdapterInterface:
    def add(self, id: str, vector: List[float], metadata: Optional[dict] = None): raise NotImplementedError
    def search(self, vector: List[float], top_k: int = 5): raise NotImplementedError
    def update(self, id: str, vector: List[float], metadata: Optional[dict] = None): raise NotImplementedError
    def remove(self, id: str): raise NotImplementedError

if _have_faiss:
    class FAISSAdapter(VectorAdapterInterface):
        def __init__(self, dim: int):
            self.index = faiss.IndexFlatIP(dim)
            self.ids = []
            self.meta = {}
        def add(self, id: str, vector: List[float], metadata: Optional[dict] = None):
            import numpy as np
            v = np.array(vector, dtype='float32').reshape(1, -1)
            self.index.add(v)
            self.ids.append(id)
            self.meta[id] = metadata or {}
        def search(self, vector: List[float], top_k: int = 5):
            import numpy as np
            q = np.array(vector, dtype='float32').reshape(1, -1)
            D, I = self.index.search(q, top_k)
            out = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0: continue
                out.append({"id": self.ids[idx], "score": float(score), "metadata": self.meta.get(self.ids[idx], {})})
            return out
        def update(self, id, vector, metadata=None):
            # FAISS flat index lacks delete; full rebuild would be needed in prod
            self.add(id, vector, metadata)
        def remove(self, id):
            # noop for flat index in this simple adapter
            pass
else:
    raise ImportError("FAISS is not installed. Please install faiss-cpu or faiss-gpu.")
