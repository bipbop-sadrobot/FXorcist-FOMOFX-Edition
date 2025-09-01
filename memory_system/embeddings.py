from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def init_model(model_name='all-MiniLM-L6-v2'):
    global _model
    _model = SentenceTransformer(model_name)

def embed_text(text: str):
    global _model
    if _model is None:
        init_model()
    v = _model.encode(text)
    return v.tolist()
