"""Shared lazy embedding model singleton.

Import via `from src.ingestion.embedder import get_embedder()` instead of
constructing `SentenceTransformer` at module import time — a single 384-dim
MiniLM instance is heavy enough on the 2vCPU/2GB VPS without loading it 3x.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)
