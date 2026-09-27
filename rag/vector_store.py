"""A minimal in-memory vector index (cosine similarity on normalised vectors)."""
from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float32)

    def search(self, query_vec: np.ndarray, k: int | None = None) -> np.ndarray:
        """Cosine similarity of the query against every stored vector."""
        sims = self.vectors @ query_vec.astype(np.float32).ravel()
        return sims if k is None else np.argsort(-sims)[:k]

    def save(self, path: str):
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        return cls(np.load(path))
