"""Okapi BM25, the standard sparse (keyword) retriever."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokenize


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1))
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out
