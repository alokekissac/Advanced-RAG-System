"""Dense embeddings.

Two providers share one interface:

* ``LSAEmbedder`` (default): TF-IDF over word n-grams, projected to a dense
  semantic space with truncated SVD (latent semantic analysis). Runs locally
  with scikit-learn, needs no API key and keeps the serverless bundle small.
* ``GeminiEmbedder``: Google's ``gemini-embedding-001`` over the REST API,
  enabled with ``EMBEDDINGS=gemini`` and ``GEMINI_API_KEY``.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class LSAEmbedder:
    name = "lsa-local"

    def __init__(self, dims: int = 128):
        self.dims = dims
        self.tfidf = None
        self.svd = None

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                     stop_words="english", token_pattern=r"(?u)\b\w[\w.]*\b")
        x = self.tfidf.fit_transform(texts)
        # too few chunks to learn a latent space: fall back to plain TF-IDF vectors
        self.svd = None
        if x.shape[0] >= 10:
            k = max(2, min(self.dims, x.shape[0] - 1, x.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=k, random_state=0).fit(x)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        x = self.tfidf.transform(texts)
        return _normalize(self.svd.transform(x) if self.svd is not None else x.toarray())


class GeminiEmbedder:
    name = "gemini-embedding-001"

    def __init__(self, api_key: str | None = None, model: str = "gemini-embedding-001", dims: int = 768):
        self.key = api_key or os.environ["GEMINI_API_KEY"]
        self.model, self.dims = model, dims

    def fit(self, texts: list[str]) -> "GeminiEmbedder":
        return self

    def _call(self, texts: list[str], task: str) -> np.ndarray:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
        body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                              "taskType": task, "outputDimensionality": self.dims} for t in texts]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        return _normalize(np.array([e["values"] for e in data["embeddings"]], dtype=float))

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        out = [self._call(texts[i:i + 100], task) for i in range(0, len(texts), 100)]
        return np.vstack(out) if out else np.zeros((0, self.dims))


class FallbackEmbedder:
    """Try the primary embedder; if it fails (no network, quota), use the local one."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback, self.active = primary, fallback, primary

    @property
    def name(self):
        return self.active.name

    def fit(self, texts):
        try:
            self.primary.fit(texts)
            self._probe = self.primary.embed(texts[:1])
            self.active = self.primary
        except Exception:
            self.active = self.fallback.fit(texts)
        return self

    def embed(self, texts, **kw):
        try:
            return self.active.embed(texts, **kw)
        except TypeError:
            return self.active.embed(texts)


def make_embedder(kind: str | None = None):
    """EMBEDDINGS=lsa|gemini. Default: Gemini when GEMINI_API_KEY is set, else local LSA."""
    kind = (kind or os.environ.get("EMBEDDINGS") or ("gemini" if os.environ.get("GEMINI_API_KEY") else "lsa")).lower()
    if kind == "gemini" and os.environ.get("GEMINI_API_KEY"):
        return FallbackEmbedder(GeminiEmbedder(), LSAEmbedder())
    return LSAEmbedder()
