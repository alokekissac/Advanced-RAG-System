"""Hybrid retrieval: BM25 + dense vectors, fused with Reciprocal Rank Fusion,
then diversified with Maximal Marginal Relevance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .embeddings import LSAEmbedder
from .text import tokenize
from .vector_store import VectorStore

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused score used for ranking
    bm25: float
    dense: float        # cosine similarity
    coverage: float     # share of the query's content words found in the chunk


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder=None):
        self.chunks = chunks
        texts = [f"{c.header}\n{c.text}" for c in chunks]
        self.bm25 = BM25(texts)
        self.embedder = (embedder or LSAEmbedder()).fit(texts)
        self.store = VectorStore(self.embedder.embed(texts))
        self._tokens = [set(tokenize(t)) for t in texts]
        self._max_idf = max(self.bm25.idf.values(), default=1.0)

    # ------------------------------------------------------------------ scoring
    def _embed_query(self, q: str) -> np.ndarray:
        try:
            return self.embedder.embed([q], task="RETRIEVAL_QUERY")[0]
        except TypeError:
            return self.embedder.embed([q])[0]

    def _coverage(self, q_terms: set[str], i: int) -> float:
        """IDF-weighted share of query terms present in chunk i.

        Words the corpus has never seen get the maximum weight: if the question
        hinges on a term no document mentions ("dress code", "Canada"), coverage
        drops and the system abstains instead of answering something nearby.
        """
        if not q_terms:
            return 0.0
        w = {t: self.bm25.idf.get(t, self._max_idf) for t in q_terms}
        return sum(w[t] for t in q_terms & self._tokens[i]) / sum(w.values())

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid", mmr_lambda: float = 0.75,
                 pool: int = 20) -> list[Hit]:
        sparse = self.bm25.scores(query)
        qv = self._embed_query(query)
        dense = self.store.search(qv)

        if mode == "bm25":
            fused = sparse
        elif mode == "dense":
            fused = dense
        else:
            fused = np.zeros(len(self.chunks))
            for scores in (sparse, dense):
                for rank, i in enumerate(np.argsort(-scores)):
                    fused[i] += 1.0 / (RRF_K + rank + 1)

        candidates = [int(i) for i in np.argsort(-fused)[:pool]]
        order = self._mmr(candidates, fused, mmr_lambda, k) if mode == "hybrid" else candidates[:k]
        q_terms = set(tokenize(query))
        return [Hit(self.chunks[i], float(fused[i]), float(sparse[i]), float(dense[i]),
                    self._coverage(q_terms, i)) for i in order]

    def retrieve_multi(self, queries: list[str], k: int = 5, mode: str = "hybrid") -> list[Hit]:
        """Retrieve for several phrasings of one question and fuse the rankings (RRF)."""
        if len(queries) == 1:
            return self.retrieve(queries[0], k=k, mode=mode)
        best: dict[str, Hit] = {}
        fused: dict[str, float] = {}
        for q in queries:
            for rank, h in enumerate(self.retrieve(q, k=k * 2, mode=mode)):
                cid = h.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                if cid not in best or h.coverage > best[cid].coverage:
                    best[cid] = h
        order = sorted(fused, key=lambda c: -fused[c])[:k]
        return [Hit(best[c].chunk, fused[c], best[c].bm25, best[c].dense, best[c].coverage) for c in order]

    def _mmr(self, candidates: list[int], relevance: np.ndarray, lam: float, k: int) -> list[int]:
        """Pick relevant chunks while penalising near-duplicates of ones already picked."""
        vecs = self.store.vectors
        rel = relevance[candidates]
        rel = (rel - rel.min()) / (np.ptp(rel) or 1)
        chosen: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(chosen) < k:
            if not chosen:
                best = remaining[int(np.argmax(rel[remaining]))]
            else:
                sel = vecs[[candidates[c] for c in chosen]]
                scores = [lam * rel[r] - (1 - lam) * float(np.max(sel @ vecs[candidates[r]]))
                          for r in remaining]
                best = remaining[int(np.argmax(scores))]
            chosen.append(best)
            remaining.remove(best)
        return [candidates[c] for c in chosen]
