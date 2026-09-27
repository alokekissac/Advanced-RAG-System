"""End-to-end pipeline: retrieve → screen → decide → generate → verify."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import chunk_document
from .documents import Document, load_directory
from .embeddings import make_embedder
from .generation import NOT_FOUND, ExtractiveGenerator, GeminiGenerator, corpus_idf
from .guardrails import confidence, looks_like_injection, strip_injection, verify_citations
from .retrieval import HybridRetriever


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    confidence: float
    generator: str
    queries: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    def __init__(self, documents: list[Document], embedder=None, use_llm: bool = True,
                 min_confidence: float = 0.65, chunk_words: int = 140, overlap: int = 25):
        t = time.perf_counter()
        self.documents = documents
        self.chunks = [c for d in documents for c in chunk_document(d, chunk_words, overlap)]
        self.retriever = HybridRetriever(self.chunks, embedder or make_embedder())
        self.extractive = ExtractiveGenerator(corpus_idf(self.chunks))
        self.llm = GeminiGenerator() if use_llm else None
        self.min_confidence = min_confidence
        self.index_ms = (time.perf_counter() - t) * 1000

    @classmethod
    def from_directory(cls, path: str | Path, **kw) -> "RAGPipeline":
        return cls(load_directory(path), **kw)

    def ask(self, question: str, k: int = 5, mode: str = "hybrid", expand: bool | None = None) -> Answer:
        timings = {}
        t0 = time.perf_counter()
        queries = [question]
        use_llm = bool(self.llm and self.llm.available())
        if (expand if expand is not None else use_llm) and use_llm:
            queries += self.llm.rewrite(question)
            timings["rewrite"] = (time.perf_counter() - t0) * 1000
        hits = self.retriever.retrieve_multi(queries, k=k, mode=mode)
        timings["retrieve"] = (time.perf_counter() - t0) * 1000 - timings.get("rewrite", 0)

        warnings, sources = [], []
        for h in hits:
            text = h.chunk.text
            if looks_like_injection(text):
                # always strip the injected paragraph; only surface a warning when the chunk is relevant
                if h.coverage >= 0.3:
                    warnings.append(f"Ignored instructions embedded in '{h.chunk.doc_title}' (possible prompt injection).")
                text = strip_injection(text)
                if not text.strip():
                    continue
            sources.append({"chunk_id": h.chunk.chunk_id, "title": h.chunk.doc_title, "section": h.chunk.section,
                            "text": text, "score": round(h.score, 4), "bm25": round(h.bm25, 3),
                            "dense": round(h.dense, 3), "coverage": round(h.coverage, 3)})

        conf = confidence(hits)
        base = dict(question=question, confidence=round(conf, 3), sources=sources, warnings=warnings,
                    queries=queries)
        if conf < self.min_confidence or not sources:
            timings["total"] = (time.perf_counter() - t0) * 1000
            return Answer(answer=NOT_FOUND, abstained=True, generator="guardrail", timings_ms=timings, **base)

        t1 = time.perf_counter()
        text, gen = None, None
        if self.llm and self.llm.available():
            try:
                text, gen = self.llm.generate(question, sources)
            except RuntimeError as e:
                warnings.append(f"LLM unavailable, used extractive answer instead ({e}).")
        if text is None:
            text, gen = self.extractive.generate(question, sources)
        timings["generate"] = (time.perf_counter() - t1) * 1000

        abstained = text.strip().startswith(NOT_FOUND[:25])
        verification = {} if abstained else verify_citations(text, [s["text"] for s in sources])
        if verification.get("unsupported"):
            warnings.append(f"{len(verification['unsupported'])} sentence(s) could not be matched to their cited source.")
        timings["total"] = (time.perf_counter() - t0) * 1000
        return Answer(answer=text, abstained=abstained, generator=gen, verification=verification,
                      timings_ms={k: round(v, 1) for k, v in timings.items()}, **base)
