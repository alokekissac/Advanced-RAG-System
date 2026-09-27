"""FastAPI service + web UI for the Advanced RAG System.

    uvicorn app:app --reload        # http://127.0.0.1:8000

On Vercel this file is the entrypoint (zero config).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag import RAGPipeline
from rag.documents import from_text, load_directory, pdf_to_text
from rag.embeddings import LSAEmbedder

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "handbook"
MAX_DOCS, MAX_CHARS, MAX_UPLOAD = 5, 300_000, 5 * 1024 * 1024

app = FastAPI(title="Advanced RAG System", version="1.0.0",
              description="Hybrid retrieval, grounded answers with citations, abstention and prompt-injection guardrails.")
app.mount("/static", StaticFiles(directory=ROOT / "public" / "static"), name="static")

_lock = threading.Lock()
_pipeline: RAGPipeline | None = None


def demo_pipeline() -> RAGPipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = RAGPipeline.from_directory(CORPUS)
        return _pipeline


# ------------------------------------------------------------------ schemas
class UserDoc(BaseModel):
    name: str = Field(max_length=200)
    text: str = Field(max_length=MAX_CHARS)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=8)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"
    expand: bool | None = None
    documents: list[UserDoc] | None = Field(None, max_length=MAX_DOCS)

    @field_validator("question")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "public" / "index.html")


@app.get("/api/health")
def health():
    p = demo_pipeline()
    return {"status": "ok", "generator": "gemini" if p.llm and p.llm.available() else "extractive",
            "embeddings": getattr(p.retriever.embedder, "name", "lsa-local"),
            "documents": len(p.documents), "chunks": len(p.chunks)}


@app.get("/api/documents")
def documents():
    p = demo_pipeline()
    by_doc = {}
    for c in p.chunks:
        by_doc.setdefault(c.doc_id, set()).add(c.section)
    return [{"doc_id": d.doc_id, "title": d.title, "words": len(d.text.split()),
             "sections": sorted(s for s in by_doc.get(d.doc_id, ()) if s)} for d in p.documents]


@app.get("/api/documents/{doc_id}")
def document(doc_id: str):
    for d in demo_pipeline().documents:
        if d.doc_id == doc_id:
            return {"doc_id": d.doc_id, "title": d.title, "text": d.text}
    raise HTTPException(404, "document not found")


@app.post("/api/ask")
def ask(req: AskRequest):
    if req.documents:
        docs = [from_text(d.name, d.text, doc_id=f"user-{i}") for i, d in enumerate(req.documents) if d.text.strip()]
        if not docs:
            raise HTTPException(400, "uploaded documents are empty")
        pipe = RAGPipeline(docs, embedder=LSAEmbedder())
    else:
        pipe = demo_pipeline()
    return pipe.ask(req.question, k=req.k, mode=req.mode, expand=req.expand).to_dict()


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file is larger than 5 MB")
    name = file.filename or "document"
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        try:
            text = pdf_to_text(data)
        except Exception:
            raise HTTPException(400, "could not read this PDF")
    elif suffix in {".txt", ".md", ".markdown"}:
        text = data.decode("utf-8", errors="replace")
    else:
        raise HTTPException(415, "supported formats: .pdf, .txt, .md")
    text = text[:MAX_CHARS]
    if not text.strip():
        raise HTTPException(400, "no text found (scanned PDFs need OCR first)")
    return {"name": name, "text": text, "words": len(text.split())}


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return resp
