"""Structure-aware chunking.

Documents are split on Markdown headings first, so every chunk knows which
section it came from ("Expenses > Hotels"). Long sections are then split on
paragraph and sentence boundaries into windows of roughly `max_words`, with
`overlap` words carried over so that facts spanning a boundary survive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import Document

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str

    @property
    def header(self) -> str:
        return f"{self.doc_title} > {self.section}" if self.section else self.doc_title


def _sections(text: str):
    """Yield (section_title, body) pairs from Markdown."""
    title, buf = "", []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            if buf and "".join(buf).strip():
                yield title, "\n".join(buf).strip()
            title = "" if len(m.group(1)) == 1 else m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        yield title, "\n".join(buf).strip()


def _units(body: str):
    """Paragraphs, and sentences inside very long paragraphs."""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) > 120 and not para.lstrip().startswith(("-", "1.", "*")):
            yield from (s.strip() for s in _SENT.split(para) if s.strip())
        else:
            yield para


def chunk_document(doc: Document, max_words: int = 140, overlap: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _sections(doc.text):
        window: list[str] = []
        count = 0
        for unit in _units(body):
            n = len(unit.split())
            if window and count + n > max_words:
                chunks.append(_make(doc, section, window, len(chunks)))
                tail = " ".join(" ".join(window).split()[-overlap:]) if overlap else ""
                window, count = ([tail] if tail else []), len(tail.split())
            window.append(unit)
            count += n
        if window:
            chunks.append(_make(doc, section, window, len(chunks)))
    return chunks


def _make(doc: Document, section: str, parts: list[str], i: int) -> Chunk:
    return Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, doc_title=doc.title,
                 section=section, text="\n\n".join(parts).strip())
