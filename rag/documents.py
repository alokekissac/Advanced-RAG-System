"""Loading documents (Markdown, text, PDF) into a common structure."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""
    meta: dict = field(default_factory=dict)


def _title_from_markdown(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return m.group(1).strip() if m else fallback


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def from_text(name: str, text: str, doc_id: str | None = None) -> Document:
    text = clean_text(text)
    stem = Path(name).stem
    return Document(doc_id=doc_id or stem, title=_title_from_markdown(text, stem.replace("-", " ").title()),
                    text=text, source=name)


def pdf_to_text(data: bytes, max_pages: int = 60) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages[:max_pages]):
        pages.append(f"## Page {i + 1}\n\n{page.extract_text() or ''}")
    return "\n\n".join(pages)


def load_directory(path: str | Path) -> list[Document]:
    docs = []
    for p in sorted(Path(path).glob("**/*")):
        if p.suffix.lower() in {".md", ".txt"}:
            docs.append(from_text(p.name, p.read_text(encoding="utf-8")))
        elif p.suffix.lower() == ".pdf":
            docs.append(from_text(p.name, pdf_to_text(p.read_bytes())))
    return docs
