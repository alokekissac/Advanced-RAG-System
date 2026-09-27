import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app  # noqa: E402

client = TestClient(app)


def test_health_and_index():
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Advanced RAG" in client.get("/").text


def test_documents():
    docs = client.get("/api/documents").json()
    assert len(docs) == 9
    doc = client.get(f"/api/documents/{docs[0]['doc_id']}").json()
    assert doc["text"]
    assert client.get("/api/documents/nope").status_code == 404


def test_ask():
    r = client.post("/api/ask", json={"question": "What is the minimum password length?", "mode": "bm25"})
    assert r.status_code == 200 and "sources" in r.json()


@pytest.mark.parametrize("body", [{"question": "hi"}, {"question": "x" * 600}, {"question": "valid question", "k": 20},
                                  {"question": "valid question", "mode": "magic"}])
def test_ask_validation(body):
    assert client.post("/api/ask", json=body).status_code == 422


def test_ask_with_uploaded_documents():
    r = client.post("/api/ask", json={"question": "How long does shipping take?",
                                      "documents": [{"name": "shop.md", "text": "# Shop\n\n## Shipping\n\nShipping takes 3 to 5 working days."}]})
    assert "3 to 5" in r.json()["answer"]


def test_extract_text_and_reject_other_types():
    ok = client.post("/api/extract", files={"file": ("a.md", b"# Title\n\nSome text here.", "text/markdown")})
    assert ok.status_code == 200 and ok.json()["words"] > 0
    bad = client.post("/api/extract", files={"file": ("a.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 415
