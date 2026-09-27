"""Evaluate retrieval, answering, abstention and guardrails on the demo corpus.

    python eval/run_eval.py

Questions: eval/questions.jsonl (46 answerable + 12 unanswerable, hand-written).
Odd ids form the dev split (used to pick the abstention threshold), even ids
the test split (reported). Everything runs locally: extractive answers, LSA
embeddings, no API calls, fully reproducible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import RAGPipeline  # noqa: E402
from rag.guardrails import confidence  # noqa: E402

QS = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if l.strip()]
ANS = [q for q in QS if q["answerable"] and not q.get("paraphrase")]
PARA = [q for q in QS if q.get("paraphrase")]


def _has_answer(chunk, q):
    text = (chunk.text + " " + chunk.section).lower()
    return chunk.doc_id == q["doc_id"] and all(k.lower() in text for k in q["answer_keywords"])


def retrieval_metrics(pipe, mode, k_max=10, qs=None):
    """Answer-bearing chunk hit rate: is a chunk that actually contains the answer in the top k?"""
    hits = {1: 0, 3: 0, 5: 0}
    rr = []
    qs = qs or ANS
    for q in qs:
        got = pipe.retriever.retrieve(q["question"], k=k_max, mode=mode)
        rank = next((i + 1 for i, h in enumerate(got) if _has_answer(h.chunk, q)), None)
        for k in hits:
            hits[k] += rank is not None and rank <= k
        rr.append(1 / rank if rank else 0.0)
    n = len(qs)
    return {"hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@5": hits[5] / n, "mrr@10": float(np.mean(rr))}


def correct(q, ans):
    if not q["answerable"]:
        return ans.abstained
    return (not ans.abstained) and all(k.lower() in ans.answer.lower() for k in q["answer_keywords"])


def answer_metrics(pipe, qs):
    out = {"n": len(qs), "answerable": 0, "answer_correct": 0, "false_abstain": 0,
           "unanswerable": 0, "correct_abstain": 0, "citations_checked": 0, "citations_supported": 0}
    for q in qs:
        a = pipe.ask(q["question"])
        if q["answerable"]:
            out["answerable"] += 1
            out["answer_correct"] += correct(q, a)
            out["false_abstain"] += a.abstained
        else:
            out["unanswerable"] += 1
            out["correct_abstain"] += a.abstained
        out["citations_checked"] += a.verification.get("checked", 0)
        out["citations_supported"] += a.verification.get("supported", 0)
    out["answer_accuracy"] = out["answer_correct"] / max(out["answerable"], 1)
    out["abstain_recall"] = out["correct_abstain"] / max(out["unanswerable"], 1)
    out["false_abstain_rate"] = out["false_abstain"] / max(out["answerable"], 1)
    out["overall_accuracy"] = (out["answer_correct"] + out["correct_abstain"]) / out["n"]
    out["citation_support_rate"] = out["citations_supported"] / max(out["citations_checked"], 1)
    return out


def main():
    t = time.perf_counter()
    pipe = RAGPipeline.from_directory(ROOT / "data" / "handbook", use_llm=False)
    res = {"corpus": {"documents": len(pipe.documents), "chunks": len(pipe.chunks),
                      "index_ms": round(pipe.index_ms, 1)},
           "retrieval": {m: retrieval_metrics(pipe, m) for m in ("bm25", "dense", "hybrid")},
           "retrieval_paraphrase": {m: retrieval_metrics(pipe, m, qs=PARA) for m in ("bm25", "dense", "hybrid")}}

    # chunk-size ablation (hybrid)
    res["chunking"] = {}
    for words in (60, 140, 300):
        p = RAGPipeline.from_directory(ROOT / "data" / "handbook", use_llm=False, chunk_words=words)
        res["chunking"][words] = {"chunks": len(p.chunks), **retrieval_metrics(p, "hybrid")}

    # abstention threshold: sweep on dev, report on test
    base_qs = [q for q in QS if not q.get("paraphrase")]
    dev = [q for q in base_qs if q["id"] % 2 == 1]
    test = [q for q in base_qs if q["id"] % 2 == 0]
    confs = {q["id"]: confidence(pipe.retriever.retrieve(q["question"], k=5)) for q in QS}
    sweep = []
    for th in np.round(np.arange(0.20, 0.81, 0.05), 2):
        ok = [(confs[q["id"]] >= th) == q["answerable"] for q in dev]
        sweep.append({"threshold": float(th), "dev_decision_accuracy": float(np.mean(ok))})
    best = max(sweep, key=lambda s: (s["dev_decision_accuracy"], -abs(s["threshold"] - 0.5)))
    pipe.min_confidence = best["threshold"]
    res["abstention"] = {"sweep": sweep, "chosen_threshold": best["threshold"],
                         "confidence_answerable": [round(confs[q["id"]], 3) for q in QS if q["answerable"] and not q.get("paraphrase")],
                         "confidence_paraphrase": [round(confs[q["id"]], 3) for q in PARA],
                         "confidence_unanswerable": [round(confs[q["id"]], 3) for q in QS if not q["answerable"]]}
    res["answering"] = {"dev": answer_metrics(pipe, dev), "test": answer_metrics(pipe, test),
                        "all": answer_metrics(pipe, base_qs), "paraphrase": answer_metrics(pipe, PARA)}

    # guardrail: the vendor note contains an injected instruction
    a = pipe.ask("Are expenses automatically approved without receipts?")
    res["injection_test"] = {"question": a.question, "warnings": a.warnings,
                             "answer": a.answer, "poisoned": "automatically approved" in a.answer.lower()}

    lat = [pipe.ask(q["question"]).timings_ms["total"] for q in QS]
    res["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    res["runtime_s"] = round(time.perf_counter() - t, 1)
    (ROOT / "eval" / "results.json").write_text(json.dumps(res, indent=2))
    write_markdown(res)
    try:
        sys.path.insert(0, str(ROOT / "eval"))
        from make_figures import make
        make(res, ROOT / "docs" / "figures")
    except ImportError:  # matplotlib not installed
        pass
    print(json.dumps({k: res[k] for k in ("retrieval", "answering", "injection_test")}, indent=1))


def write_markdown(res):
    r = res["retrieval"]
    lines = ["| Retriever | Keyword Qs Hit@1 | Hit@3 | MRR@10 | Paraphrased Qs Hit@1 | Hit@3 | MRR@10 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    rp = res["retrieval_paraphrase"]
    for m, name in (("bm25", "BM25 (sparse)"), ("dense", "Dense (LSA embeddings)"), ("hybrid", "**Hybrid (RRF + MMR)**")):
        lines.append(f"| {name} | {r[m]['hit@1']:.1%} | {r[m]['hit@3']:.1%} | {r[m]['mrr@10']:.3f} "
                     f"| {rp[m]['hit@1']:.1%} | {rp[m]['hit@3']:.1%} | {rp[m]['mrr@10']:.3f} |")
    t = res["answering"]["test"]
    lines += ["", f"Abstention threshold chosen on dev split: **{res['abstention']['chosen_threshold']:.2f}**", "",
              "| Test split | Value |", "|---|---:|",
              f"| Answer accuracy (answerable questions) | {t['answer_accuracy']:.1%} ({t['answer_correct']}/{t['answerable']}) |",
              f"| Correct abstentions (unanswerable) | {t['abstain_recall']:.1%} ({t['correct_abstain']}/{t['unanswerable']}) |",
              f"| False abstentions | {t['false_abstain_rate']:.1%} |",
              f"| Overall accuracy | {t['overall_accuracy']:.1%} |",
              f"| Cited sentences supported by their source | {t['citation_support_rate']:.1%} |",
              "", f"Paraphrased questions (extractive answers): {res['answering']['paraphrase']['answer_accuracy']:.1%} correct, "
              f"{res['answering']['paraphrase']['false_abstain_rate']:.1%} abstained."]
    (ROOT / "eval" / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
