"""Guardrails: prompt-injection screening, abstention and citation verification."""
from __future__ import annotations

import re

from .text import tokenize

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above) ",
    r"(reveal|print|show|leak) (your|the) (system )?prompt",
    r"\byou are now\b",
    r"\bsystem (note|message|prompt)\s*:",
    r"do not (tell|inform) the user",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.I)


def looks_like_injection(text: str) -> bool:
    return bool(_INJ.search(text))


def strip_injection(text: str) -> str:
    """Remove paragraphs that contain instructions aimed at the model."""
    return "\n\n".join(p for p in re.split(r"\n\s*\n", text) if not looks_like_injection(p))


def confidence(hits) -> float:
    """How well the best evidence covers the question (0–1).

    Combines IDF-weighted keyword coverage (are the question's informative words
    present in the chunk?) with semantic similarity. Used to decide whether to
    answer at all.
    """
    if not hits:
        return 0.0
    return max(0.85 * h.coverage + 0.15 * max(h.dense, 0.0) for h in hits[:3])


_CITE = re.compile(r"\[(\d+)\]")


def verify_citations(answer: str, sources: list[str], threshold: float = 0.5) -> dict:
    """Check that each cited sentence is lexically supported by the source it cites.

    Returns {"checked": n, "supported": m, "unsupported": [sentences...]}.
    """
    checked, supported, bad = 0, 0, []
    # a sentence ends after its citation group: "... 25 days. [2] Next ..." -> split after "[2]"
    for sent in re.split(r"\n+", re.sub(r"((?:\[\d+\])+)[ \t]+", r"\1\n", answer)):
        refs = [int(n) for n in _CITE.findall(sent)]
        words = set(tokenize(_CITE.sub("", sent)))
        if not refs or not words:
            continue
        checked += 1
        best = max((len(words & set(tokenize(sources[r - 1]))) / len(words)
                    for r in refs if 0 < r <= len(sources)), default=0.0)
        if best >= threshold:
            supported += 1
        else:
            bad.append(sent.strip())
    return {"checked": checked, "supported": supported, "unsupported": bad}
