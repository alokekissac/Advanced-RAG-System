"""Figures for the README, from eval/results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2,
                     "axes.grid": True, "grid.color": "#e8e7e2", "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 12.5})


def make(res: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    # 1. retrieval: keyword vs paraphrased questions
    names = [("bm25", "BM25"), ("dense", "Dense (LSA)"), ("hybrid", "Hybrid")]
    kw = [res["retrieval"][m]["hit@3"] for m, _ in names]
    pp = [res["retrieval_paraphrase"][m]["hit@3"] for m, _ in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    b1 = ax.bar(x - 0.19, kw, 0.36, color=BLUE, label="Keyword-style questions (46)", edgecolor=SURF, linewidth=2)
    b2 = ax.bar(x + 0.19, pp, 0.36, color=ORANGE, label="Paraphrased questions (15)", edgecolor=SURF, linewidth=2)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.0%}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [n for _, n in names], color=INK2)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Answer-bearing chunk in top 3")
    ax.set_title("Retrieval: exact wording is easy, paraphrases are not")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "retrieval.png", dpi=160)
    plt.close(fig)

    # 2. confidence distributions + threshold
    a = np.array(res["abstention"]["confidence_answerable"])
    u = np.array(res["abstention"]["confidence_unanswerable"])
    pp = np.array(res["abstention"].get("confidence_paraphrase", []))
    th = res["abstention"]["chosen_threshold"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for yv, vals, col in ((2, a, BLUE), (1, pp, AQUA), (0, u, ORANGE)):
        ax.scatter(vals, yv + rng.uniform(-0.13, 0.13, len(vals)), s=44, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.axvline(th, color=INK2, lw=1.5, ls=(0, (4, 3)))
    ax.text(th - 0.01, 2.45, f"threshold {th:.2f}", color=INK2, fontsize=9.5, ha="right")
    ax.text(th + 0.01, 2.45, "answer →", color=INK2, fontsize=9.5)
    ax.set_yticks([0, 1, 2], [f"Unanswerable ({len(u)})", f"Paraphrased ({len(pp)})", f"Keyword-style ({len(a)})"], color=INK2)
    ax.set_ylim(-0.5, 2.7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Evidence confidence (threshold chosen on the dev split)")
    ax.set_title("Abstention: answer only when the evidence is there")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "abstention.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make(json.loads((root / "eval" / "results.json").read_text()), root / "docs" / "figures")
