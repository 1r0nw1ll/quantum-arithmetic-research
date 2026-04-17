"""Beta-B visualization — 4 figures from beta_results.json.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

1. factor_decomposition.png  — 38 queries × top-1 stacked bars; detect
                               degenerate-formula patterns.
2. contradiction_boost_ablation.png — per-prior rank shift on 8 C-pairs.
3. provenance_depth_score.png — score ratio by provenance depth; validate
                               exp(-depth/3) curve.
4. authority_domain_heatmap.png — 4×6 authority × domain node counts for
                               corpus-gap detection (Phase 4.7 targeting).

matplotlib only. No seaborn / pandas. Reads beta_results.json, queries
the live qa_kg.db for authority×domain counts.

Run: python -m tools.qa_kg.analysis.beta_visualize
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import json
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.qa_kg.schema import DEFAULT_DB

_REPO = Path(__file__).resolve().parents[3]
_RESULTS_PATH = _REPO / "tools" / "qa_kg" / "analysis" / "beta_results.json"
_FIG_DIR = _REPO / "tools" / "qa_kg" / "analysis" / "figures"

_FACTOR_KEYS = (
    "authority", "lifecycle", "bm25_norm", "confidence",
    "time_decay", "contradiction", "prov_decay",
)
_FACTOR_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
)


def _load_results() -> dict:
    return json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Figure 1 — factor decomposition
# ---------------------------------------------------------------------------

def fig_factor_decomposition(report: dict, out_path: Path) -> None:
    """Each query's top-1 hit's log-space factor contribution as stacked bars.

    Uses |log(f)| across the 7 factors, normalized to share per query. A
    single-factor-dominated query shows one tall band; well-tuned queries
    show multi-factor mixtures.
    """
    per_query = report["per_query"]
    labels = []
    shares_by_factor: dict[str, list[float]] = {k: [] for k in _FACTOR_KEYS}
    for e in per_query:
        bd = e.get("top1_breakdown")
        if not bd:
            continue
        labels.append(e["id"])
        contribs = {}
        for k in _FACTOR_KEYS:
            v = bd.get(k, 1.0)
            contribs[k] = abs(math.log(v)) if v > 0.0 else 0.0
        total = sum(contribs.values()) or 1.0
        for k in _FACTOR_KEYS:
            shares_by_factor[k].append(contribs[k] / total)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = list(range(len(labels)))
    bottom = [0.0] * len(labels)
    for idx, k in enumerate(_FACTOR_KEYS):
        vals = shares_by_factor[k]
        ax.bar(x, vals, bottom=bottom, color=_FACTOR_COLORS[idx], label=k, width=0.85)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Log-space contribution share")
    ax.set_title(
        "Beta-B factor decomposition — 38 queries × top-1 ranker score\n"
        f"dominated_fraction = {report['aggregate']['factor_dominance']['dominated_fraction']:.3f} "
        f"(gate: ≤ 0.50)"
    )
    ax.axhline(0.80, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="80% dominance threshold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — contradiction-boost ablation
# ---------------------------------------------------------------------------

def fig_ablation(report: dict, out_path: Path) -> None:
    """Per-prior (1.0 … 2.0) rank histogram: src_rank and dst_rank of 8 C-pairs."""
    ablation = report["contradiction_boost_ablation"]
    priors = (1.0, 1.25, 1.5, 1.75, 2.0)

    fig, axes = plt.subplots(1, len(priors), figsize=(16, 4.2), sharey=True)
    for i, prior in enumerate(priors):
        detail = ablation[f"prior_{prior}"]["per_pair_detail"]
        src_ranks = [d["src_rank"] for d in detail]
        dst_ranks = [d["dst_rank"] for d in detail]
        ax = axes[i]
        width = 0.35
        x = list(range(len(detail)))
        ax.bar([xi - width/2 for xi in x], src_ranks, width=width, color="#1f77b4", label="src")
        ax.bar([xi + width/2 for xi in x], dst_ranks, width=width, color="#ff7f0e", label="dst")
        ax.axhline(4.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(
            f"α={prior}\nrecall={ablation[f'prior_{prior}']['contradiction_recall_per_pair']:.2f}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([d["id"] for d in detail], rotation=45, fontsize=7)
        if i == 0:
            ax.set_ylabel("rank in top-10 (−1 = not present)")
            ax.legend(fontsize=8)
        ax.set_ylim(-2, 12)
    pareto = ablation.get("pareto_summary", {})
    fig.suptitle(
        f"Beta-B contradiction-boost ablation — prior α ∈ {{1.0, 1.25, 1.5, 1.75, 2.0}}\n"
        f"1.5 Pareto-optimal = {pareto.get('prior_1.5_is_pareto_optimal')}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — provenance depth vs score
# ---------------------------------------------------------------------------

def fig_provenance_depth_score(report: dict, out_path: Path) -> None:
    """Box plot: score distribution bucketed by provenance_depth across all
    ranked hits in all queries. Verifies exp(-depth/3) curve shape.
    """
    buckets: dict[int, list[float]] = {}
    for qid, hits in report["ranked_hits_raw_by_qid"].items():
        for h in hits:
            d = int(h.get("provenance_depth", -1))
            s = float(h.get("score", 0.0))
            buckets.setdefault(d, []).append(s)
    depths = sorted(buckets.keys())
    vals = [buckets[d] for d in depths]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(vals, tick_labels=[str(d) for d in depths], showfliers=True)
    ax.set_xlabel("provenance_depth (−1 = no axiom-rooted path)")
    ax.set_ylabel("composed ranker score")
    ax.set_title(
        "Beta-B ranker score vs provenance depth\n"
        "expected: exp(-depth/3) curve if provenance_decay dominant"
    )
    # Overlay theoretical curve at unit-scale (ordinal reference)
    xs = [d for d in depths if d >= 0]
    if xs:
        max_score = max(max(b) for b in vals) if vals else 1.0
        ys = [math.exp(-x / 3.0) * max_score for x in xs]
        # xticks are 1-indexed in boxplot
        x_idx = [depths.index(x) + 1 for x in xs]
        ax.plot(x_idx, ys, color="red", linestyle="--", marker="o",
                alpha=0.6, label="exp(-depth/3) × max_score")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — authority × domain heatmap
# ---------------------------------------------------------------------------

_AUTHORITIES = ("primary", "derived", "internal", "agent")
_DOMAINS = ("qa_core", "svp", "geometry", "biology", "physics", "rsf")


def fig_authority_domain_heatmap(report: dict, out_path: Path) -> None:
    conn = sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    matrix: list[list[int]] = []
    for a in _AUTHORITIES:
        row = []
        for d in _DOMAINS:
            n = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE authority=? AND domain=?",
                (a, d),
            ).fetchone()[0]
            row.append(int(n))
        matrix.append(row)
    conn.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(len(_DOMAINS)))
    ax.set_xticklabels(_DOMAINS, rotation=30)
    ax.set_yticks(range(len(_AUTHORITIES)))
    ax.set_yticklabels(_AUTHORITIES)
    ax.set_xlabel("domain")
    ax.set_ylabel("authority")
    ax.set_title(
        "Beta-B authority × domain node counts\n"
        "empty cells = Phase 4.7 corpus-expansion targets"
    )
    for i in range(len(_AUTHORITIES)):
        for j in range(len(_DOMAINS)):
            v = matrix[i][j]
            color = "white" if v > max(max(r) for r in matrix) * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, label="node count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = _load_results()
    fig_factor_decomposition(report, _FIG_DIR / "factor_decomposition.png")
    fig_ablation(report, _FIG_DIR / "contradiction_boost_ablation.png")
    fig_provenance_depth_score(report, _FIG_DIR / "provenance_depth_score.png")
    fig_authority_domain_heatmap(report, _FIG_DIR / "authority_domain_heatmap.png")
    print(f"4 figures → {_FIG_DIR.relative_to(_REPO)}/")
    for p in sorted(_FIG_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024}KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
