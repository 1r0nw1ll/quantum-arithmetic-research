"""Beta-A metric functions.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

All functions pure. No DB access. Input shapes are explicit dicts so the
Beta-B benchmark runner can assemble them from ranker outputs without
coupling to sqlite3.Row.

Primary quality signal: graph_structural_pass_rate (binary per-query hit@k)
plus contradiction_recall_per_pair (both-endpoints-in-top-k per pair) and
authority_presence_at_k (expected primary in top-3 per A-query). Secondary
consistency checks (authority_monotonicity, provenance_spearman) report
shape of ranking but are NOT gate-driving per the decision matrix.

Factor-dominance is log-space contribution-share, matching the ranker's
multiplicative composition (score = ∏ factors → log score = ∑ log(factor)).
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Any


# ---------------------------------------------------------------------------
# Core per-query predicates
# ---------------------------------------------------------------------------

def gold_hit_at_k(result_ids: Sequence[str], gold_ids: Iterable[str], k: int = 5) -> bool:
    """True iff ≥1 gold node appears in the top-k results."""
    gold = set(gold_ids)
    if not gold:
        return True  # empty gold ↔ edge case ↔ no hit required
    top = set(result_ids[:k])
    return bool(top & gold)


def graph_structural_recall_at_k(
    result_ids: Sequence[str], gold_ids: Iterable[str], k: int = 5,
) -> float:
    """Fraction of gold nodes that appear in the top-k results.

    For domain queries where |gold| > k, this is capped at k/|gold|; the
    decision matrix uses the per-query BINARY hit@k (gold_hit_at_k) as the
    pass predicate. This function is for reporting only.
    """
    gold = set(gold_ids)
    if not gold:
        return 1.0
    top = set(result_ids[:k])
    return len(top & gold) / len(gold)


def graph_structural_pass_rate(
    per_query: Sequence[dict], k: int = 5,
) -> float:
    """Fraction of queries whose top-k surfaces ≥1 gold node.

    per_query: list of {"result_ids": [...], "gold_ids": [...]}.
    Decision-matrix gate: ≥ 0.80 over the 28 graph-structural queries.
    """
    if not per_query:
        return 0.0
    hits = sum(
        1 for q in per_query
        if gold_hit_at_k(q["result_ids"], q["gold_ids"], k)
    )
    return hits / len(per_query)


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------

def contradiction_recall_per_pair(
    per_pair: Sequence[dict], k: int = 5,
) -> float:
    """Fraction of C-pairs where BOTH endpoints appear in top-k.

    per_pair: list of {"result_ids": [...], "pair": [src_id, dst_id]}.
    Decision-matrix gate: ≥ 6/8 pairs.
    """
    if not per_pair:
        return 0.0
    both_hits = 0
    for entry in per_pair:
        top = set(entry["result_ids"][:k])
        src, dst = entry["pair"]
        if src in top and dst in top:
            both_hits += 1
    return both_hits / len(per_pair)


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------

def authority_presence_at_k(
    per_query: Sequence[dict], k: int = 3,
) -> float:
    """Fraction of A-queries with the curator-specified primary in top-k.

    per_query: list of {"result_ids": [...], "expected_primary_id": "..."}.
    Decision-matrix gate: ≥ 6/8 on A-queries.
    """
    if not per_query:
        return 0.0
    hits = 0
    for q in per_query:
        if q["expected_primary_id"] in q["result_ids"][:k]:
            hits += 1
    return hits / len(per_query)


# ---------------------------------------------------------------------------
# NDCG (for cross-domain blind-labeled queries)
# ---------------------------------------------------------------------------

def _dcg(grades: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(
    result_ids: Sequence[str],
    grades: dict[str, int],
    k: int = 10,
) -> float:
    """Standard nDCG@k with integer grades in [1,5] (missing → 0).

    Gain = 2**grade - 1; discount = log2(rank + 1). Ideal order = grades
    sorted descending. Returns 0.0 if no gradings exist.
    """
    if not grades:
        return 0.0
    predicted = [grades.get(nid, 0) for nid in result_ids[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return _dcg(predicted) / idcg


# ---------------------------------------------------------------------------
# Factor dominance (log-space contribution share)
# ---------------------------------------------------------------------------

_FACTOR_KEYS = (
    "authority", "lifecycle", "bm25_norm", "confidence",
    "time_decay", "contradiction", "prov_decay",
)


def _per_query_dominance(breakdown: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Return (max share, {factor: share}) for one query's score.

    Uses |log(factor)| so factors pulling score toward 1.0 (bm25_norm near
    max, contradiction = 1.0) contribute little, and factors lifting far
    from 1.0 (authority=10, bm25_norm near 0) contribute a lot. If every
    factor is exactly 1.0, returns (0.0, {all: 0.0}).
    """
    contribs: dict[str, float] = {}
    for k in _FACTOR_KEYS:
        v = breakdown.get(k, 1.0)
        if v <= 0.0:
            contribs[k] = 0.0
        else:
            contribs[k] = abs(math.log(v))
    total = sum(contribs.values())
    if total <= 0.0:
        return 0.0, {k: 0.0 for k in _FACTOR_KEYS}
    shares = {k: contribs[k] / total for k in _FACTOR_KEYS}
    return max(shares.values()), shares


def factor_dominance(
    per_query_breakdowns: Sequence[dict[str, float]],
    dominance_threshold: float = 0.80,
) -> dict[str, Any]:
    """Aggregate factor-dominance over a set of queries' top-1 breakdowns.

    per_query_breakdowns: list of score_breakdown dicts (from RankedHit).
    Returns {
      "per_query_max_share": [...],
      "dominated_fraction": float,      # fraction of queries with max > threshold
      "mean_share_by_factor": {factor: mean across queries},
    }
    Decision-matrix gate: dominated_fraction ≤ 0.50 (no single factor
    dominates > 80% of the score for > 50% of queries).
    """
    if not per_query_breakdowns:
        return {
            "per_query_max_share": [],
            "dominated_fraction": 0.0,
            "mean_share_by_factor": {k: 0.0 for k in _FACTOR_KEYS},
        }
    max_shares: list[float] = []
    share_accumulators: dict[str, list[float]] = {k: [] for k in _FACTOR_KEYS}
    for bd in per_query_breakdowns:
        max_share, shares = _per_query_dominance(bd)
        max_shares.append(max_share)
        for k in _FACTOR_KEYS:
            share_accumulators[k].append(shares[k])
    dominated = sum(1 for s in max_shares if s > dominance_threshold)
    return {
        "per_query_max_share": max_shares,
        "dominated_fraction": dominated / len(max_shares),
        "mean_share_by_factor": {
            k: statistics.mean(v) if v else 0.0
            for k, v in share_accumulators.items()
        },
    }


# ---------------------------------------------------------------------------
# Consistency checks (reporting-only, not gate-driving)
# ---------------------------------------------------------------------------

_AUTHORITY_ORDER = {"primary": 3, "derived": 2, "internal": 1, "agent": 0}


def authority_monotonicity(
    per_query: Sequence[dict], epsilon: float = 0.5,
) -> float:
    """Fraction of queries where authority tier ordering within top-k is
    consistent with authority_weight (primary ≥ derived ≥ internal ≥ agent)
    in score order, up to ε tolerance on adjacent-rank authority equality.

    per_query: list of {"ranked_hits": [{"authority": str, "score": float}, ...]}.
    Consistency check only — large inversions don't pass the gate but don't
    fail it either; reported for the editorial section.
    """
    if not per_query:
        return 0.0
    consistent = 0
    for q in per_query:
        hits = q["ranked_hits"]
        if len(hits) < 2:
            consistent += 1
            continue
        bad = False
        for i in range(len(hits) - 1):
            a1, a2 = hits[i]["authority"], hits[i + 1]["authority"]
            r1, r2 = _AUTHORITY_ORDER.get(a1, 0), _AUTHORITY_ORDER.get(a2, 0)
            if r1 < r2:
                # lower authority above higher — inversion
                s1, s2 = hits[i]["score"], hits[i + 1]["score"]
                if s1 - s2 < epsilon:
                    # inside tolerance; skip
                    continue
                bad = True
                break
        if not bad:
            consistent += 1
    return consistent / len(per_query)


def provenance_spearman(per_query: Sequence[dict]) -> float:
    """Mean Spearman correlation of (rank, provenance_depth) across queries.

    per_query: list of {"ranked_hits": [{"rank": int, "provenance_depth": int}, ...]}.
    Within each query, positive correlation means lower-rank (better) hits
    have shallower provenance depth — expected if provenance_decay dominates.
    Ignores queries with all depths = -1 or < 3 hits.
    """
    if not per_query:
        return 0.0
    corrs: list[float] = []
    for q in per_query:
        hits = q["ranked_hits"]
        if len(hits) < 3:
            continue
        depths = [h["provenance_depth"] for h in hits]
        if all(d == -1 for d in depths):
            continue
        ranks = [h["rank"] for h in hits]
        corrs.append(_spearman(ranks, depths))
    if not corrs:
        return 0.0
    return statistics.mean(corrs)


def _spearman(x: Sequence[int], y: Sequence[int]) -> float:
    """Pearson correlation on rank-transformed x, y — standard Spearman."""
    def _rank(vs: Sequence[int]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-indexed
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    rx, ry = _rank(x), _rank(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Lifecycle gate
# ---------------------------------------------------------------------------

def lifecycle_ordering_pass(per_query: Sequence[dict]) -> float:
    """Fraction of L-queries where current-lifecycle gold outranks all
    superseded anti-gold in the top-k window.

    per_query: list of {"result_ids": [...], "gold_ids": [current_id],
                        "anti_gold_ids": [superseded_ids...], "k": int}.
    Decision-matrix gate: 2/2 (both L-queries must pass).
    """
    if not per_query:
        return 0.0
    hits = 0
    for q in per_query:
        ids = q["result_ids"]
        k = q.get("k", 5)
        top = ids[:k]
        gold_id = q["gold_ids"][0] if q["gold_ids"] else None
        if gold_id is None or gold_id not in top:
            continue
        gold_rank = top.index(gold_id)
        anti_ranks = [top.index(a) for a in q["anti_gold_ids"] if a in top]
        if all(gold_rank < ar for ar in anti_ranks):
            hits += 1
    return hits / len(per_query)
