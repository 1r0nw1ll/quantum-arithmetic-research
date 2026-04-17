"""Beta-B benchmark orchestrator.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Runs the 38 pre-registered queries through KG.search_authority_ranked and raw
KG.search (BM25 baseline), applies the pre-committed metrics from
tools.qa_kg.analysis.metrics, runs the 5 agent tasks, runs the contradiction-
boost ablation, and measures latency. Writes beta_results.json with raw
per-query breakdowns so the decision matrix can be applied without re-running.

Conforms to the Beta-B execution contract in docs/specs/QA_MEM_BETA_DECISION_MATRIX.md:
no methodology changes; no post-hoc threshold adjustment; no fixture edits beyond
N1 scope; no A-RAG on T3–T5.

Run: python -m tools.qa_kg.analysis.run_beta_benchmark
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import dataclasses
import datetime as _dt
import json
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

from tools.qa_kg.canonicalize import graph_hash
from tools.qa_kg.kg import KG
from tools.qa_kg.ranker import RankerSpec, load_spec
from tools.qa_kg.schema import DEFAULT_DB
from tools.qa_kg.analysis import metrics as M
from tools.qa_kg.analysis.agent_tasks import TASKS as AGENT_TASKS

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO / "tools" / "qa_kg" / "fixtures"
_ANALYSIS_DIR = _REPO / "tools" / "qa_kg" / "analysis"
_QUERIES_PATH = _FIXTURE_DIR / "beta_prereg_queries.json"
_GOLD_PATH = _FIXTURE_DIR / "beta_prereg_gold.json"
_BLIND_PATH = _FIXTURE_DIR / "beta_blind_gold.json"
_RESULTS_PATH = _ANALYSIS_DIR / "beta_results.json"

# ---------------------------------------------------------------------------
# FTS5 sanitizer — mirrored from pilot_validate to keep query shape identical
# ---------------------------------------------------------------------------

_FTS5_SANITIZE_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_FTS5_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "was", "were", "be", "by", "for", "with", "about", "what", "how",
    "does", "do", "did", "can", "that", "this", "these", "those", "which",
    "vs", "versus", "but", "if", "not", "no", "yes", "its", "it", "i",
    "as", "at", "so", "us", "we", "both",
})


def _sanitize_for_fts5(query_text: str) -> str:
    cleaned = _FTS5_SANITIZE_RE.sub(" ", query_text).lower()
    tokens = [t for t in cleaned.split() if t and t not in _FTS5_STOPWORDS]
    if not tokens:
        return query_text.strip()
    return " OR ".join(tokens) if len(tokens) > 1 else tokens[0]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Query execution helpers
# ---------------------------------------------------------------------------

def _row_summary(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "authority": row["authority"],
        "epistemic_status": row["epistemic_status"],
        "domain": row["domain"],
        "lifecycle_state": row["lifecycle_state"],
        "node_type": row["node_type"],
    }


def _run_ranker(kg: KG, q: dict, *, spec: RankerSpec | None = None, k: int = 10) -> dict:
    """Execute KG.search_authority_ranked and return the hit list as dicts."""
    qtext = _sanitize_for_fts5(q["query_text"])
    kwargs: dict = {"k": k}
    if q.get("domain_filter") is not None:
        kwargs["domain"] = q["domain_filter"]
    if q.get("min_authority"):
        kwargs["min_authority"] = q["min_authority"]
    if q.get("valid_at"):
        kwargs["valid_at"] = _dt.datetime.fromisoformat(
            q["valid_at"].replace("Z", "+00:00"),
        )
    if spec is not None:
        kwargs["spec"] = spec

    t0 = time.perf_counter()
    try:
        hits = kg.search_authority_ranked(qtext, **kwargs)
        err = None
    except sqlite3.OperationalError as exc:
        hits = []
        err = str(exc)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    out_hits = []
    for rank, h in enumerate(hits):
        out_hits.append({
            "rank": rank,
            "id": h.node["id"],
            "authority": h.authority,
            "epistemic_status": h.node["epistemic_status"],
            "domain": h.node["domain"],
            "lifecycle_state": h.node["lifecycle_state"],
            "score": h.score,
            "contradiction_state": h.contradiction_state,
            "provenance_depth": h.provenance_depth,
            "score_breakdown": h.score_breakdown,
        })
    return {
        "sanitized_query": qtext,
        "hits": out_hits,
        "latency_ms": dt_ms,
        "error": err,
    }


def _run_bm25_baseline(kg: KG, q: dict, *, k: int = 10) -> dict:
    """Raw BM25 search — no authority awareness, no ranker factors."""
    qtext = _sanitize_for_fts5(q["query_text"])
    t0 = time.perf_counter()
    try:
        rows = kg.search(qtext, k=k)
        err = None
    except sqlite3.OperationalError as exc:
        rows = []
        err = str(exc)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    hits = [
        {"rank": i, **_row_summary(r)} for i, r in enumerate(rows)
    ]
    return {
        "sanitized_query": qtext,
        "hits": hits,
        "latency_ms": dt_ms,
        "error": err,
    }


# ---------------------------------------------------------------------------
# Per-query categorised evaluation
# ---------------------------------------------------------------------------

def _eval_query(q: dict, gold: dict, blind: dict, ranker_out: dict, bm25_out: dict) -> dict:
    """Fold ranker result + gold into per-query metrics."""
    hits = ranker_out["hits"]
    result_ids = [h["id"] for h in hits]
    category = q["category"]

    entry = {
        "id": q["id"],
        "category": category,
        "query_text": q["query_text"],
        "sanitized_query": ranker_out["sanitized_query"],
        "domain_filter": q.get("domain_filter"),
        "min_authority": q.get("min_authority"),
        "valid_at": q.get("valid_at"),
        "result_count": len(hits),
        "top_k_ids": result_ids[:10],
        "top_k_authorities": [h["authority"] for h in hits[:10]],
        "ranker_latency_ms": ranker_out["latency_ms"],
        "bm25_latency_ms": bm25_out["latency_ms"],
        "ranker_error": ranker_out["error"],
    }

    gold_ids = gold.get("gold_ids") or []
    anti_gold_ids = gold.get("anti_gold_ids") or []
    entry["gold_ids"] = gold_ids
    entry["anti_gold_ids"] = anti_gold_ids

    # Per-category metric computation
    if category == "provenance":
        entry["hit_at_5"] = M.gold_hit_at_k(result_ids, gold_ids, k=5)
        entry["recall_at_5"] = M.graph_structural_recall_at_k(result_ids, gold_ids, k=5)

    elif category == "contradiction":
        # per_pair: both endpoints must be in top-5
        if len(gold_ids) == 2:
            entry["pair_in_top_5"] = all(g in result_ids[:5] for g in gold_ids)
        else:
            entry["pair_in_top_5"] = False
        entry["hit_at_5"] = M.gold_hit_at_k(result_ids, gold_ids, k=5)

    elif category == "domain":
        entry["hit_at_5"] = M.gold_hit_at_k(result_ids, gold_ids, k=5)
        entry["recall_at_5"] = M.graph_structural_recall_at_k(result_ids, gold_ids, k=5)
        # Domain filter purity: every returned hit should have q.domain_filter
        domain_target = q.get("domain_filter") or ""
        if domain_target and domain_target != "nonexistent_domain":
            entry["domain_purity"] = (
                all(h["domain"] == domain_target for h in hits)
                if hits else True
            )

    elif category == "authority":
        expected = q.get("expected_primary_id")
        entry["expected_primary_id"] = expected
        entry["primary_in_top_3"] = expected in result_ids[:3]
        entry["primary_in_top_5"] = expected in result_ids[:5]
        entry["primary_rank"] = (
            result_ids.index(expected) if expected in result_ids else -1
        )

    elif category == "lifecycle":
        current_id = gold_ids[0] if gold_ids else None
        if current_id and current_id in result_ids[:5]:
            current_rank = result_ids.index(current_id)
            anti_ranks_in_top5 = [
                result_ids.index(a) for a in anti_gold_ids if a in result_ids[:5]
            ]
            entry["lifecycle_pass"] = all(
                current_rank < ar for ar in anti_ranks_in_top5
            )
            entry["current_rank"] = current_rank
            entry["anti_gold_ranks_in_top5"] = anti_ranks_in_top5
        else:
            entry["lifecycle_pass"] = False
            entry["current_rank"] = -1
            entry["anti_gold_ranks_in_top5"] = []

    elif category == "cross_domain":
        blind_entry = blind.get(q["id"], {})
        grades = blind_entry.get("parsed_grades", {}) or {}
        entry["n_graded"] = len(grades)
        entry["ndcg_at_10_qa_mem"] = M.ndcg_at_k(result_ids, grades, k=10)
        bm25_result_ids = [h["id"] for h in bm25_out["hits"]]
        entry["ndcg_at_10_bm25"] = M.ndcg_at_k(bm25_result_ids, grades, k=10)
        entry["ndcg_delta_qa_mem_vs_bm25"] = (
            entry["ndcg_at_10_qa_mem"] - entry["ndcg_at_10_bm25"]
        )
        entry["bm25_top_k_ids"] = bm25_result_ids[:10]

    elif category == "edge_case":
        expected = q.get("expected_result")
        entry["expected_result"] = expected
        if expected == "empty":
            entry["edge_pass"] = len(hits) == 0
        elif expected == "no_derived_in_top_k":
            # min_authority='primary' → candidate pool excludes derived
            top5_auth = [h["authority"] for h in hits[:5]]
            entry["edge_pass"] = all(a == "primary" for a in top5_auth) or not top5_auth
        elif expected == "valid_at_passthrough":
            # valid_at far-future should pass-through all valid_until='' nodes;
            # compared below against the no-valid_at run appended on the runner
            entry["edge_pass"] = len(hits) > 0
        else:
            entry["edge_pass"] = False

    return entry


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(per_query: list[dict]) -> dict:
    """Apply metrics.py aggregators across the 38 queries."""
    # Graph-structural queries: P(6) + C(8) + D(6) + A(8) — 28 total.
    # Gate: ≥1 gold_id in top-5. Domain queries use same predicate via gold_hit_at_k.
    # Authority queries use top-3 specifically; here graph_structural_pass_rate
    # uses top-5 per metrics.py. Authority presence @3 is a separate gate.
    gs_qs = [
        {"result_ids": e["top_k_ids"], "gold_ids": e["gold_ids"]}
        for e in per_query
        if e["category"] in ("provenance", "contradiction", "domain", "authority")
    ]
    graph_structural_pass_rate = M.graph_structural_pass_rate(gs_qs, k=5)

    # Contradiction: per-pair both-in-top-5
    contra_pairs = [
        {"result_ids": e["top_k_ids"], "pair": e["gold_ids"]}
        for e in per_query if e["category"] == "contradiction" and len(e["gold_ids"]) == 2
    ]
    contradiction_recall = M.contradiction_recall_per_pair(contra_pairs, k=5)

    # Authority: curator-specified primary in top-3
    auth_qs = [
        {"result_ids": e["top_k_ids"], "expected_primary_id": e["expected_primary_id"]}
        for e in per_query if e["category"] == "authority" and e.get("expected_primary_id")
    ]
    authority_presence = M.authority_presence_at_k(auth_qs, k=3)

    # Lifecycle: current outranks all superseded in top-5
    lc_qs = [
        {
            "result_ids": e["top_k_ids"],
            "gold_ids": e["gold_ids"],
            "anti_gold_ids": e["anti_gold_ids"],
            "k": 5,
        }
        for e in per_query if e["category"] == "lifecycle"
    ]
    lifecycle_pass = M.lifecycle_ordering_pass(lc_qs)

    # NDCG@10 cross-domain (QA-MEM ≥ BM25)
    xd = [e for e in per_query if e["category"] == "cross_domain"]
    ndcg_wins = sum(1 for e in xd if e["ndcg_at_10_qa_mem"] >= e["ndcg_at_10_bm25"])
    ndcg_cross_domain_wins = ndcg_wins
    ndcg_cross_domain_total = len(xd)

    # Factor dominance: over all 38 queries' top-1 hits that exist
    top1_breakdowns = [
        e["top_k_ids"] and
        next(
            (h for h in ((r for r in [e] for r in [None])) if h), None,
        )
        for e in per_query
    ]
    # Simpler: each per_query row carries its hits under top_k_ids; we need
    # the first hit's score_breakdown. Pull from the raw hits slot we saved
    # on the entry under "top1_breakdown".
    breakdowns = [
        e["top1_breakdown"] for e in per_query if e.get("top1_breakdown")
    ]
    fd = M.factor_dominance(breakdowns, dominance_threshold=0.80)

    # Consistency metrics
    auth_mon_qs = [
        {"ranked_hits": [{"authority": h["authority"], "score": h["score"]}
                         for h in e["ranked_hits_raw"]]}
        for e in per_query if e.get("ranked_hits_raw")
    ]
    authority_monotonicity = M.authority_monotonicity(auth_mon_qs)

    prov_qs = [
        {"ranked_hits": [{"rank": h["rank"], "provenance_depth": h["provenance_depth"]}
                         for h in e["ranked_hits_raw"]]}
        for e in per_query if e.get("ranked_hits_raw")
    ]
    provenance_spearman = M.provenance_spearman(prov_qs)

    return {
        "graph_structural_pass_rate": graph_structural_pass_rate,
        "graph_structural_hits_over_28": sum(
            1 for q in gs_qs if M.gold_hit_at_k(q["result_ids"], q["gold_ids"], 5)
        ),
        "graph_structural_denominator": len(gs_qs),
        "contradiction_recall_per_pair": contradiction_recall,
        "contradiction_pair_hits_over_8": sum(
            1 for p in contra_pairs
            if all(g in p["result_ids"][:5] for g in p["pair"])
        ),
        "contradiction_pair_denominator": len(contra_pairs),
        "authority_presence_at_3": authority_presence,
        "authority_hits_over_8": sum(
            1 for q in auth_qs if q["expected_primary_id"] in q["result_ids"][:3]
        ),
        "authority_denominator": len(auth_qs),
        "lifecycle_ordering_pass": lifecycle_pass,
        "lifecycle_hits_over_2": int(lifecycle_pass * len(lc_qs)),
        "lifecycle_denominator": len(lc_qs),
        "ndcg_cross_domain_wins": ndcg_cross_domain_wins,
        "ndcg_cross_domain_denominator": ndcg_cross_domain_total,
        "factor_dominance": fd,
        "authority_monotonicity": authority_monotonicity,
        "provenance_spearman": provenance_spearman,
    }


# ---------------------------------------------------------------------------
# Contradiction-boost ablation
# ---------------------------------------------------------------------------

_BOOST_LADDER = (1.0, 1.25, 1.5, 1.75, 2.0)


def _derived_spec(base: RankerSpec, *, contradiction_prior: float) -> RankerSpec:
    return dataclasses.replace(base, contradiction_prior=contradiction_prior)


def _run_ablation(
    kg: KG, queries: list[dict], gold_map: dict, blind_map: dict, base_spec: RankerSpec,
) -> dict:
    """Run the benchmark at each contradiction_prior setting; report per-setting
    contradiction_recall + graph_structural_pass_rate + per-pair rank changes."""
    c_queries = [q for q in queries if q["category"] == "contradiction"]
    gs_queries = [
        q for q in queries
        if q["category"] in ("provenance", "contradiction", "domain", "authority")
    ]

    ablation = {}
    for prior in _BOOST_LADDER:
        spec = _derived_spec(base_spec, contradiction_prior=prior)

        # Contradiction recall
        pair_results = []
        pair_detail = []
        for q in c_queries:
            r = _run_ranker(kg, q, spec=spec, k=10)
            result_ids = [h["id"] for h in r["hits"]]
            pair = gold_map[q["id"]]["gold_ids"]
            pair_results.append({"result_ids": result_ids, "pair": pair})
            src_rank = result_ids.index(pair[0]) if pair[0] in result_ids else -1
            dst_rank = result_ids.index(pair[1]) if pair[1] in result_ids else -1
            pair_detail.append({
                "id": q["id"], "pair": pair,
                "src_rank": src_rank, "dst_rank": dst_rank,
                "both_in_top_5": (
                    0 <= src_rank < 5 and 0 <= dst_rank < 5
                ),
            })
        contra_recall = M.contradiction_recall_per_pair(pair_results, k=5)

        # Graph-structural pass rate across all 28 structural queries
        gs_hits = []
        for q in gs_queries:
            r = _run_ranker(kg, q, spec=spec, k=10)
            result_ids = [h["id"] for h in r["hits"]]
            g = gold_map.get(q["id"], {})
            # Authority queries: gold is [expected_primary_id]; domain: gold_ids list;
            # provenance/contradiction: gold_ids from derive_gold.
            gold_ids = g.get("gold_ids") or []
            if q["category"] == "authority":
                exp = q.get("expected_primary_id")
                gold_ids = [exp] if exp else []
            gs_hits.append({"result_ids": result_ids, "gold_ids": gold_ids})
        gs_pass = M.graph_structural_pass_rate(gs_hits, k=5)

        ablation[f"prior_{prior}"] = {
            "contradiction_prior": prior,
            "contradiction_recall_per_pair": contra_recall,
            "graph_structural_pass_rate": gs_pass,
            "per_pair_detail": pair_detail,
        }

    # Pareto check: 1.5 vs alternatives
    baseline = ablation["prior_1.5"]
    pareto = {"prior_1.5_is_pareto_optimal": True, "notes": []}
    for prior in _BOOST_LADDER:
        if prior == 1.5:
            continue
        alt = ablation[f"prior_{prior}"]
        # 1.5 Pareto-optimal over alt iff weakly worse at prior OR strictly better at prior.
        worse_metric = False
        if alt["contradiction_recall_per_pair"] > baseline["contradiction_recall_per_pair"]:
            worse_metric = True
        if alt["graph_structural_pass_rate"] > baseline["graph_structural_pass_rate"]:
            worse_metric = True
        if worse_metric:
            pareto["prior_1.5_is_pareto_optimal"] = False
            pareto["notes"].append(
                f"prior={prior}: contradiction={alt['contradiction_recall_per_pair']:.3f} "
                f"gs_pass={alt['graph_structural_pass_rate']:.3f} dominates 1.5="
                f"{baseline['contradiction_recall_per_pair']:.3f}/"
                f"{baseline['graph_structural_pass_rate']:.3f}"
            )
    ablation["pareto_summary"] = pareto
    return ablation


# ---------------------------------------------------------------------------
# Agent tasks (T1–T5)
# ---------------------------------------------------------------------------

def _task_t1_t2(kg: KG) -> dict:
    """Head-to-head for T1/T2: run both QA-MEM and A-RAG; binary pass/fail on
    whether the required identifiers can be surfaced in the top-k output."""
    try:
        from tools.qa_retrieval.query import retrieve_pipeline
        arag_available = True
    except Exception as exc:  # import guard — A-RAG index may be absent
        arag_available = False
        _arag_import_error = str(exc)

    # T1 — Keely triune claim
    t1_q = (
        "Keely triune force law 11 primary quote "
        "cert qa_keely_triune_cert"
    )
    t1_qa_mem = _run_ranker(kg, {"query_text": t1_q, "category": "agent_task_t1"}, k=10)
    t1_qa_ids = {h["id"] for h in t1_qa_mem["hits"]}
    t1_qa_mem_pass = (
        "sc:keely_law_11_triune_force" in t1_qa_ids
        and "cert:fs:qa_keely_triune_cert_v1" in t1_qa_ids
    )

    t1_arag_pass = None
    t1_arag_result = None
    if arag_available:
        try:
            t1_arag_result = retrieve_pipeline(t1_q, limit=10, use_semantic=False)
            # A-RAG returns conversation/message objects from chat/obsidian corpus.
            # Pass = output text plainly identifies the Keely Law 11 triune force
            # quote AND the cert family id/name. We check lexical markers.
            blob = " ".join(
                (r.get("body") or "") + " " + (r.get("title") or "")
                for r in t1_arag_result.get("final", [])
            ).lower()
            has_quote = ("triune" in blob and "law 11" in blob) or (
                "atomole" in blob and "law 11" in blob
            )
            has_cert = "qa_keely_triune_cert" in blob or "cert:fs:qa_keely_triune" in blob
            t1_arag_pass = bool(has_quote and has_cert)
        except Exception as exc:
            t1_arag_pass = False
            t1_arag_result = {"error": str(exc)}

    # T2 — Keely Law 17 dispute
    t2_q = (
        "Keely Law 17 transformation structural reclassification "
        "Vibes dispute 21 octaves"
    )
    t2_qa_mem = _run_ranker(kg, {"query_text": t2_q, "category": "agent_task_t2"}, k=10)
    t2_qa_ids = {h["id"] for h in t2_qa_mem["hits"]}
    t2_qa_mem_pass = (
        "sc:keely_law_17_transformation_21_octaves" in t2_qa_ids
        and "obs:keely_law_17_vibes_structural_reclassification" in t2_qa_ids
    )

    t2_arag_pass = None
    t2_arag_result = None
    if arag_available:
        try:
            t2_arag_result = retrieve_pipeline(t2_q, limit=10, use_semantic=False)
            blob = " ".join(
                (r.get("body") or "") + " " + (r.get("title") or "")
                for r in t2_arag_result.get("final", [])
            ).lower()
            has_original = "law 17" in blob and "21" in blob
            has_dispute = (
                "vibes" in blob
                or "structural reclassification" in blob
                or "category 1" in blob
            )
            t2_arag_pass = bool(has_original and has_dispute)
        except Exception as exc:
            t2_arag_pass = False
            t2_arag_result = {"error": str(exc)}

    return {
        "T1": {
            "prompt": t1_q,
            "qa_mem_top_10": [h["id"] for h in t1_qa_mem["hits"]],
            "qa_mem_pass": t1_qa_mem_pass,
            "a_rag_pass": t1_arag_pass,
            "a_rag_final_ids": (
                [r.get("msg_id") for r in t1_arag_result.get("final", [])]
                if isinstance(t1_arag_result, dict) else None
            ),
            "head_to_head_both_pass": bool(t1_qa_mem_pass and t1_arag_pass),
        },
        "T2": {
            "prompt": t2_q,
            "qa_mem_top_10": [h["id"] for h in t2_qa_mem["hits"]],
            "qa_mem_pass": t2_qa_mem_pass,
            "a_rag_pass": t2_arag_pass,
            "a_rag_final_ids": (
                [r.get("msg_id") for r in t2_arag_result.get("final", [])]
                if isinstance(t2_arag_result, dict) else None
            ),
            "head_to_head_both_pass": bool(t2_qa_mem_pass and t2_arag_pass),
        },
        "_a_rag_available": arag_available,
    }


def _task_t3_t4_t5(kg: KG) -> dict:
    """Capability-only agent tasks — QA-MEM internal consistency."""
    # T3 — provenance via why() back to an axiom
    chain = kg.why("cert:fs:qa_keely_triune_cert_v1", max_depth=3)
    dsts = [(row["src"], row["dst"], row["edge_type"]) for row in chain]
    axiom_found = False
    reached_axioms: list[str] = []
    if chain:
        dst_ids = sorted({r["dst"] for r in chain})
        if dst_ids:
            placeholders = ",".join("?" * len(dst_ids))
            for row in kg.conn.execute(
                f"SELECT id, epistemic_status FROM nodes WHERE id IN ({placeholders})",
                dst_ids,
            ).fetchall():
                if row["epistemic_status"] == "axiom":
                    axiom_found = True
                    reached_axioms.append(row["id"])
    t3 = {
        "chain_length": len(chain),
        "chain_triples": dsts[:20],
        "reached_axioms": reached_axioms,
        "pass": bool(axiom_found),
    }

    # T4 — domain filter purity on geometry
    t4_hits = kg.search_authority_ranked("Parker", domain="geometry", k=10)
    t4_domains = [h.node["domain"] for h in t4_hits]
    t4 = {
        "result_count": len(t4_hits),
        "returned_domains": t4_domains,
        "pass": bool(t4_hits) and all(d == "geometry" for d in t4_domains),
    }

    # T5 — min_authority='internal' excludes agent
    t5_hits = kg.search_authority_ranked("keely", min_authority="internal", k=10)
    t5_authorities = [h.authority for h in t5_hits]
    agent_count = kg.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE authority='agent'"
    ).fetchone()[0]
    t5 = {
        "result_count": len(t5_hits),
        "returned_authorities": t5_authorities,
        "agent_nodes_in_db": int(agent_count),
        "pass": not any(a == "agent" for a in t5_authorities),
        "pass_kind": (
            "vacuous" if agent_count == 0 else "real_exclusion"
        ),
    }

    return {"T3": t3, "T4": t4, "T5": t5}


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _measure_real_latency(kg: KG, queries: list[dict]) -> dict:
    # Warm-up + three passes
    lat_ms: list[float] = []
    for _ in range(3):
        for q in queries:
            r = _run_ranker(kg, q, k=10)
            if r["error"] is None:
                lat_ms.append(r["latency_ms"])
    return {
        "n_samples": len(lat_ms),
        "p50_ms": _percentile(lat_ms, 50),
        "p95_ms": _percentile(lat_ms, 95),
        "p99_ms": _percentile(lat_ms, 99),
        "mean_ms": (sum(lat_ms) / len(lat_ms)) if lat_ms else 0.0,
        "max_ms": max(lat_ms) if lat_ms else 0.0,
    }


def _synthetic_latency_only(kg: KG) -> dict:
    """Insert 5000 method=script synthetic nodes, measure p95 search latency,
    delete after. LATENCY ONLY — ranking-quality claims on synthetic data are
    explicitly forbidden by the Beta-B execution contract.

    # noqa: T2-D-6, reason: synthetic latency simulation only, no QA ranking
    # claims — observer-layer measurement per Theorem NT + Beta-B hard rule.
    """
    conn = kg.conn
    n_before = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    inserted_ids: list[str] = []
    try:
        NOW_TS = _now()
        SYNTH_PREFIX = "synth:latency_beta_b:"
        cols = (
            "id", "node_type", "tier", "title", "body",
            "idx_b", "idx_e",
            "authority", "epistemic_status",
            "method", "source_locator",
            "lifecycle_state", "confidence", "domain",
            "created_ts", "updated_ts",
        )
        sql = (
            f"INSERT INTO nodes ({', '.join(cols)}) VALUES "
            f"({', '.join('?' * len(cols))})"
        )
        for i in range(5000):
            nid = f"{SYNTH_PREFIX}{i:05d}"
            body_words = [f"keely_{i%17}", f"parker_{i%13}", f"wildberger_{i%11}"]
            row = (
                nid, "SourceClaim", "cosmos", f"synth latency {i}",
                " ".join(body_words * 4),
                (i % 9) + 1, (i % 9) + 1,
                "internal", "source_claim",
                "script", f"synthetic:idx={i}",
                "current", 1.0, "",
                NOW_TS, NOW_TS,
            )
            conn.execute(sql, row)
            inserted_ids.append(nid)
        conn.commit()

        # Measure latency at ~5500 nodes — use a generic query token.
        import time as _time
        n_after = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        # FTS5 auto-maintained trigger should keep nodes_fts in sync (per schema)
        # Run a handful of generic queries
        lat_ms: list[float] = []
        test_queries = [
            "keely", "parker", "wildberger", "keely OR parker",
            "triune OR chromogeometry", "digital root", "synthetic",
        ]
        for _ in range(10):
            for q in test_queries:
                t0 = _time.perf_counter()
                try:
                    _ = kg.search_authority_ranked(q, k=10)
                except sqlite3.OperationalError:
                    continue
                lat_ms.append((_time.perf_counter() - t0) * 1000.0)
        result = {
            "synthetic_node_count_inserted": len(inserted_ids),
            "nodes_before": int(n_before),
            "nodes_during_measurement": int(n_after),
            "p50_ms": _percentile(lat_ms, 50),
            "p95_ms": _percentile(lat_ms, 95),
            "p99_ms": _percentile(lat_ms, 99),
            "mean_ms": (sum(lat_ms) / len(lat_ms)) if lat_ms else 0.0,
            "n_samples": len(lat_ms),
            "disclaimer": (
                "LATENCY ONLY — synthetic nodes have method=script and carry "
                "no authority/provenance signal. Ranking quality on this "
                "configuration is not claimed. Nodes deleted post-measurement."
            ),
        }
    finally:
        # Clean up synthetic nodes
        if inserted_ids:
            placeholders = ",".join("?" * len(inserted_ids))
            conn.execute(
                f"DELETE FROM nodes WHERE id IN ({placeholders})", inserted_ids,
            )
            conn.commit()
        n_final = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        result["nodes_after_cleanup"] = int(n_final)
        result["cleanup_verified"] = (n_final == n_before)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_fixtures() -> tuple[list[dict], dict, dict]:
    queries_file = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    queries = queries_file["queries"]
    gold_file = json.loads(_GOLD_PATH.read_text(encoding="utf-8"))
    gold = {e["id"]: e for e in gold_file["gold_entries"]}
    blind_file = json.loads(_BLIND_PATH.read_text(encoding="utf-8"))
    blind = {r["id"]: r for r in blind_file["results"]}
    return queries, gold, blind


def main() -> int:
    queries, gold_map, blind_map = _load_fixtures()

    # Read-write sqlite so the synthetic-latency step can insert+delete.
    conn = sqlite3.connect(DEFAULT_DB)
    conn.row_factory = sqlite3.Row
    try:
        gh_before = graph_hash(conn)
        kg = KG(conn)
        base_spec = load_spec()

        # --- B1: full 38-query benchmark ----------------------------------
        per_query: list[dict] = []
        for q in queries:
            ranker_out = _run_ranker(kg, q, spec=base_spec, k=10)
            bm25_out = _run_bm25_baseline(kg, q, k=10)
            # For authority category, gold_map does not carry gold_ids of the
            # expected primary directly — derive_gold stores them conditional
            # on primary validation. We also add expected_primary_id from the
            # query fixture for the metric.
            gold_entry = gold_map.get(q["id"], {})
            # Construct a gold_ids list from authority queries where missing.
            if q["category"] == "authority" and not gold_entry.get("gold_ids"):
                exp = q.get("expected_primary_id")
                gold_entry = dict(gold_entry)
                if exp:
                    gold_entry["gold_ids"] = [exp]
            entry = _eval_query(q, gold_entry, blind_map, ranker_out, bm25_out)
            # Attach raw breakdown for aggregation
            entry["top1_breakdown"] = (
                ranker_out["hits"][0]["score_breakdown"]
                if ranker_out["hits"] else None
            )
            entry["ranked_hits_raw"] = ranker_out["hits"]
            per_query.append(entry)

        # E04 secondary run: no valid_at for comparison
        e04 = next((e for e in per_query if e["id"] == "E04"), None)
        if e04 is not None:
            e04_q_no_valid = {
                "id": "E04_no_valid_at",
                "query_text": "Keely",
                "category": "edge_case",
            }
            r_noval = _run_ranker(kg, e04_q_no_valid, k=10)
            e04["valid_at_passthrough_delta"] = {
                "with_valid_at_ids": e04["top_k_ids"],
                "without_valid_at_ids": [h["id"] for h in r_noval["hits"]],
                "identical_ordering": (
                    e04["top_k_ids"] == [h["id"] for h in r_noval["hits"]]
                ),
            }
            # Updated edge_pass: identical (all valid_until='' should pass-through)
            e04["edge_pass"] = e04["valid_at_passthrough_delta"]["identical_ordering"]

        aggregate = _aggregate(per_query)

        # --- B2: agent tasks -----------------------------------------------
        t12 = _task_t1_t2(kg)
        t345 = _task_t3_t4_t5(kg)
        agent_task_summary = {
            **t12, **t345,
            "head_to_head_both_pass": (
                t12["T1"]["head_to_head_both_pass"]
                and t12["T2"]["head_to_head_both_pass"]
            ),
        }

        # --- B3: contradiction-boost ablation -----------------------------
        ablation = _run_ablation(kg, queries, gold_map, blind_map, base_spec)

        # --- B5: latency --------------------------------------------------
        real_latency = _measure_real_latency(kg, queries)
        synth_latency = _synthetic_latency_only(kg)

        gh_after = graph_hash(conn)
        graph_hash_stable = (gh_before == gh_after)

        # --- Compose output -----------------------------------------------
        # Strip heavy fields from top-level per_query to stay human-inspectable;
        # keep ranked_hits_raw in a side dict for visualization.
        ranked_raw = {
            e["id"]: e.pop("ranked_hits_raw", []) for e in per_query
        }
        # top1_breakdown already inline

        report = {
            "_exempt": (
                "<!-- PRIMARY-SOURCE-EXEMPT: reason=Beta-B benchmark output; "
                "reproducible via `python -m tools.qa_kg.analysis.run_beta_benchmark`. "
                "graph_hash pins DB state pre- and post-synthetic-latency. -->"
            ),
            "phase": "beta_b",
            "schema_version": 1,
            "run_ts": _now(),
            "graph_hash_before": gh_before,
            "graph_hash_after": gh_after,
            "graph_hash_stable": graph_hash_stable,
            "queries_fixture": _QUERIES_PATH.name,
            "gold_fixture": _GOLD_PATH.name,
            "blind_fixture": _BLIND_PATH.name,
            "n_queries": len(queries),
            "per_query": per_query,
            "aggregate": aggregate,
            "agent_tasks": agent_task_summary,
            "contradiction_boost_ablation": ablation,
            "latency_real_graph": real_latency,
            "latency_synthetic_5000_nodes": synth_latency,
            "ranked_hits_raw_by_qid": ranked_raw,
        }
        _RESULTS_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=False, default=str) + "\n",
            encoding="utf-8",
        )

        # Sanity
        print(f"graph_hash stable: {graph_hash_stable}")
        print(f"graph_structural_pass_rate: {aggregate['graph_structural_pass_rate']:.3f} "
              f"({aggregate['graph_structural_hits_over_28']}/{aggregate['graph_structural_denominator']})")
        print(f"contradiction_recall_per_pair: {aggregate['contradiction_recall_per_pair']:.3f} "
              f"({aggregate['contradiction_pair_hits_over_8']}/{aggregate['contradiction_pair_denominator']})")
        print(f"authority_presence_at_3: {aggregate['authority_presence_at_3']:.3f} "
              f"({aggregate['authority_hits_over_8']}/{aggregate['authority_denominator']})")
        print(f"lifecycle_ordering_pass: {aggregate['lifecycle_ordering_pass']}")
        print(f"ndcg cross-domain wins: "
              f"{aggregate['ndcg_cross_domain_wins']}/{aggregate['ndcg_cross_domain_denominator']}")
        print(f"factor dominance: "
              f"{aggregate['factor_dominance']['dominated_fraction']:.3f}")
        print(f"T1 QA-MEM/A-RAG pass: "
              f"{t12['T1']['qa_mem_pass']}/{t12['T1']['a_rag_pass']}")
        print(f"T2 QA-MEM/A-RAG pass: "
              f"{t12['T2']['qa_mem_pass']}/{t12['T2']['a_rag_pass']}")
        print(f"T3/T4/T5: {t345['T3']['pass']}/{t345['T4']['pass']}/{t345['T5']['pass']}")
        print(f"real latency p95: {real_latency['p95_ms']:.2f}ms")
        print(f"synthetic 5000-node latency p95: {synth_latency['p95_ms']:.2f}ms "
              f"(disclaimer: ranking-quality not claimed)")
        print(f"results → {_RESULTS_PATH.relative_to(_REPO)}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
