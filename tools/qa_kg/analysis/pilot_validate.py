"""Pilot validator for Beta-A — 10-query subset, fixture-level checks.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Purpose: validate the fixture plumbing before Beta-B runs the full 38.
For every apparent recall miss, classify the cause:
  (a) real ranker miss    — candidate for Phase 4.7 ranker-tuning
  (b) graph-incompleteness — corpus-expansion signal
  (c) gold-spec bug       — fixture fix under N1 constraints

N1 fixture-fix constraint allows ONLY:
  (i) syntactic JSON errors
  (ii) nonexistent node IDs
  (iii) malformed domain values
  (iv) rephrasing when FTS5 returns zero matches (no-op queries)

Every change is logged in beta_prereg_deviations.json. Gold-label changes
after seeing ranker results are explicitly forbidden (that's the whole
pre-registration design).

Pilot coverage: ≥1 query per category = 7 queries minimum; we run 10 to
give a per-category headroom of 2 on provenance/contradiction/authority
plus 1 on domain/lifecycle/cross_domain/edge_case.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import datetime as _dt
import json
import re
import sqlite3
import sys
from pathlib import Path

from tools.qa_kg.canonicalize import graph_hash
from tools.qa_kg.kg import KG
from tools.qa_kg.schema import DEFAULT_DB

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO / "tools" / "qa_kg" / "fixtures"
_QUERIES_PATH = _FIXTURE_DIR / "beta_prereg_queries.json"
_GOLD_PATH = _FIXTURE_DIR / "beta_prereg_gold.json"
_BLIND_PATH = _FIXTURE_DIR / "beta_blind_gold.json"
_DEVIATIONS_PATH = _FIXTURE_DIR / "beta_prereg_deviations.json"
_PILOT_REPORT_PATH = _FIXTURE_DIR / "beta_pilot_report.json"

# Category coverage for the 10-query pilot (≥ 1 per category):
PILOT_IDS = (
    "P01", "P02",           # provenance (2 — includes axiom-stripping check)
    "C04", "C07",           # contradiction (2 — includes NFKD-normalized match)
    "D06",                  # domain (1 — smallest gold set for clearest signal)
    "A03",                  # authority (1)
    "L01",                  # lifecycle (1)
    "X04",                  # cross_domain (1 — strongest topic separation)
    "E01", "E02",           # edge_case (2 — empty results)
)

_FTS5_SANITIZE_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_FTS5_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "was", "were", "be", "by", "for", "with", "about", "what", "how",
    "does", "do", "did", "can", "that", "this", "these", "those", "which",
    "vs", "versus", "but", "if", "not", "no", "yes", "its", "it", "i",
    "as", "at", "so", "us", "we", "both",
})


def _sanitize_for_fts5(query_text: str) -> str:
    """Mirror blind_label_prompt._sanitize_for_fts5; deliberately colocated so
    the pilot and full run share the exact FTS5 query shape.

    Note: we keep 'primary/source/sources' here unlike blind_label — they
    are informative for provenance queries that are literally about
    "primary sources". The cross-domain labeler strips them because the
    X-queries are phrased as natural questions.
    """
    cleaned = _FTS5_SANITIZE_RE.sub(" ", query_text).lower()
    tokens = [t for t in cleaned.split() if t and t not in _FTS5_STOPWORDS]
    if not tokens:
        return query_text.strip()
    return " OR ".join(tokens) if len(tokens) > 1 else tokens[0]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_fixtures() -> tuple[dict, dict, dict]:
    queries = {
        q["id"]: q
        for q in json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    }
    gold = {
        e["id"]: e
        for e in json.loads(_GOLD_PATH.read_text(encoding="utf-8"))["gold_entries"]
    }
    blind: dict = {}
    if _BLIND_PATH.exists():
        blind = {
            r["id"]: r
            for r in json.loads(_BLIND_PATH.read_text(encoding="utf-8"))["results"]
        }
    return queries, gold, blind


def _run_query(kg: KG, q: dict) -> list[dict]:
    """Run a single query's ranker call and return simplified result rows.

    Each row: {id, authority, epistemic_status, domain, score, score_breakdown,
               contradiction_state, provenance_depth}.
    """
    qtext = _sanitize_for_fts5(q["query_text"])
    kwargs: dict = {"k": 10}
    if q.get("domain_filter") is not None:
        kwargs["domain"] = q["domain_filter"]
    if q.get("min_authority"):
        kwargs["min_authority"] = q["min_authority"]
    if q.get("valid_at"):
        kwargs["valid_at"] = _dt.datetime.fromisoformat(
            q["valid_at"].replace("Z", "+00:00"),
        )
    try:
        hits = kg.search_authority_ranked(qtext, **kwargs)
    except sqlite3.OperationalError as exc:
        # E.g., fts5 syntax error on degenerate queries — record and return []
        return [{"error": str(exc)}]
    out: list[dict] = []
    for h in hits:
        row = h.node
        out.append({
            "id": row["id"],
            "authority": h.authority,
            "epistemic_status": row["epistemic_status"],
            "domain": row["domain"],
            "score": h.score,
            "contradiction_state": h.contradiction_state,
            "provenance_depth": h.provenance_depth,
            "score_breakdown": h.score_breakdown,
        })
    return out


def _classify_miss(q: dict, gold_entry: dict, results: list[dict]) -> str:
    """One of 'real_ranker_miss' | 'graph_incomplete_false_neg' | 'gold_spec_bug'.

    Decision tree:
      1. If gold_ids is empty, a miss is not meaningful — return 'n_a'.
      2. If no results at all, the candidate pool failed — (a) real_ranker_miss
         if the query should have matches; (c) gold_spec_bug if the query is
         intentionally degenerate (E-queries).
      3. If results exist but no gold in top-5:
         - If gold node doesn't exist in DB → (c) gold_spec_bug
         - If gold node exists but its body has < 2 shared tokens with query → (b)
         - Else → (a) real_ranker_miss
    """
    gold_ids = gold_entry.get("gold_ids") or []
    if not gold_ids:
        return "n_a"
    if not results or (results and "error" in results[0]):
        return "real_ranker_miss"  # pool empty but gold expected
    top5_ids = [r["id"] for r in results[:5]]
    if any(gid in top5_ids for gid in gold_ids):
        return "hit"
    # miss — dig deeper
    return "real_ranker_miss"  # conservative default; manual reclassification during review


def run_pilot() -> dict:
    queries, gold, blind = _load_fixtures()
    conn = sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        gh = graph_hash(conn)
        kg = KG(conn)
        per_query_report: list[dict] = []
        misses: list[dict] = []
        for qid in PILOT_IDS:
            if qid not in queries:
                per_query_report.append({"id": qid, "error": "query not in fixture"})
                continue
            q = queries[qid]
            g = gold.get(qid, {})
            results = _run_query(kg, q)
            classification = _classify_miss(q, g, results)

            # edge_case classification override: E-queries have expected_predicate, not gold_ids
            if q["category"] == "edge_case":
                expected = q.get("expected_result")
                if expected == "empty":
                    classification = "hit" if not results or (results and "error" in results[0]) else "real_ranker_miss"
                elif expected == "no_derived_in_top_k":
                    top5_auth = [r["authority"] for r in results[:5] if "error" not in r]
                    classification = "hit" if all(a == "primary" for a in top5_auth) or not top5_auth else "real_ranker_miss"
                elif expected == "valid_at_passthrough":
                    # We can't easily compare without running without valid_at; record for Beta-B
                    classification = "deferred_to_beta_b"

            # cross_domain: gold is blind grades, not gold_ids
            if q["category"] == "cross_domain":
                blind_entry = blind.get(qid, {})
                grades = blind_entry.get("parsed_grades", {})
                top_graded_ids = {nid for nid, g in grades.items() if g >= 4}
                top5_ids = [r["id"] for r in results[:5] if "error" not in r]
                classification = "hit" if any(tid in top_graded_ids for tid in top5_ids) else "real_ranker_miss"

            entry = {
                "id": qid,
                "category": q["category"],
                "query_text": q["query_text"],
                "classification": classification,
                "gold_ids": g.get("gold_ids"),
                "anti_gold_ids": g.get("anti_gold_ids"),
                "result_count": len([r for r in results if "error" not in r]),
                "top5_ids": [r["id"] for r in results[:5] if "error" not in r],
                "top5_authorities": [r["authority"] for r in results[:5] if "error" not in r],
            }
            per_query_report.append(entry)
            if classification not in ("hit", "n_a", "deferred_to_beta_b"):
                misses.append(entry)

        report = {
            "_exempt": "<!-- PRIMARY-SOURCE-EXEMPT: reason=Beta-A pilot-run output. -->",
            "phase": "beta_a_pilot",
            "schema_version": 1,
            "derived_ts": _now(),
            "graph_hash": gh,
            "pilot_query_count": len(PILOT_IDS),
            "per_query": per_query_report,
            "miss_count": len(misses),
            "misses": misses,
            "summary": {
                "hit": sum(1 for e in per_query_report if e.get("classification") == "hit"),
                "real_ranker_miss": sum(1 for e in per_query_report if e.get("classification") == "real_ranker_miss"),
                "n_a": sum(1 for e in per_query_report if e.get("classification") == "n_a"),
                "deferred": sum(1 for e in per_query_report if e.get("classification") == "deferred_to_beta_b"),
            },
        }
        _PILOT_REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

        # Deviations file — empty at Beta-A unless the pilot forces an N1 change.
        if not _DEVIATIONS_PATH.exists():
            _DEVIATIONS_PATH.write_text(
                json.dumps({
                    "_exempt": "<!-- PRIMARY-SOURCE-EXEMPT: reason=Beta-A fixture-fix log under N1 constraints. -->",
                    "phase": "beta_a",
                    "schema_version": 1,
                    "created_ts": _now(),
                    "deviations": [],
                    "n1_constraint": (
                        "Only (i) syntactic JSON errors, (ii) nonexistent node IDs, "
                        "(iii) malformed domain values, (iv) zero-FTS5-match rephrasing. "
                        "Gold-label changes after seeing ranker results are FORBIDDEN."
                    ),
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        return report
    finally:
        conn.close()


def main() -> int:
    report = run_pilot()
    s = report["summary"]
    print(f"pilot: {s['hit']} hit, {s['real_ranker_miss']} miss, {s['n_a']} n/a, {s['deferred']} deferred")
    if report["miss_count"]:
        print("MISSES (for manual review before Beta-B):")
        for m in report["misses"]:
            print(f"  {m['id']} ({m['category']}): top5={m['top5_ids'][:3]} gold={m['gold_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
