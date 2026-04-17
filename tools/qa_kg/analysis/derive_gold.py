"""Mechanical gold-label derivation for Beta-A pre-registered queries.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Reads tools/qa_kg/fixtures/beta_prereg_queries.json, derives per-query
gold from the live qa_kg.db (read-only), writes:

  tools/qa_kg/fixtures/beta_prereg_gold.json
    Per-query: gold_ids, gold_source, derivation_method,
    derivation_sql_or_edges, timestamp, graph_hash.

  tools/qa_kg/fixtures/beta_prereg_contradiction_audit.json
    Lexical pre-flight: per C-query, per-endpoint shared-token count
    with query_text minus stopwords. Flags endpoints with < 2 shared.

Gold-source semantics:
  graph_structural  — derived by this script from graph edges/columns
  curator_specified — expected_primary_id from the fixture, validated here
  llm_blind         — deferred to blind_label_prompt.py (gold_ids left null)
  mechanical        — empty/filter-passthrough/min_authority-exclusion

Axioms are filtered from provenance gold — the ranker is BM25-based and
cannot surface axioms via query text lexically. Axioms are tested only
via T3 (why() traversal) in agent_tasks.py.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import datetime as _dt
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

from tools.qa_kg.canonicalize import graph_hash
from tools.qa_kg.schema import DEFAULT_DB

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO / "tools" / "qa_kg" / "fixtures"
_QUERIES_PATH = _FIXTURE_DIR / "beta_prereg_queries.json"
_GOLD_PATH = _FIXTURE_DIR / "beta_prereg_gold.json"
_AUDIT_PATH = _FIXTURE_DIR / "beta_prereg_contradiction_audit.json"

_PROVENANCE_EDGES = ("derived-from", "validates", "extends", "instantiates")

# Stopwords include generic query connectives AND domain-generic tokens
# that carry no disambiguating signal. Deliberately small — we want the
# pre-flight to catch truly lexically-isolated endpoints, not nitpick
# common content words.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "was", "were", "be", "by", "for", "with", "about", "what", "how",
    "does", "do", "did", "can", "that", "this", "these", "those", "which",
    "vs", "versus", "but", "if", "not", "no", "yes", "its", "it", "i",
    "cert", "claim", "primary", "source", "sources", "ground", "grounds",
    "grounding", "derived", "as", "at", "so", "us", "we",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """NFKD-normalized + lowercase + alphanumeric-split + stopword-strip.

    NFKD decomposes superscripts (a² → a2) so the pre-flight aligns with
    what FTS5 would roughly see after unicode61 + remove_diacritics. Single
    characters are kept — variable names like C/b/h are content in math
    bodies — but common short English words are in the stopword list.
    """
    if not text:
        return set()
    normalized = unicodedata.normalize("NFKD", text).lower()
    return {t for t in _TOKEN_RE.findall(normalized) if t not in _STOPWORDS}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Per-category derivation
# ---------------------------------------------------------------------------

def _derive_provenance(
    conn: sqlite3.Connection, target_id: str, max_depth: int = 3,
) -> dict:
    """BFS via provenance edges; strip Axioms from output.

    Returns {"gold_ids": [...], "axioms_stripped": [...], "edges_traversed": [...]}.
    """
    edge_placeholders = ",".join("?" * len(_PROVENANCE_EDGES))
    frontier = {target_id}
    visited = {target_id}
    all_reached: list[str] = []
    traversed: list[tuple[str, str, str]] = []
    for depth in range(max_depth):
        if not frontier:
            break
        fronter_list = list(frontier)
        placeholders = ",".join("?" * len(fronter_list))
        rows = conn.execute(
            f"SELECT src_id, dst_id, edge_type FROM edges "
            f"WHERE src_id IN ({placeholders}) "
            f"  AND edge_type IN ({edge_placeholders}) "
            f"  AND method != 'keyword'",
            (*fronter_list, *_PROVENANCE_EDGES),
        ).fetchall()
        next_frontier: set[str] = set()
        for r in rows:
            traversed.append((r["src_id"], r["dst_id"], r["edge_type"]))
            if r["dst_id"] not in visited:
                visited.add(r["dst_id"])
                all_reached.append(r["dst_id"])
                next_frontier.add(r["dst_id"])
        frontier = next_frontier

    # Filter Axioms: ranker is search-based, cannot surface via body text.
    axioms_stripped: list[str] = []
    gold: list[str] = []
    if all_reached:
        placeholders = ",".join("?" * len(all_reached))
        rows = conn.execute(
            f"SELECT id, node_type FROM nodes WHERE id IN ({placeholders})",
            all_reached,
        ).fetchall()
        node_type_map = {r["id"]: r["node_type"] for r in rows}
        for nid in all_reached:
            if node_type_map.get(nid) == "Axiom":
                axioms_stripped.append(nid)
            else:
                gold.append(nid)
    return {
        "gold_ids": sorted(gold),
        "axioms_stripped": sorted(axioms_stripped),
        "edges_traversed": traversed,
    }


def _derive_contradiction(
    conn: sqlite3.Connection, target_edge: list[str], query_text: str,
) -> dict:
    """Return {gold_ids, lexical_audit, reason}."""
    assert len(target_edge) == 2
    src_id, dst_id = target_edge
    row = conn.execute(
        "SELECT provenance FROM edges "
        "WHERE src_id=? AND dst_id=? AND edge_type='contradicts'",
        (src_id, dst_id),
    ).fetchone()
    if row is None:
        return {
            "gold_ids": [],
            "lexical_audit": {"error": f"contradicts edge not found: {src_id}→{dst_id}"},
            "reason": None,
        }
    provenance = json.loads(row["provenance"])
    reason = provenance.get("reason")

    q_tokens = _tokenize(query_text)
    node_rows = conn.execute(
        "SELECT id, title, body FROM nodes WHERE id IN (?, ?)", (src_id, dst_id),
    ).fetchall()
    per_endpoint: dict[str, dict] = {}
    for nr in node_rows:
        e_tokens = _tokenize((nr["title"] or "") + " " + (nr["body"] or ""))
        shared = sorted(q_tokens & e_tokens)
        per_endpoint[nr["id"]] = {
            "shared_tokens": shared,
            "shared_count": len(shared),
            "pass": len(shared) >= 2,
        }
    return {
        "gold_ids": sorted([src_id, dst_id]),
        "lexical_audit": {
            "query_tokens": sorted(q_tokens),
            "per_endpoint": per_endpoint,
            "both_pass": all(v["pass"] for v in per_endpoint.values()),
        },
        "reason": reason,
    }


def _derive_domain(conn: sqlite3.Connection, domain: str) -> dict:
    rows = conn.execute(
        "SELECT id FROM nodes "
        "WHERE domain=? AND epistemic_status='source_claim' "
        "ORDER BY id",
        (domain,),
    ).fetchall()
    return {
        "gold_ids": [r["id"] for r in rows],
        "count": len(rows),
        "derivation_sql": (
            "SELECT id FROM nodes WHERE domain=? AND epistemic_status='source_claim'"
        ),
    }


def _derive_authority(
    conn: sqlite3.Connection, expected_primary_id: str,
) -> dict:
    """Validate curator-specified primary SourceClaim exists and is primary."""
    row = conn.execute(
        "SELECT id, authority, epistemic_status, domain FROM nodes WHERE id=?",
        (expected_primary_id,),
    ).fetchone()
    if row is None:
        return {
            "gold_ids": [],
            "validation": {"error": f"expected_primary_id {expected_primary_id!r} not found"},
        }
    validation = {
        "id": row["id"],
        "authority": row["authority"],
        "epistemic_status": row["epistemic_status"],
        "domain": row["domain"],
        "is_primary_source_claim": (
            row["authority"] == "primary"
            and row["epistemic_status"] == "source_claim"
        ),
    }
    return {
        "gold_ids": [expected_primary_id] if validation["is_primary_source_claim"] else [],
        "validation": validation,
    }


def _derive_lifecycle(
    conn: sqlite3.Connection, target_topic: str,
) -> dict:
    """target_topic is a substring matched against id (cert:fs:<topic>_vN).

    Returns gold_ids=[current-lifecycle cert], anti_gold=[superseded certs].
    """
    rows = conn.execute(
        "SELECT id, lifecycle_state FROM nodes "
        "WHERE id LIKE ? AND node_type='Cert' "
        "ORDER BY id",
        (f"cert:fs:{target_topic}_v%",),
    ).fetchall()
    current = [r["id"] for r in rows if r["lifecycle_state"] == "current"]
    superseded = [r["id"] for r in rows if r["lifecycle_state"] == "superseded"]
    return {
        "gold_ids": current,
        "anti_gold_ids": superseded,
        "all_versions": [r["id"] for r in rows],
        "gate": "gold[0] rank in top-5 < min(anti_gold ranks)",
    }


def _derive_edge_case(query: dict) -> dict:
    expected = query["expected_result"]
    if expected == "empty":
        return {"gold_ids": [], "expected_predicate": "results == []"}
    if expected == "no_derived_in_top_k":
        return {
            "gold_ids": [],
            "expected_predicate": (
                "all(hit.authority == 'primary' for hit in results[:5]) "
                "OR results == []"
            ),
        }
    if expected == "valid_at_passthrough":
        return {
            "gold_ids": [],
            "expected_predicate": (
                "results_with_valid_at == results_without_valid_at "
                "(all valid_until='' → filter passes all)"
            ),
        }
    return {"gold_ids": [], "expected_predicate": f"unknown:{expected!r}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def derive_all(
    db_path: Path = DEFAULT_DB,
    queries_path: Path = _QUERIES_PATH,
    gold_path: Path = _GOLD_PATH,
    audit_path: Path = _AUDIT_PATH,
) -> dict:
    """Run full derivation. Returns summary dict; writes two JSON files."""
    fixture = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = fixture["queries"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        gh = graph_hash(conn)
        gold_entries: list[dict] = []
        contradiction_audit: list[dict] = []
        summary = {"derived": 0, "contradiction_lex_fails": 0, "errors": []}

        for q in queries:
            qid = q["id"]
            category = q["category"]
            entry: dict = {
                "id": qid,
                "category": category,
                "gold_source": q["gold_source"],
                "derived_ts": _now(),
            }

            try:
                if category == "provenance":
                    result = _derive_provenance(conn, q["target_id"])
                    entry.update(result)
                elif category == "contradiction":
                    result = _derive_contradiction(conn, q["target_edge"], q["query_text"])
                    entry.update(result)
                    contradiction_audit.append({
                        "id": qid,
                        "query_text": q["query_text"],
                        "target_edge": q["target_edge"],
                        "lexical_audit": result["lexical_audit"],
                        "reason": result["reason"],
                    })
                    if not result["lexical_audit"].get("both_pass", False):
                        summary["contradiction_lex_fails"] += 1
                elif category == "domain":
                    result = _derive_domain(conn, q["domain_filter"])
                    entry.update(result)
                elif category == "authority":
                    result = _derive_authority(conn, q["expected_primary_id"])
                    entry.update(result)
                elif category == "lifecycle":
                    result = _derive_lifecycle(conn, q["target_topic"])
                    entry.update(result)
                elif category == "cross_domain":
                    entry["gold_ids"] = None
                    entry["note"] = "deferred to blind_label_prompt.py (llm_blind)"
                elif category == "edge_case":
                    result = _derive_edge_case(q)
                    entry.update(result)
                else:
                    raise ValueError(f"unknown category: {category!r}")
                gold_entries.append(entry)
                summary["derived"] += 1
            except Exception as exc:  # noqa: BLE001 — summary-level error catch for batch script
                summary["errors"].append({"id": qid, "error": str(exc)})
                raise

        output = {
            "_exempt": (
                "<!-- PRIMARY-SOURCE-EXEMPT: reason=mechanical gold output "
                "from derive_gold.py against live qa_kg.db; reproducible "
                "via `python -m tools.qa_kg.analysis.derive_gold`; "
                "graph_hash pins the DB state. -->"
            ),
            "phase": fixture["phase"],
            "schema_version": 1,
            "derived_ts": _now(),
            "graph_hash": gh,
            "queries_fixture_name": queries_path.name,
            "gold_entries": gold_entries,
            "summary": summary,
        }
        gold_path.write_text(
            json.dumps(output, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

        audit_output = {
            "_exempt": "<!-- PRIMARY-SOURCE-EXEMPT: reason=Beta-A contradiction lexical pre-flight output. -->",
            "derived_ts": _now(),
            "graph_hash": gh,
            "audit": contradiction_audit,
            "fail_count": summary["contradiction_lex_fails"],
            "total_contradictions": len(contradiction_audit),
        }
        audit_path.write_text(
            json.dumps(audit_output, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        conn.close()


def main() -> int:
    summary = derive_all()
    print(f"derived {summary['derived']} gold entries")
    print(f"contradiction lexical fails: {summary['contradiction_lex_fails']}/8")
    if summary["errors"]:
        print(f"ERRORS: {summary['errors']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
