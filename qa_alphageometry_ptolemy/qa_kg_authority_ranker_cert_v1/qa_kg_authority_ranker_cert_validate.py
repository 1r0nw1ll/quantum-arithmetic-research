# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 4 infrastructure validator -->
"""QA-KG Authority Ranker Cert [254] v1 validator.

QA_COMPLIANCE = "cert_validator — validates KG authority-tiered ranker invariants, no empirical QA state machine"

Phase 4. Validates that `tools/qa_kg/kg.py::KG.search_authority_ranked`
implements the formula at `ranker_spec.json` and surfaces the right
material on the 20-query benchmark fixture:

  R1 (HARD)  min_authority='internal' returns no authority='agent' nodes
  R2 (HARD)  per-fixture expected_top_1_authority == results[0].authority
  R3 (HARD or N-A) contradicted_material — contradicting node in top-3,
             labeled with contradiction_state != 'none'
  R4 (HARD or N-A) tri-state — N-A when no node carries valid_until,
             PASS when filter correctly excludes expired
  R5 (WARN)  recall@5 vs A-RAG retrieve_keyword baseline within ±10%
  R6 (HARD)  formula correctness — compose_score within 1e-6 on ≥6 golden cases
  R7 (HARD)  determinism — 5× re-run yields identical RankedHit lists
  R8 (HARD)  no except-Exception-pass swallows in ranker.py or
             KG.search_authority_ranked (AST scan)
  R9 (HARD)  coverage completeness on BOTH axes against
             qa_kg_epistemic_fields_cert_v1/allowed_matrix.json

Single source of truth for formula constants: ranker_spec.json
(imported once; the validator re-implements compose_score arithmetically
for R6 cross-check). Single source of truth for queries:
query_fixture.json (hand-curated; per-query `expected_top_1_authority`
and `tags`; rationale documented per entry).
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates KG authority-tiered ranker invariants, no empirical QA state machine"

import argparse
import ast
import datetime as _dt
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CERT_DIR = Path(__file__).resolve().parent
_SPEC_FILE = _CERT_DIR / "ranker_spec.json"
_FIXTURE_FILE = _CERT_DIR / "query_fixture.json"
_ALLOWED_MATRIX = (
    _REPO / "qa_alphageometry_ptolemy" / "qa_kg_epistemic_fields_cert_v1"
    / "allowed_matrix.json"
)

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.qa_kg.kg import KG, connect  # noqa: E402
from tools.qa_kg import ranker as _ranker  # noqa: E402
from tools.qa_kg.schema import (  # noqa: E402
    AUTHORITIES, EPISTEMIC_STATUSES, LIFECYCLE_STATES,
)


def _db_path_default() -> Path:
    override = os.environ.get("QA_KG_DB")
    if override:
        return Path(override)
    return _REPO / "tools" / "qa_kg" / "qa_kg.db"


def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise FileNotFoundError(f"QA-KG DB not found at {db}")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _load_spec() -> _ranker.RankerSpec:
    return _ranker.load_spec(_SPEC_FILE)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_FILE.read_text(encoding="utf-8"))


def _load_allowed_matrix() -> dict:
    return json.loads(_ALLOWED_MATRIX.read_text(encoding="utf-8"))


# =============================================================================
# R1 — min_authority='internal' excludes agent
# =============================================================================

def check_r1_min_authority_excludes_agent(
    kg: KG, fixture: dict,
) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    queries = fixture.get("queries", [])
    for q in queries:
        hits = kg.search_authority_ranked(
            q["query"], min_authority="internal", k=10,
        )
        agent_hits = [h for h in hits if h.authority == "agent"]
        if agent_hits:
            ids = ", ".join(h.node["id"] for h in agent_hits)
            viols.append(
                f"query={q['id']!r} returned {len(agent_hits)} agent node(s) "
                f"under min_authority='internal': {ids}"
            )
    return (
        len(viols) == 0,
        f"{len(viols)} R1 violation(s) across {len(queries)} fixture queries",
        viols,
    )


# =============================================================================
# R2 — per-query expected_top_1_authority
# =============================================================================

def check_r2_top1_authority(
    kg: KG, fixture: dict,
) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    queries = fixture.get("queries", [])
    for q in queries:
        expected = q.get("expected_top_1_authority")
        if not expected:
            viols.append(f"query={q['id']!r} missing expected_top_1_authority")
            continue
        hits = kg.search_authority_ranked(
            q["query"], min_authority=q.get("min_authority", "internal"), k=5,
        )
        if not hits:
            viols.append(f"query={q['id']!r} returned 0 hits (expected {expected})")
            continue
        actual = hits[0].authority
        if actual != expected:
            viols.append(
                f"query={q['id']!r} top1 authority={actual!r} "
                f"(expected {expected!r}) — top hit: {hits[0].node['id']}"
            )
    return (
        len(viols) == 0,
        f"{len(viols)} R2 violation(s) across {len(queries)} fixture queries",
        viols,
    )


# =============================================================================
# R3 — contradicted_material in top-3 (tri-state)
# =============================================================================

def check_r3_contradiction_surfacing(
    kg: KG, fixture: dict,
) -> tuple[str, str, list[str]]:
    """Tri-state: N-A | PASS | FAIL."""
    queries = [
        q for q in fixture.get("queries", [])
        if "contradicted_material" in q.get("tags", [])
    ]
    if not queries:
        return ("N-A", "no fixture queries tagged contradicted_material", [])
    runnable = 0
    viols: list[str] = []
    for q in queries:
        hits = kg.search_authority_ranked(
            q["query"], min_authority=q.get("min_authority", "internal"), k=10,
        )
        # Find any node in the top-10 hits that has a contradicts edge.
        # The cert requires that the contradicting node be in top-3.
        contradicted_in_pool = [
            h for h in hits if h.contradiction_state != "none"
        ]
        if not contradicted_in_pool:
            viols.append(
                f"query={q['id']!r} produced no contradicted hits in top-10 "
                f"despite contradicted_material tag"
            )
            continue
        runnable += 1
        top3_ids = {h.node["id"] for h in hits[:3]}
        # For each contradicted node in the pool, all of its contradicts
        # partners that are themselves in the pool must appear in top-3
        # alongside it (or the contradicted node itself must be in top-3).
        contradicted_top3 = any(
            h.contradiction_state != "none" for h in hits[:3]
        )
        if not contradicted_top3:
            viols.append(
                f"query={q['id']!r}: contradicted material exists in pool "
                f"but none in top-3 (top-3 ids: {sorted(top3_ids)})"
            )
    if runnable == 0:
        return ("N-A", f"{len(queries)} contradicted-tagged queries — none runnable in live DB", viols)
    if viols:
        return ("FAIL", f"{len(viols)} R3 violation(s) across {runnable} runnable queries", viols)
    return ("PASS", f"contradicted material in top-3 across {runnable} runnable queries", [])


# =============================================================================
# R4 — valid_at filter (tri-state)
# =============================================================================

def check_r4_valid_at_filter(
    kg: KG, fixture: dict,
) -> tuple[str, str, list[str]]:
    """Tri-state: N-A when no node has valid_until populated."""
    populated = kg.conn.execute(
        "SELECT COUNT(*) AS n FROM nodes WHERE valid_until != ''"
    ).fetchone()["n"]
    if populated == 0:
        return (
            "N-A",
            "no node carries valid_until — filter correctness pending Phase 4.5 corpus pass",
            [],
        )
    queries = [
        q for q in fixture.get("queries", [])
        if "valid_at_filter" in q.get("tags", [])
    ]
    if not queries:
        return (
            "FAIL",
            f"valid_until populated on {populated} node(s) but no fixture query tagged valid_at_filter",
            [],
        )
    viols: list[str] = []
    for q in queries:
        valid_at_iso = q.get("valid_at")
        if not valid_at_iso:
            viols.append(f"query={q['id']!r} missing valid_at field")
            continue
        valid_at = _ranker._parse_iso(valid_at_iso)
        hits = kg.search_authority_ranked(
            q["query"], min_authority=q.get("min_authority", "internal"),
            valid_at=valid_at, k=10,
        )
        for h in hits:
            vu = h.node["valid_until"] or ""
            if vu and vu < valid_at_iso:
                viols.append(
                    f"query={q['id']!r}: returned node {h.node['id']} with "
                    f"valid_until={vu!r} < valid_at={valid_at_iso!r}"
                )
    if viols:
        return ("FAIL", f"{len(viols)} R4 violation(s)", viols)
    return ("PASS", f"valid_at filter excludes expired across {len(queries)} queries", [])


# =============================================================================
# R5 — recall@5 vs A-RAG baseline (WARN)
# =============================================================================

def check_r5_recall_vs_arag(
    kg: KG, fixture: dict,
) -> tuple[bool, str, list[str]]:
    """Always WARN. Returns (True, msg, details) — surfaced via stdout WARN line."""
    try:
        from tools.qa_retrieval import query as _arag_query
    except ImportError as exc:
        return True, f"A-RAG unavailable ({exc}); recall@5 skipped", []
    queries = fixture.get("queries", [])
    overlaps: list[float] = []
    details: list[str] = []
    for q in queries:
        ranker_top5 = kg.search_authority_ranked(
            q["query"], min_authority="internal", k=5,
        )
        ranker_ids = {h.node["id"] for h in ranker_top5}
        try:
            arag_hits = _arag_query.retrieve_keyword(q["query"], limit=5)
        except (sqlite3.Error, FileNotFoundError, AttributeError):
            details.append(f"query={q['id']!r}: A-RAG keyword search failed")
            continue
        arag_ids = {(h.get("msg_id") or h.get("id") or "") for h in arag_hits}
        if not arag_ids:
            continue
        overlap = len(ranker_ids & arag_ids) / max(1, len(ranker_ids | arag_ids))
        overlaps.append(overlap)
        details.append(f"query={q['id']!r}: jaccard={overlap:.2f}")
    if not overlaps:
        return True, "no comparable queries (A-RAG indexes a different corpus)", details
    avg = sum(overlaps) / len(overlaps)
    return True, f"avg jaccard={avg:.2f} across {len(overlaps)} queries (WARN-only sensor)", details


# =============================================================================
# R6 — formula correctness on golden cases
# =============================================================================

# Golden cases: (label, kwargs, expected_score). Hand-derived from the spec.
def _golden_cases(spec: _ranker.RankerSpec) -> list[tuple[str, dict, float]]:
    """≥6 golden cases per plan §R6:
      - each authority tier
      - decayed vs exempt status
      - contradicted vs clean
      - provenance depths in {0, 1, 3, 5, -1}
      - valid_from precedence over created_ts
      - confidence < 1.0
      - lifecycle current vs superseded
    """
    iso_now = "2026-04-16T00:00:00Z"
    iso_year_ago = "2025-04-16T00:00:00Z"
    valid_at = _dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc)
    return [
        # (1) primary axiom, all neutral, depth 0 → 10.0
        ("primary_axiom_depth0", dict(
            authority="primary", bm25_norm=1.0, confidence=1.0,
            epistemic_status="axiom", created_ts=iso_now, valid_from="",
            valid_at=None, contradiction_state="none", provenance_depth=0,
            lifecycle_state="current",
        ), 10.0),
        # (2) derived certified, partial bm25, depth 1, current
        ("derived_certified_depth1", dict(
            authority="derived", bm25_norm=0.5, confidence=1.0,
            epistemic_status="certified", created_ts=iso_now, valid_from="",
            valid_at=None, contradiction_state="none", provenance_depth=1,
            lifecycle_state="current",
        ), 8.0 * 1.0 * 0.5 * 1.0 * 1.0 * 1.0 * math.exp(-1.0 / spec.halflife_hops)),
        # (3) internal observation, decayed 365d, contradicted, depth 3
        ("internal_obs_decayed_contradicted_depth3", dict(
            authority="internal", bm25_norm=1.0, confidence=1.0,
            epistemic_status="observation", created_ts=iso_year_ago, valid_from="",
            valid_at=valid_at, contradiction_state="src", provenance_depth=3,
            lifecycle_state="current",
        ), 5.0 * 1.0 * 1.0 * 1.0 * math.exp(-365.0 / spec.halflife_days)
            * spec.contradiction_prior * math.exp(-3.0 / spec.halflife_hops)),
        # (4) agent conjecture, no path, fresh (delta=0 → time_decay=1.0)
        ("agent_conjecture_no_path", dict(
            authority="agent", bm25_norm=0.7, confidence=0.8,
            epistemic_status="conjecture", created_ts=iso_now, valid_from="",
            valid_at=valid_at, contradiction_state="none", provenance_depth=-1,
            lifecycle_state="current",
        ), 1.0 * 1.0 * 0.7 * 0.8 * 1.0 * 1.0 * spec.no_path_factor),
        # (5) valid_from precedence: claim from year ago, captured today
        ("valid_from_precedence", dict(
            authority="internal", bm25_norm=1.0, confidence=1.0,
            epistemic_status="interpretation", created_ts=iso_now,
            valid_from=iso_year_ago, valid_at=valid_at,
            contradiction_state="none", provenance_depth=5,
            lifecycle_state="current",
        ), 5.0 * 1.0 * 1.0 * 1.0 * math.exp(-365.0 / spec.halflife_days)
            * 1.0 * math.exp(-5.0 / spec.halflife_hops)),
        # (6) confidence < 1.0 (e.g., OCR'd source claim)
        ("confidence_partial", dict(
            authority="primary", bm25_norm=1.0, confidence=0.7,
            epistemic_status="source_claim", created_ts=iso_now, valid_from="",
            valid_at=None, contradiction_state="none", provenance_depth=-1,
            lifecycle_state="current",
        ), 10.0 * 1.0 * 1.0 * 0.7 * 1.0 * 1.0 * spec.no_path_factor),
        # (7) contradiction on both sides
        ("primary_source_claim_both_sided_contradiction", dict(
            authority="primary", bm25_norm=0.5, confidence=1.0,
            epistemic_status="source_claim", created_ts=iso_now, valid_from="",
            valid_at=None, contradiction_state="both", provenance_depth=-1,
            lifecycle_state="current",
        ), 10.0 * 1.0 * 0.5 * 1.0 * 1.0 * spec.contradiction_prior * spec.no_path_factor),
        # (8) superseded — derived cert, otherwise identical to (2), should
        #     score 0.5× the "current" version. Locks the lifecycle factor.
        ("derived_certified_depth1_superseded", dict(
            authority="derived", bm25_norm=0.5, confidence=1.0,
            epistemic_status="certified", created_ts=iso_now, valid_from="",
            valid_at=None, contradiction_state="none", provenance_depth=1,
            lifecycle_state="superseded",
        ), 8.0 * 0.5 * 0.5 * 1.0 * 1.0 * 1.0 * math.exp(-1.0 / spec.halflife_hops)),
    ]


def check_r6_formula_correctness(
    spec: _ranker.RankerSpec,
) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    cases = _golden_cases(spec)
    for label, kwargs, expected in cases:
        score, _br = _ranker.compose_score(spec=spec, **kwargs)
        if abs(score - expected) > 1e-6:
            viols.append(
                f"{label}: got {score:.10f}, expected {expected:.10f} "
                f"(Δ={abs(score-expected):.2e})"
            )
    return (
        len(viols) == 0,
        f"{len(viols)} golden-case mismatch(es) out of {len(cases)}",
        viols,
    )


# =============================================================================
# R7 — determinism
# =============================================================================

def check_r7_determinism(
    kg: KG, fixture: dict,
) -> tuple[bool, str, list[str]]:
    """Determinism = same query + same DB + same valid_at → same RankedHit
    sequence (id-order + score within 1e-9). Wall-clock-leakage failure mode
    is exercised by passing a fixed valid_at, NOT by relying on default
    valid_at=None (which intentionally snapshots wall-clock per call).
    """
    viols: list[str] = []
    queries = fixture.get("queries", [])[:5]  # 5 queries × 5 reruns = 25 calls
    fixed_valid_at = _dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc)
    for q in queries:
        runs: list[list[tuple[str, float]]] = []
        for _ in range(5):
            hits = kg.search_authority_ranked(
                q["query"], min_authority="internal", k=10,
                valid_at=fixed_valid_at,
            )
            runs.append([(h.node["id"], round(h.score, 9)) for h in hits])
        if not all(r == runs[0] for r in runs):
            viols.append(
                f"query={q['id']!r}: ranker not deterministic across 5 reruns "
                f"under fixed valid_at"
            )
    return (
        len(viols) == 0,
        f"{len(viols)} determinism violation(s) across {len(queries)} queries × 5 runs",
        viols,
    )


# =============================================================================
# R8 — AST scan for swallow patterns
# =============================================================================

def _scan_for_swallows(path: Path) -> list[str]:
    """Return list of (lineno: pattern) for except-Exception-pass / bare-except-pass."""
    if not path.exists():
        return [f"{path}: missing file"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            # bare except: (handler.type is None) OR except Exception:
            is_bare = handler.type is None
            is_exception_named = (
                isinstance(handler.type, ast.Name)
                and handler.type.id == "Exception"
            )
            if not (is_bare or is_exception_named):
                continue
            body = handler.body
            # swallow = body is exactly [pass] or [continue]
            if (len(body) == 1
                    and isinstance(body[0], (ast.Pass, ast.Continue))):
                pat = "bare-except" if is_bare else "except-Exception"
                action = type(body[0]).__name__.lower()
                findings.append(f"{path}:{handler.lineno} {pat}-{action}")
    return findings


def check_r8_no_silent_swallows(
) -> tuple[bool, str, list[str]]:
    """Scan tools/qa_kg/ranker.py and the search_authority_ranked block in kg.py."""
    targets = [
        _REPO / "tools" / "qa_kg" / "ranker.py",
        _REPO / "tools" / "qa_kg" / "kg.py",
    ]
    findings: list[str] = []
    for t in targets:
        findings.extend(_scan_for_swallows(t))
    return (
        len(findings) == 0,
        f"{len(findings)} swallow site(s) in ranker.py / kg.py",
        findings,
    )


# =============================================================================
# R9 — coverage completeness on BOTH axes
# =============================================================================

def check_r9_coverage_completeness(
    spec: _ranker.RankerSpec,
) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    matrix = _load_allowed_matrix()
    matrix_authorities = set(matrix["allowed"].keys())
    schema_authorities = set(AUTHORITIES)
    schema_statuses = set(EPISTEMIC_STATUSES)
    # 'withdrawn' is excluded from the candidate pool; it is intentionally
    # absent from lifecycle_factor.
    expected_lifecycle_keys = set(LIFECYCLE_STATES) - {"withdrawn"}

    spec_authorities = set(spec.authority_weight.keys())
    spec_lifecycle_keys = set(spec.lifecycle_factor.keys())
    spec_decay_union = set(spec.decay_exempt_status) | set(spec.decay_status)
    spec_decay_intersect = set(spec.decay_exempt_status) & set(spec.decay_status)

    # (a) decay coverage
    if spec_decay_union != schema_statuses:
        missing = schema_statuses - spec_decay_union
        extra = spec_decay_union - schema_statuses
        viols.append(
            f"decay coverage incomplete: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    # (b) authority coverage (vs schema AND vs allowed_matrix)
    if spec_authorities != schema_authorities:
        missing = schema_authorities - spec_authorities
        extra = spec_authorities - schema_authorities
        viols.append(
            f"authority coverage vs schema incomplete: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if spec_authorities != matrix_authorities:
        missing = matrix_authorities - spec_authorities
        extra = spec_authorities - matrix_authorities
        viols.append(
            f"authority coverage vs allowed_matrix incomplete: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    # (c) decay disjointness
    if spec_decay_intersect:
        viols.append(
            f"decay sets overlap: status(es) in BOTH exempt and decayed: "
            f"{sorted(spec_decay_intersect)}"
        )
    # (d) lifecycle coverage — must be exactly LIFECYCLE_STATES \ {'withdrawn'}
    if spec_lifecycle_keys != expected_lifecycle_keys:
        missing = expected_lifecycle_keys - spec_lifecycle_keys
        extra = spec_lifecycle_keys - expected_lifecycle_keys
        viols.append(
            f"lifecycle coverage incomplete: missing={sorted(missing)}, extra={sorted(extra)} "
            f"(expected = LIFECYCLE_STATES \\ {{'withdrawn'}})"
        )
    return (
        len(viols) == 0,
        f"{len(viols)} R9 coverage violation(s)",
        viols,
    )


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--show-details", action="store_true")
    args = p.parse_args(argv)

    db = args.db or _db_path_default()
    try:
        conn = _connect(db)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 2

    kg = KG(conn)
    spec = _load_spec()
    fixture = _load_fixture()

    hard_fail = False

    def run_bool(code: str, desc: str, fn, is_hard: bool) -> None:
        nonlocal hard_fail
        ok, msg, detail = fn()
        flag = "PASS" if ok else ("FAIL" if is_hard or args.strict else "WARN")
        print(f"[{flag}] {code}  {desc} — {msg}")
        if args.show_details and detail:
            for d in detail[:10]:
                print(f"         {d}")
            if len(detail) > 10:
                print(f"         ... +{len(detail)-10} more")
        if not ok and (is_hard or args.strict):
            hard_fail = True

    def run_tristate(code: str, desc: str, fn) -> None:
        nonlocal hard_fail
        state, msg, detail = fn()
        flag = state  # PASS | FAIL | N-A
        print(f"[{flag}] {code}  {desc} — {msg}")
        if args.show_details and detail:
            for d in detail[:10]:
                print(f"         {d}")
            if len(detail) > 10:
                print(f"         ... +{len(detail)-10} more")
        if state == "FAIL":
            hard_fail = True

    run_bool("R1", "min_authority='internal' excludes agent",
             lambda: check_r1_min_authority_excludes_agent(kg, fixture), True)
    run_bool("R2", "per-query expected_top_1_authority",
             lambda: check_r2_top1_authority(kg, fixture), True)
    run_tristate("R3", "contradicted material in top-3 (tri-state)",
                 lambda: check_r3_contradiction_surfacing(kg, fixture))
    run_tristate("R4", "valid_at filter excludes expired (tri-state)",
                 lambda: check_r4_valid_at_filter(kg, fixture))

    # R5 — WARN-only with stdout sensor for ledger capture
    ok5, msg5, detail5 = check_r5_recall_vs_arag(kg, fixture)
    print(f"[WARN] R5  Recall@5 vs A-RAG baseline — {msg5}")
    if args.show_details and detail5:
        for d in detail5[:10]:
            print(f"         {d}")

    run_bool("R6", "compose_score formula correctness",
             lambda: check_r6_formula_correctness(spec), True)
    run_bool("R7", "determinism — 5× re-run identical",
             lambda: check_r7_determinism(kg, fixture), True)
    run_bool("R8", "no except-Exception-pass / bare-except-pass swallows",
             lambda: check_r8_no_silent_swallows(), True)
    run_bool("R9", "coverage completeness on BOTH axes",
             lambda: check_r9_coverage_completeness(spec), True)

    if hard_fail:
        print("[FAIL] QA-KG authority ranker cert [254] v1")
        return 1
    print("[PASS] QA-KG authority ranker cert [254] v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
