# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 4 ranker unit tests -->
"""Phase 4 ranker unit tests — formula correctness + spec loading + invariants.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Cert [254] runs against the live qa_kg.db; these tests run against
fresh-built ephemeral DBs and golden-input arithmetic. They isolate
ranker.py from kg.py so a regression in either layer is localized.

Run: python -m tools.qa_kg.tests.test_ranker
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import datetime as _dt
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.qa_kg import ranker as _ranker
from tools.qa_kg.kg import Node, connect
from tools.qa_kg.schema import AUTHORITIES, EPISTEMIC_STATUSES, LIFECYCLE_STATES


# ---------- spec loading + caching ----------

def test_load_spec_returns_rankerspec():
    spec = _ranker.load_spec()
    assert isinstance(spec, _ranker.RankerSpec)
    assert spec.authority_weight["primary"] == 10.0
    assert spec.contradiction_prior == 1.5
    assert spec.no_path_factor == 0.5
    assert spec.halflife_hops == 3.0
    assert spec.halflife_days == 365.0


def test_load_spec_cached():
    a = _ranker.load_spec()
    b = _ranker.load_spec()
    assert a is b, "load_spec should return the cached instance on repeat calls"


def test_reload_spec_resets_cache():
    a = _ranker.load_spec()
    b = _ranker.reload_spec()
    assert a is not b, "reload_spec must drop the cached instance"


# ---------- normalize_bm25 ----------

def test_normalize_bm25_inverts_fts5_sign():
    """SQLite FTS5 returns NEGATIVE bm25 (smaller = better). Normalized
    output must invert: smallest input → 1.0, largest input → 0.0."""
    out = _ranker.normalize_bm25([-2.0, -5.0, -1.0])
    assert abs(out[1] - 1.0) < 1e-6, f"min=-5 should map to 1.0, got {out[1]}"
    assert abs(out[2] - 0.0) < 1e-6, f"max=-1 should map to 0.0, got {out[2]}"
    assert abs(out[0] - 0.25) < 1e-6, f"-2 should map to 0.25, got {out[0]}"


def test_normalize_bm25_empty():
    assert _ranker.normalize_bm25([]) == []


def test_normalize_bm25_single():
    out = _ranker.normalize_bm25([-3.5])
    assert out == [1.0]


def test_normalize_bm25_all_tied():
    out = _ranker.normalize_bm25([-2.0, -2.0, -2.0])
    assert out == [1.0, 1.0, 1.0]


# ---------- authority_weight + lifecycle_factor: KeyError on unknown ----------

def test_authority_weight_raises_keyerror_on_unknown():
    spec = _ranker.load_spec()
    try:
        _ranker.authority_weight(spec, "secondary")
    except KeyError:
        pass
    else:
        raise AssertionError("authority_weight must raise KeyError on unknown authority")


def test_authority_weight_raises_on_none():
    spec = _ranker.load_spec()
    try:
        _ranker.authority_weight(spec, None)  # type: ignore[arg-type]
    except KeyError:
        pass
    else:
        raise AssertionError("authority_weight(None) must raise")


def test_lifecycle_factor_raises_on_unknown():
    spec = _ranker.load_spec()
    try:
        _ranker.lifecycle_factor(spec, "withdrawn")
    except KeyError:
        pass
    else:
        raise AssertionError(
            "lifecycle_factor('withdrawn') must raise — withdrawn is excluded "
            "from the candidate pool, not factored"
        )


# ---------- coverage invariants (mirrors cert [254] R9 in unit-test scope) ----------

def test_decay_coverage_matches_schema():
    spec = _ranker.load_spec()
    union = set(spec.decay_exempt_status) | set(spec.decay_status)
    assert union == set(EPISTEMIC_STATUSES), (
        f"decay coverage mismatch: union={sorted(union)} vs "
        f"EPISTEMIC_STATUSES={sorted(EPISTEMIC_STATUSES)}"
    )


def test_authority_weight_keys_match_schema():
    spec = _ranker.load_spec()
    assert set(spec.authority_weight.keys()) == set(AUTHORITIES)


def test_lifecycle_factor_keys_match_schema_minus_withdrawn():
    spec = _ranker.load_spec()
    assert set(spec.lifecycle_factor.keys()) == set(LIFECYCLE_STATES) - {"withdrawn"}


def test_decay_sets_disjoint():
    spec = _ranker.load_spec()
    assert not (set(spec.decay_exempt_status) & set(spec.decay_status))


# ---------- compose_score — golden cases (subset of cert R6) ----------

def test_compose_score_primary_axiom_full_signal():
    """Primary axiom, perfect bm25, depth 0 → exactly 10.0."""
    spec = _ranker.load_spec()
    score, br = _ranker.compose_score(
        authority="primary", bm25_norm=1.0, confidence=1.0,
        epistemic_status="axiom", created_ts="2026-04-16T00:00:00Z",
        valid_from="", valid_at=None,
        contradiction_state="none", provenance_depth=0,
        lifecycle_state="current", spec=spec,
    )
    assert abs(score - 10.0) < 1e-9, f"expected 10.0, got {score}"
    assert br["authority"] == 10.0
    assert br["lifecycle"] == 1.0
    assert br["prov_decay"] == 1.0


def test_compose_score_superseded_halves_score():
    """Same inputs, lifecycle_state='superseded' → score halved."""
    spec = _ranker.load_spec()
    base, _ = _ranker.compose_score(
        authority="derived", bm25_norm=0.5, confidence=1.0,
        epistemic_status="certified", created_ts="2026-04-16T00:00:00Z",
        valid_from="", valid_at=None,
        contradiction_state="none", provenance_depth=1,
        lifecycle_state="current", spec=spec,
    )
    superseded, _ = _ranker.compose_score(
        authority="derived", bm25_norm=0.5, confidence=1.0,
        epistemic_status="certified", created_ts="2026-04-16T00:00:00Z",
        valid_from="", valid_at=None,
        contradiction_state="none", provenance_depth=1,
        lifecycle_state="superseded", spec=spec,
    )
    assert abs(superseded - 0.5 * base) < 1e-9


def test_compose_score_no_path_factor():
    """provenance_depth=-1 applies spec.no_path_factor (0.5)."""
    spec = _ranker.load_spec()
    score, br = _ranker.compose_score(
        authority="primary", bm25_norm=1.0, confidence=1.0,
        epistemic_status="source_claim", created_ts="2026-04-16T00:00:00Z",
        valid_from="", valid_at=None,
        contradiction_state="none", provenance_depth=-1,
        lifecycle_state="current", spec=spec,
    )
    assert br["prov_decay"] == 0.5
    assert abs(score - 10.0 * 0.5) < 1e-9


def test_compose_score_contradiction_boost():
    """contradiction_state != 'none' applies 1.5× boost."""
    spec = _ranker.load_spec()
    base, _ = _ranker.compose_score(
        authority="primary", bm25_norm=0.5, confidence=1.0,
        epistemic_status="source_claim", created_ts="2026-04-16T00:00:00Z",
        valid_from="", valid_at=None,
        contradiction_state="none", provenance_depth=-1,
        lifecycle_state="current", spec=spec,
    )
    boosted, _ = _ranker.compose_score(
        authority="primary", bm25_norm=0.5, confidence=1.0,
        epistemic_status="source_claim", created_ts="2026-04-16T00:00:00Z",
        valid_from="", valid_at=None,
        contradiction_state="src", provenance_depth=-1,
        lifecycle_state="current", spec=spec,
    )
    assert abs(boosted - 1.5 * base) < 1e-9


def test_compose_score_valid_from_precedence():
    """valid_from takes precedence over created_ts as decay anchor."""
    spec = _ranker.load_spec()
    valid_at = _dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc)
    fresh, _ = _ranker.compose_score(
        authority="internal", bm25_norm=1.0, confidence=1.0,
        epistemic_status="observation", created_ts="2025-04-16T00:00:00Z",
        valid_from="2026-04-16T00:00:00Z",  # fresh
        valid_at=valid_at, contradiction_state="none", provenance_depth=0,
        lifecycle_state="current", spec=spec,
    )
    aged, _ = _ranker.compose_score(
        authority="internal", bm25_norm=1.0, confidence=1.0,
        epistemic_status="observation", created_ts="2026-04-16T00:00:00Z",
        valid_from="2025-04-16T00:00:00Z",  # 1 year old per valid_from
        valid_at=valid_at, contradiction_state="none", provenance_depth=0,
        lifecycle_state="current", spec=spec,
    )
    assert fresh > aged
    expected_aged = 5.0 * 1.0 * 1.0 * math.exp(-365.0 / 365.0) * 1.0 * 1.0
    assert abs(aged - expected_aged) < 1e-6


# ---------- end-to-end via KG.search_authority_ranked ----------

def test_search_authority_ranked_excludes_agent_under_internal():
    """Insert an agent node alongside a primary one; min_authority='internal'
    must NOT return the agent node. Cert [254] R1 in unit-test scope."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        kg = connect(db)
        kg.upsert_node(Node(
            id="axiom:T_TEST", node_type="Axiom",
            title="T_TEST testing axiom",
            body="anchor token unique_marker_xyz testing",
            authority="primary", epistemic_status="axiom",
        ))
        kg.upsert_node(Node(
            id="ag:1", node_type="Thought",
            title="agent claim",
            body="anchor token unique_marker_xyz testing extra agent words",
            authority="agent", epistemic_status="conjecture",
        ))
        hits = kg.search_authority_ranked(
            "unique_marker_xyz", min_authority="internal", k=10,
        )
        ids = [h.node["id"] for h in hits]
        assert "ag:1" not in ids, f"agent leaked under min_authority='internal': {ids}"
        # And should be present when min_authority='agent'
        hits_all = kg.search_authority_ranked(
            "unique_marker_xyz", min_authority="agent", k=10,
        )
        ids_all = [h.node["id"] for h in hits_all]
        assert "ag:1" in ids_all, f"agent should appear at min_authority='agent': {ids_all}"


def test_search_authority_ranked_deterministic_under_fixed_valid_at():
    """5 reruns under fixed valid_at must yield identical (id, score) lists."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        kg = connect(db)
        for i in range(3):
            kg.upsert_node(Node(
                id=f"obs:det_{i}", node_type="Thought",
                title=f"deterministic test obs {i}",
                body="repeatable_query_marker body words " + ("x" * (i + 1)),
                authority="internal", epistemic_status="observation",
            ))
        fixed = _dt.datetime(2026, 4, 16, tzinfo=_dt.timezone.utc)
        runs = []
        for _ in range(5):
            hits = kg.search_authority_ranked(
                "repeatable_query_marker", min_authority="internal",
                k=10, valid_at=fixed,
            )
            runs.append([(h.node["id"], round(h.score, 9)) for h in hits])
        assert all(r == runs[0] for r in runs), f"non-deterministic: {runs}"


def test_export_clean_subset_strips_contradictions():
    """_export_clean_subset is the only path that drops contradicted hits."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        kg = connect(db)
        kg.upsert_node(Node(
            id="sc:test_contra", node_type="Claim",
            title="test contra claim",
            body="contra_marker body of claim",
            authority="primary", epistemic_status="source_claim",
        ))
        kg.upsert_node(Node(
            id="obs:test_correction", node_type="Thought",
            title="test correction",
            body="contra_marker correction body",
            authority="internal", epistemic_status="observation",
        ))
        # Add a contradicts edge from internal correction → primary claim.
        from tools.qa_kg.kg import Edge
        import json as _json
        kg.upsert_edge(Edge(
            src_id="obs:test_correction", dst_id="sc:test_contra",
            edge_type="contradicts", confidence=1.0,
            method="test",
            provenance=_json.dumps({"reason": "true"}),
        ))
        full = kg.search_authority_ranked("contra_marker", min_authority="internal", k=10)
        clean = kg._export_clean_subset("contra_marker", min_authority="internal", k=10)
        assert any(h.contradiction_state != "none" for h in full), (
            "search_authority_ranked must surface contradictions by default"
        )
        assert all(h.contradiction_state == "none" for h in clean), (
            "_export_clean_subset must strip contradicted hits"
        )


# ---------- runner ----------

def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            fails.append(fn.__name__)
    print(f"\n{len(fns) - len(fails)}/{len(fns)} ranker tests PASS")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(_run_all())
