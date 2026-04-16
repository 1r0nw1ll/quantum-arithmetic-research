# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG test harness -->
"""Candidate F [202] + Phase 1 epistemic fields + Phase 2 promote — QA-KG tests.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Run: python -m tools.qa_kg.tests.test_kg_basic
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.qa_kg import (
    Tier, NODE_TYPE_RANK, compute_index, dr, char_ord_sum,
    tier_for_index, connect, Index, PROMOTED_FROM_EDGE,
)
from tools.qa_kg.kg import Edge, FirewallViolation, Node, _now
from tools.qa_kg.orbit import qa_step, edge_allowed
from tools.qa_kg.predicate import run as run_predicate
from tools.qa_kg.schema import SCHEMA_VERSION


def test_index_a1():
    """Index enforces A1 bounds on idx_b/idx_e and has no .d/.a properties."""
    c = Index(3, 7)
    assert c.idx_b == 3 and c.idx_e == 7
    assert not hasattr(c, "d")
    assert not hasattr(c, "a")
    try:
        Index(0, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("A1 must reject 0")


def test_qa_step_a1():
    for b in range(1, 10):
        for e in range(1, 10):
            r = qa_step(b, e, 9)
            assert 1 <= r <= 9


def test_dr_a1():
    for n in (1, 9, 10, 18, 27, 100, 1000):
        assert 1 <= dr(n) <= 9


def test_candidate_f_deterministic():
    c1 = compute_index("hello world", "Cert")
    c2 = compute_index("hello world", "Cert")
    assert c1 == c2
    c3 = compute_index("hello world", "Axiom")
    assert c1.idx_b == c3.idx_b
    assert c1.idx_e != c3.idx_e
    assert c1.idx_e == NODE_TYPE_RANK["Cert"]
    assert c3.idx_e == NODE_TYPE_RANK["Axiom"]


def test_candidate_f_matches_arag_formula():
    text = "Quantum Arithmetic has canonical orbits"
    c = compute_index(text, "Thought")
    assert c.idx_b == dr(char_ord_sum(text))
    assert c.idx_e == NODE_TYPE_RANK["Thought"]


def test_tier_unassigned_on_none():
    assert tier_for_index(None, None) is Tier.UNASSIGNED


def test_firewall_archive_to_cosmos():
    assert not edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=False)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=True)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "cites", via_cert=False)


def test_firewall_agent_authority_blocked():
    """Phase 2: authority=agent blocks causal edges unconditionally at policy level.
    via_cert alone is NOT sufficient — DB-backed promoted-from is required."""
    assert not edge_allowed(
        Tier.COSMOS, Tier.COSMOS, "validates", via_cert=False,
        src_authority="agent",
    )
    # Phase 2 change: via_cert=True is NO LONGER sufficient for agent
    assert not edge_allowed(
        Tier.COSMOS, Tier.COSMOS, "validates", via_cert=True,
        src_authority="agent",
    )
    # Non-causal edges still pass for agent
    assert edge_allowed(
        Tier.COSMOS, Tier.COSMOS, "keyword-co-occurs", via_cert=False,
        src_authority="agent",
    )
    # Non-agent authorities still pass with via_cert
    assert edge_allowed(
        Tier.COSMOS, Tier.COSMOS, "validates", via_cert=False,
        src_authority="derived",
    )


def test_schema_roundtrip_epistemic():
    """Phase 1: epistemic fields (authority/epistemic_status/method/source_locator)
    round-trip through upsert_node → get."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(
            id="axiom:A1", node_type="Axiom",
            title="No-Zero", body="States in {1..N}",
            authority="primary", epistemic_status="axiom",
            method="axioms_block", source_locator="file:QA_AXIOMS_BLOCK.md#A1",
        ))
        a = kg.get("axiom:A1")
        assert a["authority"] == "primary"
        assert a["epistemic_status"] == "axiom"
        assert a["method"] == "axioms_block"
        assert a["source_locator"] == "file:QA_AXIOMS_BLOCK.md#A1"
        assert a["lifecycle_state"] == "current"
        assert a["idx_e"] == NODE_TYPE_RANK["Axiom"]
        assert a["idx_b"] == dr(a["char_ord_sum"])

        kg.upsert_node(Node(
            id="cert:demo", node_type="Cert",
            title="Demo", body="Demo body",
            authority="derived", epistemic_status="certified",
            method="cert_validator", source_locator="file:qa_alphageometry_ptolemy/demo",
        ))
        c = kg.get("cert:demo")
        assert c["authority"] == "derived"
        assert c["idx_e"] == NODE_TYPE_RANK["Cert"]

        kg.upsert_node(Node(
            id="empty:x", node_type="Thought",
            title="", body="",
            authority="internal", epistemic_status="observation",
        ))
        u = kg.get("empty:x")
        assert u["tier"] == "unassigned" and u["idx_b"] is None


def test_epistemic_validation_rejects_bad_values():
    """Application-layer CHECK rejects invalid authority/epistemic_status."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        try:
            kg.upsert_node(Node(
                id="bad:1", node_type="Cert", title="X", body="Y",
                authority="bogus",
            ))
        except ValueError:
            pass
        else:
            raise AssertionError("must reject invalid authority")
        try:
            kg.upsert_node(Node(
                id="bad:2", node_type="Cert", title="X", body="Y",
                authority="derived", epistemic_status="bogus",
            ))
        except ValueError:
            pass
        else:
            raise AssertionError("must reject invalid epistemic_status")


def test_firewall_violation_raised():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(
            id="arch:old", node_type="Thought", title="", body="",
            authority="internal", epistemic_status="observation",
        ))
        kg.upsert_node(Node(
            id="cos:new", node_type="Cert", title="X", body="body",
            authority="derived", epistemic_status="certified",
        ))
        try:
            kg.upsert_edge(Edge(src_id="arch:old", dst_id="cos:new", edge_type="validates"))
        except FirewallViolation:
            pass
        else:
            raise AssertionError("Theorem NT must block archive→canonical 'validates'")
        kg.upsert_edge(Edge(src_id="arch:old", dst_id="cos:new",
                            edge_type="validates", via_cert="cert:demo"))


def test_firewall_agent_edge_blocked():
    """Phase 2: agent causal edges blocked even with via_cert string.
    Only a DB-backed promoted-from edge allows agent causal edges."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(
            id="agent:note1", node_type="Thought",
            title="Agent observation", body="something",
            authority="agent", epistemic_status="observation",
        ))
        kg.upsert_node(Node(
            id="cert:target", node_type="Cert",
            title="Target cert", body="body here",
            authority="derived", epistemic_status="certified",
        ))
        # Without via_cert — blocked
        try:
            kg.upsert_edge(Edge(
                src_id="agent:note1", dst_id="cert:target",
                edge_type="validates",
            ))
        except FirewallViolation:
            pass
        else:
            raise AssertionError("agent → causal without via_cert must be blocked")
        # With via_cert but NO promoted-from in DB — STILL blocked (Phase 2 fix)
        try:
            kg.upsert_edge(Edge(
                src_id="agent:note1", dst_id="cert:target",
                edge_type="validates", via_cert="cert:252",
            ))
        except FirewallViolation:
            pass
        else:
            raise AssertionError("agent → causal with via_cert but no promoted-from must be blocked")
        # Non-causal edge still allowed
        kg.upsert_edge(Edge(
            src_id="agent:note1", dst_id="cert:target",
            edge_type="keyword-co-occurs",
            confidence=0.3, method="keyword",
        ))
        # After inserting promoted-from edge, causal edge is allowed
        kg.upsert_node(Node(
            id="rule:promoter1", node_type="Rule",
            title="Promoter rule", body="internal rule",
            authority="internal", epistemic_status="interpretation",
        ))
        kg.conn.execute(
            """INSERT INTO edges (src_id, dst_id, edge_type, confidence, method,
                                  provenance, via_cert, created_ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("agent:note1", "rule:promoter1", PROMOTED_FROM_EDGE,
             1.0, "promote", '{"test": true}', "cert:252", _now()),
        )
        kg.conn.commit()
        # NOW the causal edge should succeed
        kg.upsert_edge(Edge(
            src_id="agent:note1", dst_id="cert:target",
            edge_type="validates", via_cert="cert:252",
        ))


def test_search_with_authority_filter():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(
            id="a:1", node_type="Axiom", title="No-Zero test",
            body="states in 1..N", authority="primary", epistemic_status="axiom",
        ))
        kg.upsert_node(Node(
            id="r:1", node_type="Rule", title="No-Zero rule",
            body="hard rule about zero", authority="internal",
            epistemic_status="interpretation",
        ))
        primary_only = kg.search("zero", authority=["primary"])
        assert all(r["authority"] == "primary" for r in primary_only)
        all_results = kg.search("zero")
        assert len(all_results) >= len(primary_only)


def test_back_compat_aliases_scheduled_for_removal_in_v3():
    """PIN: back-compat aliases removed at schema v3.

    This test asserts aliases MUST be absent at v3+. The Phase 0 pin is
    honored: this commit bumps SCHEMA_VERSION to 3 AND removes aliases."""
    from tools.qa_kg import orbit as _orbit_mod

    if SCHEMA_VERSION >= 3:
        assert not hasattr(_orbit_mod, "Coord"), (
            "Coord alias still present at schema v3 — delete from orbit.py")
        assert not hasattr(_orbit_mod, "compute_be"), (
            "compute_be alias still present at schema v3 — delete from orbit.py")
        assert not hasattr(_orbit_mod, "tier_for_coord"), (
            "tier_for_coord alias still present at schema v3 — delete from orbit.py")
    else:
        raise AssertionError(f"unexpected SCHEMA_VERSION={SCHEMA_VERSION}")


def _stub_pred_ok() -> tuple[bool, str]:
    return True, "stub"


def test_predicate_runtime():
    ref = f"{__name__}:_stub_pred_ok"
    res = run_predicate(ref)
    assert res.ok and res.msg == "stub"


def _make_promote_fixtures(tmp):
    """Shared fixture builder for promote tests. Returns (kg, ledger_path)."""
    db = Path(tmp) / "t.db"
    kg = connect(db)
    kg.upsert_node(Node(
        id="agent:note_promo", node_type="Thought",
        title="Agent thought for promotion", body="test content",
        authority="agent", epistemic_status="observation",
    ))
    kg.upsert_node(Node(
        id="rule:promoter_promo", node_type="Rule",
        title="Internal rule", body="this is a rule",
        authority="internal", epistemic_status="interpretation",
    ))
    kg.upsert_node(Node(
        id="cert:target_promo", node_type="Cert",
        title="Target cert", body="certified thing",
        authority="derived", epistemic_status="certified",
    ))
    ledger_path = Path(tmp) / "_meta_ledger.json"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    from tools.qa_kg.kg import _current_git_head
    ledger = {
        "225": {"status": "PASS", "ts": now_iso, "git_head": _current_git_head()},
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return kg, ledger_path


def _fresh_broadcast() -> dict:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session": "test-session",
        "event_type": "kg_promotion",
    }


def test_promote_happy_path():
    """promote() with valid ledger + promoter succeeds, creates promoted-from edge."""
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        kg.promote(
            agent_note_id="agent:note_promo",
            via_cert="225",
            promoter_node_id="rule:promoter_promo",
            broadcast_payload=_fresh_broadcast(),
            ledger_path=lpath,
        )
        edge = kg.conn.execute(
            "SELECT * FROM edges WHERE src_id=? AND edge_type=?",
            ("agent:note_promo", PROMOTED_FROM_EDGE),
        ).fetchone()
        assert edge is not None, "promoted-from edge must exist"
        assert edge["via_cert"] == "225"
        prov = json.loads(edge["provenance"])
        assert prov["promoter_node_id"] == "rule:promoter_promo"
        assert prov["broadcast_payload_snapshot"] is not None
        # Now agent can emit causal edges
        kg.upsert_edge(Edge(
            src_id="agent:note_promo", dst_id="cert:target_promo",
            edge_type="validates", via_cert="225",
        ))


def test_promote_rejects_non_agent():
    """promote() raises ValueError when target is not authority=agent."""
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        try:
            kg.promote(
                agent_note_id="rule:promoter_promo",
                via_cert="225",
                promoter_node_id="rule:promoter_promo",
                broadcast_payload=_fresh_broadcast(),
                ledger_path=lpath,
            )
        except ValueError as e:
            assert "authority=" in str(e)
        else:
            raise AssertionError("must reject non-agent target")


def test_promote_rejects_bad_promoter():
    """promote() raises ValueError when promoter is authority=agent."""
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        kg.upsert_node(Node(
            id="agent:other", node_type="Thought",
            title="Another agent", body="x",
            authority="agent", epistemic_status="observation",
        ))
        try:
            kg.promote(
                agent_note_id="agent:note_promo",
                via_cert="225",
                promoter_node_id="agent:other",
                broadcast_payload=_fresh_broadcast(),
                ledger_path=lpath,
            )
        except ValueError as e:
            assert "agent" in str(e)
        else:
            raise AssertionError("must reject agent promoter")


def test_promote_rejects_stale_ledger():
    """promote() raises FirewallViolation when ledger entry is >14d old."""
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        # Overwrite ledger with stale timestamp
        import datetime as dt
        old_ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=20)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        from tools.qa_kg.kg import _current_git_head
        ledger = {"225": {"status": "PASS", "ts": old_ts, "git_head": _current_git_head()}}
        lpath.write_text(json.dumps(ledger), encoding="utf-8")
        try:
            kg.promote(
                agent_note_id="agent:note_promo",
                via_cert="225",
                promoter_node_id="rule:promoter_promo",
                broadcast_payload=_fresh_broadcast(),
                ledger_path=lpath,
            )
        except FirewallViolation as e:
            assert "stale" in str(e)
        else:
            raise AssertionError("must reject stale ledger")


def test_promote_rejects_mismatched_git_head():
    """promote() raises FirewallViolation when ledger git_head != HEAD."""
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ledger = {"225": {"status": "PASS", "ts": now_iso, "git_head": "deadbeef"}}
        lpath.write_text(json.dumps(ledger), encoding="utf-8")
        try:
            kg.promote(
                agent_note_id="agent:note_promo",
                via_cert="225",
                promoter_node_id="rule:promoter_promo",
                broadcast_payload=_fresh_broadcast(),
                ledger_path=lpath,
            )
        except FirewallViolation as e:
            assert "git_head" in str(e)
        else:
            raise AssertionError("must reject mismatched git_head")


TESTS = [
    test_index_a1,
    test_qa_step_a1,
    test_dr_a1,
    test_candidate_f_deterministic,
    test_candidate_f_matches_arag_formula,
    test_tier_unassigned_on_none,
    test_firewall_archive_to_cosmos,
    test_firewall_agent_authority_blocked,
    test_schema_roundtrip_epistemic,
    test_epistemic_validation_rejects_bad_values,
    test_firewall_violation_raised,
    test_firewall_agent_edge_blocked,
    test_search_with_authority_filter,
    test_back_compat_aliases_scheduled_for_removal_in_v3,
    test_predicate_runtime,
    test_promote_happy_path,
    test_promote_rejects_non_agent,
    test_promote_rejects_bad_promoter,
    test_promote_rejects_stale_ledger,
    test_promote_rejects_mismatched_git_head,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"---\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
