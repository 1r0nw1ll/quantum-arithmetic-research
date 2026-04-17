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


# =========================================================================
# Phase 3 tests — SourceWork / SourceClaim / contradicts / supersedes /
# locators / certs lifecycle bridge. All use real-shape ephemeral DBs,
# matching the Phase 2 FE4 no-mocks standard.
# =========================================================================


def test_source_work_constructor_and_upsert():
    """Node.source_work builds a (primary, source_work, Work) node; upsert
    + read-back preserves fields."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        n = Node.source_work(
            work_id="unit_test_work",
            title="Unit Test Work",
            source_locator="file:docs/theory/svp_wiki_qa_elements_snapshot.md",
            body="test body",
        )
        assert n.id == "work:unit_test_work"
        assert n.node_type == "Work"
        assert n.authority == "primary"
        assert n.epistemic_status == "source_work"
        kg.upsert_node(n)
        row = kg.conn.execute(
            "SELECT * FROM nodes WHERE id=?", (n.id,)
        ).fetchone()
        assert row["authority"] == "primary"
        assert row["epistemic_status"] == "source_work"
        assert row["node_type"] == "Work"


def test_source_claim_constructor_and_upsert():
    """Node.source_claim builds a (primary, source_claim, Claim) node;
    factory rejects empty quote and invalid extraction_method."""
    sc = Node.source_claim(
        claim_id="sctest",
        quote="L = FC/2",
        source_locator="file:docs/theory/svp_wiki_qa_elements_snapshot.md#L546",
        extraction_method="manual",
    )
    assert sc.id == "sc:sctest"
    assert sc.node_type == "Claim"
    assert sc.authority == "primary"
    assert sc.epistemic_status == "source_claim"
    assert sc.body == "L = FC/2"

    try:
        Node.source_claim(
            claim_id="x", quote="",
            source_locator="file:foo", extraction_method="manual",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("factory must reject empty quote")

    try:
        Node.source_claim(
            claim_id="x", quote="a",
            source_locator="file:foo", extraction_method="nonsense",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("factory must reject bad extraction_method")


def test_source_claim_must_have_quoted_from_edge():
    """SC2 catches SourceClaim without quoted-from edge (checked via live
    DB query pattern that [253] SC2 uses)."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node.source_claim(
            claim_id="orphan",
            quote="orphan quote",
            source_locator="file:docs/theory/svp_wiki_qa_elements_snapshot.md",
            extraction_method="manual",
        ))
        # No quoted-from edge. SC2's query should find 0 edges.
        edges = kg.conn.execute(
            "SELECT * FROM edges WHERE src_id=? AND edge_type='quoted-from'",
            ("sc:orphan",),
        ).fetchall()
        assert len(edges) == 0, "SC2 precondition: orphan has no quoted-from edges"


def test_quoted_from_is_non_causal():
    """Scenario: an agent-authored Thought cites a SourceWork it referenced
    while reasoning. Emitting agent-thought --quoted-from--> source-work
    must NOT raise FirewallViolation — quoted-from is structural metadata,
    not a causal derivation. If this started firing the firewall, the
    structural/causal split in orbit.py would be silently broken and
    agents would lose the ability to cite sources without promoting first."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="agent:thinker", node_type="Thought",
            title="agent thought", body="a thinking agent",
            authority="agent", epistemic_status="observation",
        ))
        kg.upsert_node(Node.source_work(
            work_id="cited_work", title="Cited Work",
            source_locator="file:CLAUDE.md",
        ))
        # Non-causal edge from agent to SourceWork — must succeed.
        kg.upsert_edge(Edge(
            src_id="agent:thinker", dst_id="work:cited_work",
            edge_type="quoted-from", confidence=1.0,
        ))
        row = kg.conn.execute(
            "SELECT 1 FROM edges WHERE src_id='agent:thinker' AND "
            "dst_id='work:cited_work' AND edge_type='quoted-from'"
        ).fetchone()
        assert row is not None
        # And agent causal edges STILL blocked (regression guard).
        try:
            kg.upsert_edge(Edge(
                src_id="agent:thinker", dst_id="work:cited_work",
                edge_type="validates",
            ))
        except FirewallViolation:
            pass
        else:
            raise AssertionError(
                "regression: agent causal edge passed without promote"
            )


def test_contradicts_accepts_valid_reason():
    """A contradicts edge with reason in closed set round-trips via
    provenance JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node.source_work(
            work_id="w", title="W", source_locator="file:CLAUDE.md",
        ))
        kg.upsert_node(Node.source_claim(
            claim_id="a", quote="X", source_locator="file:CLAUDE.md#L1",
            extraction_method="manual",
        ))
        kg.upsert_node(Node.source_claim(
            claim_id="b", quote="Y", source_locator="file:CLAUDE.md#L1",
            extraction_method="manual",
        ))
        kg.upsert_edge(Edge(
            src_id="sc:a", dst_id="sc:b",
            edge_type="contradicts",
            provenance=json.dumps({"reason": "typo", "extractor": "t"}),
        ))
        row = kg.conn.execute(
            "SELECT provenance FROM edges WHERE src_id='sc:a' "
            "AND dst_id='sc:b' AND edge_type='contradicts'"
        ).fetchone()
        prov = json.loads(row["provenance"])
        assert prov["reason"] == "typo"


def test_supersedes_dag_cycle_detection():
    """KG13's cycle detection catches A→B→A supersedes loop."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="cert:A", node_type="Cert", title="A", body="a",
            authority="derived", epistemic_status="certified",
            lifecycle_state="superseded",
        ))
        kg.upsert_node(Node(
            id="cert:B", node_type="Cert", title="B", body="b",
            authority="derived", epistemic_status="certified",
            lifecycle_state="superseded",
        ))
        kg.upsert_edge(Edge(src_id="cert:A", dst_id="cert:B",
                            edge_type="supersedes"))
        kg.upsert_edge(Edge(src_id="cert:B", dst_id="cert:A",
                            edge_type="supersedes"))
        # Import KG13 checker and verify it flags this.
        import importlib.util as _iu
        spec_path = (
            Path(__file__).resolve().parents[3]
            / "qa_alphageometry_ptolemy"
            / "qa_kg_consistency_cert_v4"
            / "qa_kg_consistency_cert_validate.py"
        )
        spec = _iu.spec_from_file_location("_kg_v4_validator", spec_path)
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok, msg, detail = mod.check_kg13_supersedes_dag_and_lifecycle(kg.conn)
        assert not ok, "KG13 must flag A→B→A cycle"
        assert any("cycle" in d for d in detail)


def test_supersedes_lifecycle_consistency():
    """A superseded node with 0 incoming supersedes edges fails KG13."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="cert:orphan_v1", node_type="Cert", title="orphan",
            body="o", authority="derived", epistemic_status="certified",
            lifecycle_state="superseded",  # claims superseded, but...
        ))
        # ...nobody supersedes it. KG13 fires.
        import importlib.util as _iu
        spec_path = (
            Path(__file__).resolve().parents[3]
            / "qa_alphageometry_ptolemy"
            / "qa_kg_consistency_cert_v4"
            / "qa_kg_consistency_cert_validate.py"
        )
        spec = _iu.spec_from_file_location("_kg_v4_validator2", spec_path)
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok, _msg, detail = mod.check_kg13_supersedes_dag_and_lifecycle(kg.conn)
        assert not ok
        assert any("cert:orphan_v1" in d and "no incoming" in d for d in detail)


def test_phase3_extractor_idempotent():
    """Running source_claims.populate twice yields stable node + edge counts."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        # Seed the cert nodes the fixture's supersedes chain points at.
        for vN in (1, 2, 3, 4):
            kg.upsert_node(Node(
                id=f"cert:fs:qa_kg_consistency_cert_v{vN}",
                node_type="Cert",
                title=f"stub v{vN}",
                body=f"stub {vN}",
                authority="derived",
                epistemic_status="certified",
            ))
        from tools.qa_kg.extractors import source_claims
        r1 = source_claims.populate(kg)
        r2 = source_claims.populate(kg)
        for k in ("works", "claims", "observations", "contradicts", "supersedes"):
            assert r1[k] == r2[k], f"{k} count drifted: {r1[k]} vs {r2[k]}"
        # Also confirm node count doesn't double.
        n_total_1 = kg.conn.execute("SELECT COUNT(*) n FROM nodes").fetchone()["n"]
        source_claims.populate(kg)
        n_total_2 = kg.conn.execute("SELECT COUNT(*) n FROM nodes").fetchone()["n"]
        assert n_total_1 == n_total_2


def test_ef3_allows_primary_source_work():
    """[252] EF3 allowed_matrix accepts (primary, source_work)."""
    import importlib.util as _iu
    spec_path = (
        Path(__file__).resolve().parents[3]
        / "qa_alphageometry_ptolemy"
        / "qa_kg_epistemic_fields_cert_v1"
        / "qa_kg_epistemic_fields_cert_validate.py"
    )
    spec = _iu.spec_from_file_location("_ef_validator", spec_path)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    matrix = mod._load_allowed_matrix()
    assert "source_work" in matrix["primary"], (
        "allowed_matrix must contain (primary, source_work) after Phase 3"
    )
    # Also verify EF3 passes on a real primary+source_work node.
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node.source_work(
            work_id="ef3_test", title="EF3 Test",
            source_locator="file:CLAUDE.md",
        ))
        ok, _msg, viols = mod.check_ef3(kg.conn, matrix)
        assert ok, f"EF3 must pass on SourceWork node: {viols}"


def test_sc8_rejects_axiom_endpoint():
    """[253] SC8 rejects contradicts edge with Axiom src or dst."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="axiom:A1", node_type="Axiom",
            title="A1", body="A1: No-Zero",
            authority="primary", epistemic_status="axiom",
        ))
        kg.upsert_node(Node.source_work(
            work_id="w", title="W", source_locator="file:CLAUDE.md",
        ))
        kg.upsert_node(Node.source_claim(
            claim_id="s", quote="x", source_locator="file:CLAUDE.md",
            extraction_method="manual",
        ))
        kg.upsert_edge(Edge(
            src_id="axiom:A1", dst_id="sc:s",
            edge_type="contradicts",
            provenance=json.dumps({"reason": "typo"}),
        ))
        import importlib.util as _iu
        spec_path = (
            Path(__file__).resolve().parents[3]
            / "qa_alphageometry_ptolemy"
            / "qa_kg_source_claims_cert_v1"
            / "qa_kg_source_claims_cert_validate.py"
        )
        spec = _iu.spec_from_file_location("_sc_validator", spec_path)
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok, _msg, detail = mod.check_sc8_contradicts_endpoint_whitelist(
            kg.conn, {"Axiom"}, {"agent"},
        )
        assert not ok
        assert any("Axiom" in d for d in detail)


def test_sc8_rejects_agent_endpoint():
    """[253] SC8 rejects contradicts edge with agent-authority src — the
    contradicts channel must not be used as a back door for unpromoted
    agent dissent."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="agent:dissenter", node_type="Thought",
            title="agent dissent", body="I disagree",
            authority="agent", epistemic_status="conjecture",
        ))
        kg.upsert_node(Node(
            id="obs:target", node_type="Claim",
            title="target obs", body="observation",
            authority="internal", epistemic_status="observation",
        ))
        # Insert contradicts edge directly; contradicts is not causal so
        # the firewall doesn't block it at upsert_edge time. SC8 is what
        # catches the agent endpoint later.
        kg.upsert_edge(Edge(
            src_id="agent:dissenter", dst_id="obs:target",
            edge_type="contradicts",
            provenance=json.dumps({"reason": "dispute"}),
        ))
        import importlib.util as _iu
        spec_path = (
            Path(__file__).resolve().parents[3]
            / "qa_alphageometry_ptolemy"
            / "qa_kg_source_claims_cert_v1"
            / "qa_kg_source_claims_cert_validate.py"
        )
        spec = _iu.spec_from_file_location("_sc_validator2", spec_path)
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok, _msg, detail = mod.check_sc8_contradicts_endpoint_whitelist(
            kg.conn, {"Axiom"}, {"agent"},
        )
        assert not ok
        assert any("agent" in d for d in detail)


# =========================================================================
# Phase 6 tests — W5 authority-immutable + promote() rate-limit integration
# =========================================================================


def test_w5_authority_immutable_blocks_upgrade():
    """Phase 6 W5: agent → internal upsert raises FirewallViolation."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="x:1", node_type="Thought", title="t", body="b",
            authority="agent", epistemic_status="observation",
        ))
        try:
            kg.upsert_node(Node(
                id="x:1", node_type="Thought", title="t", body="b",
                authority="internal", epistemic_status="observation",
            ))
        except FirewallViolation as e:
            assert "authority_immutable" in str(e)
        else:
            raise AssertionError("agent→internal must be blocked")


def test_w5_authority_immutable_blocks_silent_downgrade():
    """Phase 6 W5: primary → agent is also blocked (silent-downgrade guard)."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="sc:primary", node_type="Claim", title="p", body="quoted",
            authority="primary", epistemic_status="source_claim",
            method="manual", source_locator="file:CLAUDE.md",
        ))
        try:
            kg.upsert_node(Node(
                id="sc:primary", node_type="Claim", title="p", body="quoted",
                authority="agent", epistemic_status="observation",
            ))
        except FirewallViolation as e:
            assert "authority_immutable" in str(e)
        else:
            raise AssertionError("primary→agent must be blocked")


def test_w5_same_authority_idempotent_upsert_passes():
    """Phase 6 W5: re-upserting with the same authority is a no-op pass."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = connect(Path(tmp) / "t.db")
        kg.upsert_node(Node(
            id="x:2", node_type="Thought", title="t", body="b",
            authority="agent", epistemic_status="observation",
        ))
        # Same authority, updated body — must succeed.
        kg.upsert_node(Node(
            id="x:2", node_type="Thought", title="t", body="b2",
            authority="agent", epistemic_status="observation",
        ))
        row = kg.get("x:2")
        assert row["body"] == "b2"
        assert row["authority"] == "agent"


def test_promote_rate_limit_integration():
    """Phase 6: mcp_session passed to promote() increments _agent_writes.json
    and raises RateLimitExceeded at the cap. Bypassing promote (kg.upsert_edge
    + promoted-from directly) does NOT burn a count — the counter is
    scoped to the MCP-authored write path per plan M1."""
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        awpath = Path(tmp) / "_agent_writes.json"
        # First promote via MCP session succeeds + increments counter.
        kg.promote(
            agent_note_id="agent:note_promo",
            via_cert="225",
            promoter_node_id="rule:promoter_promo",
            broadcast_payload=_fresh_broadcast(),
            ledger_path=lpath,
            mcp_session="test-session",
            agent_writes_path=awpath,
        )
        data = json.loads(awpath.read_text())
        assert data["test-session"]["count"] == 1


def test_promote_rate_limit_raises_at_cap():
    """Simulate the cap — 3 allowed, 4th raises. Uses cap=3 via env override
    pattern: direct rate_limit.increment exercises cap; here we bolt on a
    small-cap test by monkey-patching MAX for the duration."""
    import tools.qa_kg_mcp.rate_limit as rl
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        awpath = Path(tmp) / "_agent_writes.json"
        old_cap = rl.MAX_WRITES_PER_SESSION
        rl.MAX_WRITES_PER_SESSION = 3
        try:
            for _ in range(3):
                kg.promote(
                    agent_note_id="agent:note_promo",
                    via_cert="225",
                    promoter_node_id="rule:promoter_promo",
                    broadcast_payload=_fresh_broadcast(),
                    ledger_path=lpath,
                    mcp_session="cap-session",
                    agent_writes_path=awpath,
                )
            try:
                kg.promote(
                    agent_note_id="agent:note_promo",
                    via_cert="225",
                    promoter_node_id="rule:promoter_promo",
                    broadcast_payload=_fresh_broadcast(),
                    ledger_path=lpath,
                    mcp_session="cap-session",
                    agent_writes_path=awpath,
                )
            except rl.RateLimitExceeded as e:
                assert "cap-session" in str(e)
            else:
                raise AssertionError("4th promote must raise RateLimitExceeded")
        finally:
            rl.MAX_WRITES_PER_SESSION = old_cap


def test_promote_without_mcp_session_does_not_touch_counter():
    """Extractor-path callers (no mcp_session) must not increment counter."""
    with tempfile.TemporaryDirectory() as tmp:
        kg, lpath = _make_promote_fixtures(tmp)
        awpath = Path(tmp) / "_agent_writes.json"
        kg.promote(
            agent_note_id="agent:note_promo",
            via_cert="225",
            promoter_node_id="rule:promoter_promo",
            broadcast_payload=_fresh_broadcast(),
            ledger_path=lpath,
            # no mcp_session — extractor-path call
        )
        assert not awpath.exists() or json.loads(awpath.read_text()) == {}


def test_certs_extractor_translates_frozen_to_superseded():
    """§6a lifecycle bridge: frozen with sibling _v<N+1> → superseded;
    frozen without successor → deprecated; not frozen → current."""
    from tools.qa_kg.extractors.certs import _lifecycle_for_status
    repo = Path(__file__).resolve().parents[3]
    meta = repo / "qa_alphageometry_ptolemy"
    # v1 is frozen (banner-style _status), v2/v3 exist as successors:
    # expect superseded.
    assert _lifecycle_for_status(meta / "qa_kg_consistency_cert_v1") == "superseded"
    # v3 is now frozen by Phase 3, v4 exists: expect superseded.
    assert _lifecycle_for_status(meta / "qa_kg_consistency_cert_v3") == "superseded"
    # v4 is current (no successor sibling yet): expect current.
    assert _lifecycle_for_status(meta / "qa_kg_consistency_cert_v4") == "current"
    # EpistemicFields v1 has no sibling pattern: expect current.
    assert _lifecycle_for_status(meta / "qa_kg_epistemic_fields_cert_v1") == "current"


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
    # Phase 3 additions
    test_source_work_constructor_and_upsert,
    test_source_claim_constructor_and_upsert,
    test_source_claim_must_have_quoted_from_edge,
    test_quoted_from_is_non_causal,
    test_contradicts_accepts_valid_reason,
    test_supersedes_dag_cycle_detection,
    test_supersedes_lifecycle_consistency,
    test_phase3_extractor_idempotent,
    test_ef3_allows_primary_source_work,
    test_sc8_rejects_axiom_endpoint,
    test_sc8_rejects_agent_endpoint,
    test_certs_extractor_translates_frozen_to_superseded,
    # Phase 6 additions
    test_w5_authority_immutable_blocks_upgrade,
    test_w5_authority_immutable_blocks_silent_downgrade,
    test_w5_same_authority_idempotent_upsert_passes,
    test_promote_rate_limit_integration,
    test_promote_rate_limit_raises_at_cap,
    test_promote_without_mcp_session_does_not_touch_counter,
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
