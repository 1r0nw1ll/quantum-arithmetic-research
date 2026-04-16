# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG test harness -->
"""Candidate F [202] — basic QA-KG tests.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Run: python -m tools.qa_kg.tests.test_kg_basic
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.qa_kg import (
    Coord, Tier, NODE_TYPE_RANK, compute_be, dr, char_ord_sum,
    tier_for_coord, connect,
)
from tools.qa_kg.kg import Edge, FirewallViolation, Node
from tools.qa_kg.orbit import qa_step, edge_allowed
from tools.qa_kg.predicate import run as run_predicate


def test_index_a1():
    """Phase 0: Index (formerly Coord) enforces A1 bounds on idx_b/idx_e and
    has no .d/.a properties — those would imply a QA state derivation which
    a retrieval index does not support. See orbit.py module docstring."""
    c = Coord(3, 7)  # Coord is a back-compat alias for Index
    assert c.idx_b == 3 and c.idx_e == 7
    assert not hasattr(c, "d")
    assert not hasattr(c, "a")
    try:
        Coord(0, 3)
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
    c1 = compute_be("hello world", "Cert")
    c2 = compute_be("hello world", "Cert")
    assert c1 == c2
    # Same content, different type → different e
    c3 = compute_be("hello world", "Axiom")
    assert c1.idx_b == c3.idx_b
    assert c1.idx_e != c3.idx_e
    # Integrity
    assert c1.idx_e == NODE_TYPE_RANK["Cert"]
    assert c3.idx_e == NODE_TYPE_RANK["Axiom"]


def test_candidate_f_matches_arag_formula():
    """Our compute_be must produce the same (b=dr(char_ord_sum), e=type_rank) shape as A-RAG."""
    text = "Quantum Arithmetic has canonical orbits"
    c = compute_be(text, "Thought")
    assert c.idx_b == dr(char_ord_sum(text))
    assert c.idx_e == NODE_TYPE_RANK["Thought"]


def test_tier_unassigned_on_none():
    assert tier_for_coord(None, None) is Tier.UNASSIGNED


def test_firewall_archive_to_cosmos():
    assert not edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=False)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=True)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "cites", via_cert=False)


def test_schema_roundtrip_candidate_f():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(id="axiom:A1", node_type="Axiom", title="No-Zero",
                            body="States in {1..N}"))
        kg.upsert_node(Node(id="cert:demo", node_type="Cert", title="Demo",
                            body="Demo body"))
        # Empty content → unassigned
        kg.upsert_node(Node(id="empty:x", node_type="Thought", title="", body=""))
        a = kg.get("axiom:A1")
        assert a["idx_e"] == NODE_TYPE_RANK["Axiom"]
        assert a["idx_b"] == dr(a["char_ord_sum"])
        c = kg.get("cert:demo")
        assert c["idx_e"] == NODE_TYPE_RANK["Cert"]
        u = kg.get("empty:x")
        assert u["tier"] == "unassigned" and u["idx_b"] is None


def test_firewall_violation_raised():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(id="arch:old", node_type="Thought", title="", body=""))
        kg.upsert_node(Node(id="cos:new", node_type="Cert", title="X", body="body"))
        try:
            kg.upsert_edge(Edge(src_id="arch:old", dst_id="cos:new", edge_type="validates"))
        except FirewallViolation:
            pass
        else:
            raise AssertionError("Theorem NT must block archive→canonical 'validates'")
        kg.upsert_edge(Edge(src_id="arch:old", dst_id="cos:new",
                            edge_type="validates", via_cert="cert:demo"))


def test_back_compat_aliases_scheduled_for_removal_in_v3():
    """PIN: back-compat aliases Coord / compute_be / tier_for_coord are
    deprecated and scheduled for removal at [225] v3 / schema v3.

    This test asserts two things:
      (a) while schema_version == 2 (current), the aliases MUST still exist
          (so in-flight callers don't break mid-Phase-0).
      (b) when schema_version advances past 2, the aliases MUST be removed
          in the same commit.

    When this test fails under schema v3, the fix is to delete the aliases
    from orbit.py and delete this test. Do NOT silence it; the failure IS
    the forcing function."""
    from tools.qa_kg.schema import SCHEMA_VERSION
    from tools.qa_kg import orbit as _orbit_mod

    if SCHEMA_VERSION == 2:
        # In-flight Phase 0: aliases MUST be present so callers can migrate.
        assert hasattr(_orbit_mod, "Coord"), "Coord alias removed too early"
        assert hasattr(_orbit_mod, "compute_be"), "compute_be alias removed too early"
        assert hasattr(_orbit_mod, "tier_for_coord"), "tier_for_coord alias removed too early"
    elif SCHEMA_VERSION >= 3:
        # v3+: aliases MUST be gone.
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


TESTS = [
    test_index_a1,
    test_qa_step_a1,
    test_dr_a1,
    test_candidate_f_deterministic,
    test_candidate_f_matches_arag_formula,
    test_tier_unassigned_on_none,
    test_firewall_archive_to_cosmos,
    test_schema_roundtrip_candidate_f,
    test_firewall_violation_raised,
    test_predicate_runtime,
    test_back_compat_aliases_scheduled_for_removal_in_v3,
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
