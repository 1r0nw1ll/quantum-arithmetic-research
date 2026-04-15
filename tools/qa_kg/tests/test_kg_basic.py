"""Basic QA-KG tests — schema round-trip, firewall, coord math, predicate runtime.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Run: python -m tools.qa_kg.tests.test_kg_basic
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.qa_kg.kg import Edge, FirewallViolation, KG, Node, connect
from tools.qa_kg.orbit import Coord, Tier, coord_for, edge_allowed, qa_step
from tools.qa_kg.predicate import run as run_predicate


def test_coord_constraints():
    c = Coord(3, 7)
    assert c.d == 10 and c.a == 17, "A2 raw derivation"
    try:
        Coord(0, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("A1 must reject 0")


def test_qa_step_a1():
    # A1: result in {1..9}, never 0
    for b in range(1, 10):
        for e in range(1, 10):
            r = qa_step(b, e, 9)
            assert 1 <= r <= 9, f"A1 violated: qa_step({b},{e})={r}"


def test_coord_for_stable():
    assert coord_for("cert:225") == coord_for("cert:225")
    assert coord_for("cert:225") != coord_for("cert:226")


def test_firewall_archive_to_cosmos():
    assert not edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=False)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=True)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "cites", via_cert=False)  # non-causal
    assert edge_allowed(Tier.SATELLITE, Tier.COSMOS, "validates", via_cert=False)  # in-flight ok


def test_schema_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(id="axiom:A1", node_type="Axiom", title="No-Zero",
                            tier=Tier.SINGULARITY, coord=Coord(9, 9)))
        kg.upsert_node(Node(id="cert:demo", node_type="Cert", title="Demo cert",
                            tier=Tier.COSMOS))
        kg.upsert_edge(Edge(src_id="cert:demo", dst_id="axiom:A1", edge_type="derived-from"))
        row = kg.get("cert:demo")
        assert row is not None and row["title"] == "Demo cert"
        ns = kg.neighbors("cert:demo")
        assert len(ns) == 1 and ns[0]["edge_type"] == "derived-from"
        st = kg.stats()
        assert st.get("singularity") == 1 and st.get("cosmos") == 1


def test_firewall_violation_raised():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        kg = connect(db)
        kg.upsert_node(Node(id="arch:old", node_type="Thought", title="archive",
                            tier=Tier.UNASSIGNED))
        kg.upsert_node(Node(id="cos:new", node_type="Cert", title="canonical",
                            tier=Tier.COSMOS))
        try:
            kg.upsert_edge(Edge(src_id="arch:old", dst_id="cos:new", edge_type="validates"))
        except FirewallViolation:
            pass
        else:
            raise AssertionError("Theorem NT firewall must block archive→cosmos 'validates'")
        # With via_cert, allowed.
        kg.upsert_edge(Edge(src_id="arch:old", dst_id="cos:new", edge_type="validates",
                            via_cert="cert:demo"))


def _stub_pred_ok() -> tuple[bool, str]:
    return True, "stub"


def test_predicate_runtime():
    # Build a reference to this module's stub
    mod = __name__
    ref = f"{mod}:_stub_pred_ok"
    res = run_predicate(ref)
    assert res.ok and res.msg == "stub"


TESTS = [
    test_coord_constraints,
    test_qa_step_a1,
    test_coord_for_stable,
    test_firewall_archive_to_cosmos,
    test_schema_roundtrip,
    test_firewall_violation_raised,
    test_predicate_runtime,
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
