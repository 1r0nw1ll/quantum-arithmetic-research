"""Populate Singularity — the six QA axioms + Theorem NT.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Every axiom carries a runnable predicate that re-checks the axiom invariant
against the repo at check-time. Failures diagnose the mapping/implementation,
never QA itself (Hard Rule 2026-04-15).
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from tools.qa_kg.kg import KG, Node
from tools.qa_kg.orbit import Coord, Tier


AXIOMS = [
    ("A1", "No-Zero", "States in {1..N}, never {0..N-1}. qa_step = ((b+e-1) % m) + 1.",
     "tools.qa_kg.predicates.axioms:check_a1"),
    ("A2", "Derived Coords", "d = b+e, a = b+2e — derived, raw for elements; mod only via T-op.",
     "tools.qa_kg.predicates.axioms:check_a2"),
    ("T2", "Observer Firewall", "Float × modulus → int cast is a QA violation. Theorem NT firewall.",
     "tools.qa_kg.predicates.axioms:check_t2"),
    ("S1", "No **2", "Write b*b, never b**2 (libm ULP drift).",
     "tools.qa_kg.predicates.axioms:check_s1"),
    ("S2", "No float state", "b, e must be int or Fraction. No np.zeros / np.random.rand as QA state.",
     "tools.qa_kg.predicates.axioms:check_s2"),
    ("T1", "Path Time", "QA time = integer path length k. No continuous time in QA logic.",
     "tools.qa_kg.predicates.axioms:check_t1"),
    ("NT", "Observer Projection Firewall",
     "Continuous functions are observer projections ONLY. Boundary crossed exactly twice.",
     "tools.qa_kg.predicates.axioms:check_nt"),
]


def populate(kg: KG) -> list[str]:
    """Insert axioms as Singularity nodes. Returns list of node ids."""
    ids: list[str] = []
    for code, title, body, pred in AXIOMS:
        nid = f"axiom:{code}"
        kg.upsert_node(Node(
            id=nid,
            node_type="Axiom",
            title=f"{code} — {title}",
            body=body,
            tier=Tier.SINGULARITY,
            coord=Coord(9, 9),
            source="CLAUDE.md",
            vetted_by=nid,  # axioms are self-vetting
            predicate_ref=pred,
        ))
        ids.append(nid)
    return ids
