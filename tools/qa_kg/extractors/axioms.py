"""Populate axioms as Axiom-typed nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Coord falls out of Candidate F: b = dr(char_ord_sum(title+body)), e = NODE_TYPE_RANK['Axiom'] = 4.
Axioms land wherever the formula puts them. No (9,9) override.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from tools.qa_kg.kg import KG, Node


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
    ids: list[str] = []
    for code, title, body, pred in AXIOMS:
        nid = f"axiom:{code}"
        # Axioms' authority is the axiom system itself. Represented by node_type='Axiom'
        # and source pointing at the authority docs — not a self-vetted edge.
        kg.upsert_node(Node(
            id=nid,
            node_type="Axiom",
            title=f"{code} — {title}",
            body=body,
            source="CLAUDE.md + QA_AXIOMS_BLOCK.md",
            vetted_by="",
            predicate_ref=pred,
        ))
        ids.append(nid)
    return ids
