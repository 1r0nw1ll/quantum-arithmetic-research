"""Populate axioms as Axiom-typed nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Canonical axiom source: CLAUDE.md §"QA Axiom Compliance" (has the codes
A1/A2/T2/S1/S2/T1/NT). This is the single source of truth for KG9 —
edges.py imports CANONICAL_AXIOM_CODES from here.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import re
from pathlib import Path

from tools.qa_kg.kg import KG, Node


_REPO = Path(__file__).resolve().parents[3]
_CLAUDE_MD = _REPO / "CLAUDE.md"
_AXIOMS_BLOCK = _REPO / "QA_AXIOMS_BLOCK.md"


def _parse_axiom_codes_from_claude_md() -> tuple[str, ...]:
    """Parse axiom codes from CLAUDE.md — the canonical source per KG9."""
    if not _CLAUDE_MD.exists():
        return ("A1", "A2", "T2", "S1", "S2", "T1", "NT")
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    codes: list[str] = []
    for m in re.finditer(r"^\s*-\s+\*\*([A-Z][A-Z0-9]+)\s+\(", text, re.M):
        codes.append(m.group(1))
    nt_match = re.search(r"\*\*Theorem\s+NT\b", text)
    if nt_match and "NT" not in codes:
        codes.append("NT")
    return tuple(dict.fromkeys(codes))


CANONICAL_AXIOM_CODES = _parse_axiom_codes_from_claude_md()

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
    source_locator = "file:QA_AXIOMS_BLOCK.md" if _AXIOMS_BLOCK.exists() else "file:CLAUDE.md"
    for code, title, body, pred in AXIOMS:
        nid = f"axiom:{code}"
        kg.upsert_node(Node(
            id=nid,
            node_type="Axiom",
            title=f"{code} — {title}",
            body=body,
            source="CLAUDE.md + QA_AXIOMS_BLOCK.md",
            vetted_by="",
            predicate_ref=pred,
            authority="primary",
            epistemic_status="axiom",
            method="axioms_block",
            source_locator=f"{source_locator}#{code}",
        ))
        ids.append(nid)
    return ids
