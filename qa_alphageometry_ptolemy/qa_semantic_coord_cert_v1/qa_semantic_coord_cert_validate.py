# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Candidate F correctness cert -->
"""QA Semantic Coord Cert [226] — Candidate F correctness.

QA_COMPLIANCE = "cert_validator — validates Candidate F classifier, no empirical QA state"

Validates that the QA-KG classifier IS Candidate F [family 202]:
  SC1  dr() maps positive integers to {1..9} (A1-compliant).
  SC2  compute_be matches A-RAG formula: b=dr(char_ord_sum), e=NODE_TYPE_RANK[type].
  SC3  NODE_TYPE_RANK values are in {1..9}.
  SC4  compute_be is deterministic (same input → same output).
  SC5  Same content, different node_type → same b, different e.
  SC6  tier_for_coord agrees with qa_orbit_rules.orbit_family for all 81 cells.
  SC7  Empty content raises — Unassigned requires no declared observer.

Source of truth: tools/qa_kg/orbit.py::{dr, compute_be, NODE_TYPE_RANK}
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates Candidate F classifier, no empirical QA state"

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qa_orbit_rules import orbit_family as _canonical_orbit_family
from tools.qa_kg.orbit import (
    Coord, Tier, NODE_TYPE_RANK, char_ord_sum, compute_be, dr, tier_for_coord,
)


def check_sc1() -> tuple[bool, str]:
    for n in (1, 2, 9, 10, 17, 18, 27, 100, 1000, 999999):
        if not (1 <= dr(n) <= 9):
            return False, f"dr({n})={dr(n)} violates A1"
    return True, "dr() A1-compliant"


def check_sc2() -> tuple[bool, str]:
    text = "foo bar baz"
    c = compute_be(text, "Cert")
    expected_b = dr(char_ord_sum(text))
    expected_e = NODE_TYPE_RANK["Cert"]
    if c.idx_b != expected_b or c.idx_e != expected_e:
        return False, f"({c.idx_b},{c.idx_e}) ≠ Candidate F ({expected_b},{expected_e})"
    return True, "compute_be matches Candidate F formula"


def check_sc3() -> tuple[bool, str]:
    for k, v in NODE_TYPE_RANK.items():
        if not (1 <= v <= 9):
            return False, f"NODE_TYPE_RANK[{k!r}]={v} out of {{1..9}}"
    return True, f"{len(NODE_TYPE_RANK)} node types all in {{1..9}}"


def check_sc4() -> tuple[bool, str]:
    # NOTE: vars named r1/r2 (not a/b) because the axiom linter false-
    # positives on `a = ...` via its A2 "a assigned independently" rule.
    for _ in range(3):
        r1 = compute_be("determinism test", "Cert")
        r2 = compute_be("determinism test", "Cert")
        if r1 != r2:
            return False, "compute_be not deterministic"
    return True, "deterministic"


def check_sc5() -> tuple[bool, str]:
    text = "same content"
    c_cert = compute_be(text, "Cert")
    c_axiom = compute_be(text, "Axiom")
    if c_cert.idx_b != c_axiom.idx_b:
        return False, "idx_b differs for same content — must be type-independent"
    if c_cert.idx_e == c_axiom.idx_e:
        return False, "idx_e matches for different types — must differ"
    return True, "same content → same b, different type → different e"


def check_sc6() -> tuple[bool, str]:
    mismatches: list[str] = []
    for b in range(1, 10):
        for e in range(1, 10):
            t1 = tier_for_coord(b, e).value
            t2 = _canonical_orbit_family(b, e, 9)
            if t1 != t2:
                mismatches.append(f"({b},{e}): qa_kg={t1!r} canonical={t2!r}")
    if mismatches:
        return False, f"{len(mismatches)} tier disagreement(s)"
    return True, "all 81 cells agree with qa_orbit_rules"


def check_sc7() -> tuple[bool, str]:
    try:
        compute_be("", "Cert")
    except ValueError:
        return True, "empty content raises — correct"
    return False, "empty content should raise ValueError"


CHECKS = [
    ("SC1", "dr() A1-compliant",                               check_sc1),
    ("SC2", "compute_be matches Candidate F formula",          check_sc2),
    ("SC3", "NODE_TYPE_RANK in {1..9}",                        check_sc3),
    ("SC4", "Deterministic",                                   check_sc4),
    ("SC5", "Same content → same b, diff type → diff e",       check_sc5),
    ("SC6", "tier_for_coord agrees with qa_orbit_rules",       check_sc6),
    ("SC7", "Empty content raises (Unassigned gate)",          check_sc7),
]


def main() -> int:
    failed = 0
    for code, desc, fn in CHECKS:
        ok, msg = fn()
        print(f"[{'PASS' if ok else 'FAIL'}] {code}  {desc} — {msg}")
        if not ok:
            failed += 1
    if failed:
        print(f"[FAIL] QA-Semantic-Coord cert [226] ({failed} failure(s))"); return 1
    print("[PASS] QA-Semantic-Coord cert [226]"); return 0


if __name__ == "__main__":
    sys.exit(main())
