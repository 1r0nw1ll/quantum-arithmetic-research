"""Canonical failure-algebra anchor for QA failure tag composition.

This module provides a single source of truth for the minimal finite failure
algebra used by QA cert families.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple


FAILURE_ALGEBRA_ANCHOR_REF = "QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1"
FAILURE_TYPES: Tuple[str, ...] = (
    "OK",
    "PARITY_BLOCK",
    "INVARIANT_VIOLATION",
    "OUT_OF_DOMAIN",
)
UNIT = "OK"

# Ordered relation rows (including reflexive edges).
LEQ_ROWS: Tuple[Tuple[str, str], ...] = (
    ("OK", "OK"),
    ("PARITY_BLOCK", "PARITY_BLOCK"),
    ("INVARIANT_VIOLATION", "INVARIANT_VIOLATION"),
    ("OUT_OF_DOMAIN", "OUT_OF_DOMAIN"),
    ("OK", "PARITY_BLOCK"),
    ("OK", "INVARIANT_VIOLATION"),
    ("OK", "OUT_OF_DOMAIN"),
    ("PARITY_BLOCK", "OUT_OF_DOMAIN"),
    ("INVARIANT_VIOLATION", "OUT_OF_DOMAIN"),
)

# Canonical triangular join table rows.
JOIN_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("OK", "OK", "OK"),
    ("OK", "PARITY_BLOCK", "PARITY_BLOCK"),
    ("OK", "INVARIANT_VIOLATION", "INVARIANT_VIOLATION"),
    ("OK", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN"),
    ("PARITY_BLOCK", "PARITY_BLOCK", "PARITY_BLOCK"),
    ("PARITY_BLOCK", "INVARIANT_VIOLATION", "OUT_OF_DOMAIN"),
    ("PARITY_BLOCK", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN"),
    ("INVARIANT_VIOLATION", "INVARIANT_VIOLATION", "INVARIANT_VIOLATION"),
    ("INVARIANT_VIOLATION", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN"),
    ("OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN"),
)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _join_lookup(a: str, b: str) -> str:
    for x, y, z in JOIN_ROWS:
        if (x == a and y == b) or (x == b and y == a):
            return z
    raise KeyError(f"join undefined for ({a},{b})")


def join(fail_a: str, fail_b: str) -> str:
    """Return canonical failure-tag join fail_a ∨ fail_b."""
    return _join_lookup(fail_a, fail_b)


def compose(fail_a: str, fail_b: str) -> str:
    """Canonical sequencing operator for failure tags (compose = join for v1)."""
    return join(fail_a, fail_b)


def compose_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for a in FAILURE_TYPES:
        for b in FAILURE_TYPES:
            rows.append({"a": a, "b": b, "comp": compose(a, b)})
    return rows


def build_failure_algebra_payload() -> Dict[str, Any]:
    """Canonical payload used for optional soft-pin verification in other families."""
    return {
        "anchor_ref": FAILURE_ALGEBRA_ANCHOR_REF,
        "carrier": list(FAILURE_TYPES),
        "leq": [{"a": a, "b": b} for a, b in LEQ_ROWS],
        "join_table": [{"a": a, "b": b, "join": j} for a, b, j in JOIN_ROWS],
        "compose_table": compose_rows(),
        "unit": UNIT,
    }


def compute_failure_algebra_anchor_rollup_sha256() -> str:
    return hashlib.sha256(_canonical_json(build_failure_algebra_payload()).encode("utf-8")).hexdigest()


FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256 = compute_failure_algebra_anchor_rollup_sha256()
