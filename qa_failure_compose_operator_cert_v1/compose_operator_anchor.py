"""Canonical compose(Fi, Fj, form) anchor for Family [87]."""
from __future__ import annotations

from typing import Dict, List, Tuple

CARRIER: Tuple[str, ...] = (
    "OK",
    "PARITY_BLOCK",
    "INVARIANT_VIOLATION",
    "OUT_OF_DOMAIN",
)
FORMS: Tuple[str, ...] = ("serial", "parallel", "feedback")

_JOIN_TRIANGULAR: Dict[Tuple[str, str], str] = {
    ("OK", "OK"): "OK",
    ("OK", "PARITY_BLOCK"): "PARITY_BLOCK",
    ("OK", "INVARIANT_VIOLATION"): "INVARIANT_VIOLATION",
    ("OK", "OUT_OF_DOMAIN"): "OUT_OF_DOMAIN",
    ("PARITY_BLOCK", "PARITY_BLOCK"): "PARITY_BLOCK",
    ("PARITY_BLOCK", "INVARIANT_VIOLATION"): "OUT_OF_DOMAIN",
    ("PARITY_BLOCK", "OUT_OF_DOMAIN"): "OUT_OF_DOMAIN",
    ("INVARIANT_VIOLATION", "INVARIANT_VIOLATION"): "INVARIANT_VIOLATION",
    ("INVARIANT_VIOLATION", "OUT_OF_DOMAIN"): "OUT_OF_DOMAIN",
    ("OUT_OF_DOMAIN", "OUT_OF_DOMAIN"): "OUT_OF_DOMAIN",
}


def join(f_i: str, f_j: str) -> str:
    key = (f_i, f_j)
    if key in _JOIN_TRIANGULAR:
        return _JOIN_TRIANGULAR[key]
    return _JOIN_TRIANGULAR[(f_j, f_i)]


def _feedback_escalate(f: str) -> str:
    return "OK" if f == "OK" else "OUT_OF_DOMAIN"


def compose(f_i: str, f_j: str, form: str) -> str:
    """Formal composition operator compose(Fi, Fj, form).

    serial:   Fi ∨ Fj
    parallel: Fi ∨ Fj
    feedback: escalate(Fi) ∨ escalate(Fj), with escalate(non-OK)=OUT_OF_DOMAIN
    """
    if form in {"serial", "parallel"}:
        return join(f_i, f_j)
    if form == "feedback":
        return join(_feedback_escalate(f_i), _feedback_escalate(f_j))
    raise KeyError(f"unknown form: {form}")


def compose_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for form in FORMS:
        for a in CARRIER:
            for b in CARRIER:
                rows.append({"form": form, "a": a, "b": b, "comp": compose(a, b, form)})
    return rows
