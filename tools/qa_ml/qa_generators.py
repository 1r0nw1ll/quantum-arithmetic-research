"""QA generator transformations on (b, e) discrete states under modulus m.

Four generators (matching the QA-ML synthesis spec and qa_structural_algebra
naming):

    sigma   : (b, e) -> (e, qa_step second-coord)        always defined
    mu      : (b, e) -> (e, b)                           always defined
    lambda_2: (b, e) -> (2*b, 2*e)                       defined when 2b <= m and 2e <= m
    nu      : (b, e) -> (b // 2, e // 2)                 defined when b and e are both even

sigma uses qa_orbit_rules.qa_step so the result lands in {1,...,m} (A1).
lambda_2 and nu return None outside their domain (no mod wrapping — preserves
generator identity for the reachability-graph builder).

QA_COMPLIANCE = "qa_ml_generators — int (b, e, m) -> tuple|None; A1 via qa_step"
"""

from __future__ import annotations

from typing import Optional

from qa_orbit_rules import qa_step


def sigma(b: int, e: int, m: int) -> tuple[int, int]:
    """sigma: (b, e) -> (e, ((b+e-1) % m) + 1)  — A1-compliant Fibonacci step."""
    return qa_step(b, e, m)


def mu(b: int, e: int, m: int) -> tuple[int, int]:
    """mu: (b, e) -> (e, b)  — coordinate swap."""
    assert 1 <= b <= m and 1 <= e <= m, f"A1: ({b},{e}) out of {{1,...,{m}}}"
    return (e, b)


def lambda_2(b: int, e: int, m: int) -> Optional[tuple[int, int]]:
    """lambda_2: (b, e) -> (2*b, 2*e)  — defined only when output stays in {1,...,m}."""
    assert 1 <= b <= m and 1 <= e <= m, f"A1: ({b},{e}) out of {{1,...,{m}}}"
    new_b, new_e = 2 * b, 2 * e
    if new_b > m or new_e > m:
        return None
    return (new_b, new_e)


def nu(b: int, e: int, m: int) -> Optional[tuple[int, int]]:
    """nu: (b, e) -> (b // 2, e // 2)  — defined only when both coords are even."""
    assert 1 <= b <= m and 1 <= e <= m, f"A1: ({b},{e}) out of {{1,...,{m}}}"
    if b % 2 != 0 or e % 2 != 0:
        return None
    return (b // 2, e // 2)


GENERATORS: dict[str, callable] = {
    "sigma": sigma,
    "mu": mu,
    "lambda_2": lambda_2,
    "nu": nu,
}
