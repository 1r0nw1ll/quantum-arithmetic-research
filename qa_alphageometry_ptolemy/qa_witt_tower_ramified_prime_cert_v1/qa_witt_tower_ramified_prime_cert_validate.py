#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical p-adic ramification theory and Jordan canonical form over Z/p^kZ; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 (ramified extensions, e=2); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano period pi(5)=20); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (primitive roots, Hensel lifting) -->
"""QA Witt Tower Ramified Prime Closed Form -- cert [434].

Closes the ONE remaining gap left open by the entire Witt-tower ladder.
Every cert in this chain -- [389], [421], [424], [429], [431], [432],
[433] -- restricts its closed forms to p != 5, explicitly excluding the
ramified prime (the only p for which x^2-x-1 has a DOUBLE root mod p,
rather than two distinct roots (split) or none (inert)). [432]'s own
comment states this plainly: "no case analysis on prime class
(inert/split/ramified) is needed" for the embedding ISOMORPHISM, but that
cert (and [433]) never derive the actual period/multiplicity closed form
for p=5 -- only a single fixture-level spot check that the commuting
square still holds.

WHY p=5 IS STRUCTURALLY DIFFERENT:
  sigma_m(a,b) = (a+b mod m, a mod m) is driven by M = [[1,1],[1,0]],
  char. poly x^2-x-1. Mod 5, x^2-x-1 = (x-3)^2 -- a DOUBLE root at the
  primitive root 3. Unlike the split case (two distinct simple roots,
  diagonalizable) or the inert case (irreducible, diagonalizable over the
  unramified quadratic extension), M mod 5 is NOT diagonalizable: it is a
  genuine 2x2 JORDAN BLOCK, M = 3*I + N mod 5, with N^2 = 0 mod 5 and
  N != 0 mod 5 (verified directly below). None of the split/inert
  eigen-stratification arguments in [432]/[433] apply to a Jordan block.

CLAIM 1 (PERIOD_SET_LAW_RAMIFIED): the period set of sigma on
(Z/5^k Z)^2 is exactly

    {1, 4}  union  { 4 * 5^L : L = 1, ..., k }

for every k >= 1. The fixed point (period 1) and a single eigenline
(period 4, the literal kernel of N -- the one direction Hensel-lifts
exactly) are the base case; every other period is 4*5^L for some
L <= k. (Brute-force orbit decomposition, k=1..5 in self-test, k=6 cited
as additional dev-time evidence.)

CLAIM 2 (BIRTH_JUMP_FREEZE_MULTIPLICITY): for L >= 1, define
period P_L = 4*5^L. Its orbit count as a function of tower level k is

    count(P_L, k) = 0                   if k < L   (not yet born)
    count(P_L, k) = 5^(L-1)             if k == L  (birth)
    count(P_L, k) = 6 * 5^(L-1)         if k > L   (one jump, then frozen)

Unlike the split/inert case (where a new period appears once and its
count freezes immediately, [433]'s C1), a Jordan-block period is born
with count 5^(L-1), then receives exactly ONE more batch of 5^L new
orbits at level L+1 (total 5^(L-1) + 5^L = 6*5^(L-1)), then freezes for
good. This extra "delayed jump" is exactly what the Jordan block's
nilpotent part forces: the genuinely new (non-5-divisible) vectors at
level L+1 split into two strata sharing the SAME two period values
(4*5^L and 4*5^(L+1)) rather than producing one brand-new period each
level, because there is only one eigenvalue (with multiplicity 2), not
two as in the split-unequal case.

CLAIM 3 (EIGENLINE_PERSISTENCE): periods 1 and 4 each have orbit count
exactly 1 at EVERY level k >= 1, with no birth/jump behavior at all --
consistent with (and a direct instance of) [432]'s embedding-isomorphism
mechanism, which that cert proves holds unconditionally regardless of
prime class. This confirms p=5 is NOT an exception to [432]'s mechanism;
it is only the NEW-part multiplicity law (Claim 2, novel to this cert)
that differs from the split/inert case.

CLAIM 4 (JORDAN_BLOCK_NONDEGENERACY): the algebraic facts underlying
Claims 1-2 -- (a) N = M - 3*I mod 5 is nonzero but N^2 = 0 mod 5 (genuine
non-split Jordan block, not an accidental diagonalization), and (b) 3 is
a primitive root mod 5^k for every k (ord(3 mod 5^k) = 4*5^(k-1), no
Wieferich-like stalling) -- are checked directly via fast modular
arithmetic (no brute-force orbit enumeration) up to k=30.

LINEAGE: completes the prime-class trichotomy left open by [389],
[421], [424], [429], [431], [432], [433] (all restrict p != 5). Uses the
identical map sigma_m as every cert in this chain.

PRIMARY SOURCES:
  Serre (1979) doi.org/10.1007/978-1-4757-5673-9 -- ramification theory in local fields
  Wall (1960) doi.org/10.1080/00029890.1960.11989541 -- Pisano period table (pi(5)=20)
  Ireland & Rosen (1990) ISBN 978-0-387-97329-6 -- primitive roots, Hensel lifting
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Core dynamics (identical map to certs [389], [432], [433])
# ---------------------------------------------------------------------------

def _sigma(a: int, b: int, m: int) -> tuple[int, int]:
    return (a + b) % m, a % m


def _compute_periods(p: int, power: int) -> dict[int, int]:
    """Return {period: orbit_count} for sigma on (Z/p^power Z)^2."""
    m = p ** power
    counts: dict[int, int] = {}
    visited = bytearray(m * m)
    for a in range(m):
        for b in range(m):
            if visited[a * m + b]:
                continue
            ca, cb = a, b
            k = 0
            while True:
                visited[ca * m + cb] = 1
                ca, cb = _sigma(ca, cb, m)
                k += 1
                if ca == a and cb == b:
                    break
            counts[k] = counts.get(k, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# C1: period set law for p=5
# ---------------------------------------------------------------------------

def _predicted_period_set(k: int) -> set[int]:
    return {1, 4} | {4 * 5 ** L for L in range(1, k + 1)}


def _check_period_set_law_ramified(k_list: list[int]) -> dict[str, Any]:
    failures = []
    for k in k_list:
        d = _compute_periods(5, k)
        predicted = _predicted_period_set(k)
        actual = set(d.keys())
        if actual != predicted:
            failures.append(f"k={k}: predicted={sorted(predicted)} actual={sorted(actual)}")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C2: birth/jump/freeze multiplicity law
# ---------------------------------------------------------------------------

def _predicted_count_ramified(L: int, k: int, p: int = 5) -> int:
    if k < L:
        return 0
    if k == L:
        return p ** (L - 1)
    return (p + 1) * p ** (L - 1)


def _check_birth_jump_freeze_multiplicity(k_list: list[int]) -> dict[str, Any]:
    failures = []
    for k in k_list:
        d = _compute_periods(5, k)
        for L in range(1, k + 1):
            period_L = 4 * 5 ** L
            expected = _predicted_count_ramified(L, k)
            actual = d.get(period_L, 0)
            if actual != expected:
                failures.append(
                    f"k={k},L={L}: period={period_L} expected_count={expected} actual_count={actual}"
                )
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C3: eigenline persistence (periods 1 and 4, frozen at count 1 forever)
# ---------------------------------------------------------------------------

def _check_eigenline_persistence(k_list: list[int]) -> dict[str, Any]:
    failures = []
    for k in k_list:
        d = _compute_periods(5, k)
        if d.get(1) != 1:
            failures.append(f"k={k}: period=1 count={d.get(1)} expected=1")
        if d.get(4) != 1:
            failures.append(f"k={k}: period=4 count={d.get(4)} expected=1")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C4: Jordan-block nondegeneracy (fast modular arithmetic, no brute force)
# ---------------------------------------------------------------------------

def _check_jordan_block_nondegeneracy(k_max: int) -> dict[str, Any]:
    failures: list[str] = []

    # M = [[1,1],[1,0]] mod 5; N = M - 3*I mod 5; expect N != 0, N^2 == 0.
    n00, n01, n10, n11 = (1 - 3) % 5, 1 % 5, 1 % 5, (0 - 3) % 5
    if (n00, n01, n10, n11) == (0, 0, 0, 0):
        failures.append("N is zero mod 5 -- not a genuine Jordan block")
    sq00 = (n00 * n00 + n01 * n10) % 5
    sq01 = (n00 * n01 + n01 * n11) % 5
    sq10 = (n10 * n00 + n11 * n10) % 5
    sq11 = (n10 * n01 + n11 * n11) % 5
    if (sq00, sq01, sq10, sq11) != (0, 0, 0, 0):
        failures.append(f"N^2 mod 5 != 0: {(sq00, sq01, sq10, sq11)}")

    for k in range(1, k_max + 1):
        mod = 5 ** k
        order = 4 * 5 ** (k - 1)
        for q in (2, 5):
            while order % q == 0 and pow(3, order // q, mod) == 1:
                order //= q
        predicted = 4 * 5 ** (k - 1)
        if order != predicted:
            failures.append(f"k={k}: ord(3 mod 5^{k})={order} predicted={predicted}")

    return {"ok": not failures, "failures": failures, "k_max": k_max}


# ---------------------------------------------------------------------------
# Self-test (--self-test mode)
# ---------------------------------------------------------------------------

C1_K_LIST = [1, 2, 3, 4, 5]
C2_K_LIST = [2, 3, 4, 5]
C3_K_LIST = [1, 2, 3, 4, 5]
C4_K_MAX = 30


def run_self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    r1 = _check_period_set_law_ramified(C1_K_LIST)
    checks["PERIOD_SET_LAW_RAMIFIED"] = r1["ok"]
    if not r1["ok"]:
        details["PERIOD_SET_LAW_RAMIFIED"] = r1["failures"]

    r2 = _check_birth_jump_freeze_multiplicity(C2_K_LIST)
    checks["BIRTH_JUMP_FREEZE_MULTIPLICITY"] = r2["ok"]
    if not r2["ok"]:
        details["BIRTH_JUMP_FREEZE_MULTIPLICITY"] = r2["failures"]

    r3 = _check_eigenline_persistence(C3_K_LIST)
    checks["EIGENLINE_PERSISTENCE"] = r3["ok"]
    if not r3["ok"]:
        details["EIGENLINE_PERSISTENCE"] = r3["failures"]

    r4 = _check_jordan_block_nondegeneracy(C4_K_MAX)
    checks["JORDAN_BLOCK_NONDEGENERACY"] = r4["ok"]
    details["JORDAN_BLOCK_NONDEGENERACY_COVERAGE"] = {"k_max": r4["k_max"]}
    if not r4["ok"]:
        details["JORDAN_BLOCK_NONDEGENERACY"] = r4["failures"]

    ok = all(checks.values())
    result: dict[str, Any] = {"ok": ok, "checks": checks}
    if details:
        result["details"] = details
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "id": "PERIOD_SET_P5_K3",
        "description": "p=5,k=3: period set is exactly {1,4,20,100,500}",
        "expected": True,
        "fn": lambda: set(_compute_periods(5, 3).keys()) == {1, 4, 20, 100, 500},
    },
    {
        "id": "BIRTH_COUNT_P5_K4_L4",
        "description": "p=5,k=4: period=4*5^4=2500 is freshly born with count 5^3=125",
        "expected": True,
        "fn": lambda: _compute_periods(5, 4).get(4 * 5 ** 4) == 125,
    },
    {
        "id": "FROZEN_COUNT_P5_K4_L3",
        "description": "p=5,k=4: period=4*5^3=500 (born at k=3 with count 25) has jumped+frozen to 6*5^2=150",
        "expected": True,
        "fn": lambda: _compute_periods(5, 4).get(4 * 5 ** 3) == 150,
    },
    {
        "id": "EIGENLINE_P5_K5",
        "description": "p=5,k=5: periods 1 and 4 both still have count exactly 1",
        "expected": True,
        "fn": lambda: _compute_periods(5, 5).get(1) == 1 and _compute_periods(5, 5).get(4) == 1,
    },
    {
        "id": "WRONG_NO_JUMP_ASSUMPTION",
        "description": "DESIGNED FAIL: if periods froze immediately at birth like the split/inert case ([433] C1), count(500) at k=4 would stay at its birth value 25 -- it does not, it jumps to 150",
        "expected": False,
        "fn": lambda: _compute_periods(5, 4).get(4 * 5 ** 3) == 25,
    },
]


def run_fixtures() -> dict[str, Any]:
    results = {}
    for f in FIXTURES:
        try:
            actual = f["fn"]()
            passed = actual == f["expected"]
        except Exception as e:
            actual = f"ERROR: {e}"
            passed = False
        results[f["id"]] = {
            "description": f["description"],
            "expected": f["expected"],
            "actual": actual,
            "passed": passed,
        }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QA Witt Tower Ramified Prime Closed Form cert [434] validator"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()

    if args.fixtures:
        out = {"fixtures": run_fixtures()}
        print(json.dumps(out, indent=2))
        sys.exit(0)

    result = run_self_test()
    if args.self_test:
        fixture_result = run_fixtures()
        result["fixtures"] = fixture_result
        fixture_pass = sum(1 for v in fixture_result.values() if v["passed"])
        fixture_total = len(fixture_result)
        result["fixture_summary"] = f"{fixture_pass}/{fixture_total} passed"

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
