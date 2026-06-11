#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical p-adic number theory and Witt vector structure; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 §II.4 (Witt vectors, tower structure); Neukirch (1999) ISBN 978-3-540-65399-8 §II.3 (p-adic valuations and local fields); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano period tower: pi(p^k)=p^(k-1)*pi(p)); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (Hensel lifting and Newton polygons) -->
"""QA Witt Tower Orbit Refinement — cert [389].

Certifies the period-refinement law for the sigma-dynamics on Z[phi]/(p^2)
vs Z[phi]/(p), where sigma(a,b) = (a+b mod m, a) is the Fibonacci shift.

CLAIM (narrow, falsifiable):
  For any prime p != 5, the periods of sigma on Z[phi]/(p^2) are EXACTLY:
    {1} union {k : k is a non-trivial period at level p}
        union {p*k : k is a non-trivial period at level p}
  In symbols: Periods(p^2) = {1} u Periods_nt(p) u p*Periods_nt(p)
  where Periods_nt(p) = Periods(p) - {1}.

COROLLARY — tier count at level p^2 distinguishes prime types:
  (T1) Inert p:            Periods(p) = {1, pi(p)}
                           Periods(p^2) = {1, pi(p), p*pi(p)}  [3 tiers]
  (T2) Split p, unequal root orders:
                           Periods(p) = {1, ord_min, pi(p)}
                           Periods(p^2) = {1, ord_min, pi(p), p*ord_min, p*pi(p)}  [5 tiers]
  (T3) Split p, equal root orders (e.g. p=41):
                           Periods(p) = {1, pi(p)}
                           Periods(p^2) = {1, pi(p), p*pi(p)}  [3 tiers]

KEY STRUCTURAL FACT: period-1 (the zero element) does NOT lift to period-p.
Only non-trivial periods lift. The Witt multiplier is exactly p.

HECKE INTERPRETATION: the map Periods(p) -> Periods(p^2) via k -> p*k is the
local Hecke "level-raising" operator. The period-k orbits at level p^2 that do
NOT come via this map are the "old-part" (embedded from level p). The p*k orbits
are the "new-part" (genuine level-p^2 orbits in the Witt sense). This mirrors
the old-form/new-form decomposition in the Atkin-Lehner theory, where:
  - old-part at level p^2 = image of level-p forms under the two degeneracy maps
  - new-part at level p^2 = forms with conductor exactly p^2

The tier-count difference (5 vs 3) at level p^2 is a FINER discriminant than
the split/inert test at level p: it separates split-unequal (5 tiers) from
both inert and split-equal (3 tiers each).

LINEAGE: extends [388] (split prime orbit geometry at level p) and [387] (Witt
carry sub-orbit for inert p=3). Together [387]+[389] cover the full Witt tower
for both inert and split primes.

PRIMARY SOURCES:
  Wall (1960) doi.org/10.1080/00029890.1960.11989541 — Pisano tower pi(p^k)=p^(k-1)*pi(p)
  Serre (1979) doi.org/10.1007/978-1-4757-5673-9 — local fields, Witt vectors
  Neukirch (1999) ISBN 978-3-540-65399-8 — algebraic number theory, p-adic
  Ireland & Rosen (1990) ISBN 978-0-387-97329-6 — classical number theory
"""

from __future__ import annotations

import argparse
import json
import sys
from math import gcd
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Core dynamics
# ---------------------------------------------------------------------------

def _sigma(a: int, b: int, m: int) -> tuple[int, int]:
    return (a + b) % m, a % m


def _compute_periods(p: int, power: int) -> dict[int, int]:
    """Return {period: orbit_count} for sigma on Z[phi]/(p^power)."""
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


def _poly_roots_mod_p(p: int) -> list[int]:
    """Roots of x^2 - x - 1 mod p."""
    return [x for x in range(p) if (x * x - x - 1) % p == 0]


def _multiplicative_order(a: int, p: int) -> int:
    if a % p == 0:
        return 0
    o, cur = 1, a % p
    while cur != 1:
        cur = (cur * a) % p
        o += 1
    return o


def _prime_class(p: int) -> str:
    """inert / split_unequal / split_equal / ramified"""
    if p == 5:
        return "ramified"
    roots = _poly_roots_mod_p(p)
    if len(roots) == 0:
        return "inert"
    if len(set(roots)) == 2:
        r1, r2 = sorted(roots)
        o1 = _multiplicative_order(r1, p)
        o2 = _multiplicative_order(r2, p)
        return "split_equal" if o1 == o2 else "split_unequal"
    return "ramified"  # double root


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_witt_refinement_law(
    primes: list[int],
) -> dict[str, Any]:
    """C1: Periods(p^2) = {1} u Periods_nt(p) u p*Periods_nt(p) for each p."""
    failures = []
    for p in primes:
        d1 = _compute_periods(p, 1)
        d2 = _compute_periods(p, 2)
        periods1 = set(d1)
        periods_nt1 = periods1 - {1}
        expected2 = {1} | periods_nt1 | {p * k for k in periods_nt1}
        actual2 = set(d2)
        if actual2 != expected2:
            failures.append(
                f"p={p}: expected periods {sorted(expected2)}, got {sorted(actual2)}"
            )
    return {"ok": not failures, "failures": failures}


def _check_tier_counts(
    inert_primes: list[int],
    split_unequal_primes: list[int],
    split_equal_primes: list[int],
) -> dict[str, Any]:
    """C2: tier count at level p^2 matches type: inert/split-equal=3, split-unequal=5."""
    failures = []
    for p, expected_tiers, cls in (
        [(p, 3, "inert") for p in inert_primes]
        + [(p, 5, "split_unequal") for p in split_unequal_primes]
        + [(p, 3, "split_equal") for p in split_equal_primes]
    ):
        d2 = _compute_periods(p, 2)
        actual_tiers = len(d2)
        if actual_tiers != expected_tiers:
            failures.append(
                f"p={p} ({cls}): expected {expected_tiers} tiers, got {actual_tiers}: {sorted(d2)}"
            )
    return {"ok": not failures, "failures": failures}


def _check_witt_multiplier_is_p(primes: list[int]) -> dict[str, Any]:
    """C3: Every period in Periods_new(p^2) is exactly p times a Periods_nt(p) period."""
    failures = []
    for p in primes:
        d1 = _compute_periods(p, 1)
        d2 = _compute_periods(p, 2)
        periods_nt1 = set(d1) - {1}
        for k in set(d2) - {1}:
            if k not in periods_nt1 and k % p != 0:
                failures.append(f"p={p}: period {k} in level-p^2 is neither in Periods_nt(p) nor a p-multiple")
            if k not in periods_nt1 and k % p == 0 and k // p not in periods_nt1:
                failures.append(f"p={p}: period {k} = p*{k//p} but {k//p} not in Periods_nt(p)")
    return {"ok": not failures, "failures": failures}


def _check_zero_does_not_lift(primes: list[int]) -> dict[str, Any]:
    """C4: period-p is NOT a period at level p^2 (zero does not lift to period-p)."""
    failures = []
    for p in primes:
        d2 = _compute_periods(p, 2)
        if p in d2:
            failures.append(f"p={p}: period-p={p} unexpectedly appears at level p^2")
    return {"ok": not failures, "failures": failures}


def _check_orbit_counts(primes: list[int]) -> dict[str, Any]:
    """C5: orbit counts at level p^2 are consistent (total elements = p^4)."""
    failures = []
    for p in primes:
        d2 = _compute_periods(p, 2)
        total = sum(k * v for k, v in d2.items())
        expected = p ** 4
        if total != expected:
            failures.append(f"p={p}: total elements {total} != p^4={expected}")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# Self-test (--self-test mode)
# ---------------------------------------------------------------------------

INERT_PRIMES = [7, 13, 17]
SPLIT_UNEQUAL_PRIMES = [11, 19, 29, 31]
SPLIT_EQUAL_PRIMES = [41]
ALL_PRIMES = INERT_PRIMES + SPLIT_UNEQUAL_PRIMES + SPLIT_EQUAL_PRIMES


def run_self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    r1 = _check_witt_refinement_law(ALL_PRIMES)
    checks["WITT_REFINEMENT_LAW"] = r1["ok"]
    if not r1["ok"]:
        details["WITT_REFINEMENT_LAW"] = r1["failures"]

    r2 = _check_tier_counts(INERT_PRIMES, SPLIT_UNEQUAL_PRIMES, SPLIT_EQUAL_PRIMES)
    checks["TIER_COUNTS"] = r2["ok"]
    if not r2["ok"]:
        details["TIER_COUNTS"] = r2["failures"]

    r3 = _check_witt_multiplier_is_p(ALL_PRIMES)
    checks["WITT_MULTIPLIER_IS_P"] = r3["ok"]
    if not r3["ok"]:
        details["WITT_MULTIPLIER_IS_P"] = r3["failures"]

    r4 = _check_zero_does_not_lift(ALL_PRIMES)
    checks["ZERO_DOES_NOT_LIFT"] = r4["ok"]
    if not r4["ok"]:
        details["ZERO_DOES_NOT_LIFT"] = r4["failures"]

    r5 = _check_orbit_counts(ALL_PRIMES)
    checks["ORBIT_COUNTS_P4"] = r5["ok"]
    if not r5["ok"]:
        details["ORBIT_COUNTS_P4"] = r5["failures"]

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
        "id": "INERT_P7_THREE_TIERS",
        "description": "p=7 (inert): level p^2 has exactly 3 period tiers {1,16,112}",
        "expected": True,
        "fn": lambda: (
            set(_compute_periods(7, 2).keys()) == {1, 16, 112}
        ),
    },
    {
        "id": "SPLIT_P11_FIVE_TIERS",
        "description": "p=11 (split-unequal): level p^2 has exactly 5 period tiers {1,5,10,55,110}",
        "expected": True,
        "fn": lambda: (
            set(_compute_periods(11, 2).keys()) == {1, 5, 10, 55, 110}
        ),
    },
    {
        "id": "SPLIT_EQUAL_P41_THREE_TIERS",
        "description": "p=41 (split-equal): level p^2 has exactly 3 tiers {1,40,1640}",
        "expected": True,
        "fn": lambda: (
            set(_compute_periods(41, 2).keys()) == {1, 40, 1640}
        ),
    },
    {
        "id": "WITT_MULTIPLIER_P7",
        "description": "p=7 inert: new period at level p^2 is 7*16=112, not 2*16 or 3*16",
        "expected": True,
        "fn": lambda: (
            112 in _compute_periods(7, 2) and 32 not in _compute_periods(7, 2)
        ),
    },
    {
        "id": "ZERO_NOT_PERIOD_P_AT_P2",
        "description": "p=11: period 11 does NOT appear at level 121 (zero does not lift)",
        "expected": True,
        "fn": lambda: 11 not in _compute_periods(11, 2),
    },
    {
        "id": "WRONG_TIER_COUNT",
        "description": "DESIGNED FAIL: inert p=7 at level p^2 does NOT have 5 tiers",
        "expected": False,
        "fn": lambda: len(_compute_periods(7, 2)) == 5,
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
        description="QA Witt Tower Orbit Refinement cert [389] validator"
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
