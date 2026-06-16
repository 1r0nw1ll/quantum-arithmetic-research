#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical p-adic number theory and Witt vector structure; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 SS II.4 (Witt vectors, tower structure); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano period tower); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (Hensel lifting, eigenvalue orders) -->
"""QA Witt Tower Scaling Isomorphism — cert [432].

Closes two gaps that cert [389] (QA Witt Tower Orbit Refinement) explicitly
states it does NOT cover:
  (a) "Does not identify WHICH period-k orbits at level p^2 are old-part vs
      new-part -- only that both classes exist and have the stated periods."
  (b) "Does not certify the tower structure at level p^3 or beyond."
  (c) [389] restricts to p != 5 (excludes the ramified prime).

CLAIM 1 (SCALING EMBEDDING, proved, not just verified):
  Let sigma_m(a,b) = (a+b mod m, a mod m) be the Fibonacci pair-shift (the
  same map as [389]). For ANY prime p (including p=5) and ANY k >= 2, define

      iota(a,b) = (p*a mod p^k, p*b mod p^k)        for (a,b) in (Z/p^(k-1)Z)^2

  Then iota is an injective map whose image is EXACTLY the "old part"
  sublattice {(x,y) in (Z/p^kZ)^2 : p|x and p|y}, and it intertwines the
  dynamics at consecutive tower levels:

      sigma_{p^k} ( iota(a,b) ) = iota ( sigma_{p^(k-1)}(a,b) )      -- ALWAYS

  This holds unconditionally because sigma is LINEAR (a matrix map): for any
  integers a,b, p*((a+b) mod p^(k-1)) === p*a + p*b (mod p^k) identically --
  no case analysis on prime class (inert/split/ramified) is needed. This
  answers gap (a): old-part = precisely the image of iota, with EXACT
  multiplicity preservation (not just matching periods, but matching orbit
  COUNTS), and it removes restriction (c): nothing here excludes p=5.

CLAIM 2 (TOWER INDUCTION, answers gap (b) for the structural part):
  Because the intertwining holds for every consecutive pair (p^(k-1), p^k),
  it composes: the old-part of level p^k is isomorphic to the FULL level
  p^(k-1) decomposition, which itself splits into its own old/new parts via
  the same map one level down. This gives an unconditional structural
  induction up the tower (verified here through k=3). What remains open
  (and is NOT claimed here) is the classical, still-conjectural fact that
  the genuinely NEW periods introduced at each level are nonzero multiples
  of p times the prior level's nontrivial periods -- this is the
  non-degenerate / non-Wall-Sun-Sun assumption already used throughout the
  rest of this cert chain (e.g. [429], [431]), not re-derived here.

CLAIM 3 (EXACT NEW-PART MULTIPLICITY FORMULA, split-unequal primes, proved):
  [389] establishes which TWO new periods occur at level p^2 for a
  split-unequal prime p (p*ord_min and p*ord_max, where ord_min < ord_max
  are the multiplicative orders mod p of the two roots of x^2-x-1), but does
  not give their orbit COUNTS. Total-count conservation alone cannot recover
  both counts (one equation, two unknowns). Stratifying the new part by
  eigen-coordinate (c1,c2) -- exactly one of which can vanish on the nose,
  not just be a multiple of p -- gives a second equation and pins down both
  counts in closed form:

      count_new(p*ord_min) = (p-1) / ord_min
      count_new(p*ord_max) = (p-1)*(p^2+p-1) / ord_max

  Both divide exactly (ord_min and ord_max both divide p-1 by Lagrange, so
  the right-hand sides are integers). For inert and split-equal primes
  there is only ONE new period (p*pi(p)), so the analogous count is forced
  by total-count conservation alone: count_new(p*pi(p)) = p * count_old(pi(p))
  -- a corollary of CLAIM 1, not an independent new fact, included here only
  for completeness (C5).

LINEAGE: extends [389] (Witt tower refinement law, period SETS only) and
[385] (orbit/prime-ideal filtration at a single fixed level). [387] (Witt
carry invariant, p=3 inert) and [388] (split eigenspace, level p) are
unaffected -- this cert is about the level-to-level scaling map, not the
internal coset structure within one level.

PRIMARY SOURCES:
  Wall (1960) doi.org/10.1080/00029890.1960.11989541 -- Pisano tower
  Serre (1979) doi.org/10.1007/978-1-4757-5673-9 -- local fields, Witt vectors
  Ireland & Rosen (1990) ISBN 978-0-387-97329-6 -- eigenvalue orders, Lagrange
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Core dynamics (identical map to cert [389])
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


def _compute_periods_with_members(p: int, power: int) -> tuple[dict[int, int], dict[int, list[list[tuple[int, int]]]]]:
    m = p ** power
    counts: dict[int, int] = {}
    members: dict[int, list[list[tuple[int, int]]]] = {}
    visited = bytearray(m * m)
    for a in range(m):
        for b in range(m):
            if visited[a * m + b]:
                continue
            orbit = []
            ca, cb = a, b
            k = 0
            while True:
                visited[ca * m + cb] = 1
                orbit.append((ca, cb))
                ca, cb = _sigma(ca, cb, m)
                k += 1
                if ca == a and cb == b:
                    break
            counts[k] = counts.get(k, 0) + 1
            members.setdefault(k, []).append(orbit)
    return counts, members


def _iota(a: int, b: int, p: int, k: int) -> tuple[int, int]:
    pk = p ** k
    return (p * a) % pk, (p * b) % pk


def _poly_roots_mod_p(p: int) -> list[int]:
    return [x for x in range(p) if (x * x - x - 1) % p == 0]


def _multiplicative_order(a: int, p: int) -> int:
    if a % p == 0:
        return 0
    o, cur = 1, a % p
    while cur != 1:
        cur = (cur * a) % p
        o += 1
    return o


def _prime_class(p: int) -> Any:
    """inert / ('split_equal', o1, o2) / ('split_unequal', o1, o2) / ramified"""
    if p == 5:
        return "ramified"
    roots = _poly_roots_mod_p(p)
    if len(roots) == 0:
        return "inert"
    if len(set(roots)) == 2:
        r1, r2 = sorted(roots)
        o1, o2 = _multiplicative_order(r1, p), _multiplicative_order(r2, p)
        return ("split_equal", o1, o2) if o1 == o2 else ("split_unequal", o1, o2)
    return "ramified"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_scaling_embedding_commutes(pk_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """C1: sigma_{p^k}(iota(a,b)) == iota(sigma_{p^(k-1)}(a,b)) for ALL (a,b)."""
    failures = []
    for p, k in pk_pairs:
        m_low = p ** (k - 1)
        m_high = p ** k
        for a in range(m_low):
            for b in range(m_low):
                lhs = _sigma(*_iota(a, b, p, k), m_high)
                rhs = _iota(*_sigma(a, b, m_low), p, k)
                if lhs != rhs:
                    failures.append(f"p={p},k={k}: (a,b)=({a},{b}) lhs={lhs} rhs={rhs}")
    return {"ok": not failures, "failures": failures[:10]}


def _check_old_part_is_exact_image(pk_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """C2: image of iota == exactly the sublattice {p|x and p|y}, injectively."""
    failures = []
    for p, k in pk_pairs:
        m_low = p ** (k - 1)
        m_high = p ** k
        image = set()
        for a in range(m_low):
            for b in range(m_low):
                image.add(_iota(a, b, p, k))
        if len(image) != m_low * m_low:
            failures.append(f"p={p},k={k}: iota not injective, |image|={len(image)} != {m_low*m_low}")
            continue
        sublattice = {(x, y) for x in range(0, m_high, p) for y in range(0, m_high, p)}
        if image != sublattice:
            failures.append(f"p={p},k={k}: image != sublattice (|image|={len(image)}, |sublattice|={len(sublattice)})")
    return {"ok": not failures, "failures": failures}


def _check_isomorphic_orbit_structure(pk_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """C3: old-part period/count decomposition at level p^k == FULL decomposition at level p^(k-1), exactly."""
    failures = []
    for p, k in pk_pairs:
        d_low = _compute_periods(p, k - 1)
        _, members_high = _compute_periods_with_members(p, k)
        d_old_part: dict[int, int] = {}
        for per, orbits in members_high.items():
            n_old = sum(1 for orb in orbits if all(x % p == 0 and y % p == 0 for x, y in orb))
            if n_old:
                d_old_part[per] = n_old
        if d_old_part != d_low:
            failures.append(f"p={p},k={k}: old_part={d_old_part} != level(k-1)={d_low}")
    return {"ok": not failures, "failures": failures}


def _check_split_unequal_new_multiplicity(primes: list[int]) -> dict[str, Any]:
    """C4: count_new(p*ord_min) = (p-1)/ord_min ; count_new(p*ord_max) = (p-1)(p^2+p-1)/ord_max."""
    failures = []
    for p in primes:
        cls = _prime_class(p)
        if not (isinstance(cls, tuple) and cls[0] == "split_unequal"):
            failures.append(f"p={p}: not split_unequal (class={cls})")
            continue
        _, o1, o2 = cls
        ord_min, ord_max = min(o1, o2), max(o1, o2)
        d2 = _compute_periods(p, 2)
        expected_min = (p - 1) // ord_min
        expected_max = (p - 1) * (p * p + p - 1) // ord_max
        actual_min = d2.get(p * ord_min)
        actual_max = d2.get(p * ord_max)
        if actual_min != expected_min or actual_max != expected_max:
            failures.append(
                f"p={p}: ord_min={ord_min} ord_max={ord_max} "
                f"expected=({expected_min},{expected_max}) actual=({actual_min},{actual_max})"
            )
    return {"ok": not failures, "failures": failures}


def _check_inert_split_equal_new_multiplicity(primes: list[int]) -> dict[str, Any]:
    """C5: count_new(p*pi(p)) = p * count_old(pi(p)) for inert/split_equal primes (corollary of C1)."""
    failures = []
    for p in primes:
        cls = _prime_class(p)
        is_single_new_period = cls == "inert" or (isinstance(cls, tuple) and cls[0] == "split_equal")
        if not is_single_new_period:
            failures.append(f"p={p}: expected inert or split_equal (class={cls})")
            continue
        d1 = _compute_periods(p, 1)
        d2 = _compute_periods(p, 2)
        pi_p = max(k for k in d1 if k != 1)
        count_old = d1[pi_p]
        expected_new = p * count_old
        actual_new = d2.get(p * pi_p)
        if actual_new != expected_new:
            failures.append(f"p={p}: pi(p)={pi_p} expected_new={expected_new} actual_new={actual_new}")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# Self-test (--self-test mode)
# ---------------------------------------------------------------------------

K2_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]  # p=5 included (excluded by [389])
K2_PAIRS = [(p, 2) for p in K2_PRIMES]
K3_PRIMES = [2, 3, 5, 7]  # cost-bounded (p^3 state space = p^6 visits)
K3_PAIRS = [(p, 3) for p in K3_PRIMES]
ALL_PK_PAIRS = K2_PAIRS + K3_PAIRS

SPLIT_UNEQUAL_PRIMES = [11, 19, 29, 31]
INERT_SPLIT_EQUAL_PRIMES = [7, 13, 17, 41]  # 7,13,17 inert; 41 split_equal


def run_self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    r1 = _check_scaling_embedding_commutes(ALL_PK_PAIRS)
    checks["SCALING_EMBEDDING_COMMUTES"] = r1["ok"]
    if not r1["ok"]:
        details["SCALING_EMBEDDING_COMMUTES"] = r1["failures"]

    r2 = _check_old_part_is_exact_image(ALL_PK_PAIRS)
    checks["OLD_PART_IS_EXACT_IMAGE"] = r2["ok"]
    if not r2["ok"]:
        details["OLD_PART_IS_EXACT_IMAGE"] = r2["failures"]

    r3 = _check_isomorphic_orbit_structure(ALL_PK_PAIRS)
    checks["ISOMORPHIC_ORBIT_STRUCTURE"] = r3["ok"]
    if not r3["ok"]:
        details["ISOMORPHIC_ORBIT_STRUCTURE"] = r3["failures"]

    r4 = _check_split_unequal_new_multiplicity(SPLIT_UNEQUAL_PRIMES)
    checks["SPLIT_UNEQUAL_NEW_MULTIPLICITY"] = r4["ok"]
    if not r4["ok"]:
        details["SPLIT_UNEQUAL_NEW_MULTIPLICITY"] = r4["failures"]

    r5 = _check_inert_split_equal_new_multiplicity(INERT_SPLIT_EQUAL_PRIMES)
    checks["INERT_SPLIT_EQUAL_NEW_MULTIPLICITY"] = r5["ok"]
    if not r5["ok"]:
        details["INERT_SPLIT_EQUAL_NEW_MULTIPLICITY"] = r5["failures"]

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
        "id": "IOTA_COMMUTES_P7_K2",
        "description": "p=7,k=2: sigma_49(iota(3,4)) == iota(sigma_7(3,4))",
        "expected": True,
        "fn": lambda: _sigma(*_iota(3, 4, 7, 2), 49) == _iota(*_sigma(3, 4, 7), 7, 2),
    },
    {
        "id": "IOTA_COMMUTES_P5_RAMIFIED",
        "description": "p=5 (ramified, excluded by [389]): commuting square still holds",
        "expected": True,
        "fn": lambda: _sigma(*_iota(2, 3, 5, 2), 25) == _iota(*_sigma(2, 3, 5), 5, 2),
    },
    {
        "id": "OLD_PART_P7_K2_MATCHES_LEVEL1",
        "description": "p=7: old-part decomposition at level 49 == full decomposition at level 7",
        "expected": True,
        "fn": lambda: (
            lambda counts_high, members_high: (
                {per: sum(1 for orb in orbs if all(x % 7 == 0 and y % 7 == 0 for x, y in orb))
                 for per, orbs in members_high.items()
                 if any(all(x % 7 == 0 and y % 7 == 0 for x, y in orb) for orb in orbs)}
                == _compute_periods(7, 1)
            )
        )(*_compute_periods_with_members(7, 2)),
    },
    {
        "id": "SPLIT_UNEQUAL_P11_NEW_COUNTS",
        "description": "p=11 split-unequal: count_new(55)=2, count_new(110)=131 (closed form)",
        "expected": True,
        "fn": lambda: _compute_periods(11, 2).get(55) == 2 and _compute_periods(11, 2).get(110) == 131,
    },
    {
        "id": "INERT_P7_NEW_COUNT_IS_P_TIMES_OLD",
        "description": "p=7 inert: count_new(112) = 7 * count_old(16) = 7*3 = 21",
        "expected": True,
        "fn": lambda: _compute_periods(7, 2).get(112) == 21,
    },
    {
        "id": "WRONG_MULTIPLICITY_FORMULA",
        "description": "DESIGNED FAIL: p=11 new-min count is NOT (p-1) = 10",
        "expected": False,
        "fn": lambda: _compute_periods(11, 2).get(55) == 10,
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
        description="QA Witt Tower Scaling Isomorphism cert [432] validator"
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
