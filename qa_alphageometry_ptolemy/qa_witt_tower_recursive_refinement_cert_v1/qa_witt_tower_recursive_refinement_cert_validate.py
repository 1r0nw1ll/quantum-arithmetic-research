#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical p-adic number theory and Witt vector tower structure; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 SS II.4 (Witt vectors, tower structure); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano period tower); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (Hensel lifting, multiplicative order, Lagrange) -->
"""QA Witt Tower Recursive Refinement Law -- cert [433].

Closes the gap cert [432] (QA Witt Tower Scaling Isomorphism) explicitly
leaves open in its own "What this cert does NOT claim" section:

  "Does not claim the exact list of new periods at level p^3 and beyond is
   fully classified -- that final step ... still rests on the same
   non-degenerate (non-Wall-Sun-Sun) assumption used throughout this chain."

[432] proves the embedding/isomorphism mechanism (old-part = exact image of
iota, isomorphic to the full lower level) holds unconditionally at EVERY
level k, but only gives the new-part orbit-COUNT closed form at k=2.

CLAIM 1 (RECURSIVE_PERIOD_SET_LAW, generalizes [389] from a single
transition to every transition):
  For sigma_m(a,b) = (a+b mod m, a mod m), and ANY k >= 1:

      Periods_nt(p^(k+1)) = Periods_nt(p^k)  union  p * Periods_nt(p^k)

  (Periods_nt = nontrivial period set, excluding the fixed point at period
  1), with the old periods' orbit COUNTS exactly unchanged between level k
  and level k+1. This is a direct consequence of [432]'s C1-C3 (embedding
  commutes + old-part is exactly the lower-level decomposition, at every
  k, proved there unconditionally) composed with the classical fact that
  multiplicative order in (Z/p^kZ)* grows by exactly one factor of p per
  level for a non-exceptional (non-Wieferich-like) lift -- the same
  non-degeneracy hypothesis already used throughout this chain (e.g. [429],
  [431]), not re-derived from scratch here.

CLAIM 2 (SPLIT_UNEQUAL_GENERAL_K_MULTIPLICITY, generalizes [432]'s C4 from
k=2 only to every k >= 2):
  For a split-unequal prime p (ord_min < ord_max, the multiplicative orders
  mod p of the two roots of x^2-x-1):

      count_new(p^(k-1) * ord_min) = (p-1) / ord_min                    [constant in k]
      count_new(p^(k-1) * ord_max) = (p-1) * (p^k + p^(k-1) - 1) / ord_max

  Derivation (conservation, generalizing [432]'s eigen-stratification
  argument inductively): the c2=0-exactly stratum (eigenbasis) has size
  p^(k-1)*(p-1) and EVERY unit c1 in it shares the identical period
  p^(k-1)*ord_min (multiplying any unit by a fixed-order element cycles
  with that element's own order, regardless of the unit) -- count =
  size/period = (p-1)/ord_min, with the p^(k-1) factor cancelling exactly,
  giving a closed form CONSTANT in k. The complementary stratum has size
  [p^(2k)-p^(2k-2)] - p^(k-1)*(p-1) = (p-1)*p^(k-1)*(p^k+p^(k-1)-1), giving
  the second formula by the same division.

CLAIM 3 (INERT_SPLIT_EQUAL_GENERAL_K_MULTIPLICITY, generalizes [432]'s C5
from a one-step corollary to an explicit closed form at every k):

      count_new(p^(k-1) * pi(p)) = p^(k-1) * (p^2-1) / pi(p)

  Since there is only one new period per level (no eigen-stratification
  needed), this follows directly from total-count conservation: new-part
  size p^(2k)-p^(2k-2) divided by period p^(k-1)*pi(p).

CLAIM 4 (NONDEGENERACY_WIDE_SCAN): the non-degenerate Hensel-lift
assumption underlying claims 1-3 -- that multiplicative order grows by
EXACTLY a factor of p at every tower level, never stalling -- is checked
directly (fast modular arithmetic, no brute-force orbit enumeration) for
78 split primes (split_unequal + split_equal) < 1000 at k=2..10, and 31
inert primes < 300 at k=2..8 (via Galois-ring GR(p^k,2) arithmetic for the
inert eigenvalue). Zero exceptions found. This is independent of, and far
wider than, the brute-force orbit decomposition used in checks 1-3.

LINEAGE: generalizes [432] (which proved the k=2 case and the
all-k embedding mechanism) and [389] (period sets at k=1->2 only).

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
# Core dynamics (identical map to certs [389] and [432])
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
# C1: recursive period-set law, every level (not just k=1->2)
# ---------------------------------------------------------------------------

def _check_recursive_period_set_law(pk_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Periods_nt(p^(k+1)) == Periods_nt(p^k) | p*Periods_nt(p^k), old counts frozen exactly."""
    failures = []
    for p, k in pk_pairs:
        d_low = _compute_periods(p, k)
        d_high = _compute_periods(p, k + 1)
        old_periods = set(d_low) - {1}
        predicted = old_periods | {p * per for per in old_periods}
        actual = set(d_high) - {1}
        if actual != predicted:
            failures.append(f"p={p},k={k}->{k+1}: predicted={sorted(predicted)} actual={sorted(actual)}")
            continue
        for per in old_periods:
            if d_high.get(per) != d_low[per]:
                failures.append(
                    f"p={p},k={k}->{k+1}: period={per} old_count={d_low[per]} "
                    f"new_level_count={d_high.get(per)} (not frozen)"
                )
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C2: split-unequal general-k closed forms
# ---------------------------------------------------------------------------

def _check_split_unequal_general_k(cases: list[tuple[int, int]]) -> dict[str, Any]:
    """cases: (p, k) pairs, k >= 2. Checks count_new(p^(k-1)*ord_min/ord_max) against closed forms."""
    failures = []
    for p, k in cases:
        cls = _prime_class(p)
        if not (isinstance(cls, tuple) and cls[0] == "split_unequal"):
            failures.append(f"p={p}: not split_unequal (class={cls})")
            continue
        _, o1, o2 = cls
        ord_min, ord_max = min(o1, o2), max(o1, o2)
        dk = _compute_periods(p, k)
        new_min_period = p ** (k - 1) * ord_min
        new_max_period = p ** (k - 1) * ord_max
        expected_min = (p - 1) // ord_min
        expected_max = (p - 1) * (p ** k + p ** (k - 1) - 1) // ord_max
        actual_min = dk.get(new_min_period, 0)
        actual_max = dk.get(new_max_period, 0)
        if actual_min != expected_min or actual_max != expected_max:
            failures.append(
                f"p={p},k={k}: ord_min={ord_min} ord_max={ord_max} "
                f"expected=({expected_min},{expected_max}) actual=({actual_min},{actual_max})"
            )
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C3: inert/split-equal general-k closed form
# ---------------------------------------------------------------------------

def _check_inert_split_equal_general_k(cases: list[tuple[int, int]]) -> dict[str, Any]:
    """cases: (p, k) pairs, k >= 2. Checks count_new(p^(k-1)*pi(p)) = p^(k-1)*(p^2-1)/pi(p)."""
    failures = []
    for p, k in cases:
        cls = _prime_class(p)
        is_single_new_period = cls == "inert" or (isinstance(cls, tuple) and cls[0] == "split_equal")
        if not is_single_new_period:
            failures.append(f"p={p}: expected inert or split_equal (class={cls})")
            continue
        d1 = _compute_periods(p, 1)
        pi_p = max(per for per in d1 if per != 1)
        dk = _compute_periods(p, k)
        new_period = p ** (k - 1) * pi_p
        expected = p ** (k - 1) * (p * p - 1) // pi_p
        actual = dk.get(new_period, 0)
        if actual != expected:
            failures.append(f"p={p},k={k}: pi(p)={pi_p} expected={expected} actual={actual}")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C4: wide non-degeneracy scan (fast modular arithmetic, no orbit enumeration)
# ---------------------------------------------------------------------------

def _trial_factor(n: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    nn, d = n, 2
    while d * d <= nn:
        while nn % d == 0:
            factors[d] = factors.get(d, 0) + 1
            nn //= d
        d += 1
    if nn > 1:
        factors[nn] = factors.get(nn, 0) + 1
    return factors


def _hensel_lift_root(r0: int, p: int, target_k: int) -> int:
    r = r0 % p
    for j in range(1, target_k):
        nm = p ** (j + 1)
        fprime = (2 * r - 1) % nm
        inv = pow(fprime, -1, nm)
        fr = (r * r - r - 1) % nm
        r = (r - fr * inv) % nm
    return r


def _order_modpk_split(r0: int, p: int, k: int, base_factors: set[int]) -> int:
    rk = _hensel_lift_root(r0, p, k) if k > 1 else r0 % p
    mod = p ** k
    factors = base_factors | ({p} if k > 1 else set())
    order = (p ** (k - 1)) * (p - 1)
    for q in factors:
        while order % q == 0 and pow(rk, order // q, mod) == 1:
            order //= q
    return order


def _gr_mul(x: tuple[int, int], y: tuple[int, int], mod: int) -> tuple[int, int]:
    a, b = x
    c, d = y
    return ((a * c + 5 * b * d) % mod, (a * d + b * c) % mod)


def _gr_pow(x: tuple[int, int], e: int, mod: int) -> tuple[int, int]:
    result, base = (1, 0), x
    while e > 0:
        if e & 1:
            result = _gr_mul(result, base, mod)
        base = _gr_mul(base, base, mod)
        e >>= 1
    return result


def _inert_lambda(p: int, k: int) -> tuple[int, int]:
    mod = p ** k
    inv2 = pow(2, -1, mod)
    return (inv2 % mod, inv2 % mod)


def _pi_p_inert_gr(p: int) -> int:
    one = (1, 0)
    lam1 = _inert_lambda(p, 1)
    cur, o = lam1, 1
    while cur != one:
        cur = _gr_mul(cur, lam1, p)
        o += 1
    return o


def _order_gr(p: int, k: int, base_factors: set[int]) -> int:
    mod = p ** k
    one = (1, 0)
    lamk = _inert_lambda(p, k)
    factors = base_factors | ({p} if k > 1 else set())
    order = (p * p - 1) * p ** (2 * (k - 1))
    for q in factors:
        while order % q == 0 and _gr_pow(lamk, order // q, mod) == one:
            order //= q
    return order


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _check_nondegeneracy_wide_scan(split_max: int, split_max_k: int,
                                    inert_max: int, inert_max_k: int) -> dict[str, Any]:
    failures: list[str] = []
    n_split, n_inert = 0, 0
    for p in range(7, split_max + 1):
        if not _is_prime(p):
            continue
        cls = _prime_class(p)
        if isinstance(cls, tuple):
            n_split += 1
            roots = _poly_roots_mod_p(p)
            base_factors = set(_trial_factor(p - 1).keys())
            for r0 in roots:
                ord_p = _multiplicative_order(r0, p)
                for k in range(2, split_max_k + 1):
                    ordk = _order_modpk_split(r0, p, k, base_factors)
                    predicted = p ** (k - 1) * ord_p
                    if ordk != predicted:
                        failures.append(f"split p={p} r0={r0} k={k}: order={ordk} predicted={predicted}")
        elif cls == "inert":
            if p > inert_max:
                continue
            n_inert += 1
            pi_p = _pi_p_inert_gr(p)
            base_factors = set(_trial_factor(p * p - 1).keys())
            for k in range(2, inert_max_k + 1):
                ordk = _order_gr(p, k, base_factors)
                predicted = p ** (k - 1) * pi_p
                if ordk != predicted:
                    failures.append(f"inert p={p} k={k}: order={ordk} predicted={predicted}")
    return {
        "ok": not failures,
        "n_split_primes": n_split,
        "n_inert_primes": n_inert,
        "failures": failures[:10],
    }


# ---------------------------------------------------------------------------
# Self-test (--self-test mode)
# ---------------------------------------------------------------------------

C1_PK_PAIRS = [(7, 1), (7, 2), (11, 1), (11, 2), (13, 1), (13, 2), (17, 1), (17, 2), (19, 1), (19, 2)]

C2_CASES = [(11, 2), (11, 3), (19, 2), (19, 3), (29, 2), (31, 2)]

C3_CASES = [(7, 2), (7, 3), (7, 4), (13, 2), (13, 3), (17, 2), (17, 3), (23, 2)]


def run_self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    r1 = _check_recursive_period_set_law(C1_PK_PAIRS)
    checks["RECURSIVE_PERIOD_SET_LAW"] = r1["ok"]
    if not r1["ok"]:
        details["RECURSIVE_PERIOD_SET_LAW"] = r1["failures"]

    r2 = _check_split_unequal_general_k(C2_CASES)
    checks["SPLIT_UNEQUAL_GENERAL_K_MULTIPLICITY"] = r2["ok"]
    if not r2["ok"]:
        details["SPLIT_UNEQUAL_GENERAL_K_MULTIPLICITY"] = r2["failures"]

    r3 = _check_inert_split_equal_general_k(C3_CASES)
    checks["INERT_SPLIT_EQUAL_GENERAL_K_MULTIPLICITY"] = r3["ok"]
    if not r3["ok"]:
        details["INERT_SPLIT_EQUAL_GENERAL_K_MULTIPLICITY"] = r3["failures"]

    r4 = _check_nondegeneracy_wide_scan(split_max=997, split_max_k=10, inert_max=293, inert_max_k=8)
    checks["NONDEGENERACY_WIDE_SCAN"] = r4["ok"]
    details["NONDEGENERACY_WIDE_SCAN_COVERAGE"] = {
        "n_split_primes": r4["n_split_primes"],
        "n_inert_primes": r4["n_inert_primes"],
    }
    if not r4["ok"]:
        details["NONDEGENERACY_WIDE_SCAN"] = r4["failures"]

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
        "id": "RECURSIVE_LAW_P11_K1_TO_2",
        "description": "p=11: Periods_nt(121) == Periods_nt(11) | 11*Periods_nt(11)",
        "expected": True,
        "fn": lambda: (
            set(_compute_periods(11, 2)) - {1}
            == (set(_compute_periods(11, 1)) - {1}) | {11 * per for per in set(_compute_periods(11, 1)) - {1}}
        ),
    },
    {
        "id": "RECURSIVE_LAW_P11_K2_TO_3",
        "description": "p=11: same recursive law holds one level further up (k=2->3), not just k=1->2",
        "expected": True,
        "fn": lambda: (
            set(_compute_periods(11, 3)) - {1}
            == (set(_compute_periods(11, 2)) - {1}) | {11 * per for per in set(_compute_periods(11, 2)) - {1}}
        ),
    },
    {
        "id": "SPLIT_UNEQUAL_P11_K3_GENERAL_FORM",
        "description": "p=11,k=3: count_new(p^2*ord_min)=2 (same as k=2, constant in k), count_new(p^2*ord_max)=1451",
        "expected": True,
        "fn": lambda: _compute_periods(11, 3).get(11 * 11 * 5) == 2 and _compute_periods(11, 3).get(11 * 11 * 10) == 1451,
    },
    {
        "id": "INERT_P7_K4_GENERAL_FORM",
        "description": "p=7,k=4: count_new(7^3*16) = 7^3*(49-1)/16 = 343*3 = 1029",
        "expected": True,
        "fn": lambda: _compute_periods(7, 4).get(7 ** 3 * 16) == 1029,
    },
    {
        "id": "WRONG_GENERAL_K_FORMULA_NOT_CONSTANT",
        "description": "DESIGNED FAIL: split-unequal count_new(p^(k-1)*ord_min) is NOT k-dependent like p^(k-1)*(p-1)/ord_min",
        "expected": False,
        "fn": lambda: _compute_periods(11, 3).get(11 * 11 * 5) == 11 * 11 * (11 - 1) // 5,
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
        description="QA Witt Tower Recursive Refinement Law cert [433] validator"
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
