#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical p-adic ramification theory and Jordan canonical form over Z/p^kZ; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 (ramified extensions); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano-style period tables); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (primitive roots, Hensel lifting); Washington (1997) doi.org/10.1007/978-1-4612-1934-7 Ch.5 (structure of (Z/2^k Z)^*, non-cyclic for k>=3) -->
"""QA Witt Tower Ramified Prime GENERALIZATION -- cert [435].

[434] derived the ramified-prime closed form for exactly ONE case:
p=5, the unique ramified prime of the Fibonacci recurrence x^2-x-1
(Q(sqrt(5))). This cert tests whether that closed form is a property of
Fibonacci specifically, or of ramified Jordan blocks in general, by
deriving and checking the SAME closed form on a structurally different
recurrence: x^2-4x+1 (M=[[4,-1],[1,0]]), discriminant D=12=2^2*3, which
generates Q(sqrt(3)) -- a genuinely different quadratic field from
Fibonacci's Q(sqrt(5)) -- and has TWO ramified primes (2 and 3), letting
this cert probe both an odd ramified prime (clean generalization) and
p=2 (the classically exceptional prime) in one discriminant.

CLAIM 1 (PERIOD_SET_LAW_GENERALIZED): for ANY odd ramified prime p of a
unimodular companion matrix M (det(M)=+-1, char. poly has a double root
lambda0 mod p, e := ord(lambda0 mod p)), the period set of
sigma_m(v) = M*v mod m on (Z/p^k Z)^2 is exactly

    Periods(p^k) = {1, e}  union  { e * p^L : L = 1, ..., k }

-- IDENTICAL in form to [434]'s p=5-specific formula, with e=4 there
simply being a special case (ord(3 mod 5)=4). Verified directly on TWO
independent (M, p) pairs: p=3 via this cert's D=12 matrix (e=2), and a
fresh re-derivation of [434]'s own p=5 case (e=4) using the same
generic code path, to rule out the check being overfit to either
recurrence.

CLAIM 2 (BIRTH_JUMP_FREEZE_LAW_GENERALIZED): for P_L = e*p^L, the
orbit count (number of distinct cycles, matching [434]'s C2 convention
exactly -- NOT point count) as a function of tower level k is

    count(P_L, k) = 0                  if k < L   (not yet born)
    count(P_L, k) = p^(L-1)            if k == L  (birth)
    count(P_L, k) = (p+1) * p^(L-1)    if k > L    (one delayed jump, frozen)

This is [434]'s C2 formula (there written as 5^(L-1) / 6*5^(L-1) for
the special case p=5, where p+1=6) generalized to literal (p+1) -- a
clean, parameter-free closed form. Verified on the same two (M,p) pairs
as Claim 1.

CLAIM 3 (EIGENLINE_PERSISTENCE_GENERALIZED): periods 1 and e each have
orbit count exactly 1 at every level k, with no birth/jump -- [432]'s
embedding-isomorphism mechanism, confirmed prime- and discriminant-
agnostic. (When e=1, the two periods coincide and the single bucket
period=1 carries orbit count 2 instead -- see Claim 4.)

CLAIM 4 (P2_STALL_EXCEPTION -- the flagged exception): p=2 does NOT
satisfy Claims 1-3 verbatim. For this cert's D=12 matrix, lambda0 mod 2
= 1, so e = ord(1 mod 2) = 1 -- the two eigenline periods collapse into
a single bucket. Worse, the matrix order itself breaks the e*p^k
scaling law:

    ord(M mod 2^k) = 2^k        for k = 1, 2
    ord(M mod 2^k) = 2^(k-1)    for k >= 3   (one tower level "stalls")

This is verified directly (fast modular matrix exponentiation, no
brute-force orbit enumeration, k up to 30) rather than assumed. The
mechanism: at p=3 (and p=5), N = M - lambda0*I satisfies N^2 = p*I
EXACTLY as an integer matrix (a clean scalar, decoupling N from M and
giving Z[N] = Z[sqrt(p)] exactly). At p=2, N^2 = 2*M EXACTLY instead --
a RECURSIVE identity coupling N back to M itself, not a scalar -- which
breaks the one-step binomial collapse (lambda0*I+N)^n = lambda0^n*I +
n*lambda0^(n-1)*N that drives the odd-prime closed form. This matches
the classical fact that (Z/2^k Z)^* is cyclic for k<=2 but isomorphic
to Z/2 x Z/2^(k-2) (NOT cyclic) for k>=3 (Washington Ch.5) -- 2 is the
one prime where the unit-group structure itself changes shape, and
that shows up here as a one-level stall in the matrix order before the
expected x2-per-level growth resumes.

CLAIM 5 (CRT_CROSS_PRIME_INDEPENDENCE): despite p=2's anomalous
internal law, the two ramified primes of a single discriminant combine
by ORDINARY composition: ord(M mod 2^j * 3^k) = lcm(ord(M mod 2^j),
ord(M mod 3^k)) for every (j,k) tested -- zero interaction/cross terms.
This is the genuinely new question this discriminant was chosen to
probe (a single ramified prime, as in Fibonacci, cannot test it).
Verified via fast modular matrix exponentiation (CRT / Sun-Tzu, no new
mechanism) for j=1..8, k=1..5 (40 pairs).

WHY THIS DISCRIMINANT: D=12 was chosen (over the q=c=1 family, which
either gives a single boring ramified prime or D=20=2^2*5 -- the SAME
field Q(sqrt(5)) as Fibonacci, risking a relabeling objection) because
it (a) has two distinct ramified primes including p=2, the classically
exceptional case, and (b) generates Q(sqrt(3)), a genuinely different
field from Fibonacci's Q(sqrt(5)) -- ruling out "this is just
Fibonacci in disguise."

LINEAGE: generalizes [434] (p=5-only) by testing a second, structurally
distinct discriminant. Does not modify or duplicate [434]'s checks --
re-derives them generically and confirms [434]'s own numbers fall out
of the SAME formula. Uses [432]'s embedding-isomorphism mechanism
(Claim 3) and is consistent with [433]'s and [434]'s established
results throughout.

PRIMARY SOURCES:
  Serre (1979) doi.org/10.1007/978-1-4757-5673-9 -- ramification theory in local fields
  Wall (1960) doi.org/10.1080/00029890.1960.11989541 -- Pisano-style period tables
  Ireland & Rosen (1990) ISBN 978-0-387-97329-6 -- primitive roots, Hensel lifting
  Washington (1997) doi.org/10.1007/978-1-4612-1934-7 Ch.5 -- structure of (Z/2^k Z)^*
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Generic 2x2 matrix dynamics mod m (any companion matrix M)
# ---------------------------------------------------------------------------

D12_M = [[4, -1], [1, 0]]   # x^2-4x+1, D=12=2^2*3, generates Q(sqrt(3))
FIB_M = [[1, 1], [1, 0]]    # x^2-x-1,  D=5,        generates Q(sqrt(5)) -- [434]'s case


def _sigma(M: list[list[int]], a: int, b: int, m: int) -> tuple[int, int]:
    return ((M[0][0] * a + M[0][1] * b) % m, (M[1][0] * a + M[1][1] * b) % m)


def _mat_mul_mod(A: list[list[int]], B: list[list[int]], m: int) -> list[list[int]]:
    return [
        [(A[i][0] * B[0][j] + A[i][1] * B[1][j]) % m for j in range(2)]
        for i in range(2)
    ]


def _mat_pow_mod(M: list[list[int]], n: int, m: int) -> list[list[int]]:
    result = [[1, 0], [0, 1]]
    base = [[x % m for x in row] for row in M]
    while n:
        if n & 1:
            result = _mat_mul_mod(result, base, m)
        base = _mat_mul_mod(base, base, m)
        n >>= 1
    return result


def _compute_periods(M: list[list[int]], p: int, power: int) -> dict[int, int]:
    """Return {period: orbit_count} for sigma on (Z/p^power Z)^2.

    orbit_count = number of distinct CYCLES with that period (matches
    [434]'s C2 convention exactly -- not point count).
    """
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
                ca, cb = _sigma(M, ca, cb, m)
                k += 1
                if ca == a and cb == b:
                    break
            counts[k] = counts.get(k, 0) + 1
    return counts


def _lambda0_odd(M: list[list[int]], p: int) -> int:
    tr = (M[0][0] + M[1][1]) % p
    inv2 = pow(2, -1, p)
    return (tr * inv2) % p


def _order_mod_p(a: int, p: int) -> int:
    e = 1
    cur = a % p
    while cur != 1:
        cur = (cur * a) % p
        e += 1
        if e > p:
            raise RuntimeError(f"no multiplicative order found for {a} mod {p}")
    return e


# ---------------------------------------------------------------------------
# C1: period set law, generalized (tested on D12/p=3 AND Fibonacci/p=5)
# ---------------------------------------------------------------------------

def _predicted_period_set(e: int, p: int, k: int) -> set[int]:
    return {1, e} | {e * p ** L for L in range(1, k + 1)}


def _check_period_set_law_generalized(cases: list[tuple[str, list[list[int]], int, list[int]]]) -> dict[str, Any]:
    failures = []
    for name, M, p, k_list in cases:
        e = _order_mod_p(_lambda0_odd(M, p), p)
        for k in k_list:
            d = _compute_periods(M, p, k)
            predicted = _predicted_period_set(e, p, k)
            actual = set(d.keys())
            if actual != predicted:
                failures.append(f"{name} k={k}: predicted={sorted(predicted)} actual={sorted(actual)}")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C2: birth/jump/freeze orbit-count law, generalized to (p+1)*p^(L-1)
# ---------------------------------------------------------------------------

def _predicted_count_generalized(L: int, k: int, p: int) -> int:
    if k < L:
        return 0
    if k == L:
        return p ** (L - 1)
    return (p + 1) * p ** (L - 1)


def _check_birth_jump_freeze_generalized(cases: list[tuple[str, list[list[int]], int, list[int]]]) -> dict[str, Any]:
    failures = []
    for name, M, p, k_list in cases:
        e = _order_mod_p(_lambda0_odd(M, p), p)
        for k in k_list:
            d = _compute_periods(M, p, k)
            for L in range(1, k + 1):
                period_L = e * p ** L
                expected = _predicted_count_generalized(L, k, p)
                actual = d.get(period_L, 0)
                if actual != expected:
                    failures.append(
                        f"{name} k={k},L={L}: period={period_L} expected_count={expected} actual_count={actual}"
                    )
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C3: eigenline persistence, generalized
# ---------------------------------------------------------------------------

def _check_eigenline_persistence_generalized(cases: list[tuple[str, list[list[int]], int, list[int]]]) -> dict[str, Any]:
    failures = []
    for name, M, p, k_list in cases:
        e = _order_mod_p(_lambda0_odd(M, p), p)
        for k in k_list:
            d = _compute_periods(M, p, k)
            if d.get(1) != 1:
                failures.append(f"{name} k={k}: period=1 count={d.get(1)} expected=1")
            if d.get(e) != 1:
                failures.append(f"{name} k={k}: period={e} count={d.get(e)} expected=1")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# C4: p=2 stall exception (D12 matrix only) -- fast modular, no brute force
# ---------------------------------------------------------------------------

def _predicted_order_p2(k: int) -> int:
    return 2 ** k if k <= 2 else 2 ** (k - 1)


def _check_p2_stall_exception(k_max: int) -> dict[str, Any]:
    failures: list[str] = []
    M = D12_M
    I2 = [[1, 0], [0, 1]]

    # Jordan-block mechanism contrast: N^2 = p*I (scalar) at p=3,
    # N^2 = 2*M (recursive, NOT scalar) at p=2 -- the structural root
    # cause of the stall, checked as EXACT integer matrix identities
    # (no modular reduction).
    lam0_3 = 2  # lambda0 mod 3 = 2 for D12; integer lift used for exact N
    N3 = [[M[0][0] - lam0_3, M[0][1]], [M[1][0], M[1][1] - lam0_3]]
    N3_sq = [[N3[0][0] * N3[0][0] + N3[0][1] * N3[1][0], N3[0][0] * N3[0][1] + N3[0][1] * N3[1][1]],
             [N3[1][0] * N3[0][0] + N3[1][1] * N3[1][0], N3[1][0] * N3[0][1] + N3[1][1] * N3[1][1]]]
    if N3_sq != [[3, 0], [0, 3]]:
        failures.append(f"N^2 at p=3 expected exactly 3*I, got {N3_sq}")

    lam0_2 = 1  # lambda0 mod 2 = 1 for D12; integer lift used for exact N
    N2 = [[M[0][0] - lam0_2, M[0][1]], [M[1][0], M[1][1] - lam0_2]]
    N2_sq = [[N2[0][0] * N2[0][0] + N2[0][1] * N2[1][0], N2[0][0] * N2[0][1] + N2[0][1] * N2[1][1]],
             [N2[1][0] * N2[0][0] + N2[1][1] * N2[1][0], N2[1][0] * N2[0][1] + N2[1][1] * N2[1][1]]]
    two_M = [[2 * M[0][0], 2 * M[0][1]], [2 * M[1][0], 2 * M[1][1]]]
    if N2_sq != two_M:
        failures.append(f"N^2 at p=2 expected exactly 2*M={two_M}, got {N2_sq}")

    # Order stall: predicted order is exact (M^order = I) AND minimal
    # (M^(order/2) != I), checked via fast modular matrix exponentiation.
    for k in range(1, k_max + 1):
        mod = 2 ** k
        predicted = _predicted_order_p2(k)
        if _mat_pow_mod(M, predicted, mod) != I2:
            failures.append(f"k={k}: M^{predicted} != I mod 2^{k}")
            continue
        if predicted > 1 and _mat_pow_mod(M, predicted // 2, mod) == I2:
            failures.append(f"k={k}: M^{predicted // 2} == I mod 2^{k} -- predicted order not minimal")

    return {"ok": not failures, "failures": failures, "k_max": k_max}


# ---------------------------------------------------------------------------
# C5: CRT cross-prime independence (D12 matrix only) -- fast modular
# ---------------------------------------------------------------------------

def _predicted_order_3(k: int) -> int:
    e = 2  # ord(2 mod 3)
    return e * 3 ** k


def _check_crt_independence(j_max: int, k_max: int) -> dict[str, Any]:
    failures: list[str] = []
    M = D12_M
    I2 = [[1, 0], [0, 1]]

    def order_2j(j: int) -> int:
        return _predicted_order_p2(j)

    def order_3k(k: int) -> int:
        return _predicted_order_3(k)

    for j in range(1, j_max + 1):
        for k in range(1, k_max + 1):
            m = 2 ** j * 3 ** k
            predicted = math.lcm(order_2j(j), order_3k(k))
            if _mat_pow_mod(M, predicted, m) != I2:
                failures.append(f"j={j},k={k}: M^{predicted} != I mod {m}")
                continue
            if predicted > 1 and _mat_pow_mod(M, predicted // 2, m) == I2:
                failures.append(
                    f"j={j},k={k}: predicted order {predicted} not minimal (halved order also gives I)"
                )

    return {"ok": not failures, "failures": failures, "j_max": j_max, "k_max": k_max}


# ---------------------------------------------------------------------------
# Self-test (--self-test mode)
# ---------------------------------------------------------------------------

GENERALIZATION_CASES = [
    ("D12_p3", D12_M, 3, [1, 2, 3, 4, 5, 6]),
    ("FIB_p5", FIB_M, 5, [1, 2, 3, 4, 5]),
]
C4_K_MAX = 30
C5_J_MAX = 8
C5_K_MAX = 5


def run_self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    r1 = _check_period_set_law_generalized(GENERALIZATION_CASES)
    checks["PERIOD_SET_LAW_GENERALIZED"] = r1["ok"]
    if not r1["ok"]:
        details["PERIOD_SET_LAW_GENERALIZED"] = r1["failures"]

    r2 = _check_birth_jump_freeze_generalized(GENERALIZATION_CASES)
    checks["BIRTH_JUMP_FREEZE_LAW_GENERALIZED"] = r2["ok"]
    if not r2["ok"]:
        details["BIRTH_JUMP_FREEZE_LAW_GENERALIZED"] = r2["failures"]

    r3 = _check_eigenline_persistence_generalized(GENERALIZATION_CASES)
    checks["EIGENLINE_PERSISTENCE_GENERALIZED"] = r3["ok"]
    if not r3["ok"]:
        details["EIGENLINE_PERSISTENCE_GENERALIZED"] = r3["failures"]

    r4 = _check_p2_stall_exception(C4_K_MAX)
    checks["P2_STALL_EXCEPTION"] = r4["ok"]
    details["P2_STALL_EXCEPTION_COVERAGE"] = {"k_max": r4["k_max"]}
    if not r4["ok"]:
        details["P2_STALL_EXCEPTION"] = r4["failures"]

    r5 = _check_crt_independence(C5_J_MAX, C5_K_MAX)
    checks["CRT_CROSS_PRIME_INDEPENDENCE"] = r5["ok"]
    details["CRT_CROSS_PRIME_INDEPENDENCE_COVERAGE"] = {"j_max": r5["j_max"], "k_max": r5["k_max"]}
    if not r5["ok"]:
        details["CRT_CROSS_PRIME_INDEPENDENCE"] = r5["failures"]

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
        "id": "PERIOD_SET_D12_P3_K4",
        "description": "D12,p=3,k=4: period set is exactly {1,2,6,18,54,162}",
        "expected": True,
        "fn": lambda: set(_compute_periods(D12_M, 3, 4).keys()) == {1, 2, 6, 18, 54, 162},
    },
    {
        "id": "BIRTH_COUNT_D12_P3_K4_L4",
        "description": "D12,p=3,k=4: period=2*3^4=162 is freshly born with orbit count 3^3=27",
        "expected": True,
        "fn": lambda: _compute_periods(D12_M, 3, 4).get(2 * 3 ** 4) == 27,
    },
    {
        "id": "FROZEN_COUNT_D12_P3_K4_L3",
        "description": "D12,p=3,k=4: period=2*3^3=54 (born k=3, count 9) jumped+frozen to 4*9=36",
        "expected": True,
        "fn": lambda: _compute_periods(D12_M, 3, 4).get(2 * 3 ** 3) == 36,
    },
    {
        "id": "FIB_REDERIVATION_MATCHES_434",
        "description": "Generic code re-derives [434]'s own number: p=5,k=4,L=3 period=500 has orbit count 150",
        "expected": True,
        "fn": lambda: _compute_periods(FIB_M, 5, 4).get(4 * 5 ** 3) == 150,
    },
    {
        "id": "P2_ORDER_STALLS_AT_K3",
        "description": "DESIGNED CONTRAST: ord(M mod 8)=4, NOT 8 -- the naive e*p^k=1*2^3=8 prediction fails",
        "expected": True,
        "fn": lambda: _mat_pow_mod(D12_M, 4, 8) == [[1, 0], [0, 1]] and _mat_pow_mod(D12_M, 8, 8) == [[1, 0], [0, 1]],
    },
    {
        "id": "WRONG_NAIVE_P2_SCALING",
        "description": "DESIGNED FAIL: if e*p^k held at p=2 like the odd primes, ord(M mod 8) would be minimal at 8, not 4",
        "expected": False,
        "fn": lambda: _mat_pow_mod(D12_M, 4, 8) != [[1, 0], [0, 1]],
    },
    {
        "id": "CRT_SAMPLE_J3_K2",
        "description": "ord(M mod 2^3*3^2=72) = lcm(ord(M mod 8)=4, ord(M mod 9)=18) = 36",
        "expected": True,
        "fn": lambda: _mat_pow_mod(D12_M, 36, 72) == [[1, 0], [0, 1]] and _mat_pow_mod(D12_M, 18, 72) != [[1, 0], [0, 1]],
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
        description="QA Witt Tower Ramified Prime Generalization cert [435] validator"
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
