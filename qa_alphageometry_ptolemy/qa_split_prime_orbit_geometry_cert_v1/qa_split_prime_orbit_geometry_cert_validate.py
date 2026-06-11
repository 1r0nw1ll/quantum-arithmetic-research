# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical algebraic number theory and Fibonacci/Pisano theory; Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano periods, orders of roots of x^2-x-1); Neukirch (1999) ISBN 978-3-540-65399-8 §I.8 (splitting of primes in quadratic fields); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.5 (Legendre symbol, primitive roots) -->
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — integer arithmetic over F_p and Z[phi]/(p); all state exact integer; no float; no continuous QA inputs"
"""Cert [388]: QA Split Prime Orbit Geometry.

PRIMARY CLAIM:
  For the QA Fibonacci shift sigma(a,b) = (a+b mod p, a) acting on F_p x F_p:

  (C1) SPLIT_THREE_PERIODS:
       For each split prime p in {11,19,29,31} (where the two roots r1,r2 of
       x^2-x-1 mod p have UNEQUAL orders: min(ord_p(r1),ord_p(r2)) < pi(p)):
         sigma-orbit periods = {1, ord_min, pi(p)}
       where ord_min = min(ord_p(r1), ord_p(r2)) and pi(p) = Pisano period.
       This is DISTINCT from the inert case which has only {1, pi(p)}.

  (C2) EIGENSPACE_IDENTIFICATION:
       For each such split prime p, the period-ord_min orbits are EXACTLY the
       scalar multiples of the eigenvector (r_min, 1) of sigma for eigenvalue r_min:
         {(r_min*c mod p, c) : c in F_p*}  (p-1 nonzero elements)
       These form (p-1)/ord_min orbits of period ord_min.

  (C3) ORBIT_COUNTS:
       For split p in {11,19,29,31}:
         |period-1|   = 1                (zero element only)
         |period-ord_min orbs| = (p-1)/ord_min
         |period-pi(p) orbs|   = (p^2-p)/pi(p)
         Total elements = 1 + (p-1) + (p^2-p) = p^2 (exhaustive)

  (C4) EQUAL_ORDER_CASE:
       For split p=41 where ord_p(r1) = ord_p(r2) = pi(p) = 40:
         sigma-orbit periods = {1, pi(p)} ONLY (no intermediate period).
         |period-pi(p) orbs| = (p^2-1)/pi(p) = 42.
       The two eigenspaces each contribute 1 orbit among the 42, indistinguishable
       by period from generic orbits.

  (C5) INERT_CONTRAST:
       For inert primes p in {3, 7, 13, 17} (x^2-x-1 irreducible mod p):
         sigma-orbit periods = {1, pi(p)} ONLY — no intermediate period.
       The intermediate period is a SPLIT SIGNATURE, absent for inert primes.
       The presence of the eigenspace period reflects the non-locality of Z[phi]/(p)
       for split p (two maximal ideals) vs. the local ring structure for inert p.

ROOT-ORDER RULE (certified):
  For split prime p, r1*r2 = -1 (mod p, from constant term of x^2-x-1).
  If min(ord(r1),ord(r2)) is ODD (= d), then max(ord) = 2d = pi(p).
  If both orders are EVEN (as in p=41), then ord(r1)=ord(r2)=pi(p).
  The intermediate period exists iff one root has odd order < pi(p).

LINEAGE: extends cert [386] (inert/split/ramified prime classification) and
         cert [387] (Witt carry sub-orbit invariant for inert p=3).
SOURCES: Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano periods);
         Neukirch (1999) ISBN 978-3-540-65399-8; Ireland & Rosen (1990) ISBN 978-0-387-97329-6.
"""

import json
import sys
from math import gcd
from typing import Any


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

def _roots_mod_p(p: int) -> list[int]:
    return [x for x in range(p) if (x * x - x - 1) % p == 0]


def _classify_prime(p: int) -> str:
    n = len(_roots_mod_p(p))
    return {0: "inert", 1: "ramified", 2: "split"}[n]


def _ord_mod(x: int, p: int) -> int:
    if x % p == 0:
        return 0
    r, k = x % p, 1
    while r != 1:
        r = r * x % p
        k += 1
    return k


def _pisano(p: int) -> int:
    a, b = 0, 1
    for k in range(1, 20 * p + 1):
        a, b = b, (a + b) % p
        if a == 0 and b == 1:
            return k
    return -1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def _primes_up_to(n: int) -> list[int]:
    return [p for p in range(2, n + 1) if _is_prime(p)]


# ---------------------------------------------------------------------------
# Orbit computation
# ---------------------------------------------------------------------------

def _sigma(a: int, b: int, p: int) -> tuple:
    return ((a + b) % p, a)


def _orbit_period(a: int, b: int, p: int) -> int:
    start = (a, b)
    cur = _sigma(a, b, p)
    k = 1
    while cur != start:
        cur = _sigma(*cur, p)
        k += 1
    return k


def _compute_orbits(p: int) -> dict[int, list[tuple]]:
    """Returns {period: [orbit_start_elements]}."""
    seen: set = set()
    by_period: dict = {}
    for a in range(p):
        for b in range(p):
            if (a, b) in seen:
                continue
            period = _orbit_period(a, b, p)
            orb = [(a, b)]
            cur = _sigma(a, b, p)
            while cur != (a, b):
                orb.append(cur)
                cur = _sigma(*cur, p)
            seen.update(orb)
            by_period.setdefault(period, []).append((a, b))
    return by_period


def _full_orbit(a: int, b: int, p: int) -> set:
    start = (a, b)
    orb = {start}
    cur = _sigma(a, b, p)
    while cur != start:
        orb.add(cur)
        cur = _sigma(*cur, p)
    return orb


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_split_three_periods() -> dict[str, Any]:
    """C1: split primes {11,19,29,31} have orbit periods {1, ord_min, pi(p)}."""
    target_primes = [11, 19, 29, 31]
    errors = []
    results = []

    for p in target_primes:
        roots = _roots_mod_p(p)
        r1, r2 = sorted(roots)
        o1, o2 = _ord_mod(r1, p), _ord_mod(r2, p)
        ord_min = min(o1, o2)
        pi = _pisano(p)
        expected = {1, ord_min, pi}

        by_period = _compute_orbits(p)
        actual = set(by_period.keys())

        ok_p = actual == expected
        if not ok_p:
            errors.append(f"p={p}: expected {expected}, got {actual}")

        results.append({
            "p": p,
            "roots": [r1, r2],
            "orders": [o1, o2],
            "ord_min": ord_min,
            "pi": pi,
            "expected_periods": sorted(expected),
            "actual_periods": sorted(actual),
            "ok": ok_p,
        })

    ok = len(errors) == 0
    return {
        "name": "SPLIT_THREE_PERIODS",
        "ok": ok,
        "results": results,
        "errors": errors,
        "detail": "Split {11,19,29,31}: periods={1,ord_min,pi(p)}" if ok else f"FAIL: {errors}",
    }


def _check_eigenspace_identification() -> dict[str, Any]:
    """C2: period-ord_min orbits = scalar multiples of eigenvector (r_min,1)."""
    target_primes = [11, 19, 29, 31]
    errors = []
    results = []

    for p in target_primes:
        roots = _roots_mod_p(p)
        r1, r2 = sorted(roots)
        o1, o2 = _ord_mod(r1, p), _ord_mod(r2, p)
        r_min = r1 if o1 <= o2 else r2
        ord_min = min(o1, o2)

        eigenspace = frozenset((r_min * c % p, c) for c in range(1, p))
        by_period = _compute_orbits(p)

        period_min_elements: set = set()
        for start in by_period.get(ord_min, []):
            period_min_elements |= _full_orbit(*start, p)

        in_eigenspace = period_min_elements <= eigenspace
        covers_eigenspace = eigenspace <= period_min_elements
        exact = in_eigenspace and covers_eigenspace

        if not exact:
            errors.append(
                f"p={p}: eigenspace match={exact} "
                f"(in={in_eigenspace}, covers={covers_eigenspace})"
            )
        results.append({
            "p": p,
            "r_min": r_min,
            "ord_min": ord_min,
            "eigenspace_size": len(eigenspace),
            "period_min_elements": len(period_min_elements),
            "exact_match": exact,
        })

    ok = len(errors) == 0
    return {
        "name": "EIGENSPACE_IDENTIFICATION",
        "ok": ok,
        "results": results,
        "errors": errors,
        "detail": "Period-ord_min orbits = eigenspace of sigma for r_min" if ok else f"FAIL: {errors}",
    }


def _check_orbit_counts() -> dict[str, Any]:
    """C3: orbit counts satisfy formulas; total = p^2."""
    target_primes = [11, 19, 29, 31]
    errors = []
    results = []

    for p in target_primes:
        roots = _roots_mod_p(p)
        r1, r2 = sorted(roots)
        o1, o2 = _ord_mod(r1, p), _ord_mod(r2, p)
        ord_min = min(o1, o2)
        pi = _pisano(p)

        by_period = _compute_orbits(p)
        n1 = len(by_period.get(1, []))
        n_min = len(by_period.get(ord_min, []))
        n_pi = len(by_period.get(pi, []))

        exp_1 = 1
        exp_min = (p - 1) // ord_min
        exp_pi = (p * p - p) // pi

        total_elems = (n1 * 1 + n_min * ord_min + n_pi * pi)

        ok_p = (
            n1 == exp_1
            and n_min == exp_min
            and n_pi == exp_pi
            and total_elems == p * p
        )
        if not ok_p:
            errors.append(
                f"p={p}: n1={n1}(exp {exp_1}), n_min={n_min}(exp {exp_min}), "
                f"n_pi={n_pi}(exp {exp_pi}), total={total_elems}(exp {p*p})"
            )
        results.append({
            "p": p,
            "n1": n1, "exp_1": exp_1,
            "n_min": n_min, "exp_min": exp_min,
            "n_pi": n_pi, "exp_pi": exp_pi,
            "total": total_elems,
            "ok": ok_p,
        })

    ok = len(errors) == 0
    return {
        "name": "ORBIT_COUNTS",
        "ok": ok,
        "results": results,
        "errors": errors,
        "detail": "Orbit counts match (p-1)/ord_min, (p^2-p)/pi; total=p^2" if ok else f"FAIL: {errors}",
    }


def _check_equal_order_case() -> dict[str, Any]:
    """C4: split p=41 with equal root orders has only periods {1, pi(p)}."""
    p = 41
    roots = _roots_mod_p(p)
    r1, r2 = sorted(roots)
    o1, o2 = _ord_mod(r1, p), _ord_mod(r2, p)
    pi = _pisano(p)

    errors = []
    if o1 != o2:
        errors.append(f"p=41: expected equal orders, got ord({r1})={o1}, ord({r2})={o2}")

    by_period = _compute_orbits(p)
    actual = set(by_period.keys())
    expected = {1, pi}

    if actual != expected:
        errors.append(f"p=41: periods expected {expected}, got {actual}")

    n_pi = len(by_period.get(pi, []))
    exp_n_pi = (p * p - 1) // pi
    if n_pi != exp_n_pi:
        errors.append(f"p=41: n_pi={n_pi}, expected {exp_n_pi}")

    ok = len(errors) == 0
    return {
        "name": "EQUAL_ORDER_CASE",
        "ok": ok,
        "p": p,
        "root_orders": [o1, o2],
        "pi": pi,
        "actual_periods": sorted(actual),
        "n_period_pi": n_pi,
        "expected_n_pi": exp_n_pi,
        "errors": errors,
        "detail": f"p=41: equal root orders -> periods={{1,{pi}}} only" if ok else f"FAIL: {errors}",
    }


def _check_inert_contrast() -> dict[str, Any]:
    """C5: inert primes have only {1, pi(p)} orbit periods — no intermediate period."""
    inert_primes = [3, 7, 13, 17]
    errors = []
    results = []

    for p in inert_primes:
        if _classify_prime(p) != "inert":
            errors.append(f"p={p} not inert")
            continue
        pi = _pisano(p)
        by_period = _compute_orbits(p)
        actual = set(by_period.keys())
        expected = {1, pi}

        ok_p = actual == expected
        if not ok_p:
            errors.append(f"p={p}: expected {expected}, got {actual}")

        n_pi = len(by_period.get(pi, []))
        results.append({
            "p": p,
            "pi": pi,
            "actual_periods": sorted(actual),
            "n_pi_orbits": n_pi,
            "total_elements": 1 + n_pi * pi,
            "ok": ok_p,
        })

    ok = len(errors) == 0
    return {
        "name": "INERT_CONTRAST",
        "ok": ok,
        "results": results,
        "errors": errors,
        "detail": "Inert {3,7,13,17}: periods={1,pi(p)} only — intermediate period absent" if ok else f"FAIL: {errors}",
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "id": "P11_THREE_PERIODS",
        "expect": "PASS",
        "check": lambda: set(_compute_orbits(11).keys()) == {1, 5, 10},
        "desc": "p=11 (split): orbit periods = {1,5,10}",
    },
    {
        "id": "P11_EIGENSPACE_PERIOD5",
        "expect": "PASS",
        "check": lambda: (
            frozenset(e for start in _compute_orbits(11).get(5, [])
                      for e in _full_orbit(*start, 11))
            == frozenset((4 * c % 11, c) for c in range(1, 11))
        ),
        "desc": "p=11: period-5 orbits = {(4c,c): c in F*_11} = eigenspace of r_min=4",
    },
    {
        "id": "P3_INERT_NO_INTERMEDIATE",
        "expect": "PASS",
        "check": lambda: set(_compute_orbits(3).keys()) == {1, 8},
        "desc": "p=3 (inert): orbit periods = {1,8}, no intermediate",
    },
    {
        "id": "P41_EQUAL_ORDERS",
        "expect": "PASS",
        "check": lambda: (
            _ord_mod(7, 41) == _ord_mod(35, 41) == 40
            and set(_compute_orbits(41).keys()) == {1, 40}
        ),
        "desc": "p=41: equal root orders -> only {1,40}, no intermediate period",
    },
    {
        "id": "P11_INTERMEDIATE_NOT_IN_INERT",
        "expect": "FAIL",
        "check": lambda: 5 in set(_compute_orbits(3).keys()),
        "desc": "period 5 does NOT appear for inert p=3 (fixture must FAIL)",
    },
    {
        "id": "P29_ORBIT_COUNTS",
        "expect": "PASS",
        "check": lambda: (
            len(_compute_orbits(29).get(7, [])) == 4 and
            len(_compute_orbits(29).get(14, [])) == 58 and
            4 * 7 + 58 * 14 + 1 == 841
        ),
        "desc": "p=29: 4 orbits of period 7, 58 of period 14, 1 fixed, total=841=29^2",
    },
]


def _run_self_test() -> dict[str, Any]:
    checks = [
        _check_split_three_periods(),
        _check_eigenspace_identification(),
        _check_orbit_counts(),
        _check_equal_order_case(),
        _check_inert_contrast(),
    ]

    fixture_results = []
    for fix in FIXTURES:
        result = fix["check"]()
        passed = result if fix["expect"] == "PASS" else not result
        fixture_results.append({
            "id": fix["id"],
            "expect": fix["expect"],
            "actual": "PASS" if result else "FAIL",
            "ok": passed,
            "desc": fix["desc"],
        })

    all_checks_ok = all(c["ok"] for c in checks)
    all_fixtures_ok = all(f["ok"] for f in fixture_results)

    return {
        "ok": all_checks_ok and all_fixtures_ok,
        "checks": {c["name"]: c["ok"] for c in checks},
        "fixtures": fixture_results,
        "summary": {
            "checks_pass": sum(c["ok"] for c in checks),
            "checks_total": len(checks),
            "fixtures_pass": sum(f["ok"] for f in fixture_results),
            "fixtures_total": len(fixture_results),
        },
    }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        result = _run_self_test()
        print(json.dumps(result))
    else:
        result = _run_self_test()
        print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)
