# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical algebraic number theory; Neukirch (1999) ISBN 978-3-540-65399-8 §I.8 (splitting of primes in quadratic fields); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano periods); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.5 (quadratic reciprocity, Legendre symbol) -->
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — integer arithmetic in Z[phi]/(p) and Z[phi]/(p^2); no float state; all ring operations are exact modular integer arithmetic"
"""Cert [386]: QA Inert/Split/Ramified Prime Classification for Z[phi].

PRIMARY CLAIM:
  For Z[phi] = O_{Q(sqrt(5))} (ring of integers, phi^2=phi+1):

  (C1) POLY_CLASSIFY: For every prime p <= 50, the number of roots of
       x^2-x-1 in Z/pZ classifies p:
         0 roots -> inert  (p generates a prime ideal in Z[phi])
         1 root  -> ramified (p | disc(Q(sqrt(5)))=5, so p=5 only)
         2 roots -> split  (p = pi*pi-bar with distinct prime ideals)

  (C2) MOD5_CRITERION: For all odd primes p != 5 with p <= 200:
       inert  <-> p mod 5 in {2, 3}  (5 is a non-residue mod p)
       split  <-> p mod 5 in {1, 4}  (5 is a residue mod p)
       p=2 is inert (5 equiv 5 mod 8, so 2 is inert by standard criterion).

  (C3) INERT_PRIMITIVE_ELEMENT: For inert primes p in {2, 3}:
       phi is a primitive element of GF(p^2) — i.e., ord(phi) = p^2-1
       in the multiplicative group of Z[phi]/(p).
       Consequence: the 8-cycle Satellite in cert [385] (p=3) is
       literally phi cycling through all of GF(9)^*.
       Note: phi is NOT primitive for all inert primes (e.g. ord(phi)=16
       in GF(49), not 48). Primitivity is certified only for p in {2, 3}.

  (C4) SPLIT_IDEMPOTENTS: For split prime p=11:
       x^2-x-1 = (x-8)(x-4) mod 11.
       The CRT decomposition Z[phi]/(11) = F_11 x F_11 is witnessed by
       orthogonal idempotents e1 = (10,3), e2 = (2,8) satisfying
       e1+e2 = (1,0), e1^2 = e1, e2^2 = e2, e1*e2 = (0,0).
       The unit group has order 100 = (p-1)^2 (product of two F_p^*).

  (C5) RAMIFIED_NILPOTENT: For p=5 (the unique ramified prime):
       x^2-x-1 has a double root at x=3 mod 5.
       The element (phi-3) = (2,1) in Z[phi]/(5) is nilpotent:
       (2,1)^2 = (0,0) mod 5.
       Unit group has order p^2-p = 20 (not p^4-p^2 as in the inert case).

ORBIT CONSEQUENCES (linking to cert [385]):
  - Inert p=3: Z[phi]/(9) is local; orbits = {Cosmos 72, Satellite 8, Sing 1}.
  - Inert p=2: Z[phi]/(4) is local; orbits = {Cosmos 12, Satellite 3, Sing 1}.
  - Inert p=7: Z[phi]/(49) is local; |Cosmos|=2352, |Satellite|=48.
  - Split p=11: Z[phi]/(11) not local; no clean 3-orbit structure.
  - Ramified p=5: Z[phi]/(5) has nilpotent stratum; distinct orbit geometry.

SOURCES: Neukirch (1999) Algebraic Number Theory ISBN 978-3-540-65399-8 S.I.8;
         Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano periods);
         Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.5.

LINEAGE: extends cert [385] (orbit = prime ideal filtration for inert p=3);
         lays groundwork for cert [387] (Witt vector sub-orbit invariant).
"""

import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Ring Z[phi]/(m): elements (a, b) = a + b*phi, a,b in Z/mZ
# phi^2 = phi + 1  =>  (a+b*phi)(c+d*phi) = (ac+bd) + (ad+bc+bd)*phi
# ---------------------------------------------------------------------------

def _ring_mul(x: tuple[int, int], y: tuple[int, int], m: int) -> tuple[int, int]:
    a, b = x; c, d = y
    return ((a * c + b * d) % m, (a * d + b * c + b * d) % m)


def _ring_inv(x: tuple[int, int], m: int) -> tuple[int, int] | None:
    for c in range(m):
        for d in range(m):
            if _ring_mul(x, (c, d), m) == (1, 0):
                return (c, d)
    return None


def _phi_order(p: int) -> int:
    """Order of phi=(0,1) in the multiplicative group of Z[phi]/(p)."""
    phi = (0, 1)
    cur = phi
    for k in range(1, p * p + 1):
        if cur == (1, 0):
            return k
        cur = _ring_mul(cur, phi, p)
    return -1


def _roots_mod_p(p: int) -> list[int]:
    """Roots of x^2 - x - 1 in Z/pZ."""
    return [x for x in range(p) if (x * x - x - 1) % p == 0]


def _classify_prime(p: int) -> str:
    n = len(_roots_mod_p(p))
    if n == 0:
        return "inert"
    if n == 1:
        return "ramified"
    return "split"


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _primes_up_to(n: int) -> list[int]:
    return [p for p in range(2, n + 1) if _is_prime(p)]


def _in_ideal_p(a: int, b: int, p: int) -> bool:
    return (a % p == 0) and (b % p == 0)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_poly_classify() -> dict[str, Any]:
    """C1: classify all primes <= 50 by root count of x^2-x-1 mod p."""
    primes = _primes_up_to(50)
    errors = []

    expected_inert    = {2, 3, 7, 13, 17, 23, 37, 43, 47}
    expected_split    = {11, 19, 29, 31, 41}
    expected_ramified = {5}

    table: list[dict] = []
    for p in primes:
        cls = _classify_prime(p)
        table.append({"p": p, "class": cls, "roots": _roots_mod_p(p)})
        if cls == "inert" and p not in expected_inert:
            errors.append(f"p={p} got inert, unexpected")
        elif cls == "split" and p not in expected_split:
            errors.append(f"p={p} got split, unexpected")
        elif cls == "ramified" and p not in expected_ramified:
            errors.append(f"p={p} got ramified, unexpected")
        elif p in expected_inert and cls != "inert":
            errors.append(f"p={p} should be inert, got {cls}")
        elif p in expected_split and cls != "split":
            errors.append(f"p={p} should be split, got {cls}")
        elif p in expected_ramified and cls != "ramified":
            errors.append(f"p={p} should be ramified, got {cls}")

    inert_n = sum(1 for r in table if r["class"] == "inert")
    split_n = sum(1 for r in table if r["class"] == "split")
    ram_n   = sum(1 for r in table if r["class"] == "ramified")

    ok = len(errors) == 0
    return {
        "name": "POLY_CLASSIFY",
        "ok": ok,
        "inert_count": inert_n,
        "split_count": split_n,
        "ramified_count": ram_n,
        "table": table,
        "errors": errors[:3],
        "detail": (
            f"primes<=50: {inert_n} inert, {split_n} split, {ram_n} ramified"
            if ok else f"FAIL: {errors[:2]}"
        ),
    }


def _check_mod5_criterion() -> dict[str, Any]:
    """C2: inert <-> p%5 in {2,3}; split <-> p%5 in {1,4}; p=2 inert."""
    primes = _primes_up_to(200)
    errors = []
    checked = 0

    for p in primes:
        cls = _classify_prime(p)
        if p == 5:
            if cls != "ramified":
                errors.append(f"p=5 must be ramified, got {cls}")
            continue
        if p == 2:
            if cls != "inert":
                errors.append("p=2 must be inert")
            checked += 1
            continue
        r = p % 5
        expected = "inert" if r in {2, 3} else "split"
        if cls != expected:
            errors.append(f"p={p} (p%5={r}): expected {expected}, got {cls}")
        checked += 1

    ok = len(errors) == 0
    return {
        "name": "MOD5_CRITERION",
        "ok": ok,
        "primes_checked": checked,
        "errors": errors[:3],
        "detail": (
            f"All {checked} primes <=200 (excl. 5): p%5 in {{2,3}} <-> inert"
            if ok else f"FAIL: {errors[:2]}"
        ),
    }


def _check_inert_primitive_element() -> dict[str, Any]:
    """C3: phi is a primitive element of GF(p^2) for inert p in {2,3}."""
    inert_primes = [2, 3]
    results = []
    errors = []
    for p in inert_primes:
        expected_order = p * p - 1
        actual_order = _phi_order(p)
        is_prim = actual_order == expected_order
        results.append({
            "p": p,
            "gf_size": p * p,
            "expected_order": expected_order,
            "actual_order": actual_order,
            "primitive": is_prim,
        })
        if not is_prim:
            errors.append(
                f"p={p}: ord(phi) in GF({p*p}) = {actual_order}, "
                f"expected {expected_order}"
            )
    ok = len(errors) == 0
    return {
        "name": "INERT_PRIMITIVE_ELEMENT",
        "ok": ok,
        "results": results,
        "errors": errors,
        "detail": (
            "phi is primitive in GF(4) and GF(9) — "
            "Satellite 8-cycle IS phi cycling through GF(9)^*"
            if ok else f"FAIL: {errors}"
        ),
    }


def _check_split_idempotents() -> dict[str, Any]:
    """C4: CRT idempotents for split prime p=11; Z[phi]/(11) = F_11 x F_11."""
    p = 11
    roots = _roots_mod_p(p)
    if len(roots) != 2:
        return {
            "name": "SPLIT_IDEMPOTENTS",
            "ok": False,
            "detail": f"FAIL: p=11 should be split, got roots={roots}",
        }

    lam1, lam2 = sorted(roots)
    diff = (lam1 - lam2) % p
    diff_inv = pow(diff, -1, p)
    neg_lam2 = (-lam2) % p
    e1 = (diff_inv * neg_lam2 % p, diff_inv % p)
    e2 = ((1 - e1[0]) % p, (-e1[1]) % p)

    errors = []
    s = ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    if s != (1, 0):
        errors.append(f"e1+e2={s} != (1,0)")
    e1sq = _ring_mul(e1, e1, p)
    if e1sq != e1:
        errors.append(f"e1^2={e1sq} != e1={e1}")
    e2sq = _ring_mul(e2, e2, p)
    if e2sq != e2:
        errors.append(f"e2^2={e2sq} != e2={e2}")
    e1e2 = _ring_mul(e1, e2, p)
    if e1e2 != (0, 0):
        errors.append(f"e1*e2={e1e2} != (0,0)")

    unit_count = sum(
        1 for a in range(p) for b in range(p)
        if _ring_inv((a, b), p) is not None
    )
    expected_units = (p - 1) * (p - 1)
    if unit_count != expected_units:
        errors.append(f"|units|={unit_count}, expected {expected_units}")

    ok = len(errors) == 0
    return {
        "name": "SPLIT_IDEMPOTENTS",
        "ok": ok,
        "p": p,
        "roots": [lam1, lam2],
        "e1": list(e1),
        "e2": list(e2),
        "unit_count": unit_count,
        "expected_units": expected_units,
        "errors": errors,
        "detail": (
            f"Z[phi]/(11)=F_11xF_11; e1={e1},e2={e2}; |units|={unit_count}"
            if ok else f"FAIL: {errors}"
        ),
    }


def _check_ramified_nilpotent() -> dict[str, Any]:
    """C5: (phi-3)^2=0 mod 5; unit count = p^2-p = 20 (not p^4-p^2)."""
    p = 5
    roots = _roots_mod_p(p)
    if roots != [3]:
        return {
            "name": "RAMIFIED_NILPOTENT",
            "ok": False,
            "detail": f"FAIL: expected [3], got roots={roots}",
        }

    nilp = (2, 1)
    nilp_sq = _ring_mul(nilp, nilp, p)

    deriv_at_root = (2 * 3 - 1) % p

    unit_count = sum(
        1 for a in range(p) for b in range(p)
        if _ring_inv((a, b), p) is not None
    )
    expected_units = p * p - p

    errors = []
    if nilp_sq != (0, 0):
        errors.append(f"(2,1)^2={nilp_sq}, expected (0,0)")
    if nilp == (0, 0):
        errors.append("(2,1) is zero — should be nonzero nilpotent")
    if unit_count != expected_units:
        errors.append(f"|units|={unit_count}, expected {expected_units}")
    if deriv_at_root != 0:
        errors.append(f"derivative at root={deriv_at_root}, expected 0")

    ok = len(errors) == 0
    return {
        "name": "RAMIFIED_NILPOTENT",
        "ok": ok,
        "p": p,
        "root": roots[0],
        "nilpotent_element": list(nilp),
        "nilpotent_sq": list(nilp_sq),
        "derivative_at_root_mod_p": deriv_at_root,
        "unit_count": unit_count,
        "expected_units": expected_units,
        "errors": errors,
        "detail": (
            f"(phi-3)^2=0 mod 5; |units|=20=p^2-p"
            if ok else f"FAIL: {errors}"
        ),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "id": "P3_INERT",
        "expect": "PASS",
        "check": lambda: _classify_prime(3) == "inert",
        "desc": "p=3: x^2-x-1 irreducible mod 3 => inert",
    },
    {
        "id": "P5_RAMIFIED",
        "expect": "PASS",
        "check": lambda: _classify_prime(5) == "ramified" and _roots_mod_p(5) == [3],
        "desc": "p=5: double root at 3 => ramified",
    },
    {
        "id": "P11_SPLIT",
        "expect": "PASS",
        "check": lambda: _classify_prime(11) == "split" and sorted(_roots_mod_p(11)) == [4, 8],
        "desc": "p=11: roots 4,8 => split",
    },
    {
        "id": "PHI_PRIMITIVE_GF9",
        "expect": "PASS",
        "check": lambda: _phi_order(3) == 8,
        "desc": "phi is primitive in GF(9): ord(phi)=8=9-1",
    },
    {
        "id": "PHI_PRIMITIVE_GF4",
        "expect": "PASS",
        "check": lambda: _phi_order(2) == 3,
        "desc": "phi is primitive in GF(4): ord(phi)=3=4-1",
    },
    {
        "id": "P7_NOT_SPLIT",
        "expect": "FAIL",
        "check": lambda: _classify_prime(7) == "split",
        "desc": "p=7 is inert, not split (fixture must FAIL)",
    },
]


def _run_self_test() -> dict[str, Any]:
    checks = [
        _check_poly_classify(),
        _check_mod5_criterion(),
        _check_inert_primitive_element(),
        _check_split_idempotents(),
        _check_ramified_nilpotent(),
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
        "check_details": checks,
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
        sys.exit(0 if result["ok"] else 1)
    else:
        result = _run_self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)
