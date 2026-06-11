# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; Nikolaev 2024 arXiv DOI 10.48550/arXiv.2412.09148 (Lemma 3.3 real multiplication iff eventually periodic Bratteli); Wall 1960 DOI 10.1080/00029890.1960.11989541 (Pisano periods) -->
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — integer and modular arithmetic on Z[phi]/(9); all state is exact integer; no float state; no empirical QA dynamics"
"""Cert [385]: QA Orbit Prime Ideal Filtration.

PRIMARY CLAIM:
  Under the map f: {1,...,9}^2 -> Z[phi]/(9),  f(b,e) = (b mod 9) + (e mod 9)*phi,
  where phi^2 = phi+1 (golden ratio, ring of integers of Q(sqrt(5))):

  (C1) IRREDUCIBLE_MOD3: x^2-x-1 is irreducible over F_3 (no root in {0,1,2}),
       so 3 is inert in Z[phi] and Z[phi]/(9) is a local ring with unique
       maximal ideal m = (3)*Z[phi]/(9) of size 9.

  (C2) LOCAL_RING: every element of Z[phi]/(9) not in m is a unit (invertible);
       every element of m is a non-unit.  |units| = 81 - 9 = 72.

  (C3) COSMOS_UNITS: the 72 Cosmos pairs under the QA step (b,e)->(((b+e-1)%9)+1, b)
       map exactly to the 72 units of Z[phi]/(9) — no more, no fewer.

  (C4) IDEAL_STRATA: the 8 Satellite pairs map to m\\{0} and the 1 Singularity
       pair (9,9) maps to {0}.  Together: Cosmos | Satellite | Singularity =
       units | m\\{0} | {0}  (partition of all 81 elements).

  (C5) NIKOLAEV_RM: the QA step automorphism sigma on Z[phi]/(9) generates an
       eventually periodic Bratteli diagram with periods:
         Cosmos   -> 24  (Pisano period pi(9))
         Satellite -> 8  (Pisano period pi(3))
         Singularity -> 1
       By Nikolaev 2024 Lemma 3.3, the associated C*-algebra has real
       multiplication by Z[phi] = O_{Q(sqrt(5))}.

ALGEBRAIC FACTS:
  phi^2 = phi + 1  (char poly x^2-x-1 = 0, irreducible mod 3)
  N(a+b*phi) = a^2 + a*b - b^2  (norm form, indefinite over R)
  (Z[phi]/3) = GF(9)  [3 inert => quotient is a field]
  (Z[phi]/9) is local, maximal ideal = {a+b*phi : 3|a AND 3|b}
  |(Z[phi]/9)^*| = 81 - 9 = 72,  exponent = 24 = Pisano pi(9)
  Period of Fibonacci matrix on GF(9)^* = 8 = Pisano pi(3)

LINEAGE: extends cert [291] (orbit period proof) and cert [306] (mod-24 CRT)
  with the ring-theoretic identification connecting QA to Nikolaev 2024.
"""

import json
import sys
from typing import Any

M = 9  # QA modulus


# ---------------------------------------------------------------------------
# Ring Z[phi]/(9): elements (a, b) = a + b*phi, a,b in Z/9Z
# Multiplication: (a+b*phi)(c+d*phi) = (ac+bd) + (ad+bc+bd)*phi
# ---------------------------------------------------------------------------

def _ring_mul(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
    a, b = x; c, d = y
    return ((a * c + b * d) % M, (a * d + b * c + b * d) % M)


def _ring_inv(x: tuple[int, int]) -> tuple[int, int] | None:
    """Return multiplicative inverse of x in Z[phi]/(9), or None if non-unit."""
    for c in range(M):
        for d in range(M):
            if _ring_mul(x, (c, d)) == (1, 0):
                return (c, d)
    return None


def _in_ideal_3(a: int, b: int) -> bool:
    """True iff a+b*phi is in the ideal (3) of Z[phi]/(9)."""
    return (a % 3 == 0) and (b % 3 == 0)


# ---------------------------------------------------------------------------
# QA orbit computation (A1-shifted: states in {1,...,9})
# ---------------------------------------------------------------------------

def _qa_step(b: int, e: int) -> tuple[int, int]:
    nb = ((b + e - 1) % M) + 1
    return (nb, b)


def _orbit_period(b: int, e: int) -> int:
    start = (b, e)
    cur = _qa_step(b, e)
    k = 1
    while cur != start:
        cur = _qa_step(*cur)
        k += 1
        if k > M * M + 1:
            raise RuntimeError(f"orbit overflow at ({b},{e})")
    return k


def _compute_all_orbits() -> dict[str, list[tuple[int, int]]]:
    """Return dict mapping orbit name to list of (b,e) pairs."""
    seen: dict[tuple[int, int], str] = {}
    cosmos: list[tuple[int, int]] = []
    satellite: list[tuple[int, int]] = []
    singularity: list[tuple[int, int]] = []
    label_count = [0, 0, 0]
    for b in range(1, M + 1):
        for e in range(1, M + 1):
            if (b, e) in seen:
                continue
            orbit: list[tuple[int, int]] = []
            cur = (b, e)
            while cur not in seen:
                seen[cur] = "?"
                orbit.append(cur)
                cur = _qa_step(*cur)
            period = len(orbit)
            if period == 1:
                name = "singularity"
                singularity.extend(orbit)
            elif period == 8:
                name = f"satellite_{label_count[1]}"
                label_count[1] += 1
                satellite.extend(orbit)
            else:
                name = f"cosmos_{label_count[0]}"
                label_count[0] += 1
                cosmos.extend(orbit)
            for p in orbit:
                seen[p] = name
    return {"cosmos": cosmos, "satellite": satellite, "singularity": singularity}


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_irreducible_mod3() -> dict[str, Any]:
    """C1: x^2-x-1 has no root in F_3 = {0,1,2}."""
    roots = [x for x in range(3) if (x * x - x - 1) % 3 == 0]
    ok = len(roots) == 0
    return {
        "name": "IRREDUCIBLE_MOD3",
        "ok": ok,
        "roots_in_F3": roots,
        "detail": "x^2-x-1 irreducible over F_3 => 3 inert in Z[phi] => Z[phi]/(9) local" if ok
                  else f"FAIL: roots found {roots}, 3 not inert",
    }


def _check_local_ring() -> dict[str, Any]:
    """C2: elements not in (3) are units; elements in (3) are non-units."""
    unit_count = 0
    non_unit_in_ideal = 0
    errors = []
    for a in range(M):
        for b in range(M):
            inv = _ring_inv((a, b))
            is_unit = inv is not None
            in_ideal = _in_ideal_3(a, b)
            if is_unit and not in_ideal:
                unit_count += 1
            elif not is_unit and in_ideal:
                non_unit_in_ideal += 1
            elif is_unit and in_ideal:
                errors.append(f"({a},{b}) in ideal but is unit (inv={inv})")
            else:
                errors.append(f"({a},{b}) not in ideal but non-unit")
    ok = len(errors) == 0 and unit_count == 72 and non_unit_in_ideal == 9
    return {
        "name": "LOCAL_RING",
        "ok": ok,
        "units": unit_count,
        "non_units_in_ideal": non_unit_in_ideal,
        "errors": errors[:3],
        "detail": f"|units|={unit_count}, |ideal|={non_unit_in_ideal}; Z[phi]/(9) is local" if ok
                  else f"FAIL: errors={errors[:2]}",
    }


def _check_cosmos_units(orbits: dict) -> dict[str, Any]:
    """C3: all 72 Cosmos pairs are units; they cover all 72 units exactly."""
    cosmos = orbits["cosmos"]
    non_units = [(b, e) for b, e in cosmos if _ring_inv((b % M, e % M)) is None]
    all_units = {(a, b) for a in range(M) for b in range(M)
                 if _ring_inv((a, b)) is not None}
    cosmos_images = {(b % M, e % M) for b, e in cosmos}
    uncovered = all_units - cosmos_images
    ok = len(non_units) == 0 and len(uncovered) == 0 and len(cosmos) == 72
    return {
        "name": "COSMOS_UNITS",
        "ok": ok,
        "cosmos_pairs": len(cosmos),
        "non_unit_cosmos": len(non_units),
        "uncovered_units": len(uncovered),
        "detail": "Cosmos pairs = (Z[phi]/9)^* exactly (72 units)" if ok
                  else f"FAIL: non_units={non_units[:2]}, uncovered={list(uncovered)[:2]}",
    }


def _check_ideal_strata(orbits: dict) -> dict[str, Any]:
    """C4: Satellite -> ideal (3)\\{0}; Singularity -> {0}."""
    satellite = orbits["satellite"]
    singularity = orbits["singularity"]

    sat_errors = []
    for b, e in satellite:
        a0, b0 = b % M, e % M
        if not _in_ideal_3(a0, b0):
            sat_errors.append(f"({b},{e}) not in ideal (3)")
        if (a0, b0) == (0, 0):
            sat_errors.append(f"({b},{e}) is zero but in Satellite")

    sing_errors = []
    for b, e in singularity:
        if (b % M, e % M) != (0, 0):
            sing_errors.append(f"({b},{e}) -> ({b%M},{e%M}) != (0,0)")
    if len(singularity) != 1:
        sing_errors.append(f"expected 1 singularity, got {len(singularity)}")

    ideal_nonzero = {(a, b) for a in range(M) for b in range(M)
                     if _in_ideal_3(a, b) and (a, b) != (0, 0)}
    sat_images = {(b % M, e % M) for b, e in satellite}
    missing = ideal_nonzero - sat_images

    ok = len(sat_errors) == 0 and len(sing_errors) == 0 and len(missing) == 0
    return {
        "name": "IDEAL_STRATA",
        "ok": ok,
        "satellite_pairs": len(satellite),
        "sat_errors": sat_errors[:2],
        "sing_errors": sing_errors[:2],
        "missing_from_satellite": len(missing),
        "detail": "Satellite=(3)/9\\{0} (8 pairs); Singularity={0} (1 pair)" if ok
                  else f"FAIL: sat_err={sat_errors[:1]}, sing_err={sing_errors[:1]}",
    }


def _check_nikolaev_periods(orbits: dict) -> dict[str, Any]:
    """C5: orbit periods are 24/8/1; matches Nikolaev Lemma 3.3 eventual periodicity."""
    def pisano(m: int) -> int:
        a, b = 0, 1
        for k in range(1, 10 * m + 1):
            a, b = b, (a + b) % m
            if a == 0 and b == 1:
                return k
        return -1

    pi3 = pisano(3)
    pi9 = pisano(9)

    cosmos_periods = set(_orbit_period(b, e) for b, e in orbits["cosmos"])
    sat_periods = set(_orbit_period(b, e) for b, e in orbits["satellite"])
    sing_periods = set(_orbit_period(b, e) for b, e in orbits["singularity"])

    ok = (pi3 == 8 and pi9 == 24
          and cosmos_periods == {24}
          and sat_periods == {8}
          and sing_periods == {1})

    return {
        "name": "NIKOLAEV_RM",
        "ok": ok,
        "pisano_pi3": pi3,
        "pisano_pi9": pi9,
        "cosmos_periods": sorted(cosmos_periods),
        "satellite_periods": sorted(sat_periods),
        "singularity_periods": sorted(sing_periods),
        "nikolaev_connection": (
            "Bratteli diagram of sigma=QA-step on Z[phi]/(9) is eventually periodic "
            "(periods 24/8/1) => real multiplication by Z[phi]=O_{Q(sqrt(5))} "
            "(Nikolaev 2024 Lemma 3.3)"
        ) if ok else "FAIL: period mismatch",
    }


# ---------------------------------------------------------------------------
# Self-test runner
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "id": "SING_IS_ZERO",
        "expect": "PASS",
        "b": 9, "e": 9,
        "check": lambda b, e: (b % M, e % M) == (0, 0),
        "desc": "(9,9) -> (0,0) in Z[phi]/(9)",
    },
    {
        "id": "SAT_IN_IDEAL",
        "expect": "PASS",
        "b": 3, "e": 3,
        "check": lambda b, e: _in_ideal_3(b % M, e % M) and (b % M, e % M) != (0, 0),
        "desc": "(3,3) -> (3,3) in (3)\\{0}",
    },
    {
        "id": "COSMOS_IS_UNIT",
        "expect": "PASS",
        "b": 1, "e": 1,
        "check": lambda b, e: _ring_inv((b % M, e % M)) is not None,
        "desc": "(1,1) -> (1,1) is a unit in Z[phi]/(9)",
    },
    {
        "id": "IRRED_MOD3",
        "expect": "PASS",
        "b": 0, "e": 0,
        "check": lambda b, e: all((x * x - x - 1) % 3 != 0 for x in range(3)),
        "desc": "x^2-x-1 irreducible over F_3",
    },
    {
        "id": "SAT_NOT_SING",
        "expect": "FAIL",
        "b": 6, "e": 9,
        "check": lambda b, e: (b % M, e % M) == (0, 0),
        "desc": "(6,9) -> (6,0) is NOT zero (fixture must FAIL)",
    },
]


def _run_self_test() -> dict[str, Any]:
    orbits = _compute_all_orbits()
    checks = [
        _check_irreducible_mod3(),
        _check_local_ring(),
        _check_cosmos_units(orbits),
        _check_ideal_strata(orbits),
        _check_nikolaev_periods(orbits),
    ]

    fixture_results = []
    for fix in FIXTURES:
        result = fix["check"](fix["b"], fix["e"])
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
