# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Wall 1960 DOI 10.1080/00029890.1960.11989541; Wildberger 2005 ISBN 978-0-9757492-0-8) -->
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z) and Fibonacci sequences; no empirical QA state machine; all state is exact integer"
"""Cert [291]: QA Fibonacci Matrix Orbit Periods.

PRIMARY CLAIM:
  For QA mod m=9, the Fibonacci matrix M = [[0,1],[1,1]] acting on (Z/9Z)^2
  by (b,e) -> (e, b+e) mod 9 has:

  (1) FMO_PISANO_24: M^24 = I_2 (mod 9)  — the Pisano period pi(9) = 24
  (2) FMO_PISANO_MIN: no proper divisor k of 24 satisfies M^k = I_2 (mod 9)
  (3) Three orbit types partition {0,...,8}^2 = (Z/9Z)^2:
      - Singularity: {(0,0)}, period 1  [= (9,9) in {1,...,9} notation]
      - Satellite:   {(b,e) : 3|b AND 3|e} \\ {(0,0)}, 8 states, period 8
      - Cosmos:      all remaining 72 states, period 24
  (4) Partition: 1 + 8 + 72 = 81 = 9^2

KEY MATRIX FACTS (all mod 9):
  M^8  = [[4,3],[3,7]]  (not identity; kernel = {3|b AND 3|e})
  M^12 = [[8,0],[0,8]]  = -I_2  (order does not divide 12)
  M^24 = [[1,0],[0,1]]  = I_2   (order divides 24)

FIVE-FAMILIES ALIGNMENT:
  Fibonacci/Lucas/Phibonacci sequence pairs -> all period 24 (Cosmos)
  Tribonacci pairs -> period 8 (Satellite: 3|b AND 3|e)
  Ninbonacci pair  -> period 1 (Singularity: (9,9) = (0,0) mod 9)

GCD-3 STRUCTURAL REQUIREMENT (extends the cross-modulus rule above):
  The Satellite characterization {3|b AND 3|e} is only a genuine subgroup of
  (Z/mZ)^2 when 3|m (adding m never changes a residue's value mod 3, so
  "b%3==0" is a well-defined quotient predicate). When gcd(m,3)=1, 3 is a
  unit mod m, and the same literal predicate is NOT closed under addition
  mod m (e.g. m=10: 9%3==0 and 3%3==0, but (9+3)%10=2, 2%3!=0). PROVED for
  all such m (this closure failure is a direct consequence of 3 being
  invertible, not a per-m computation).

  CLAIM (narrow, falsifiable, VERIFIED not proved-in-general): for the 8
  tested moduli m in {7,10,11,13,14,16,17,20} (all gcd(m,3)=1), the
  Satellite class is EMPTY -- no non-origin state has orbit period 8, and
  the predicate-tree's period8_fixed flag never fires nontrivially. This
  is CONSISTENT with (but not a full proof of) a general theorem: M^8-I
  mod 9 = [[12,21],[21,33]] = 3*[[4,7],[7,11]]; the factor of 3 is
  invertible whenever gcd(m,3)=1, so the period8_fixed system reduces to a
  determinant-(-5) system with only the trivial solution WHEN gcd(m,5)=1
  (m=7,11,13,14,16,17 above). For m=10,20 (both divisible by 5), emptiness
  is confirmed by exhaustive enumeration, not yet by a closed-form argument
  for the 5|m case -- that gap is open, not glossed over. The orbit-period
  spectrum for all 8 moduli is instead a richer, m-specific divisor lattice
  of pi(m) (e.g. m=7: only {1,16}; m=10: six distinct periods
  {1,3,4,12,20,60}; no period-8 class anywhere).

  This is SUGGESTIVE of (not proved identical to) the structural mechanism
  cert [515] (QA Orbit-Lattice Mod-3 Collapse) proves for its own qa_step
  NTRU-coefficient recursion: both certs independently find that reduction
  mod 3 is only a well-defined quotient map when 3|m, and both find the
  m-divisible-by-3 case carries extra low-period structure the
  gcd(m,3)=1 case lacks. [515] does NOT use the Fibonacci matrix M
  directly, so this cert does not establish that its Satellite mechanism
  literally *is* [515]'s NTRU weakness -- only that they share the same
  mod-3-invertibility root cause.

  Checks: FMO_GCD3_SUBGROUP_CLOSURE / FMO_SATELLITE_EMPTY_OFF3 /
  FMO_NONMULT3_SPECTRUM.
"""

from typing import List, Tuple

M = 9  # modulus


# ---------------------------------------------------------------------------
# Matrix helpers (2x2 over Z/mZ)
# ---------------------------------------------------------------------------

Matrix = List[List[int]]


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % M,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % M],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % M,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % M],
    ]


def _mat_pow(mat: Matrix, k: int) -> Matrix:
    result: Matrix = [[1, 0], [0, 1]]
    for _ in range(k):
        result = _mat_mul(result, mat)
    return result


def _apply(b: int, e: int) -> Tuple[int, int]:
    """One step of M: (b,e) -> (e, b+e) mod 9.  Uses 0-indexed states."""
    return e % M, (b + e) % M


def _orbit_period(b: int, e: int) -> int:
    """Period of (b,e) under repeated M application."""
    start = (b % M, e % M)
    cur = _apply(*start)
    k = 1
    while cur != start:
        cur = _apply(*cur)
        k += 1
        if k > M * M + 1:
            raise RuntimeError(f"orbit period overflow for ({b},{e})")
    return k


def _orbit_type(b: int, e: int) -> str:
    b0, e0 = b % M, e % M
    if b0 == 0 and e0 == 0:
        return "singularity"
    if b0 % 3 == 0 and e0 % 3 == 0:
        return "satellite"
    return "cosmos"


def _mat_mul_int(A: Matrix, B: Matrix) -> Matrix:
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0],
         A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0],
         A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


def _mat_pow_int(mat: Matrix, k: int) -> Matrix:
    result: Matrix = [[1, 0], [0, 1]]
    for _ in range(k):
        result = _mat_mul_int(result, mat)
    return result


def _matrix_minus_identity(mat: Matrix) -> Matrix:
    return [[mat[0][0] - 1, mat[0][1]], [mat[1][0], mat[1][1] - 1]]


def _apply_mod(f0: int, f1: int, modulus: int) -> Tuple[int, int]:
    return f1 % modulus, (f0 + f1) % modulus


def _orbit_period_mod(f0: int, f1: int, modulus: int) -> int:
    start = (f0 % modulus, f1 % modulus)
    cur = _apply_mod(*start, modulus)
    k = 1
    while cur != start:
        cur = _apply_mod(*cur, modulus)
        k += 1
        if k > modulus * modulus + 1:
            raise RuntimeError(f"orbit period overflow for m={modulus}, state={start}")
    return k


def _period_family_by_orbit(b: int, e: int, modulus: int) -> str:
    period = _orbit_period_mod(b, e, modulus)
    if period == 1:
        return "singularity"
    if period == 8:
        return "satellite"
    return "cosmos"


def _period_predicate_flags(b: int, e: int, modulus: int) -> dict:
    B = b % modulus
    E = e % modulus
    period1_fixed = B == 0 and E == 0
    period4_fixed = (B + 3*E) % modulus == 0 and (3*B + 4*E) % modulus == 0
    period8_fixed = (12*B + 21*E) % modulus == 0 and (21*B + 33*E) % modulus == 0
    return {
        "period1_fixed": period1_fixed,
        "period4_fixed": period4_fixed,
        "period8_fixed": period8_fixed,
        "period8_nontrivial": period8_fixed and not period4_fixed,
    }


def _period_family_by_predicate(b: int, e: int, modulus: int) -> str:
    flags = _period_predicate_flags(b, e, modulus)
    if flags["period1_fixed"]:
        return "singularity"
    if flags["period8_nontrivial"]:
        return "satellite"
    return "cosmos"


_CROSS_MODULI = [9, 12, 15, 18, 21, 24, 27, 30]
_EXPECTED_CROSS_COUNTS = {
    9: {"cosmos": 72, "satellite": 8, "singularity": 1},
    12: {"cosmos": 135, "satellite": 8, "singularity": 1},
    15: {"cosmos": 184, "satellite": 40, "singularity": 1},
    18: {"cosmos": 315, "satellite": 8, "singularity": 1},
    21: {"cosmos": 432, "satellite": 8, "singularity": 1},
    24: {"cosmos": 567, "satellite": 8, "singularity": 1},
    27: {"cosmos": 720, "satellite": 8, "singularity": 1},
    30: {"cosmos": 859, "satellite": 40, "singularity": 1},
}


_NONMULT3_MODULI = [7, 10, 11, 13, 14, 16, 17, 20]
_NONMULT3_EXPECTED_SPECTRUM = {
    7:  {1: 1, 16: 48},
    10: {1: 1, 3: 3, 4: 4, 12: 12, 20: 20, 60: 60},
    11: {1: 1, 5: 10, 10: 110},
    13: {1: 1, 28: 168},
    14: {1: 1, 3: 3, 16: 48, 48: 144},
    16: {1: 1, 3: 3, 6: 12, 12: 48, 24: 192},
    17: {1: 1, 36: 288},
    20: {1: 1, 3: 3, 4: 4, 6: 12, 12: 60, 20: 20, 60: 300},
}


def _b_mod3_subgroup_closed(modulus: int) -> bool:
    """Is {b in range(modulus): b%3==0} closed under addition mod modulus?

    True exactly when 3|modulus (a genuine quotient-to-Z/3Z predicate).
    False when gcd(modulus,3)=1 (3 is a unit; the literal integer
    condition on canonical representatives is not a subgroup).
    """
    subset = [b for b in range(modulus) if b % 3 == 0]
    return all((a + b) % modulus in subset for a in subset for b in subset)


def _gcd3_report(moduli: List[int]) -> dict:
    """For each modulus: subgroup closure, period spectrum, satellite emptiness."""
    from collections import Counter

    report = {}
    for modulus in moduli:
        closed = _b_mod3_subgroup_closed(modulus)
        dist = Counter(
            _orbit_period_mod(b, e, modulus)
            for b in range(modulus) for e in range(modulus)
        )
        no_period8 = 8 not in dist
        no_nontrivial_period8_flag = all(
            not _period_predicate_flags(b, e, modulus)["period8_nontrivial"]
            for b in range(modulus) for e in range(modulus)
            if not (b == 0 and e == 0)
        )
        report[modulus] = {
            "closed": closed,
            "spectrum": dict(sorted(dist.items())),
            "satellite_empty": no_period8 and no_nontrivial_period8_flag,
        }
    return report


def _exact_period_rule_report(moduli: List[int]) -> dict:
    from collections import Counter

    errors = []
    counts = {}
    for modulus in moduli:
        counter = Counter()
        for b in range(modulus):
            for e in range(modulus):
                actual = _period_family_by_orbit(b, e, modulus)
                predicted = _period_family_by_predicate(b, e, modulus)
                counter[actual] += 1
                if actual != predicted:
                    errors.append({
                        "m": modulus,
                        "state": [b, e],
                        "actual": actual,
                        "predicted": predicted,
                        "flags": _period_predicate_flags(b, e, modulus),
                    })
        counts[modulus] = dict(sorted(counter.items()))
    return {"errors": errors, "counts": counts}


# ---------------------------------------------------------------------------
# Checks
# FMO_PISANO_24 — M^24 = I_2 mod 9
# FMO_PISANO_MIN — no proper divisor k of 24 gives M^k = I_2 mod 9
# FMO_SAT_CHAR  — satellite states = {3|b, 3|e} \ {(0,0)}, exactly 8
# FMO_PARTITION — 1 + 8 + 72 = 81
# FMO_ORBIT     — fixture state has the declared orbit period
# FMO_TYPE      — fixture state has the declared orbit type
# ---------------------------------------------------------------------------

_IDENTITY: Matrix = [[1, 0], [0, 1]]
_FIBO_MAT: Matrix = [[0, 1], [1, 1]]
_PROPER_DIVISORS_24 = [1, 2, 3, 4, 6, 8, 12]


def _check_global() -> dict:
    """Checks that depend only on M and m=9, not on a fixture state."""
    results: dict = {}

    # FMO_PISANO_24
    results["FMO_PISANO_24"] = _mat_pow(_FIBO_MAT, 24) == _IDENTITY

    # FMO_PISANO_MIN
    results["FMO_PISANO_MIN"] = all(
        _mat_pow(_FIBO_MAT, k) != _IDENTITY for k in _PROPER_DIVISORS_24
    )

    # FMO_SAT_CHAR
    sat = [(b, e) for b in range(M) for e in range(M)
           if b % 3 == 0 and e % 3 == 0 and not (b == 0 and e == 0)]
    results["FMO_SAT_CHAR"] = len(sat) == 8 and all(
        _orbit_period(b, e) == 8 for b, e in sat
    )

    # FMO_PARTITION
    sing = sum(1 for b in range(M) for e in range(M) if _orbit_type(b, e) == "singularity")
    sat_n = sum(1 for b in range(M) for e in range(M) if _orbit_type(b, e) == "satellite")
    cos_n = sum(1 for b in range(M) for e in range(M) if _orbit_type(b, e) == "cosmos")
    results["FMO_PARTITION"] = (sing == 1 and sat_n == 8 and cos_n == 72
                                and sing + sat_n + cos_n == M * M)

    # FMO_EXACT_PERIOD_KERNELS: integer matrix identities that define the
    # period-4 and period-8 residual predicates used by the theorem-map model.
    t4_minus_i = _matrix_minus_identity(_mat_pow_int(_FIBO_MAT, 4))
    t8_minus_i = _matrix_minus_identity(_mat_pow_int(_FIBO_MAT, 8))
    results["FMO_EXACT_PERIOD_KERNELS"] = (
        t4_minus_i == [[1, 3], [3, 4]]
        and t8_minus_i == [[12, 21], [21, 33]]
    )

    # FMO_CROSS_MODULUS_RULE: the exact predicate tree
    #   period8_fixed AND NOT period4_fixed -> satellite
    #   period1_fixed -> singularity
    #   otherwise -> cosmos
    # matches actual integer orbit periods on the pressure-test moduli.
    report = _exact_period_rule_report(_CROSS_MODULI)
    results["FMO_CROSS_MODULUS_RULE"] = (
        not report["errors"]
        and report["counts"] == _EXPECTED_CROSS_COUNTS
    )

    # FMO_GCD3_SUBGROUP_CLOSURE: {b%3==0} is closed under mod-m addition
    # iff 3|m -- true for every modulus in _CROSS_MODULI, false for every
    # modulus in _NONMULT3_MODULI.
    results["FMO_GCD3_SUBGROUP_CLOSURE"] = (
        all(_b_mod3_subgroup_closed(m) for m in _CROSS_MODULI)
        and not any(_b_mod3_subgroup_closed(m) for m in _NONMULT3_MODULI)
    )

    # FMO_SATELLITE_EMPTY_OFF3 / FMO_NONMULT3_SPECTRUM: for the tested
    # gcd(m,3)=1 moduli, the Satellite class (period 8) is empty (proved
    # via the determinant-(-5) argument when gcd(m,5)=1; confirmed only by
    # enumeration for m=10,20 -- the 5|m case is not proved in general),
    # and the orbit period spectrum matches the exhaustively-verified
    # divisor lattice.
    gcd3_report = _gcd3_report(_NONMULT3_MODULI)
    results["FMO_SATELLITE_EMPTY_OFF3"] = all(
        gcd3_report[m]["satellite_empty"] for m in _NONMULT3_MODULI
    )
    results["FMO_NONMULT3_SPECTRUM"] = all(
        gcd3_report[m]["spectrum"] == _NONMULT3_EXPECTED_SPECTRUM[m]
        for m in _NONMULT3_MODULI
    )

    return results


def validate_fixture(fixture: dict) -> dict:
    if fixture.get("kind") == "cross_modulus_exact_period_rule":
        moduli = fixture.get("moduli", _CROSS_MODULI)
        expected_counts = {int(k): v for k, v in fixture.get("expected_counts", {}).items()}
        expected_matrix = fixture.get("expected_residual_matrices", {})
        report = _exact_period_rule_report(moduli)
        t4_minus_i = _matrix_minus_identity(_mat_pow_int(_FIBO_MAT, 4))
        t8_minus_i = _matrix_minus_identity(_mat_pow_int(_FIBO_MAT, 8))
        return {
            "FMO_EXACT_PERIOD_KERNELS": (
                t4_minus_i == expected_matrix.get("T4_minus_I")
                and t8_minus_i == expected_matrix.get("T8_minus_I")
            ),
            "FMO_CROSS_MODULUS_RULE": not report["errors"],
            "FMO_CROSS_MODULUS_COUNTS": report["counts"] == expected_counts,
        }

    if fixture.get("kind") == "gcd3_satellite_collapse":
        moduli = fixture.get("moduli", _NONMULT3_MODULI)
        expected_spectrum = {
            int(k): {int(kk): vv for kk, vv in v.items()}
            for k, v in fixture.get("expected_spectrum", {}).items()
        }
        report = _gcd3_report(moduli)
        return {
            "FMO_GCD3_SUBGROUP_CLOSURE": not any(report[m]["closed"] for m in moduli),
            "FMO_SATELLITE_EMPTY_OFF3": all(report[m]["satellite_empty"] for m in moduli),
            "FMO_NONMULT3_SPECTRUM": all(
                report[m]["spectrum"] == expected_spectrum.get(m) for m in moduli
            ),
        }

    b, e = fixture["state"]
    declared_period: int = fixture["expected_period"]
    declared_type: str = fixture["expected_orbit_type"]

    results: dict = {}

    # Per-fixture state checks
    actual_period = _orbit_period(b % M, e % M)
    actual_type = _orbit_type(b, e)

    results["FMO_ORBIT"] = actual_period == declared_period
    results["FMO_TYPE"] = actual_type == declared_type

    # Global checks (same for every fixture, but included so failures are visible)
    results.update(_check_global())

    return results


def self_test() -> bool:
    failures = []

    # Exhaustive orbit period distribution
    from collections import Counter
    period_dist = Counter(_orbit_period(b, e) for b in range(M) for e in range(M))
    if period_dist != {1: 1, 8: 8, 24: 72}:
        failures.append(f"Wrong period distribution: {dict(period_dist)}")

    # Global invariants
    g = _check_global()
    for k, v in g.items():
        if not v:
            failures.append(f"Global check {k} FAIL")

    # Five-families alignment: Fibonacci pairs are all period-24
    fibs = [0, 1]
    for _ in range(26):
        fibs.append((fibs[-1] + fibs[-2]) % M)
    fib_pairs = set((fibs[i], fibs[i+1]) for i in range(len(fibs)-1))
    for b, e in fib_pairs:
        if _orbit_period(b, e) != 24:
            failures.append(f"Fibonacci pair ({b},{e}) not period-24")

    # Satellite orbit: (3,3) -> all 8 satellite states in one orbit
    orbit_33 = []
    cur = (3, 3)
    for _ in range(8):
        orbit_33.append(cur)
        cur = _apply(*cur)
    if cur != (3, 3):
        failures.append(f"Satellite orbit from (3,3) not period-8")
    sat_expected = {(b, e) for b in range(M) for e in range(M)
                   if b % 3 == 0 and e % 3 == 0 and not (b == 0 and e == 0)}
    if set(orbit_33) != sat_expected:
        failures.append(f"Satellite orbit != expected sat states")

    # M^12 = -I (order does not divide 12)
    m12 = _mat_pow(_FIBO_MAT, 12)
    if m12 != [[8, 0], [0, 8]]:
        failures.append(f"M^12 != -I: {m12}")

    cross = _exact_period_rule_report(_CROSS_MODULI)
    if cross["errors"]:
        failures.append(f"Exact period predicate rule errors: {cross['errors'][:3]}")
    if cross["counts"] != _EXPECTED_CROSS_COUNTS:
        failures.append(f"Exact period predicate counts wrong: {cross['counts']}")

    if _matrix_minus_identity(_mat_pow_int(_FIBO_MAT, 4)) != [[1, 3], [3, 4]]:
        failures.append("T^4-I residual matrix mismatch")
    if _matrix_minus_identity(_mat_pow_int(_FIBO_MAT, 8)) != [[12, 21], [21, 33]]:
        failures.append("T^8-I residual matrix mismatch")

    # GCD-3 structural requirement: satellite subgroup exists iff 3|m
    gcd3 = _gcd3_report(_NONMULT3_MODULI)
    for modulus in _NONMULT3_MODULI:
        if gcd3[modulus]["closed"]:
            failures.append(f"m={modulus}: b%3==0 subgroup wrongly closed (gcd(m,3)=1)")
        if not gcd3[modulus]["satellite_empty"]:
            failures.append(f"m={modulus}: satellite class unexpectedly nonempty")
        if gcd3[modulus]["spectrum"] != _NONMULT3_EXPECTED_SPECTRUM[modulus]:
            failures.append(f"m={modulus}: period spectrum mismatch: {gcd3[modulus]['spectrum']}")
    if not all(_b_mod3_subgroup_closed(m) for m in _CROSS_MODULI):
        failures.append("subgroup closure false for a multiple-of-3 modulus")

    # Fail fixtures must fail
    fail_cases = [
        {"state": [1, 1], "expected_period": 8, "expected_orbit_type": "cosmos", "expected": "FAIL"},
        {"state": [3, 3], "expected_period": 8, "expected_orbit_type": "cosmos", "expected": "FAIL"},
        {
            "kind": "cross_modulus_exact_period_rule",
            "moduli": [9, 12],
            "expected_residual_matrices": {
                "T4_minus_I": [[1, 3], [3, 4]],
                "T8_minus_I": [[12, 21], [21, 34]],
            },
            "expected_counts": {
                9: {"cosmos": 72, "satellite": 8, "singularity": 1},
                12: {"cosmos": 135, "satellite": 8, "singularity": 1},
            },
            "expected": "FAIL",
        },
        {
            "kind": "gcd3_satellite_collapse",
            "moduli": [7, 10],
            "expected_spectrum": {
                "7": {"1": 1, "16": 48},
                "10": {"1": 1, "3": 3, "4": 4, "12": 12, "20": 20, "60": 61},
            },
            "expected": "FAIL",
        },
    ]
    for case in fail_cases:
        checks = validate_fixture(case)
        if all(checks.values()):
            failures.append(f"Expected FAIL case passed: {case}")

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
    return len(failures) == 0


FAMILY_ID = 291
CERT_SLUG = "qa_fibonacci_matrix_orbit_periods_cert_v1"


def validate_cert_family(cert_dir) -> Tuple[bool, List[str]]:
    import json
    from pathlib import Path

    errors: List[str] = []
    fixture_dir = Path(cert_dir) / "fixtures"
    if not fixture_dir.is_dir():
        errors.append("missing fixtures/ directory")
        return False, errors

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        if all_pass == expect_pass:
            pass_count += 1
        else:
            fail_count += 1
            errors.append(
                f"fixture {path.name}: expected={'PASS' if expect_pass else 'FAIL'} "
                f"got={'PASS' if all_pass else 'FAIL'} checks={checks}"
            )

    if fail_count:
        errors.append(f"{fail_count} fixture(s) had wrong outcome")
    return fail_count == 0, errors


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="QA Fibonacci Matrix Orbit Periods Cert validator [291]"
    )
    parser.add_argument("cert_dir", nargs="?", default=str(Path(__file__).parent))
    parser.add_argument("--self-test", action="store_true", dest="selftest")
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir)
    fixture_dir = cert_dir / "fixtures"

    if args.selftest:
        st_ok = self_test()
        fam_ok, fam_errors = validate_cert_family(cert_dir)
        fix_files = list(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
        pass_files = [f for f in fix_files if "pass_" in f.name]
        fail_files = [f for f in fix_files if "fail_" in f.name]
        errors = ([] if st_ok else ["self_test FAIL"]) + fam_errors
        payload = {
            "ok": st_ok and fam_ok,
            "family_id": FAMILY_ID,
            "slug": CERT_SLUG,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": errors,
        }
        print(json.dumps(payload, sort_keys=True))
        sys.exit(0 if payload["ok"] else 1)

    if not self_test():
        print("SELF_TEST FAIL")
        sys.exit(1)
    print("SELF_TEST PASS")

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        ok = all_pass == expect_pass
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        print(f"{'PASS' if ok else 'FAIL'} {path.name}: {checks}")

    print(f"\nFixtures: {pass_count} PASS, {fail_count} FAIL")
    if fail_count:
        sys.exit(1)
