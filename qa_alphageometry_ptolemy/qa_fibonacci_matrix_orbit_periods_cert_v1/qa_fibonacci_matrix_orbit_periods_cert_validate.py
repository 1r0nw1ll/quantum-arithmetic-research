# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Wall 1960 DOI 10.1080/00029890.1960.11989541; Wildberger 2005 ISBN 978-0-9757492-0-8) -->
QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z) and Fibonacci sequences; no empirical QA state machine; all state is exact integer"
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
"""

from __future__ import annotations

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

    return results


def validate_fixture(fixture: dict) -> dict:
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

    # Fail fixtures must fail
    fail_cases = [
        {"state": [1, 1], "expected_period": 8, "expected_orbit_type": "cosmos", "expected": "FAIL"},
        {"state": [3, 3], "expected_period": 8, "expected_orbit_type": "cosmos", "expected": "FAIL"},
    ]
    for case in fail_cases:
        checks = validate_fixture(case)
        state_checks = {k: v for k, v in checks.items() if k in ("FMO_ORBIT", "FMO_TYPE")}
        if all(state_checks.values()):
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
        # For FAIL fixtures, only the state-specific checks (FMO_ORBIT, FMO_TYPE) determine pass/fail
        if expect_pass:
            all_pass = all(checks.values())
        else:
            state_checks = {k: v for k, v in checks.items() if k in ("FMO_ORBIT", "FMO_TYPE")}
            all_pass = all(state_checks.values())
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
        if expect_pass:
            all_pass = all(checks.values())
        else:
            state_checks = {k: v for k, v in checks.items() if k in ("FMO_ORBIT", "FMO_TYPE")}
            all_pass = all(state_checks.values())
        ok = all_pass == expect_pass
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        print(f"{'PASS' if ok else 'FAIL'} {path.name}: {checks}")

    print(f"\nFixtures: {pass_count} PASS, {fail_count} FAIL")
    if fail_count:
        sys.exit(1)
