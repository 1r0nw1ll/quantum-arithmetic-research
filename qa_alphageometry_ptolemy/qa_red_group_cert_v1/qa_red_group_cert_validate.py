#!/usr/bin/env python3
"""QA Red Group Cert family [126] validator — QA_RED_GROUP_CERT.v1

Certifies that the QA T-operator is the Fibonacci shift matrix

    F = [[0, 1],
         [1, 1]]

This matrix represents multiplication by φ = (1+√5)/2 in the split-complex
ring Z[√5]/mZ[√5] (the "red" chromatic isometry group of Wildberger).

Key algebraic properties:
    det(F) = 0·1 − 1·1 = −1    (red norm N_red(φ) = φ·ψ = −1, where ψ=(1−√5)/2)
    tr(F)  = 0 + 1 = 1          (φ + ψ = 1, minimal polynomial x²−x−1)

The orbit period of the affine QA map coincides with ord(F) in GL₂(Z/mZ) whenever
F^{P/2} ≡ −I mod m (which holds for m=9 and m=3), because the affine translation
sum vanishes: (I+F+...+F^{P−1}) = (I+F+...+F^{P/2−1})(I+F^{P/2}) = S·0 = 0.

Standard QA orbit periods:
    m=9:  cosmos period = 24 (= ord(F) in GL₂(Z/9Z))
           satellite period = 8  (affine period for b≡e≡0 mod 3 states)
           singularity period = 1 (fixed point at (9,9))
    m=3:  cosmos period = 8  (= ord(F) in GL₂(Z/3Z))

Validation checks:
  RG1  schema_version == 'QA_RED_GROUP_CERT.v1'            → SCHEMA_VERSION_WRONG
  RG2  T_matrix == [[0,1],[1,1]] (Fibonacci shift)          → T_MATRIX_WRONG
  RG3  det(T_matrix) mod m == m−1 (≡ −1 mod m)            → DET_NOT_MINUS_ONE
  RG4  trace(T_matrix) mod m == 1                           → TRACE_NOT_ONE
  RG5  T_matrix^orbit_period ≡ I mod m                     → ORBIT_PERIOD_WRONG
  RG6  T_matrix^(orbit_period // k) ≢ I mod m for prime    → PERIOD_NOT_MINIMAL
       divisors k of orbit_period
  RG7  orbit_type ∈ {cosmos,satellite,singularity} AND      → ORBIT_TYPE_MISMATCH
       (orbit_period==1) ↔ (orbit_type=="singularity")

ok semantics (same as family [125]):
  ok=True  — cert is internally consistent: detected failures == declared fail_ledger
             AND result field matches
  ok=False — inconsistency between detected and declared, or result field wrong

Usage:
  python qa_red_group_cert_validate.py --self-test
  python qa_red_group_cert_validate.py --file fixtures/rg_pass_m9_cosmos.json
"""

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

KNOWN_FAIL_TYPES = frozenset([
    "SCHEMA_VERSION_WRONG",
    "T_MATRIX_WRONG",
    "DET_NOT_MINUS_ONE",
    "TRACE_NOT_ONE",
    "ORBIT_PERIOD_WRONG",
    "PERIOD_NOT_MINIMAL",
    "ORBIT_TYPE_MISMATCH",
])

VALID_ORBIT_TYPES = frozenset(["cosmos", "satellite", "singularity"])

_CANONICAL_T_MATRIX = [[0, 1], [1, 1]]


class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def _matmul_mod(A: list, B: list, m: int) -> list:
    """2×2 integer matrix multiplication mod m."""
    return [
        [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % m,
         (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % m],
        [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % m,
         (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % m],
    ]


def _matpow_mod(M: list, n: int, m: int) -> list:
    """2×2 integer matrix power mod m by repeated squaring."""
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in M]
    while n > 0:
        if n & 1:
            result = _matmul_mod(result, base, m)
        base = _matmul_mod(base, base, m)
        n >>= 1
    return result


def _is_identity(M: list, m: int) -> bool:
    return M == [[1 % m, 0], [0, 1 % m]]


def _small_prime_factors(n: int) -> list:
    """Return sorted list of distinct prime factors of n (n < 10**6)."""
    factors = []
    d = 2
    x = n
    while d * d <= x:
        if x % d == 0:
            factors.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        factors.append(x)
    return factors


def validate_red_group_cert(cert: dict) -> dict:
    out = _Out()
    detected: set = set()

    # ── RG1: schema_version ─────────────────────────────────────────────────
    if cert.get("schema_version") != "QA_RED_GROUP_CERT.v1":
        detected.add("SCHEMA_VERSION_WRONG")
        out.fail(
            f"RG1 SCHEMA_VERSION_WRONG: must be 'QA_RED_GROUP_CERT.v1', "
            f"got {cert.get('schema_version')!r}"
        )

    # ── required fields ──────────────────────────────────────────────────────
    for field in ["certificate_id", "cert_type", "title", "created_utc",
                  "generator", "red_group", "validation_checks",
                  "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    # ── extract generator ────────────────────────────────────────────────────
    gen = cert.get("generator", {})
    m = gen.get("m")
    if not isinstance(m, int) or m < 2:
        out.fail("generator.m must be an integer >= 2")
        return _reconcile(cert, out, detected)

    # ── extract claimed red_group fields ─────────────────────────────────────
    rg = cert.get("red_group", {})
    T_claimed = rg.get("T_matrix")
    orbit_period = rg.get("orbit_period")
    orbit_type = rg.get("orbit_type", "")

    # ── RG2: T_matrix == [[0,1],[1,1]] ───────────────────────────────────────
    if T_claimed != _CANONICAL_T_MATRIX:
        detected.add("T_MATRIX_WRONG")
        out.fail(
            f"RG2 T_MATRIX_WRONG: T_matrix must be {_CANONICAL_T_MATRIX} "
            f"(Fibonacci shift), got {T_claimed!r}"
        )
        # Cannot continue matrix checks if T_matrix is wrong
        T_for_checks = _CANONICAL_T_MATRIX
    else:
        T_for_checks = T_claimed

    # ── RG3: det(T_matrix) ≡ −1 mod m ────────────────────────────────────────
    det_claimed = rg.get("det_T")
    det_actual = (T_for_checks[0][0] * T_for_checks[1][1]
                  - T_for_checks[0][1] * T_for_checks[1][0]) % m
    det_expected = (m - 1)  # ≡ −1 mod m
    if det_actual != det_expected:
        detected.add("DET_NOT_MINUS_ONE")
        out.fail(
            f"RG3 DET_NOT_MINUS_ONE: det(T_matrix) mod {m} should be {det_expected} "
            f"(≡ −1), got {det_actual}"
        )
    if det_claimed is not None and det_claimed != (T_for_checks[0][0] * T_for_checks[1][1]
                                                    - T_for_checks[0][1] * T_for_checks[1][0]):
        out.warn(f"det_T field {det_claimed} != computed det {T_for_checks[0][0]*T_for_checks[1][1]-T_for_checks[0][1]*T_for_checks[1][0]}")

    # ── RG4: trace(T_matrix) ≡ 1 mod m ───────────────────────────────────────
    trace_claimed = rg.get("trace_T")
    trace_actual = (T_for_checks[0][0] + T_for_checks[1][1]) % m
    trace_expected = 1 % m
    if trace_actual != trace_expected:
        detected.add("TRACE_NOT_ONE")
        out.fail(
            f"RG4 TRACE_NOT_ONE: trace(T_matrix) mod {m} should be 1, "
            f"got {trace_actual}"
        )

    # ── RG5: T_matrix^orbit_period ≡ I mod m ─────────────────────────────────
    if not isinstance(orbit_period, int) or orbit_period < 1:
        out.fail("red_group.orbit_period must be a positive integer")
        return _reconcile(cert, out, detected)

    Tp = _matpow_mod(T_for_checks, orbit_period, m)
    if not _is_identity(Tp, m):
        detected.add("ORBIT_PERIOD_WRONG")
        out.fail(
            f"RG5 ORBIT_PERIOD_WRONG: T^{orbit_period} mod {m} = {Tp} ≠ I"
        )

    # ── RG6: minimality — T^(P/k) ≢ I for all prime k|P ─────────────────────
    if orbit_period > 1:
        prime_factors = _small_prime_factors(orbit_period)
        for k in prime_factors:
            reduced = orbit_period // k
            T_reduced = _matpow_mod(T_for_checks, reduced, m)
            if _is_identity(T_reduced, m):
                detected.add("PERIOD_NOT_MINIMAL")
                out.fail(
                    f"RG6 PERIOD_NOT_MINIMAL: T^{reduced} mod {m} = I, "
                    f"so {orbit_period} is not minimal (divides by prime {k})"
                )
                break  # one violation is enough

    # ── RG7: orbit_type consistency ───────────────────────────────────────────
    rg7_issues = []
    if orbit_type not in VALID_ORBIT_TYPES:
        rg7_issues.append(
            f"orbit_type must be one of {sorted(VALID_ORBIT_TYPES)}, got {orbit_type!r}"
        )
    else:
        if orbit_period == 1 and orbit_type != "singularity":
            rg7_issues.append(
                f"orbit_period=1 implies singularity, but orbit_type='{orbit_type}'"
            )
        elif orbit_period != 1 and orbit_type == "singularity":
            rg7_issues.append(
                f"orbit_period={orbit_period} > 1 is incompatible with orbit_type='singularity'"
            )

    if rg7_issues:
        detected.add("ORBIT_TYPE_MISMATCH")
        out.fail("RG7 ORBIT_TYPE_MISMATCH: " + "; ".join(rg7_issues))

    return _reconcile(cert, out, detected)


def _reconcile(cert: dict, out: _Out, detected: set) -> dict:
    """Produce final validation result. ok=True iff detected==declared AND result consistent."""
    declared_fail_types = set(cert.get("fail_ledger", []))
    declared_result = cert.get("result", "")

    consistency_errors = []

    for f in sorted(declared_fail_types):
        if f not in KNOWN_FAIL_TYPES:
            consistency_errors.append(f"fail_ledger contains unknown fail type: {f!r}")

    undeclared = detected - declared_fail_types
    for f in sorted(undeclared):
        consistency_errors.append(f"detected fail {f!r} not declared in fail_ledger")

    overclaimed = declared_fail_types - detected
    for f in sorted(overclaimed):
        consistency_errors.append(f"fail_ledger claims {f!r} but validator did not detect it")

    if declared_result == "PASS":
        if detected or declared_fail_types:
            consistency_errors.append(
                f"result='PASS' but failures present: detected={sorted(detected)}, "
                f"ledger={sorted(declared_fail_types)}"
            )
    elif declared_result == "FAIL":
        if not declared_fail_types:
            consistency_errors.append("result='FAIL' but fail_ledger is empty")
        elif detected != declared_fail_types:
            consistency_errors.append(
                f"declared ledger {sorted(declared_fail_types)} != detected {sorted(detected)}"
            )
    else:
        consistency_errors.append(
            f"result must be 'PASS' or 'FAIL', got {declared_result!r}"
        )

    ok = len(consistency_errors) == 0
    return {
        "ok": ok,
        "label": "PASS" if ok else "FAIL",
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "detected_fails": sorted(detected),
        "detection_messages": out.errors,
        "errors": consistency_errors,
        "warnings": out.warnings,
    }


def _self_test() -> dict:
    fixtures_dir = HERE / "fixtures"
    expected = {
        "rg_pass_m9_cosmos.json":     True,
        "rg_pass_m3_cosmos.json":     True,
        "rg_fail_wrong_period.json":  True,
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        with open(fpath) as f:
            cert = json.load(f)
        r = validate_red_group_cert(cert)
        passed = r["ok"] == expect_ok
        if not passed:
            all_ok = False
        results.append({
            "fixture": fname,
            "ok": passed,
            "label": r["label"],
            "detected_fails": r["detected_fails"],
            "errors": r["errors"],
        })

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Red Group Cert [126] validator"
    )
    parser.add_argument("--self-test", action="store_true",
                        help="Run all fixtures and print JSON summary")
    parser.add_argument("--file", type=str,
                        help="Validate a single fixture file")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if args.file:
        with open(args.file) as f:
            cert = json.load(f)
        res = validate_red_group_cert(cert)
        print(json.dumps(res, indent=2, sort_keys=True))
        sys.exit(0 if res["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
