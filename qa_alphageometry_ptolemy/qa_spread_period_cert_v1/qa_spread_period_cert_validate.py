#!/usr/bin/env python3
"""QA Spread Period Cert family [128] validator — QA_SPREAD_PERIOD_CERT.v1

Certifies that the QA cosmos orbit period for modulus m equals the Pisano period
π(m) of the Fibonacci sequence modulo m.

The Pisano period π(m) is the smallest positive integer k such that:
    F_k     ≡ 0  (mod m)
    F_{k+1} ≡ 1  (mod m)

where F_n is the n-th Fibonacci number (F_0=0, F_1=1, F_2=1, ...).

This equals the order of the Fibonacci shift matrix F = [[0,1],[1,1]] in GL₂(Z/mZ),
which is the period of the QA affine cosmos orbit (proven when F^{π/2} ≡ −I mod m).

Relationship to spread polynomials:
    The spread polynomial S_n satisfies S_n(sin²θ) = sin²(nθ).
    For fixed s = sin²θ ∈ F_m, the sequence S_1(s), S_2(s), ..., S_n(s) returns
    to s after π(m) steps because the angle θ has order π(m) in the spread group.

Standard Pisano periods (also QA cosmos orbit periods):
    m=2:  π=3      m=3:  π=8      m=4:  π=6
    m=5:  π=20     m=6:  π=24     m=7:  π=16
    m=8:  π=12     m=9:  π=24     m=10: π=60
    m=24: π=24

Validation checks:
  SP1  schema_version == 'QA_SPREAD_PERIOD_CERT.v1'       → SCHEMA_VERSION_WRONG
  SP2  Fibonacci sequence period mod m == claimed period   → PISANO_PERIOD_MISMATCH
  SP3  F_matrix^period ≡ I mod m                          → MATRIX_PERIOD_WRONG
  SP4  period is minimal (F^(P/k) ≢ I for prime k|P)     → PERIOD_NOT_MINIMAL
  SP5  orbit_type ∈ {cosmos,satellite,singularity}        → ORBIT_TYPE_MISMATCH
       AND (period==1) ↔ (orbit_type=="singularity")

ok semantics: detected failures == declared fail_ledger AND result field consistent

Usage:
  python qa_spread_period_cert_validate.py --self-test
  python qa_spread_period_cert_validate.py --file fixtures/sp_pass_m9.json
"""

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

KNOWN_FAIL_TYPES = frozenset([
    "SCHEMA_VERSION_WRONG",
    "PISANO_PERIOD_MISMATCH",
    "MATRIX_PERIOD_WRONG",
    "PERIOD_NOT_MINIMAL",
    "ORBIT_TYPE_MISMATCH",
])

VALID_ORBIT_TYPES = frozenset(["cosmos", "satellite", "singularity"])

_FIB_MATRIX = [[0, 1], [1, 1]]


class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def _compute_pisano_period(m: int, max_iter: int = 100_000) -> int:
    """Compute the Pisano period π(m): smallest k>0 with F_k≡0 and F_{k+1}≡1 (mod m)."""
    a, b = 0, 1
    for k in range(1, max_iter + 1):
        a, b = b, (a + b) % m
        if a == 0 and b == 1:
            return k
    raise ValueError(f"Pisano period for m={m} exceeds max_iter={max_iter}")


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


def validate_spread_period_cert(cert: dict) -> dict:
    out = _Out()
    detected: set = set()

    # ── SP1: schema_version ──────────────────────────────────────────────────
    if cert.get("schema_version") != "QA_SPREAD_PERIOD_CERT.v1":
        detected.add("SCHEMA_VERSION_WRONG")
        out.fail(
            f"SP1 SCHEMA_VERSION_WRONG: must be 'QA_SPREAD_PERIOD_CERT.v1', "
            f"got {cert.get('schema_version')!r}"
        )

    # ── required fields ──────────────────────────────────────────────────────
    for field in ["certificate_id", "cert_type", "title", "created_utc",
                  "generator", "spread_period", "validation_checks",
                  "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    # ── extract generator ────────────────────────────────────────────────────
    gen = cert.get("generator", {})
    m = gen.get("m")
    if not isinstance(m, int) or m < 2:
        out.fail("generator.m must be an integer >= 2")
        return _reconcile(cert, out, detected)

    # ── extract claimed spread_period fields ─────────────────────────────────
    sp = cert.get("spread_period", {})
    claimed_period = sp.get("cosmos_period")
    orbit_type = sp.get("orbit_type", "")

    if not isinstance(claimed_period, int) or claimed_period < 1:
        out.fail("spread_period.cosmos_period must be a positive integer")
        return _reconcile(cert, out, detected)

    # ── SP2: Pisano period computation ────────────────────────────────────────
    try:
        actual_pisano = _compute_pisano_period(m)
    except ValueError as exc:
        out.fail(f"SP2: could not compute Pisano period for m={m}: {exc}")
        return _reconcile(cert, out, detected)

    if claimed_period != actual_pisano:
        detected.add("PISANO_PERIOD_MISMATCH")
        out.fail(
            f"SP2 PISANO_PERIOD_MISMATCH: claimed cosmos_period={claimed_period}, "
            f"actual Pisano period π({m})={actual_pisano}"
        )

    # ── SP3: F_matrix^cosmos_period ≡ I mod m ─────────────────────────────────
    Fp = _matpow_mod(_FIB_MATRIX, claimed_period, m)
    if not _is_identity(Fp, m):
        detected.add("MATRIX_PERIOD_WRONG")
        out.fail(
            f"SP3 MATRIX_PERIOD_WRONG: F^{claimed_period} mod {m} = {Fp} ≠ I"
        )

    # ── SP4: minimality ───────────────────────────────────────────────────────
    if claimed_period > 1:
        prime_factors = _small_prime_factors(claimed_period)
        for k in prime_factors:
            reduced = claimed_period // k
            F_reduced = _matpow_mod(_FIB_MATRIX, reduced, m)
            if _is_identity(F_reduced, m):
                detected.add("PERIOD_NOT_MINIMAL")
                out.fail(
                    f"SP4 PERIOD_NOT_MINIMAL: F^{reduced} mod {m} = I, "
                    f"so {claimed_period} is not minimal (divides by prime {k})"
                )
                break

    # ── SP5: orbit_type consistency ───────────────────────────────────────────
    sp5_issues = []
    if orbit_type not in VALID_ORBIT_TYPES:
        sp5_issues.append(
            f"orbit_type must be one of {sorted(VALID_ORBIT_TYPES)}, got {orbit_type!r}"
        )
    else:
        if claimed_period == 1 and orbit_type != "singularity":
            sp5_issues.append(
                f"period=1 implies singularity, but orbit_type='{orbit_type}'"
            )
        elif claimed_period != 1 and orbit_type == "singularity":
            sp5_issues.append(
                f"period={claimed_period} > 1 incompatible with orbit_type='singularity'"
            )

    if sp5_issues:
        detected.add("ORBIT_TYPE_MISMATCH")
        out.fail("SP5 ORBIT_TYPE_MISMATCH: " + "; ".join(sp5_issues))

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
        "sp_pass_m9.json":          True,
        "sp_pass_m7.json":          True,
        "sp_fail_wrong_period.json": True,
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
        r = validate_spread_period_cert(cert)
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
        description="QA Spread Period Cert [128] validator"
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
        res = validate_spread_period_cert(cert)
        print(json.dumps(res, indent=2, sort_keys=True))
        sys.exit(0 if res["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
