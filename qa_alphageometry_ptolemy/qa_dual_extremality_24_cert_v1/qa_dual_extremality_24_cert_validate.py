#!/usr/bin/env python3
"""
qa_dual_extremality_24_cert_validate.py

Validator for QA_DUAL_EXTREMALITY_24_CERT.v1  [family 192]

Certifies: the joint extremality of m=24 under the Pisano period operator pi
and the Carmichael lambda function, which together formalize m=24 as the
natural Level-III self-improvement fixed point for QA state spaces (closes
item 5 of the Bateson Learning Levels sketch).

CLASSICAL FACTS VERIFIED:

(1) pi(24) = 24 — 24 is a Pisano period fixed point (Wall, 1960).

(2) 24 is the MINIMUM non-trivial Pisano fixed point. The complete fixed
    point set is OEIS A235702 = {24 * 5^k : k >= 0} = {24, 120, 600, 3000, ...}.
    The trivial case pi(1) = 1 is excluded because Z/1Z is the zero ring.

(3) {m : lambda(m) = 2} = {3, 4, 6, 8, 12, 24}. The maximum is 24.
    STRUCTURAL PROOF: m with lambda(m) = 2 must have all prime factors
    in {2, 3} (otherwise some (p-1) | lambda with p-1 >= 4), 2-adic valuation
    at most 3 (otherwise lambda(2^k) = 2^(k-2) >= 4 for k >= 4), and 3-adic
    valuation at most 1 (otherwise lambda(9) = 6). So m divides 24. The set
    is exactly the divisors of 24 with m >= 3 (lambda(1) = lambda(2) = 1).

(4) pi(9) = 24 — the QA theoretical modulus maps to the QA applied modulus
    in exactly one Pisano step. Since pi(24) = 24, the theoretical modulus
    is a pre-fixed point that reaches the minimum fixed point in one step.

(5) pi^-1(24) intersect [1, 30] = {6, 9, 12, 16, 18, 24}. The basin of
    attraction of 24 under pi is non-trivial: six distinct moduli in [1, 30]
    map to 24. The QA choice of m=9 is one of six natural pre-images.

(6) Cannonball identity: sum_{k=1}^{24} k^2 = 4900 = 70^2 (Watson, 1918).
    This is the UNIQUE nontrivial solution to sum_{k=1}^{n} k^2 = m^2. The
    identity is the construction step that builds the Leech lattice from
    the Lorentzian lattice II_{25,1} via the Weyl vector (0,1,...,24 | 70).

(7) 24-theorem: for every prime p >= 5, p^2 - 1 is divisible by 24.
    Proof: p is odd so p^2 - 1 = (p-1)(p+1) is product of two consecutive
    evens, one divisible by 4, the other by 2, giving 8 | (p^2-1). Also
    p not divisible by 3 so p ≡ ±1 (mod 3), giving 3 | (p^2-1).
    Hence 24 = 8*3 divides p^2 - 1.

ORIGINAL CONTRIBUTION (QA project): The JOINT statement that 24 is
simultaneously the minimum non-trivial Pisano fixed point AND the maximum
Carmichael-lambda=2 modulus. Both halves are classical; the joint statement
appears to be original to this project, and provides a formal answer to the
question "why m=24?" grounded in classical number theory rather than
arbitrary convention.

Source: Wall, D.D., "Fibonacci series modulo m," Amer. Math. Monthly 67
(1960); OEIS A235702; Carmichael, R.D., "Note on a new number theory
function," Bull. AMS 16 (1910); Watson, G.N., "The problem of the square
pyramid," Messenger of Math. 48 (1918); Baez, J., "My Favorite Numbers: 24"
(Rankin Lecture, Glasgow 2008).

Checks:
    DE_1         — schema_version matches
    DE_PISANO    — pi(m) correctly computed for declared witnesses
    DE_MIN_FP    — 24 is the minimum non-trivial Pisano fixed point in [2, 200]
    DE_CARMICHAEL — lambda(m) correctly computed; {m in [1,100] : lambda=2} matches
    DE_MAX_LAM   — 24 is the maximum of {m : lambda(m) = 2}
    DE_JOINT     — joint extremality statement: pi(24)=24 AND lambda(24)=2 AND both extremal
    DE_BRIDGE    — pi(9) = 24 (QA theoretical -> applied bridge)
    DE_BASIN     — pi^-1(24) intersect [1,30] contains {6, 9, 12, 16, 18, 24}
    DE_CANNON    — cannonball identity 1^2 + ... + 24^2 = 70^2
    DE_24THM     — p^2 - 1 ≡ 0 (mod 24) for primes p in {5, 7, 11, ..., 47}
    DE_SRC       — source attribution to Wall / OEIS / classical sources present
    DE_WITNESS   — at least 5 witness moduli with verified Pisano/Carmichael values
    DE_F         — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — classical number-theoretic verification of Pisano periods, Carmichael lambda, cannonball, 24-theorem; integer-only, no observer, no floats"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_DUAL_EXTREMALITY_24_CERT.v1"

# Expected structural constants
EXPECTED_PISANO_FP_IN_200 = [1, 24, 120]  # includes trivial m=1
EXPECTED_NONTRIVIAL_PISANO_MIN = 24
EXPECTED_LAMBDA_TWO_SET = [3, 4, 6, 8, 12, 24]
EXPECTED_LAMBDA_TWO_MAX = 24
EXPECTED_BASIN_24_IN_30 = [6, 9, 12, 16, 18, 24]
EXPECTED_CANNONBALL_SUM = 4900  # = 70^2
EXPECTED_CANNONBALL_SQRT = 70


# -----------------------------------------------------------------------------
# Number theory primitives (integer-only)
# -----------------------------------------------------------------------------

def pisano_period(m: int) -> int:
    """Length of the Fibonacci sequence mod m before repeating (0, 1)."""
    if m <= 0:
        return 0
    if m == 1:
        return 1
    prev, curr = 0, 1
    # Period of pi(m) is bounded by 6m (Wall 1960).
    for k in range(1, 6 * m + 1):
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return k
    return 0


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def carmichael_lambda(n: int) -> int:
    """Carmichael function: exponent of the unit group (Z/nZ)*."""
    if n == 1:
        return 1
    factors = {}
    x = n
    p = 2
    while p * p <= x:
        while x % p == 0:
            factors[p] = factors.get(p, 0) + 1
            x //= p
        p += 1
    if x > 1:
        factors[x] = factors.get(x, 0) + 1

    result = 1
    for p, k in factors.items():
        if p == 2:
            if k == 1:
                lam_pk = 1
            elif k == 2:
                lam_pk = 2
            else:
                lam_pk = 2 ** (k - 2)
        else:
            lam_pk = (p - 1) * p ** (k - 1)
        result = lcm(result, lam_pk)
    return result


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # DE_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"DE_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # DE_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("DE_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("DE_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # DE_SRC
    src = cert.get("source_attribution", "")
    if not any(k in str(src) for k in ["Wall", "OEIS", "Carmichael", "Watson"]):
        warnings.append("DE_SRC: source_attribution should credit Wall (1960), OEIS, Carmichael, or Watson")

    # DE_MIN_FP: exhaustively verify in [2, 200] that only {24, 120} are fixed points
    actual_fp = [m for m in range(2, 201) if pisano_period(m) == m]
    if actual_fp != [24, 120]:
        errors.append(
            f"DE_MIN_FP: non-trivial Pisano fixed points in [2, 200] = {actual_fp}, expected [24, 120]"
        )
    if actual_fp and actual_fp[0] != EXPECTED_NONTRIVIAL_PISANO_MIN:
        errors.append(
            f"DE_MIN_FP: minimum non-trivial Pisano fixed point = {actual_fp[0]}, expected {EXPECTED_NONTRIVIAL_PISANO_MIN}"
        )

    # DE_CARMICHAEL: verify lambda computation on [1, 100]
    actual_lam2 = [m for m in range(1, 101) if carmichael_lambda(m) == 2]
    if actual_lam2 != EXPECTED_LAMBDA_TWO_SET:
        errors.append(
            f"DE_CARMICHAEL: {{m in [1,100] : lambda(m)=2}} = {actual_lam2}, expected {EXPECTED_LAMBDA_TWO_SET}"
        )

    # DE_MAX_LAM
    if actual_lam2 and max(actual_lam2) != EXPECTED_LAMBDA_TWO_MAX:
        errors.append(
            f"DE_MAX_LAM: max of lambda=2 set = {max(actual_lam2)}, expected {EXPECTED_LAMBDA_TWO_MAX}"
        )

    # DE_JOINT: joint extremality at 24
    if pisano_period(24) != 24:
        errors.append(f"DE_JOINT: pi(24) = {pisano_period(24)}, expected 24")
    if carmichael_lambda(24) != 2:
        errors.append(f"DE_JOINT: lambda(24) = {carmichael_lambda(24)}, expected 2")

    # DE_BRIDGE: pi(9) = 24
    if pisano_period(9) != 24:
        errors.append(f"DE_BRIDGE: pi(9) = {pisano_period(9)}, expected 24")

    # DE_BASIN: pi^-1(24) intersect [1, 30]
    actual_basin = [m for m in range(1, 31) if pisano_period(m) == 24]
    if actual_basin != EXPECTED_BASIN_24_IN_30:
        errors.append(
            f"DE_BASIN: basin of 24 in [1,30] = {actual_basin}, expected {EXPECTED_BASIN_24_IN_30}"
        )

    # DE_CANNON: cannonball identity
    square_sum = sum(k * k for k in range(1, 25))
    if square_sum != EXPECTED_CANNONBALL_SUM:
        errors.append(
            f"DE_CANNON: sum_{{k=1}}^{{24}} k^2 = {square_sum}, expected {EXPECTED_CANNONBALL_SUM}"
        )
    if square_sum != EXPECTED_CANNONBALL_SQRT * EXPECTED_CANNONBALL_SQRT:
        errors.append(
            f"DE_CANNON: cannonball sum {square_sum} != 70^2 = {EXPECTED_CANNONBALL_SQRT ** 2}"
        )

    # DE_24THM: p^2 - 1 ≡ 0 (mod 24) for primes p in [5, 50]
    for p in range(5, 51):
        if is_prime(p) and (p * p - 1) % 24 != 0:
            errors.append(f"DE_24THM: p={p}: p^2 - 1 = {p*p-1} not divisible by 24")

    # DE_WITNESS / DE_PISANO: verify declared witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 5:
        errors.append(f"DE_WITNESS: need >= 5 witnesses, got {len(witnesses)}")

    for wi, w in enumerate(witnesses):
        m = w.get("m")
        if m is None:
            errors.append(f"DE_WITNESS: witness[{wi}] missing m")
            continue
        if "pisano_period" in w:
            actual = pisano_period(int(m))
            if w["pisano_period"] != actual:
                errors.append(
                    f"DE_PISANO: witness[{wi}] m={m} declared pi={w['pisano_period']}, actual={actual}"
                )
        if "carmichael_lambda" in w:
            actual = carmichael_lambda(int(m))
            if w["carmichael_lambda"] != actual:
                errors.append(
                    f"DE_CARMICHAEL: witness[{wi}] m={m} declared lambda={w['carmichael_lambda']}, actual={actual}"
                )

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("de_pass_extremality.json", True),
        ("de_fail_bad_pisano.json", True),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Dual Extremality 24 Cert [192] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
