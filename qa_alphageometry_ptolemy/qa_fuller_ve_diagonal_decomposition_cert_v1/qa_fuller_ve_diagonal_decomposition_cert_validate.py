#!/usr/bin/env python3
"""
qa_fuller_ve_diagonal_decomposition_cert_validate.py

Validator for QA_FULLER_VE_DIAGONAL_DECOMPOSITION_CERT.v1  [family 217]

Certifies: Fuller's cuboctahedral / vector-equilibrium shell count law
    S_n = 10*n*n + 2    (n >= 1: 12, 42, 92, 162, 252, 362, ...)
decomposes across QA integer diagonals by n mod 3:
    - n not divisible by 3  ==>  S_n sits on the b=e diagonal D_1
      with tuple (S_n/3, S_n/3, 2*S_n/3, S_n)
    - n divisible by 3      ==>  S_n sits off D_1 on a sibling
      odd-divisor diagonal D_k where 2*k+1 divides S_n

The mod-3 selection is itself QA-natural: the triune partition
(n mod 3 in {1, 2} on D_1 vs n mod 3 == 0 off D_1) is the smallest
non-trivial residue-class structure of the QA canonical diagonal.

Proof: S_n mod 3 = (n*n + 2) mod 3. n*n mod 3 in {0, 1}: it is 1
when n mod 3 in {1, 2}, and 0 when n mod 3 == 0. Hence S_n mod 3 == 0
iff n not divisible by 3.

Foundation cert: b=e canonical Sierpinski diagonal (see
docs/theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md). FST match
(Briddell STF iterations) is the paradigm case of a hierarchy
entirely on D_1; Fuller VE is the first documented hierarchy
whose decomposition is MIXED across D_1 and sibling diagonals.

Checks:
    FVDD_1              schema_version matches
    FVDD_FORMULA        shell counts match S_n = 10*n*n + 2 for n = 1..9
    FVDD_MOD3           mod-3 classification vector (on-diagonal iff n mod 3 != 0)
    FVDD_DIAGONAL       on-diagonal shells have valid b=e tuple (b=S_n/3, e=S_n/3, d=2*S_n/3, a=S_n)
    FVDD_OFFDIAGONAL    off-diagonal shells have at least one odd divisor > 1 of S_n, witnessing sibling diagonal D_k
    FVDD_COMPUTATIONAL  exhaustive verification for n = 1..9 matches claims
    FVDD_SRC            source attribution present (Fuller + Will Dale)
    FVDD_WITNESS        >= 3 witnesses (on-diagonal, off-diagonal, mod-3 structural)
    FVDD_F              fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator - validates Fuller VE shell decomposition across QA integer diagonals; integer state space; A1/A2 compliant; raw d=b+e, a=b+2e (no mod reduction for element derivation); no ** operator; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_FULLER_VE_DIAGONAL_DECOMPOSITION_CERT.v1"


# -----------------------------------------------------------------------------
# QA primitives (integer-only, axiom-compliant)
# -----------------------------------------------------------------------------

def shell_count(n):
    """Fuller VE shell population at frequency n (n >= 1). S_n = 10*n*n + 2."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 10 * n * n + 2


def qa_tuple_from_be(b, e):
    """(b, e, d, a) with raw A2 derivation: d = b+e, a = b+2e."""
    return (b, e, b + e, b + 2 * e)


def on_d1(a):
    """Test whether `a` admits a b=e diagonal representation (a = 3*b for int b)."""
    return a % 3 == 0


def odd_divisors(n):
    """Odd divisors of n > 0, sorted ascending."""
    out = []
    for k in range(1, n + 1):
        if k % 2 == 1 and n % k == 0:
            out.append(k)
    return out


def sibling_diagonal_candidates(a):
    """For off-D_1 a, return list of (k, b) such that a = (2k+1) * b, k >= 1."""
    out = []
    for r in odd_divisors(a):
        if r >= 3:  # 2k+1 >= 3  <=>  k >= 1
            k = (r - 1) // 2
            b = a // r
            out.append((k, b))
    return out


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def _run_checks(fixture):
    results = {}

    # FVDD_1: schema version
    results["FVDD_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    # FVDD_FORMULA: shell counts for n=1..9 match 10n^2 + 2
    shells = fixture.get("shell_table", [])
    expected_by_n = {n: shell_count(n) for n in range(1, 10)}
    formula_ok = True
    for row in shells:
        n = row.get("n")
        s = row.get("shell_count")
        if not isinstance(n, int) or not isinstance(s, int):
            formula_ok = False
            break
        if n in expected_by_n and expected_by_n[n] != s:
            formula_ok = False
            break
    # Must cover n = 1..9
    covered = {row.get("n") for row in shells}
    formula_ok = formula_ok and all(n in covered for n in range(1, 10))
    results["FVDD_FORMULA"] = formula_ok

    # FVDD_MOD3: classification vector. For each n in 1..9, on-diagonal iff n % 3 != 0.
    mod3_ok = True
    for row in shells:
        n = row.get("n")
        claim_on_d1 = row.get("on_d1")
        if not isinstance(n, int) or not isinstance(claim_on_d1, bool):
            mod3_ok = False
            break
        expected_on_d1 = (n % 3 != 0)
        if claim_on_d1 != expected_on_d1:
            mod3_ok = False
            break
        # Also verify computationally via a % 3
        s = row.get("shell_count")
        if isinstance(s, int) and on_d1(s) != expected_on_d1:
            mod3_ok = False
            break
    results["FVDD_MOD3"] = mod3_ok

    # FVDD_DIAGONAL: on-diagonal shells have valid b=e tuple
    diag_ok = True
    for row in shells:
        if row.get("on_d1") is not True:
            continue
        s = row.get("shell_count")
        tup = row.get("d1_tuple")
        if not isinstance(s, int) or not isinstance(tup, list) or len(tup) != 4:
            diag_ok = False
            break
        b, e, d, a = tup
        expected = list(qa_tuple_from_be(s // 3, s // 3))
        if [b, e, d, a] != expected or a != s:
            diag_ok = False
            break
    results["FVDD_DIAGONAL"] = diag_ok

    # FVDD_OFFDIAGONAL: off-diagonal shells have at least one odd divisor > 1
    off_ok = True
    for row in shells:
        if row.get("on_d1") is not False:
            continue
        s = row.get("shell_count")
        claims = row.get("sibling_candidates", [])
        if not isinstance(s, int) or not isinstance(claims, list):
            off_ok = False
            break
        actual = sibling_diagonal_candidates(s)
        if len(actual) < 1:
            off_ok = False
            break
        # Claim must be a subset of actual
        actual_pairs = set((k, b) for (k, b) in actual)
        for c in claims:
            k = c.get("k")
            b = c.get("b")
            if (k, b) not in actual_pairs:
                off_ok = False
                break
        if not off_ok:
            break
        # And at least one sibling claim per off-diagonal row
        if len(claims) < 1:
            off_ok = False
            break
    results["FVDD_OFFDIAGONAL"] = off_ok

    # FVDD_COMPUTATIONAL: exhaustive check for n=1..9
    computational_ok = True
    for n in range(1, 10):
        s = shell_count(n)
        on_diag = on_d1(s)
        expected = (n % 3 != 0)
        if on_diag != expected:
            computational_ok = False
            break
        if on_diag:
            b, e, d, a = qa_tuple_from_be(s // 3, s // 3)
            if a != s or b != e or d != 2 * (s // 3):
                computational_ok = False
                break
        else:
            if not sibling_diagonal_candidates(s):
                computational_ok = False
                break
    results["FVDD_COMPUTATIONAL"] = computational_ok

    # FVDD_SRC: source attribution
    src = fixture.get("source_attribution", "")
    results["FVDD_SRC"] = (
        ("Fuller" in src or "Synergetics" in src)
        and ("Will Dale" in src or "Dale" in src)
    )

    # FVDD_WITNESS: >= 3 witnesses
    witnesses = fixture.get("witnesses", [])
    results["FVDD_WITNESS"] = len(witnesses) >= 3
    if results["FVDD_WITNESS"]:
        kinds = {w.get("kind") for w in witnesses}
        # Must include at least one of each of the three structural kinds
        required = {"on_diagonal", "off_diagonal", "mod3_structural"}
        if not required.issubset(kinds):
            results["FVDD_WITNESS"] = False

    # FVDD_F: fail ledger
    fl = fixture.get("fail_ledger")
    results["FVDD_F"] = isinstance(fl, list)

    return results


def validate_fixture(path):
    with open(path) as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    all_pass = all(checks.values())
    actual = "PASS" if all_pass else "FAIL"
    ok = actual == expected
    return {"ok": ok, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    fdir = Path(__file__).parent / "fixtures"
    results = {}
    for fp in sorted(fdir.glob("*.json")):
        results[fp.name] = validate_fixture(fp)
    all_ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    elif len(sys.argv) > 1:
        r = validate_fixture(sys.argv[1])
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    else:
        print("Usage: python qa_fuller_ve_diagonal_decomposition_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
