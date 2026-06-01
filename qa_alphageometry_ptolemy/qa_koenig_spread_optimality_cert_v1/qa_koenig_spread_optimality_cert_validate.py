# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Wildberger 2005 ISBN 978-0-9757492-0-8; Hardy+Wright 2008 ISBN 978-0-19-921986-5) -->
"""Cert [292]: QA Koenig Spread Optimality.

PRIMARY CLAIM:
  The Koenig I=1 condition from cert [289] has a purely rational trig
  reformulation that eliminates all mention of sqrt(2).

DEFINITIONS (over Z, no irrationals):
  G_tilde(b,e) = b^2 + e^2          -- blue quadrance of direction vector (b,e)
  s(b,e)       = e^2 / G_tilde       -- Wildberger spread from horizontal (rational)
  I(b,e)       = |b^2 - 2*e^2|      -- Koenig I invariant (cert [289])

KEY IDENTITY (algebraic, trivially provable):
  I(b,e) = |2e^2 - b^2| = |3e^2 - (b^2+e^2)| = G_tilde * |3*s - 1|
          = 3 * G_tilde * |s - 1/3|

CLAIMS:
  (1) SPREAD_ID:   I = 3 * G_tilde * |s - 1/3|  for all b,e >= 1
  (2) NO_EXACT:    no (b,e) in Z>0 x Z>0 has s = 1/3 exactly
                   (equivalent to: b^2 = 2*e^2 has no positive integer solution,
                    i.e. sqrt(2) is irrational -- verified for b,e <= 50)
  (3) PELL_OPT:    I(b,e) = 1  iff  |s - 1/3| = 1/(3*G_tilde)
                   (Pell solutions are the states closest to s=1/3 at their scale;
                    I >= 1 for all integer b,e > 0, so I=1 is optimal)
  (4) INTER_SPREAD: for consecutive Pell pairs (b_n,e_n),(b_{n+1},e_{n+1}),
                   spread(direction_n, direction_{n+1}) = 1/(G_tilde_n * G_tilde_{n+1})
                   (Wildberger spread of the angle between the two direction lines)
  (5) ALT_SIDE:    Pell spreads s_n alternate above/below 1/3

RATIONAL TRIG INTERPRETATION:
  The "sqrt(2) cusp" is "the direction with spread 1/3 from horizontal".
  Spread 1/3 is a rational number. No integer direction achieves it (claim 2).
  The Pell chain is the sequence of integer directions that minimise |s - 1/3|
  at each scale -- the best rational approximation to spread-1/3 (claim 3).
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple


def _I(b: int, e: int) -> int:
    return abs(b * b - 2 * e * e)


def _G(b: int, e: int) -> int:
    return b * b + e * e


def _s(b: int, e: int) -> Fraction:
    return Fraction(e * e, b * b + e * e)


_ONE_THIRD = Fraction(1, 3)

# ---------------------------------------------------------------------------
# Checks
# SPREAD_ID   — I = 3 * G_tilde * |s - 1/3|
# NO_EXACT    — no b,e in [1..50] has s = 1/3
# PELL_OPT    — I=1 iff |s-1/3| = 1/(3*G_tilde)
# INTER_SPREAD — spread between consecutive Pell direction lines = 1/(G_n * G_{n+1})
# ALT_SIDE    — Pell s_n alternates above/below 1/3
# ---------------------------------------------------------------------------


def validate_fixture(fixture: dict) -> dict:
    b, e = fixture["state"]
    declared_I: int = fixture["expected_I"]

    results: dict = {}

    actual_I = _I(b, e)
    G = _G(b, e)
    spread = _s(b, e)

    # SPREAD_ID: I = 3 * G * |s - 1/3|  (exact, using Fraction arithmetic)
    spread_formula = 3 * G * abs(spread - _ONE_THIRD)
    results["SPREAD_ID"] = (actual_I == spread_formula)

    # Fixture I matches computed I
    results["I_MATCH"] = (actual_I == declared_I)

    # PELL_OPT: if I=1 then |s-1/3| = 1/(3*G); if I>1 then |s-1/3| > 1/(3*G)
    dev = abs(spread - _ONE_THIRD)
    min_dev = Fraction(1, 3 * G)
    if actual_I == 1:
        results["PELL_OPT"] = (dev == min_dev)
    else:
        results["PELL_OPT"] = (dev > min_dev)

    return results


def self_test() -> bool:
    failures = []

    # --- SPREAD_ID: verify identity for broad range ---
    for b in range(1, 30):
        for e in range(1, 30):
            G = _G(b, e)
            spread = _s(b, e)
            lhs = _I(b, e)
            rhs = 3 * G * abs(spread - _ONE_THIRD)
            if lhs != rhs:
                failures.append(f"SPREAD_ID failed for ({b},{e}): I={lhs}, formula={rhs}")

    # --- NO_EXACT: no integer direction has s = 1/3 ---
    for b in range(1, 51):
        for e in range(1, 51):
            if _s(b, e) == _ONE_THIRD:
                failures.append(f"NO_EXACT violated at ({b},{e})")

    # --- Build Pell chain ---
    pell: List[Tuple[int, int]] = [(1, 1)]
    for _ in range(11):
        b, e = pell[-1]
        pell.append((b + 2 * e, b + e))

    # --- PELL_OPT for all Pell states ---
    for b, e in pell:
        G = _G(b, e)
        dev = abs(_s(b, e) - _ONE_THIRD)
        if dev != Fraction(1, 3 * G):
            failures.append(f"PELL_OPT failed for Pell state ({b},{e})")

    # --- INTER_SPREAD ---
    for i in range(len(pell) - 1):
        b0, e0 = pell[i]
        b1, e1 = pell[i + 1]
        det_sq = (b0 * e1 - b1 * e0) ** 2
        denom = _G(b0, e0) * _G(b1, e1)
        inter_spread = Fraction(det_sq, denom)
        expected = Fraction(1, denom)   # det=±1 from Pell Farey property
        if inter_spread != expected:
            failures.append(f"INTER_SPREAD failed at pair {i}: {inter_spread} != {expected}")
        # Must be decreasing
        if i > 0:
            prev = Fraction(1, _G(*pell[i-1]) * _G(*pell[i]))
            if inter_spread >= prev:
                failures.append(f"INTER_SPREAD not decreasing at pair {i}")

    # --- ALT_SIDE: Pell spreads alternate above/below 1/3 ---
    spreads = [_s(b, e) for b, e in pell]
    for i in range(len(spreads) - 1):
        a, b = spreads[i] - _ONE_THIRD, spreads[i + 1] - _ONE_THIRD
        if (a > 0) == (b > 0):
            failures.append(f"ALT_SIDE failed at index {i}: both {'above' if a>0 else 'below'} 1/3")

    # --- Fail fixture detection ---
    fail_cases = [
        {"state": [3, 2], "expected_I": 2, "expected": "FAIL"},
        {"state": [1, 1], "expected_I": 0, "expected": "FAIL"},
    ]
    for case in fail_cases:
        checks = validate_fixture(case)
        if checks.get("I_MATCH"):
            failures.append(f"Expected FAIL case had I_MATCH=True: {case}")

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
    return len(failures) == 0


FAMILY_ID = 292
CERT_SLUG = "qa_koenig_spread_optimality_cert_v1"


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
        description="QA Koenig Spread Optimality Cert validator [292]"
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
