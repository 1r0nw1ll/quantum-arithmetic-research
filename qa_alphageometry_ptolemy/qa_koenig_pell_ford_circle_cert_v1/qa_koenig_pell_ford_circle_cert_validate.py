# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Hardy+Wright 2008 ISBN 978-0-19-921986-5; Wildberger 2005 ISBN 978-0-9757492-0-8) -->
"""Cert [289]: QA Koenig Pell Ford Circle.

PRIMARY CLAIM:
  The Koenig I=1 BEDA sequence S = {(b,e) : |b^2 - 2e^2| = 1, b>=1, e>=1}
  equals the Pell equation solution set for x^2 - 2y^2 = +-1.
  Consecutive elements are Farey neighbors: |b_n*e_{n+1} - b_{n+1}*e_n| = 1.
  This is the Ford circle tangency condition; their circles form a chain
  converging to sqrt(2) in the Stern-Brocot tree.

KEY ALGEBRAIC IDENTITY:
  Koenig I(b,e) = |C - F|
                = |2*(b+e)*e - (b+2*e)*b|
                = |2e^2 - b^2|
                = |b^2 - 2e^2|          (same as Pell discriminant)

QA-GENERATION THEOREM:
  The map (b,e) -> (b+2e, b+e) = (a, d) sends each Pell solution to the next.
  This is multiplication by (1+sqrt(2)) in Z[sqrt(2)].
  Seed (1,1): 1^2 - 2*1^2 = -1.

FAREY PROOF (that consecutive Pell solutions are Ford-tangent):
  Let (b', e') = (b+2e, b+e). Then:
  |b*e' - b'*e| = |b*(b+e) - (b+2e)*e| = |b^2 - 2e^2| = 1. QED.
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Checks
# PELL_1  — |b^2 - 2e^2| = 1 for every (b,e) in sequence
# KOENIG_1 — koenig_I = |2(b+e)e - (b+2e)b| = 1 (same formula, QA path)
# FAREY_1 — |b_n*e_{n+1} - b_{n+1}*e_n| = 1 for all consecutive pairs
# ALT_1   — (b^2 - 2e^2) alternates sign between consecutive elements
# ---------------------------------------------------------------------------


def _koenig_I(b: int, e: int) -> int:
    d = b + e
    a = b + 2 * e
    C = 2 * d * e
    F = a * b
    return abs(C - F)


def validate_fixture(fixture: dict) -> dict:
    results: dict = {}
    seq: List[List[int]] = fixture["sequence"]
    pairs: List[Tuple[int, int]] = [(row[0], row[1]) for row in seq]

    pell_vals = [b * b - 2 * e * e for b, e in pairs]

    results["PELL_1"] = all(abs(v) == 1 for v in pell_vals)
    results["KOENIG_1"] = all(_koenig_I(b, e) == 1 for b, e in pairs)

    if len(pairs) >= 2:
        results["FAREY_1"] = all(
            abs(pairs[i][0] * pairs[i + 1][1] - pairs[i + 1][0] * pairs[i][1]) == 1
            for i in range(len(pairs) - 1)
        )
        results["ALT_1"] = all(
            pell_vals[i] * pell_vals[i + 1] == -1
            for i in range(len(pell_vals) - 1)
        )
    else:
        results["FAREY_1"] = True
        results["ALT_1"] = True

    return results


def self_test() -> bool:
    failures = []

    def pell_sequence(n: int) -> List[Tuple[int, int]]:
        """Generate first n Pell solutions via QA map (b,e)->(b+2e, b+e)."""
        pairs = [(1, 1)]
        for _ in range(n - 1):
            b, e = pairs[-1]
            pairs.append((b + 2 * e, b + e))
        return pairs

    seq = pell_sequence(12)

    # Every element satisfies Pell equation
    for b, e in seq:
        if abs(b * b - 2 * e * e) != 1:
            failures.append(f"PELL_1 failed for ({b},{e}): {b*b - 2*e*e}")

    # Koenig I = Pell discriminant for every element
    for b, e in seq:
        ki = _koenig_I(b, e)
        pell = abs(b * b - 2 * e * e)
        if ki != pell:
            failures.append(f"KOENIG_I mismatch for ({b},{e}): I={ki} pell={pell}")
        if ki != 1:
            failures.append(f"KOENIG_1 failed for ({b},{e}): I={ki}")

    # Consecutive pairs are Farey neighbors (Ford tangency)
    for i in range(len(seq) - 1):
        b0, e0 = seq[i]
        b1, e1 = seq[i + 1]
        det = abs(b0 * e1 - b1 * e0)
        if det != 1:
            failures.append(f"FAREY_1 failed for pair {i}-{i+1}: det={det}")

    # Signs alternate (convergence from both sides of sqrt(2))
    pell_vals = [b * b - 2 * e * e for b, e in seq]
    for i in range(len(pell_vals) - 1):
        if pell_vals[i] * pell_vals[i + 1] != -1:
            failures.append(f"ALT_1 failed at index {i}: {pell_vals[i]}, {pell_vals[i+1]}")

    # QA-generation theorem: (b+2e, b+e) is the correct step
    for i in range(len(seq) - 1):
        b, e = seq[i]
        b_next, e_next = seq[i + 1]
        if b_next != b + 2 * e or e_next != b + e:
            failures.append(f"QA map failed at index {i}: expected ({b+2*e},{b+e}) got ({b_next},{e_next})")

    # Fail fixtures must fail at least one check
    fail_cases = [
        {"sequence": [[1, 1], [2, 1], [3, 2]], "expected": "FAIL"},
        {"sequence": [[1, 1], [7, 5], [17, 12]], "expected": "FAIL"},
    ]
    for case in fail_cases:
        checks = validate_fixture(case)
        if all(checks.values()):
            failures.append(f"Expected FAIL case passed unexpectedly: {case['sequence']}")

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
    return len(failures) == 0


FAMILY_ID = 289
CERT_SLUG = "qa_koenig_pell_ford_circle_cert_v1"


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
        description="QA Koenig Pell Ford Circle Cert validator [289]"
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
