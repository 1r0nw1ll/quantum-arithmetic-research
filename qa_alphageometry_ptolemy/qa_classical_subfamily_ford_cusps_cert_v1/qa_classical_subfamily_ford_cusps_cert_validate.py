# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Wildberger 2005 ISBN 978-0-9757492-0-8; Hardy+Wright 2008 ISBN 978-0-19-921986-5) -->
"""Cert [290]: QA Classical Subfamily Ford Cusps.

PRIMARY CLAIM:
  The three classical subfamilies form three distinct Farey-adjacent Ford
  circle chains from the common seed (1,1), each converging to a distinct cusp:

  (A) Pythagoras (b=1): (1,1),(1,2),(1,3),...
      — Farey-adjacent: |1*(e+1) - 1*e| = 1 always
      — Ford curvatures 2*1^2, 2*2^2, 2*3^2,... strictly increasing
      — cusp: b/e = 1/e -> 0

  (B) Plato (e=1): (1,1),(2,1),(3,1),...
      — Farey-adjacent: |b*1 - (b+1)*1| = 1 always
      — Ford curvatures all 2*1^2 = 2 (uniform)
      — cusp: b/e = b/1 -> inf (periodic, no irrational limit)

  (C) Fermat (I=1): (1,1),(3,2),(7,5),(17,12),...
      — Pell equation |b^2-2e^2|=1; QA map (b,e)->(b+2e,b+e)
      — Farey-adjacent: proven in cert [289]
      — cusp: b/e -> sqrt(2) (quadratic irrational)

SUBFAMILY CONDITIONS (from Wildberger 2005 Divine Proportions):
  Pythagoras: (d-e)^2 = 1  =>  b^2 = 1  =>  b = 1  (for positive BEDA)
  Plato:      |G-F| = 2    =>  2e^2 = 2  =>  e = 1  (for positive BEDA)
  Fermat:     |C-F| = 1    =>  |b^2-2e^2| = 1       (Pell equation)

FORD CIRCLE CURVATURE:
  The Ford circle C(b/e) (with gcd(b,e)=1 or b=1 or e=1) has curvature 2e^2.
  Pythagoras: 2e^2 for e=1,2,3,... -> strictly increasing
  Plato:      2*1^2 = 2 for all b   -> uniform
  Fermat:     2e_n^2 for Pell denominators -> growing geometrically
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Checks per chain type
#
# Pythagoras:
#   PYTH_COND   — b = 1 for every element
#   PYTH_FAREY  — |b_n*e_{n+1} - b_{n+1}*e_n| = |e_{n+1}-e_n| = 1 consecutive
#   PYTH_CURV   — 2e^2 strictly increasing
#
# Plato:
#   PLATO_COND  — e = 1 for every element
#   PLATO_FAREY — |b_n*1 - b_{n+1}*1| = |b_{n+1}-b_n| = 1 consecutive
#   PLATO_CURV  — 2e^2 = 2 uniform (all equal)
#
# Fermat:
#   FERMAT_PELL — |b^2 - 2e^2| = 1 for every element
#   FERMAT_FAREY — |b_n*e_{n+1} - b_{n+1}*e_n| = 1 consecutive
#   FERMAT_MAP  — (b+2e, b+e) generates next element
# ---------------------------------------------------------------------------


def _ford_curvature(e: int) -> int:
    return 2 * e * e


def validate_pythagoras(pairs: List[Tuple[int, int]]) -> dict:
    results: dict = {}
    results["PYTH_COND"] = all(b == 1 for b, e in pairs)
    if len(pairs) >= 2:
        results["PYTH_FAREY"] = all(
            abs(pairs[i][0] * pairs[i+1][1] - pairs[i+1][0] * pairs[i][1]) == 1
            for i in range(len(pairs) - 1)
        )
        curvs = [_ford_curvature(e) for _, e in pairs]
        results["PYTH_CURV"] = all(curvs[i] < curvs[i+1] for i in range(len(curvs) - 1))
    else:
        results["PYTH_FAREY"] = True
        results["PYTH_CURV"] = True
    return results


def validate_plato(pairs: List[Tuple[int, int]]) -> dict:
    results: dict = {}
    results["PLATO_COND"] = all(e == 1 for b, e in pairs)
    if len(pairs) >= 2:
        results["PLATO_FAREY"] = all(
            abs(pairs[i][0] * pairs[i+1][1] - pairs[i+1][0] * pairs[i][1]) == 1
            for i in range(len(pairs) - 1)
        )
        curvs = [_ford_curvature(e) for _, e in pairs]
        results["PLATO_CURV"] = all(c == 2 for c in curvs)
    else:
        results["PLATO_FAREY"] = True
        results["PLATO_CURV"] = True
    return results


def validate_fermat(pairs: List[Tuple[int, int]]) -> dict:
    results: dict = {}
    results["FERMAT_PELL"] = all(abs(b * b - 2 * e * e) == 1 for b, e in pairs)
    if len(pairs) >= 2:
        results["FERMAT_FAREY"] = all(
            abs(pairs[i][0] * pairs[i+1][1] - pairs[i+1][0] * pairs[i][1]) == 1
            for i in range(len(pairs) - 1)
        )
        results["FERMAT_MAP"] = all(
            pairs[i+1] == (pairs[i][0] + 2*pairs[i][1], pairs[i][0] + pairs[i][1])
            for i in range(len(pairs) - 1)
        )
    else:
        results["FERMAT_FAREY"] = True
        results["FERMAT_MAP"] = True
    return results


_VALIDATORS = {
    "pythagoras": validate_pythagoras,
    "plato": validate_plato,
    "fermat": validate_fermat,
}


def validate_fixture(fixture: dict) -> dict:
    chain_type: str = fixture["chain_type"]
    pairs: List[Tuple[int, int]] = [(row[0], row[1]) for row in fixture["sequence"]]
    validator = _VALIDATORS.get(chain_type)
    if validator is None:
        return {"UNKNOWN_CHAIN_TYPE": False}
    return validator(pairs)


def self_test() -> bool:
    failures = []

    # --- Pythagoras self-test ---
    pyth_chain = [(1, e) for e in range(1, 13)]
    r = validate_pythagoras(pyth_chain)
    for k, v in r.items():
        if not v:
            failures.append(f"Pythagoras self-test {k} FAIL")

    # --- Plato self-test ---
    plato_chain = [(b, 1) for b in range(1, 13)]
    r = validate_plato(plato_chain)
    for k, v in r.items():
        if not v:
            failures.append(f"Plato self-test {k} FAIL")

    # --- Fermat self-test ---
    fermat_chain: List[Tuple[int, int]] = [(1, 1)]
    for _ in range(11):
        b, e = fermat_chain[-1]
        fermat_chain.append((b + 2*e, b + e))
    r = validate_fermat(fermat_chain)
    for k, v in r.items():
        if not v:
            failures.append(f"Fermat self-test {k} FAIL")

    # --- Seed (1,1) is in all three chains ---
    if pyth_chain[0] != (1, 1):
        failures.append("Pythagoras chain does not start at (1,1)")
    if plato_chain[0] != (1, 1):
        failures.append("Plato chain does not start at (1,1)")
    if fermat_chain[0] != (1, 1):
        failures.append("Fermat chain does not start at (1,1)")

    # --- Cusp directions are distinct (numeric checks) ---
    # Pythagoras: b/e = 1/e -> 0 (last element 1/12 < 1/1)
    if pyth_chain[-1][0] / pyth_chain[-1][1] >= pyth_chain[0][0] / pyth_chain[0][1]:
        failures.append("Pythagoras b/e not decreasing toward 0")
    # Plato: b/e = b/1 = b -> inf (last element 12 > 1)
    if plato_chain[-1][0] / plato_chain[-1][1] <= plato_chain[0][0] / plato_chain[0][1]:
        failures.append("Plato b/e not increasing toward inf")
    # Fermat: b/e approaches sqrt(2) ~1.41421 from alternating sides
    import math
    sqrt2 = math.sqrt(2)
    ratios = [b / e for b, e in fermat_chain]
    dists = [abs(r - sqrt2) for r in ratios]
    if not all(dists[i] > dists[i+1] for i in range(len(dists) - 1)):
        failures.append("Fermat b/e not converging to sqrt(2)")

    # --- Fail cases must fail ---
    fail_cases = [
        {"chain_type": "pythagoras", "sequence": [[1,1],[2,2],[3,3]], "expected": "FAIL"},
        {"chain_type": "plato", "sequence": [[1,1],[2,2],[3,3]], "expected": "FAIL"},
        {"chain_type": "pythagoras", "sequence": [[1,1],[1,3],[1,5]], "expected": "FAIL"},
    ]
    for case in fail_cases:
        checks = validate_fixture(case)
        if all(checks.values()):
            failures.append(f"Expected FAIL case passed: {case['sequence']}")

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
    return len(failures) == 0


FAMILY_ID = 290
CERT_SLUG = "qa_classical_subfamily_ford_cusps_cert_v1"


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
        description="QA Classical Subfamily Ford Cusps Cert validator [290]"
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
