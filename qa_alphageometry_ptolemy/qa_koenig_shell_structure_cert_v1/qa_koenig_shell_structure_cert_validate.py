# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Hardy+Wright 2008 ISBN 978-0-19-921986-5; Wildberger 2005 ISBN 978-0-9757492-0-8) -->
"""Cert [293]: QA Koenig Shell Structure (Pell-Depth Stratification).

PRIMARY CLAIM:
  The Ford circle packing is stratified by the Koenig I invariant into shells
  S_k = {(b,e) in Z>0 x Z>0 : |b^2 - 2e^2| = k}.

  For any (b,e) in S_k with QA successor (b',e') = (b+2e, b+e):

  (1) SHELL_PRESERVE:  I(b',e')  = k              (shell preserved)
  (2) SIGN_FLIP:       b'^2-2e'^2 = -(b^2-2e^2)   (sign alternates)
  (3) FAREY_K:         |be' - b'e| = k             (Farey det = shell depth)
  (4) SPREAD_K:        spread(d_n, d_{n+1}) = k^2/(G_tilde * G_tilde')
                       where G_tilde = b^2+e^2, d = direction (b,e)
  (5) SPREAD_DEV_K:    |s(b,e) - 1/3| = k/(3*G_tilde)  [cert [292] corollary]
  (6) SHELL_UNIQUE:    k=1 is the unique shell with Farey det = 1 (tangency)

ALGEBRAIC PROOFS (all trivial substitutions):
  SHELL_PRESERVE: (b+2e)^2 - 2(b+e)^2 = b^2+4be+4e^2 - 2b^2-4be-2e^2
                                        = -b^2+2e^2 = -(b^2-2e^2)
                  => |I(b',e')| = |b^2-2e^2| = k
  SIGN_FLIP:      same calculation, signed.
  FAREY_K:        |b(b+e) - (b+2e)e| = |b^2+be - be-2e^2| = |b^2-2e^2| = k
  SPREAD_K:       spread = det^2/(G*G') = k^2/(G*G')  [det = k from FAREY_K]
  SPREAD_DEV_K:   I = 3*G_tilde*|s-1/3|  =>  k = 3*G*|s-1/3|
                  =>  |s-1/3| = k/(3*G)  [cert [292] identity]

EMPTY SHELLS:
  Primes p ≡ ±3 (mod 8) are inert in Z[sqrt(2)], so b^2-2e^2 ≠ ±p for any
  integer (b,e). Verified computationally: k=3,5,6,10,11 have no solutions
  for b,e in [1..100].

SHELL EXAMPLES:
  k=1: (1,1)->(3,2)->(7,5)->...      det=1, spread=1/(G*G')   [Pell, cert [289]]
  k=2: (2,1)->(4,3)->(10,7)->...     det=2, spread=4/(G*G')
  k=7: (3,1)->(5,4)->(13,9)->...     det=7, spread=49/(G*G')
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple


def _I(b: int, e: int) -> int:
    return abs(b * b - 2 * e * e)


def _raw(b: int, e: int) -> int:
    return b * b - 2 * e * e


def _qa_next(b: int, e: int) -> Tuple[int, int]:
    return b + 2 * e, b + e


def _G(b: int, e: int) -> int:
    return b * b + e * e


def _s(b: int, e: int) -> Fraction:
    return Fraction(e * e, b * b + e * e)


_ONE_THIRD = Fraction(1, 3)

# ---------------------------------------------------------------------------
# Checks per fixture
# SHELL_I      — I(b,e) = shell_k
# SHELL_PRES   — I(b',e') = shell_k  (QA successor in same shell)
# SIGN_FLIP    — b'^2-2e'^2 = -(b^2-2e^2)
# FAREY_K      — |be'-b'e| = shell_k
# SPREAD_K     — spread = Fraction(k^2, G*G')
# SPREAD_DEV_K — |s-1/3| = Fraction(k, 3*G)  [cert [292] corollary]
# ---------------------------------------------------------------------------


def validate_fixture(fixture: dict) -> dict:
    b, e = fixture["state"]
    k: int = fixture["shell_k"]
    b2, e2 = _qa_next(b, e)

    results: dict = {}
    G = _G(b, e)
    G2 = _G(b2, e2)

    results["SHELL_I"]      = (_I(b, e) == k)
    results["SHELL_PRES"]   = (_I(b2, e2) == k)
    results["SIGN_FLIP"]    = (_raw(b2, e2) == -_raw(b, e))
    results["FAREY_K"]      = (abs(b * e2 - b2 * e) == k)
    results["SPREAD_K"]     = (Fraction((b * e2 - b2 * e) ** 2, G * G2)
                                == Fraction(k * k, G * G2))
    dev = abs(_s(b, e) - _ONE_THIRD)
    results["SPREAD_DEV_K"] = (dev == Fraction(k, 3 * G))

    return results


def self_test() -> bool:
    failures = []

    # --- Verify all checks for shells k=1,2,7 chains ---
    shell_seeds = {1: (1, 1), 2: (2, 1), 7: (3, 1)}
    for k, seed in shell_seeds.items():
        chain = [seed]
        for _ in range(8):
            chain.append(_qa_next(*chain[-1]))
        for b, e in chain[:8]:
            checks = validate_fixture({"state": [b, e], "shell_k": k})
            for check, val in checks.items():
                if not val:
                    failures.append(f"Shell k={k} ({b},{e}) check {check} FAIL")

    # --- SHELL_UNIQUE: k=1 is the only shell with Farey det=1 ---
    # For k>1, consecutive pairs have det=k>1, never 1.
    for k in range(2, 20):
        seeds_k = [(b, e) for b in range(1, 30) for e in range(1, 30)
                   if _I(b, e) == k]
        for b, e in seeds_k[:3]:
            b2, e2 = _qa_next(b, e)
            det = abs(b * e2 - b2 * e)
            if det != k:
                failures.append(f"FAREY_K violated for k={k} ({b},{e}): det={det}")
            if det == 1:
                failures.append(f"SHELL_UNIQUE violated: k={k} has det=1 at ({b},{e})")

    # --- Empty shells: k=3,5,6,10,11 have no solutions ---
    empty_k = [3, 5, 6, 10, 11]
    for k in empty_k:
        hits = [(b, e) for b in range(1, 101) for e in range(1, 101)
                if _I(b, e) == k]
        if hits:
            failures.append(f"Empty shell k={k} violated: found {hits[:3]}")

    # --- Non-empty shells: k=1,2,7,8,14 have solutions ---
    nonempty = {1: (1, 1), 2: (2, 1), 7: (3, 1), 8: (4, 2), 14: (4, 1)}
    for k, expected_seed in nonempty.items():
        if _I(*expected_seed) != k:
            failures.append(f"Non-empty shell k={k}: expected seed {expected_seed} has I={_I(*expected_seed)}")

    # --- Sign flip alternation along a chain ---
    b, e = 2, 1
    for step in range(6):
        raw = _raw(b, e)
        b2, e2 = _qa_next(b, e)
        raw2 = _raw(b2, e2)
        if raw2 != -raw:
            failures.append(f"Sign flip failed at step {step}: raw={raw}, next raw={raw2}")
        b, e = b2, e2

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
    return len(failures) == 0


FAMILY_ID = 293
CERT_SLUG = "qa_koenig_shell_structure_cert_v1"


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
        description="QA Koenig Shell Structure Cert validator [293]"
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
