#!/usr/bin/env python3
"""QA Mod-24 Quadrance 2-adic Signature Cert validator.

Family [287]. Primary sources:
  Wildberger, N. J. (2005). Divine Proportions. Wild Egg Books.
    Chapter 1: quadrance Q(A,B) = (x2-x1)^2 + (y2-y1)^2.
  Wall, D. D. (1960). Fibonacci series modulo m. Amer. Math. Monthly 67(6).
    DOI: 10.1080/00029890.1960.11989541.
  Mechanism: cert [279] (Orbit Access Theorem);
             cert [283] (RT Quadrance v3 signature for mod-9).

CLAIM (narrow, falsifiable, exhaustively verified):

  For (b,e) in {1,...,24}^2, let G = b^2+e^2 (quadrance),
  v2(n) = 2-adic valuation of n, and orbit_family via qa_orbit_rules.

  (1) FORMULA: v2(G) = 2*min(v2(b),v2(e)) + delta
      where delta = 1 if v2(b) == v2(e), else 0.

  (2) THRESHOLD: orbit_family determines v2(G) range exactly:
      Cosmos      -> v2(G) in {0,...,5}  (v2(G) <= 5)
      Satellite   -> v2(G) in {6,7,9}   (v2(G) >= 6)
      Singularity -> v2(G) = 7           (b=24,e=24: G=1152=2^7*9)

  (3) CONTRAST with mod-9 cert [283]: the 3-adic formula for mod-9 has
      v3(G) = 2*v3(gcd(b,e)) with NO delta term, because odd squares
      satisfy x^2 ≡ 1 (mod 3) and 1+1=2 ≢ 0 (mod 3). For p=2:
      odd squares satisfy x^2 ≡ 1 (mod 8), so 1+1=2 ≡ 2 (mod 8)
      gives exactly one extra factor of 2, hence delta=1.

  Algebraic proof (not just empirical):
    If v2(b) != v2(e): WLOG v2(b) < v2(e). Then v2(b^2) < v2(e^2),
    so v2(G) = v2(b^2) = 2*v2(b) = 2*min(v2(b),v2(e)).
    If v2(b) == v2(e) = k: write b=2^k*b'', e=2^k*e'' with b'',e'' odd.
    G = 2^(2k)*(b''^2+e''^2). Since b'',e'' odd: b''^2 ≡ 1 (mod 8)
    and e''^2 ≡ 1 (mod 8), so b''^2+e''^2 ≡ 2 (mod 8), giving
    v2(b''^2+e''^2) = 1. Therefore v2(G) = 2k+1 = 2*min(v2(b),v2(e))+1.

  The threshold (claim 2) follows from the satellite condition:
    Satellite/Singularity requires v2(b)>=3 AND v2(e)>=3 (since 8|b,8|e),
    giving v2(G) = 2*min(3,3)+delta = 6+delta >= 6.
    Cosmos requires NOT(v2(b)>=3 AND v2(e)>=3), i.e., v2(b)<3 OR v2(e)<3,
    giving v2(G) <= 2*2+1 = 5 (achieved by e.g. (4,4): G=32, v2=5).

Checks:
  V2Q_1 -- formula: v2(G) = 2*min(v2(b),v2(e)) + delta exhaustively.
  V2Q_2 -- threshold: cosmos -> v2(G) <= 5.
  V2Q_3 -- threshold: satellite/singularity -> v2(G) >= 6.
  V2Q_4 -- delta=1 witness: equal-valuation pairs yield v2(G) = 2k+1.
  V2Q_5 -- delta=0 witness: unequal-valuation satellite has v2(G) = 2k.
  SRC   -- mapping_protocol_ref.json present and well-formed.
  F     -- FAIL fixtures trigger.
"""

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic: G=b*b+e*e; v2(G) via trial "
    "division; orbit_family on (b,e,24) from qa_orbit_rules; "
    "no float feedback into QA layer"
)

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family  # noqa: E402

SCHEMA_VERSION = "QA_MOD24_QUADRANCE_V2_CERT.v1"
CERT_SLUG = "qa_mod24_quadrance_v2_signature_cert_v1"
CANDIDATE_FAMILY_ID = 287
MODULUS = 24


def _v2(n: int) -> int:
    """2-adic valuation of positive integer n."""
    if n <= 0:
        raise ValueError(f"v2 requires positive integer; got {n}")
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    return count


def _expected_v2(b: int, e: int) -> tuple[int, int]:
    """Return (expected_v2_G, delta) per the algebraic formula."""
    vb = _v2(b)
    ve = _v2(e)
    delta = 1 if vb == ve else 0
    return 2 * min(vb, ve) + delta, delta


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema(fix: dict) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_version", "fixture_kind",
        "b_value", "e_value", "expected_quadrance",
        "expected_v2_G", "expected_family",
    ]
    for field in required:
        if field not in fix:
            errors.append(f"V2Q_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"V2Q_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("V2Q_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _check_pass_fixture(fix: dict) -> list[str]:
    errors = _validate_schema(fix)
    if errors:
        return errors

    b = fix["b_value"]
    e = fix["e_value"]
    declared_G = fix["expected_quadrance"]
    declared_v2 = fix["expected_v2_G"]
    declared_fam = fix["expected_family"]

    actual_G = b * b + e * e
    if declared_G != actual_G:
        errors.append(
            f"V2Q_1: ({b},{e}) declared G={declared_G} but computed G={actual_G}"
        )
        return errors

    actual_fam = orbit_family(b, e, MODULUS)
    if declared_fam != actual_fam:
        errors.append(
            f"V2Q_1: ({b},{e}) declared family={declared_fam!r} but actual={actual_fam!r}"
        )
        return errors

    actual_v2 = _v2(actual_G)
    expected_v2, delta = _expected_v2(b, e)

    # V2Q_1: formula check
    if actual_v2 != declared_v2:
        errors.append(
            f"V2Q_1: ({b},{e}) declared v2(G)={declared_v2} but actual v2(G)={actual_v2}"
        )
    if actual_v2 != expected_v2:
        errors.append(
            f"V2Q_1: ({b},{e}) formula v2(G)={expected_v2} != actual v2(G)={actual_v2}"
        )

    # V2Q_2: cosmos threshold
    if actual_fam == "cosmos" and actual_v2 > 5:
        errors.append(
            f"V2Q_2: cosmos ({b},{e}) has v2(G)={actual_v2} > 5 (threshold violated)"
        )

    # V2Q_3: satellite/singularity threshold
    if actual_fam in ("satellite", "singularity") and actual_v2 < 6:
        errors.append(
            f"V2Q_3: {actual_fam} ({b},{e}) has v2(G)={actual_v2} < 6 (threshold violated)"
        )

    # V2Q_4/V2Q_5: delta witness checks
    vb = _v2(b)
    ve = _v2(e)
    if vb == ve:
        if actual_v2 != 2 * vb + 1:
            errors.append(
                f"V2Q_4: equal-valuation ({b},{e}): v2(G)={actual_v2} != "
                f"2*v2(b)+1={2*vb+1} (diagonal enhancement missing)"
            )
    else:
        if actual_v2 != 2 * min(vb, ve):
            errors.append(
                f"V2Q_5: unequal-valuation ({b},{e}): v2(G)={actual_v2} != "
                f"2*min(v2(b),v2(e))={2*min(vb,ve)}"
            )

    return errors


def _check_fail_fixture(fix: dict) -> list[str]:
    errors: list[str] = []
    expected = fix.get("expected_fail_type")
    if not expected:
        errors.append("F: FAIL fixture must declare expected_fail_type")
        return errors

    if expected == "MISSING_FIELD":
        if not _validate_schema(fix):
            errors.append("F: MISSING_FIELD fixture passed schema check (should have failed)")
        return errors

    schema_errs = _validate_schema(fix)
    if schema_errs:
        return errors + [f"F: FAIL fixture {expected!r} cannot evaluate: {schema_errs}"]

    b = fix["b_value"]
    e = fix["e_value"]
    declared_G = fix["expected_quadrance"]
    declared_v2 = fix["expected_v2_G"]

    actual_G = b * b + e * e
    actual_v2 = _v2(actual_G)
    expected_formula_v2, _ = _expected_v2(b, e)

    if expected == "WRONG_V2":
        if declared_v2 == actual_v2:
            errors.append(
                f"F: WRONG_V2: declared v2(G)={declared_v2} matches actual; does not fail"
            )
    elif expected == "WRONG_QUADRANCE":
        if declared_G == actual_G:
            errors.append(
                f"F: WRONG_QUADRANCE: declared G={declared_G} matches actual; does not fail"
            )
    elif expected == "WRONG_FAMILY":
        actual_fam = orbit_family(b, e, MODULUS)
        if fix.get("expected_family") == actual_fam:
            errors.append(
                f"F: WRONG_FAMILY: declared family matches actual for ({b},{e})"
            )
    else:
        errors.append(f"F: unknown expected_fail_type {expected!r}")

    return errors


def _check_src(cert_dir: Path) -> list[str]:
    ref_path = cert_dir / "mapping_protocol_ref.json"
    if not ref_path.exists():
        return ["SRC: mapping_protocol_ref.json missing"]
    try:
        ref = _load_json(ref_path)
    except Exception as exc:
        return [f"SRC: mapping_protocol_ref.json malformed: {exc}"]
    errors: list[str] = []
    if ref.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
        errors.append(f"SRC: wrong protocol_version {ref.get('protocol_version')!r}")
    for field in ("ref_path", "ref_sha256", "scope_note"):
        if not ref.get(field):
            errors.append(f"SRC: mapping_protocol_ref.json missing {field!r}")
    return errors


def validate_cert_family(cert_dir: Path, verbose: bool = False) -> tuple[bool, list[str]]:
    errors: list[str] = []
    errors.extend(_check_src(cert_dir))

    fixtures_dir = cert_dir / "fixtures"
    if not fixtures_dir.is_dir():
        errors.append("V2Q_1: fixtures/ directory missing")
        return False, errors

    fix_files = sorted(fixtures_dir.glob("*.json"))
    if not fix_files:
        errors.append("V2Q_1: no fixture files found")
        return False, errors

    pass_count = fail_count = 0
    for fix_path in fix_files:
        try:
            fix = _load_json(fix_path)
        except Exception as exc:
            errors.append(f"V2Q_1: could not load {fix_path.name}: {exc}")
            continue
        kind = fix.get("fixture_kind")
        if kind == "pass":
            pass_count += 1
            errs = _check_pass_fixture(fix)
            for err in errs:
                errors.append(f"{fix_path.name}: {err}")
        elif kind == "fail":
            fail_count += 1
            errs = _check_fail_fixture(fix)
            for err in errs:
                errors.append(f"{fix_path.name}: {err}")
        else:
            errors.append(f"{fix_path.name}: unknown fixture_kind {kind!r}")

    if pass_count == 0:
        errors.append("V2Q_1: no PASS fixtures found")
    if fail_count == 0:
        errors.append("F: no FAIL fixtures found")

    return len(errors) == 0, errors


def self_test() -> list[str]:
    """Exhaustively verify v2(G)=formula and threshold for all 576 pairs in {1,...,24}^2."""
    errors = []
    cosmos_v2_max = 0
    sat_v2_min = 99

    for b in range(1, MODULUS + 1):
        for e in range(1, MODULUS + 1):
            G = b * b + e * e
            actual_v2 = _v2(G)
            expected_v2, delta = _expected_v2(b, e)
            fam = orbit_family(b, e, MODULUS)

            # V2Q_1: formula
            if actual_v2 != expected_v2:
                errors.append(
                    f"V2Q_1: ({b},{e}) formula gives {expected_v2}, actual v2(G)={actual_v2}"
                )

            # V2Q_4/V2Q_5: delta witness
            vb, ve = _v2(b), _v2(e)
            if vb == ve and actual_v2 != 2 * vb + 1:
                errors.append(
                    f"V2Q_4: ({b},{e}) equal v2(b)=v2(e)={vb}: "
                    f"v2(G)={actual_v2} != {2*vb+1}"
                )
            if vb != ve and actual_v2 != 2 * min(vb, ve):
                errors.append(
                    f"V2Q_5: ({b},{e}) unequal v2(b)={vb},v2(e)={ve}: "
                    f"v2(G)={actual_v2} != {2*min(vb,ve)}"
                )

            # V2Q_2/V2Q_3: thresholds
            if fam == "cosmos":
                if actual_v2 > 5:
                    errors.append(
                        f"V2Q_2: cosmos ({b},{e}) v2(G)={actual_v2} > 5"
                    )
                cosmos_v2_max = max(cosmos_v2_max, actual_v2)
            else:
                if actual_v2 < 6:
                    errors.append(
                        f"V2Q_3: {fam} ({b},{e}) v2(G)={actual_v2} < 6"
                    )
                sat_v2_min = min(sat_v2_min, actual_v2)

    # Tightness witnesses
    if cosmos_v2_max != 5:
        errors.append(
            f"V2Q_2: cosmos v2(G) max should be 5 (achieved by (4,4)); got {cosmos_v2_max}"
        )
    if sat_v2_min != 6:
        errors.append(
            f"V2Q_3: satellite/singularity v2(G) min should be 6; got {sat_v2_min}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QA Mod-24 Quadrance 2-adic Signature Cert validator [287]"
    )
    parser.add_argument(
        "cert_dir", nargs="?",
        default=str(Path(__file__).parent),
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--self-test", action="store_true", dest="selftest")
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir)

    if args.selftest:
        st_errors = self_test()
        ok, fam_errors = validate_cert_family(cert_dir)
        all_errors = st_errors + fam_errors
        fix_dir = cert_dir / "fixtures"
        fix_files = list(fix_dir.glob("*.json")) if fix_dir.is_dir() else []
        pass_files = [f for f in fix_files if "pass_" in f.name]
        fail_files = [f for f in fix_files if "fail_" in f.name]
        payload = {
            "ok": len(all_errors) == 0,
            "family_id": CANDIDATE_FAMILY_ID,
            "slug": CERT_SLUG,
            "schema_version": SCHEMA_VERSION,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": all_errors,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["ok"] else 1

    if args.demo:
        print(f"QA Mod-24 Quadrance 2-adic Signature Cert [287] — v2 formula demo")
        print(f"{'(b,e)':12} {'family':12} {'G':6} {'v2(G)':6} {'formula':8} {'delta':5}")
        print("-" * 55)
        demo_pairs = [
            (1, 1), (4, 4), (2, 4), (8, 8), (8, 16), (16, 16), (8, 24), (24, 24)
        ]
        for b, e in demo_pairs:
            G = b * b + e * e
            fam = orbit_family(b, e, MODULUS)
            actual_v2 = _v2(G)
            ev2, delta = _expected_v2(b, e)
            ok_str = "✓" if actual_v2 == ev2 else "✗"
            print(f"({b},{e}){'':8} {fam:12} {G:<6} {actual_v2:<6} {ev2:<8} {delta}  {ok_str}")
        return 0

    ok, errors = validate_cert_family(cert_dir)
    for e in errors:
        print(f"FAIL: {e}")
    print(f"{'PASS' if ok else 'FAIL'}: {len(errors)} error(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
