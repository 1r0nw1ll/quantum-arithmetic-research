#!/usr/bin/env python3
"""QA RT Quadrance Orbit Divisibility Cert validator.

Family [283]. Primary source:
  Wildberger, N. J. (2005). Divine Proportions: Rational Trigonometry
  to Universal Geometry. Wild Egg Books. ISBN 978-0-9757492-0-8.
  Chapter 1: quadrance Q(A,B) = (x2-x1)^2 + (y2-y1)^2.
Mechanism: QA Orbit Access Theorem cert [279];
           orbit classifier qa_orbit_rules.orbit_family on (b, e, 9).

Certifies the claim:

    For (b,e) in {1,...,9}^2, the Wildberger quadrance G = b^2+e^2 has
    3-adic valuation determined exactly by the orbit family:

      Cosmos      -> v3(G) = 0  (G coprime to 3)
      Satellite   -> v3(G) = 2  (G = 9k, gcd(k,3) = 1)
      Singularity -> v3(G) = 4  (G = 162 = 2*81)

    Equivalently: v3(G) = 2 * v3(gcd(b,e)) for all 81 pairs in {1,...,9}^2.

    Corollary (scope boundary, NOT certified here): the spread
    s = (b1*e2-b2*e1)^2 / (G1*G2) is invariant under 3-adic scaling of
    direction vectors (Lagrange identity). Therefore spread denominators
    carry no orbit-class 3-adic signature; orbit-class structure lives in G.

    Cert does NOT certify spread denominator structure.
    Cert does NOT claim G values outside {1,...,9}^2.
    Cert does NOT claim the v3(G) result extends to mod-24 QA.

Checks:
  RTQ_1 -- cosmos: v3(G) = 0.
  RTQ_2 -- satellite: v3(G) = 2.
  RTQ_3 -- singularity: v3(G) = 4 (G = 162).
  RTQ_4 -- v3(G) = 2*v3(gcd(b,e)) exhaustive for all 81 pairs.
  SRC   -- mapping_protocol_ref.json present and well-formed.
  F     -- every FAIL fixture declares expected_fail_type and fires.
"""

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic: G = b*b+e*e; v3(G) via trial "
    "division; gcd(b,e) via math.gcd; orbit_family on (b, e, 9) from "
    "qa_orbit_rules; no float feedback into QA layer"
)

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family  # noqa: E402

SCHEMA_VERSION = "QA_RT_QUADRANCE_ORBIT_CERT.v1"
CERT_SLUG = "qa_rt_quadrance_orbit_cert_v1"
CANDIDATE_FAMILY_ID = 283
MODULUS = 9

EXPECTED_V3 = {"cosmos": 0, "satellite": 2, "singularity": 4}
SINGULARITY_G = 162  # (9,9): G = 81+81 = 162 = 2*3^4


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _v3(n: int) -> int:
    """3-adic valuation of positive integer n."""
    if n <= 0:
        raise ValueError(f"v3 requires positive integer; got {n}")
    count = 0
    while n % 3 == 0:
        n //= 3
        count += 1
    return count


def _validate_schema(fix: dict) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_version", "fixture_kind",
        "b_value", "e_value", "expected_quadrance", "expected_family",
    ]
    for field in required:
        if field not in fix:
            errors.append(f"RTQ_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"RTQ_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("RTQ_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _check_pass_fixture(fix: dict) -> list[str]:
    errors = _validate_schema(fix)
    if errors:
        return errors

    b = fix["b_value"]
    e = fix["e_value"]
    declared_G = fix["expected_quadrance"]
    declared_fam = fix["expected_family"]

    # Verify declared G = b^2+e^2
    actual_G = b * b + e * e
    if declared_G != actual_G:
        errors.append(
            f"RTQ_1: ({b},{e}) declared quadrance={declared_G} "
            f"but computed G={actual_G}"
        )
        return errors

    # Verify orbit family
    actual_fam = orbit_family(b, e, MODULUS)
    if declared_fam != actual_fam:
        errors.append(
            f"RTQ_1: ({b},{e}) declared family={declared_fam!r} "
            f"but actual={actual_fam!r}"
        )
        return errors

    # Gate checks on v3(G)
    v3_G = _v3(actual_G)
    exp_v3 = EXPECTED_V3[actual_fam]
    if v3_G != exp_v3:
        gate = {"cosmos": "RTQ_1", "satellite": "RTQ_2", "singularity": "RTQ_3"}[actual_fam]
        errors.append(
            f"{gate}: ({b},{e}) {actual_fam}: G={actual_G}, v3(G)={v3_G}, "
            f"expected v3={exp_v3}"
        )

    # RTQ_3: singularity exact value
    if actual_fam == "singularity" and actual_G != SINGULARITY_G:
        errors.append(
            f"RTQ_3: singularity G={actual_G} != {SINGULARITY_G} (expected 162=2*81)"
        )

    # RTQ_4: formula v3(G) = 2*v3(gcd(b,e))
    g = math.gcd(b, e)
    v3_gcd = _v3(g) if g % 3 == 0 else 0
    if v3_G != 2 * v3_gcd:
        errors.append(
            f"RTQ_4: ({b},{e}): v3(G={actual_G})={v3_G} != "
            f"2*v3(gcd({b},{e})={g})={2*v3_gcd}"
        )

    return errors


def _check_fail_fixture(fix: dict) -> list[str]:
    errors: list[str] = []
    expected = fix.get("expected_fail_type")
    if not expected:
        errors.append("F: FAIL fixture must declare expected_fail_type")
        return errors

    if expected == "MISSING_FIELD":
        schema_errs = _validate_schema(fix)
        if not schema_errs:
            errors.append("F: MISSING_FIELD fixture passed schema check (should have failed)")
        return errors

    schema_errs = _validate_schema(fix)
    if schema_errs:
        return errors + [f"F: FAIL fixture {expected!r} cannot evaluate: {schema_errs}"]

    b = fix["b_value"]
    e = fix["e_value"]
    declared_G = fix["expected_quadrance"]
    declared_fam = fix["expected_family"]

    actual_G = b * b + e * e
    actual_fam = orbit_family(b, e, MODULUS)

    if expected == "WRONG_FAMILY":
        if declared_fam == actual_fam:
            errors.append(
                f"F: WRONG_FAMILY: declared family={declared_fam!r} matches actual "
                f"for ({b},{e}); does not fail"
            )

    elif expected == "WRONG_QUADRANCE":
        if declared_G == actual_G:
            errors.append(
                f"F: WRONG_QUADRANCE: declared G={declared_G} matches actual "
                f"G={actual_G} for ({b},{e}); does not fail"
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
        errors.append("RTQ_1: fixtures/ directory missing")
        return False, errors

    fix_files = sorted(fixtures_dir.glob("*.json"))
    if not fix_files:
        errors.append("RTQ_1: no fixture files found")
        return False, errors

    pass_count = fail_count = 0
    for fix_path in fix_files:
        try:
            fix = _load_json(fix_path)
        except Exception as exc:
            errors.append(f"RTQ_1: could not load {fix_path.name}: {exc}")
            continue
        kind = fix.get("fixture_kind")
        if kind == "pass":
            pass_count += 1
            errs = _check_pass_fixture(fix)
            for err in errs:
                errors.append(f"{fix_path.name}: {err}")
            if verbose and not errs:
                print(f"  PASS fixture {fix_path.name}: ok")
        elif kind == "fail":
            fail_count += 1
            errs = _check_fail_fixture(fix)
            for err in errs:
                errors.append(f"{fix_path.name}: {err}")
            if verbose and not errs:
                print(f"  FAIL fixture {fix_path.name}: ok (fail mode confirmed)")
        else:
            errors.append(f"{fix_path.name}: unknown fixture_kind {kind!r}")

    if pass_count == 0:
        errors.append("RTQ_1: no PASS fixtures found")
    if fail_count == 0:
        errors.append("F: no FAIL fixtures found")

    return len(errors) == 0, errors


def self_test() -> list[str]:
    """Verify v3(G)=2*v3(gcd(b,e)) for all 81 pairs in {1,...,9}^2."""
    errors = []
    for b in range(1, MODULUS + 1):
        for e in range(1, MODULUS + 1):
            fam = orbit_family(b, e, MODULUS)
            G = b * b + e * e
            v3_G = _v3(G)
            exp_v3 = EXPECTED_V3[fam]
            if v3_G != exp_v3:
                errors.append(
                    f"({b},{e}) {fam}: v3(G={G})={v3_G}, expected {exp_v3}"
                )
            g = math.gcd(b, e)
            v3_gcd = _v3(g) if g % 3 == 0 else 0
            if v3_G != 2 * v3_gcd:
                errors.append(
                    f"({b},{e}): v3(G={G})={v3_G} != 2*v3(gcd={g})={2*v3_gcd}"
                )
            if fam == "singularity" and G != SINGULARITY_G:
                errors.append(
                    f"({b},{e}) singularity: G={G} != {SINGULARITY_G}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QA RT Quadrance Orbit Divisibility Cert validator [283]"
    )
    parser.add_argument(
        "cert_dir", nargs="?",
        default=str(Path(__file__).parent),
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--self-test", action="store_true", dest="selftest",
                        help="Emit JSON {ok, ...} for the meta-validator")
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

    ok, errors = validate_cert_family(cert_dir, verbose=args.demo)
    if ok:
        print(f"PASS [{CERT_SLUG}]: RTQ_1/RTQ_2/RTQ_3/RTQ_4/SRC/F all ok")
        return 0
    print(f"FAIL [{CERT_SLUG}]: {len(errors)} error(s)")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
