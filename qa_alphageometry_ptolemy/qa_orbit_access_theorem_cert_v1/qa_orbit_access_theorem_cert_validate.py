#!/usr/bin/env python3
"""QA Orbit Access Theorem Cert validator.

Family [279]. Primary source for Pisano-period framing:
Wall, D. D. (1960). Fibonacci series modulo m. American Mathematical
Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

Certifies the claim:

    For mod-9 route enumeration — all (b, e) with b+2e=a, b>=1, e>=1,
    A1-reduced via bq=((b-1)%9)+1, eq=((e-1)%9)+1 — the orbit classes
    reachable for a given a are determined exclusively by gcd(a, 3):

    (1) coprime_to_3: gcd(a,3)=1 → all routes Cosmos, zero Satellite,
        zero Singularity.
    (2) mul_3_not_9: 3|a and 9 not divides a → Cosmos + Satellite,
        zero Singularity.
    (3) mul_9: 9|a → Cosmos + Satellite + Singularity.

Algebraic mechanism (not certified here, but motivating):
    Satellite period = 8 = Pisano(3).
    Cosmos period   = 24 = Pisano(9).
    Divisibility by 3 (resp. 9) controls which Pisano orbit classes
    the A1-reduced route set can reach.

Checks:
  OAT_1 — coprime_to_3: satellite=0 AND singularity=0.
  OAT_2 — mul_3_not_9: satellite>0 AND singularity=0.
  OAT_3 — mul_9: satellite>0 AND singularity>0.
  SRC   — mapping_protocol_ref.json present and well-formed.
  F     — every FAIL fixture declares expected_fail_type and the
          declared mode actually fires.
"""

QA_COMPLIANCE = (
    "cert_validator — integer arithmetic on (b, e); A1-reduction via "
    "((v-1)%9)+1; orbit_family on (bq, eq, 9); no float feedback into QA layer"
)

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family  # noqa: E402

SCHEMA_VERSION = "QA_ORBIT_ACCESS_THEOREM_CERT.v1"
CERT_SLUG = "qa_orbit_access_theorem_cert_v1"
CANDIDATE_FAMILY_ID = 279
MODULUS = 9


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _a1(v: int, m: int = MODULUS) -> int:
    r = v % m
    return r if r else m


def _enumerate_routes(a: int) -> dict:
    """Return orbit counts for all (b,e) with b+2e=a, b>=1, e>=1."""
    cosmos = satellite = singularity = 0
    for e in range(1, (a - 1) // 2 + 1):
        b = a - 2 * e
        if b < 1:
            continue
        bq = _a1(b)
        eq = _a1(e)
        orb = orbit_family(bq, eq, MODULUS)
        if orb == "cosmos":
            cosmos += 1
        elif orb == "satellite":
            satellite += 1
        elif orb == "singularity":
            singularity += 1
    total = cosmos + satellite + singularity
    return {
        "total": total,
        "cosmos": cosmos,
        "satellite": satellite,
        "singularity": singularity,
    }


def _divisibility_class(a: int) -> str:
    if a % 9 == 0:
        return "mul_9"
    if a % 3 == 0:
        return "mul_3_not_9"
    return "coprime_to_3"


def _validate_schema(fix: dict) -> list[str]:
    required = [
        "schema_version", "fixture_kind", "a_value", "divisibility_class",
        "expected_total_routes", "expected_cosmos",
        "expected_satellite", "expected_singularity",
    ]
    errors: list[str] = []
    for field in required:
        if field not in fix:
            errors.append(f"OAT_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"OAT_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("OAT_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _check_pass_fixture(fix: dict) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_schema(fix))
    if errors:
        return errors

    a = fix["a_value"]
    declared_class = fix["divisibility_class"]
    actual_class = _divisibility_class(a)

    if declared_class != actual_class:
        errors.append(
            f"OAT_1: fixture declares divisibility_class={declared_class!r} "
            f"but gcd analysis gives {actual_class!r} for a={a}"
        )
        return errors

    counts = _enumerate_routes(a)

    # Check totals
    if counts["total"] != fix["expected_total_routes"]:
        errors.append(
            f"OAT_1: a={a} total_routes={counts['total']}, "
            f"expected={fix['expected_total_routes']}"
        )
    if counts["cosmos"] != fix["expected_cosmos"]:
        errors.append(
            f"OAT_1: a={a} cosmos={counts['cosmos']}, "
            f"expected={fix['expected_cosmos']}"
        )
    if counts["satellite"] != fix["expected_satellite"]:
        errors.append(
            f"OAT_1: a={a} satellite={counts['satellite']}, "
            f"expected={fix['expected_satellite']}"
        )
    if counts["singularity"] != fix["expected_singularity"]:
        errors.append(
            f"OAT_1: a={a} singularity={counts['singularity']}, "
            f"expected={fix['expected_singularity']}"
        )

    # Gate-specific invariants
    if declared_class == "coprime_to_3":
        if counts["satellite"] != 0:
            errors.append(
                f"OAT_1: coprime_to_3 a={a} must have satellite=0; "
                f"got {counts['satellite']}"
            )
        if counts["singularity"] != 0:
            errors.append(
                f"OAT_1: coprime_to_3 a={a} must have singularity=0; "
                f"got {counts['singularity']}"
            )
    elif declared_class == "mul_3_not_9":
        if counts["satellite"] == 0:
            errors.append(
                f"OAT_2: mul_3_not_9 a={a} must have satellite>0; "
                f"got {counts['satellite']}"
            )
        if counts["singularity"] != 0:
            errors.append(
                f"OAT_2: mul_3_not_9 a={a} must have singularity=0; "
                f"got {counts['singularity']}"
            )
    elif declared_class == "mul_9":
        if counts["satellite"] == 0:
            errors.append(
                f"OAT_3: mul_9 a={a} must have satellite>0; "
                f"got {counts['satellite']}"
            )
        if counts["singularity"] == 0:
            errors.append(
                f"OAT_3: mul_9 a={a} must have singularity>0; "
                f"got {counts['singularity']}"
            )

    return errors


def _check_fail_fixture(fix: dict) -> list[str]:
    errors: list[str] = []
    expected = fix.get("expected_fail_type")
    if not expected:
        errors.append("F: FAIL fixture must declare expected_fail_type")
        return errors

    schema_errs = _validate_schema(fix)
    if expected == "MISSING_FIELD":
        if not schema_errs:
            errors.append("F: MISSING_FIELD fixture passed schema check (should have failed)")
        return errors

    if schema_errs:
        return errors + [
            f"F: FAIL fixture {expected!r} cannot evaluate due to schema errors: {schema_errs}"
        ]

    a = fix["a_value"]
    declared_class = fix["divisibility_class"]
    counts = _enumerate_routes(a)

    if expected == "WRONG_SATELLITE_FOR_COPRIME":
        # Fixture claims satellite>0 for a coprime to 3 — must fail OAT_1
        if declared_class != "coprime_to_3":
            errors.append(
                f"F: WRONG_SATELLITE_FOR_COPRIME fixture must have "
                f"divisibility_class=coprime_to_3; got {declared_class!r}"
            )
        if fix["expected_satellite"] == 0:
            errors.append(
                "F: WRONG_SATELLITE_FOR_COPRIME fixture must claim expected_satellite>0"
            )
        actual_sat = counts["satellite"]
        if actual_sat != 0:
            errors.append(
                f"F: WRONG_SATELLITE_FOR_COPRIME expected actual satellite=0 "
                f"for coprime a={a}; got {actual_sat} (cert claim itself is broken)"
            )
        # The fixture should cause OAT_1 to fire: claimed != actual
        if fix["expected_satellite"] == actual_sat:
            errors.append(
                f"F: WRONG_SATELLITE_FOR_COPRIME fixture claims satellite="
                f"{fix['expected_satellite']} but actual={actual_sat} "
                f"(they match — FAIL fixture does not actually fail)"
            )

    elif expected == "SINGULARITY_FOR_MUL3":
        # Fixture claims singularity>0 for a mul_3_not_9 — must fail OAT_2
        if declared_class != "mul_3_not_9":
            errors.append(
                f"F: SINGULARITY_FOR_MUL3 fixture must have "
                f"divisibility_class=mul_3_not_9; got {declared_class!r}"
            )
        if fix["expected_singularity"] == 0:
            errors.append(
                "F: SINGULARITY_FOR_MUL3 fixture must claim expected_singularity>0"
            )
        actual_sing = counts["singularity"]
        if actual_sing != 0:
            errors.append(
                f"F: SINGULARITY_FOR_MUL3 expected actual singularity=0 "
                f"for mul_3_not_9 a={a}; got {actual_sing}"
            )
        if fix["expected_singularity"] == actual_sing:
            errors.append(
                f"F: SINGULARITY_FOR_MUL3 fixture does not actually fail "
                f"(claimed={fix['expected_singularity']}, actual={actual_sing})"
            )

    elif expected == "ZERO_SATELLITE_FOR_MUL9":
        # Fixture claims satellite=0 for a mul_9 value — must fail OAT_3
        if declared_class != "mul_9":
            errors.append(
                f"F: ZERO_SATELLITE_FOR_MUL9 fixture must have "
                f"divisibility_class=mul_9; got {declared_class!r}"
            )
        if fix["expected_satellite"] != 0:
            errors.append(
                "F: ZERO_SATELLITE_FOR_MUL9 fixture must claim expected_satellite=0"
            )
        actual_sat = counts["satellite"]
        if actual_sat == 0:
            errors.append(
                f"F: ZERO_SATELLITE_FOR_MUL9 expected actual satellite>0 "
                f"for mul_9 a={a}; got {actual_sat}"
            )
        if fix["expected_satellite"] == actual_sat:
            errors.append(
                f"F: ZERO_SATELLITE_FOR_MUL9 fixture does not actually fail "
                f"(claimed={fix['expected_satellite']}, actual={actual_sat})"
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
        errors.append(
            f"SRC: wrong protocol_version {ref.get('protocol_version')!r}"
        )
    for field in ("ref_path", "ref_sha256", "scope_note"):
        if not ref.get(field):
            errors.append(f"SRC: mapping_protocol_ref.json missing {field!r}")
    return errors


def validate_cert_family(cert_dir: Path, verbose: bool = False) -> tuple[bool, list[str]]:
    errors: list[str] = []

    # SRC gate
    errors.extend(_check_src(cert_dir))

    # Fixtures gate
    fixtures_dir = cert_dir / "fixtures"
    if not fixtures_dir.is_dir():
        errors.append("OAT_1: fixtures/ directory missing")
        return False, errors

    fix_files = sorted(fixtures_dir.glob("*.json"))
    if not fix_files:
        errors.append("OAT_1: no fixture files found in fixtures/")
        return False, errors

    pass_count = fail_count = 0
    for fix_path in fix_files:
        try:
            fix = _load_json(fix_path)
        except Exception as exc:
            errors.append(f"OAT_1: could not load {fix_path.name}: {exc}")
            continue

        kind = fix.get("fixture_kind")
        if kind == "pass":
            pass_count += 1
            errs = _check_pass_fixture(fix)
            for e in errs:
                errors.append(f"{fix_path.name}: {e}")
            if verbose and not errs:
                print(f"  PASS fixture {fix_path.name}: ok")
        elif kind == "fail":
            fail_count += 1
            errs = _check_fail_fixture(fix)
            for e in errs:
                errors.append(f"{fix_path.name}: {e}")
            if verbose and not errs:
                print(f"  FAIL fixture {fix_path.name}: ok (fail mode confirmed)")
        else:
            errors.append(f"{fix_path.name}: unknown fixture_kind {kind!r}")

    if pass_count == 0:
        errors.append("OAT_1: no PASS fixtures found")
    if fail_count == 0:
        errors.append("F: no FAIL fixtures found")

    return len(errors) == 0, errors


def self_test() -> dict:
    """Inline self-test: verify orbit counts for each divisibility class."""
    cases = [
        (8, "coprime_to_3", 3, 3, 0, 0),
        (28, "coprime_to_3", 13, 13, 0, 0),
        (21, "mul_3_not_9", 10, 7, 3, 0),
        (15, "mul_3_not_9", 7, 5, 2, 0),
        (27, "mul_9", 13, 9, 3, 1),
        (126, "mul_9", 62, 42, 14, 6),
        (144, "mul_9", 71, 48, 16, 7),
    ]
    errors = []
    for a, cls, n, cos, sat, sg in cases:
        actual_cls = _divisibility_class(a)
        if actual_cls != cls:
            errors.append(f"a={a}: expected class {cls}, got {actual_cls}")
            continue
        counts = _enumerate_routes(a)
        if counts["total"] != n:
            errors.append(f"a={a}: total {counts['total']} != {n}")
        if counts["cosmos"] != cos:
            errors.append(f"a={a}: cosmos {counts['cosmos']} != {cos}")
        if counts["satellite"] != sat:
            errors.append(f"a={a}: satellite {counts['satellite']} != {sat}")
        if counts["singularity"] != sg:
            errors.append(f"a={a}: singularity {counts['singularity']} != {sg}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QA Orbit Access Theorem Cert validator [279]"
    )
    parser.add_argument(
        "cert_dir", nargs="?",
        default=str(Path(__file__).parent),
        help="Path to cert family root (default: this file's directory)",
    )
    parser.add_argument("--demo", action="store_true", help="Run with verbose output")
    parser.add_argument(
        "--self-test", action="store_true", dest="selftest",
        help="Emit JSON {ok, errors, ...} for the meta-validator",
    )
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir)

    if args.selftest:
        st_errors = self_test()
        ok, fam_errors = validate_cert_family(cert_dir)
        all_errors = st_errors + fam_errors
        # Count fixtures
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
        print(f"PASS [{CERT_SLUG}]: OAT_1/OAT_2/OAT_3/SRC/F all ok")
        return 0
    else:
        print(f"FAIL [{CERT_SLUG}]: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
