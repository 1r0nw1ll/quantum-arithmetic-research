#!/usr/bin/env python3
"""QA Nuclear Magic Orbit Cert validator.

Family [280]. Primary sources:
  Mayer, M. G. (1949). On Closed Shells in Nuclei. Physical Review,
  75(12), 1969-1970. DOI: 10.1103/PhysRev.75.1969.
  Haxel, O., Jensen, J. H. D., & Suess, H. E. (1949). On the "Magic
  Numbers" in Nuclear Structure. Physical Review, 75(11), 1766.
  DOI: 10.1103/PhysRev.75.1766.2.
Mechanism: QA Orbit Access Theorem cert [279] (Wall 1960).

Certifies the claim:

    The canonical nuclear magic numbers {2, 8, 20, 28, 50, 82, 126, 184}
    partition under mod-9 route enumeration into exactly three classes:

    (A) no_routes (a=2): zero valid (b,e) pairs with b+2e=a, b>=1, e>=1.
    (B) coprime_to_3 ({8,20,28,50,82,184}): pure Cosmos, zero Satellite,
        zero Singularity.
    (C) mul_9 (a=126=14*9): 62 routes — 42 Cosmos, 14 Satellite,
        6 Singularity.

    a=126 is the unique magic number in the set divisible by 3.

Cert does NOT claim QA causes nuclear shell closure, predict binding
energy, or address the physical mechanism of magic number stability.

Checks:
  NMO_1 — no_routes: total_routes=0.
  NMO_2 — coprime_to_3: satellite=0 AND singularity=0.
  NMO_3 — mul_9: exact counts match (total, cosmos, satellite, singularity).
  NMO_4 — magic_number_class matches gcd(a,3) classification.
  SRC   — mapping_protocol_ref.json present and well-formed.
  F     — every FAIL fixture declares expected_fail_type and fires.
"""

QA_COMPLIANCE = (
    "cert_validator — integer arithmetic on (b, e); A1-reduction via "
    "((v-1)%9)+1; orbit_family on (bq, eq, 9); no float feedback into QA layer"
)

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family  # noqa: E402

SCHEMA_VERSION = "QA_NUCLEAR_MAGIC_ORBIT_CERT.v1"
CERT_SLUG = "qa_nuclear_magic_orbit_cert_v1"
CANDIDATE_FAMILY_ID = 280
MODULUS = 9

KNOWN_MAGIC_NUMBERS = frozenset({2, 8, 20, 28, 50, 82, 126, 184})


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _a1(v: int, m: int = MODULUS) -> int:
    r = v % m
    return r if r else m


def _enumerate_routes(a: int) -> dict:
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
    return {"total": total, "cosmos": cosmos, "satellite": satellite, "singularity": singularity}


def _magic_number_class(a: int) -> str:
    counts = _enumerate_routes(a)
    if counts["total"] == 0:
        return "no_routes"
    if a % 9 == 0:
        return "mul_9"
    if a % 3 == 0:
        return "mul_3_not_9"
    return "coprime_to_3"


def _validate_schema(fix: dict) -> list[str]:
    required = [
        "schema_version", "fixture_kind", "a_value", "magic_number_class",
        "expected_total_routes", "expected_cosmos",
        "expected_satellite", "expected_singularity",
    ]
    errors: list[str] = []
    for field in required:
        if field not in fix:
            errors.append(f"NMO_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"NMO_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("NMO_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _check_pass_fixture(fix: dict) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_schema(fix))
    if errors:
        return errors

    a = fix["a_value"]
    declared_class = fix["magic_number_class"]

    # NMO_4: class must match actual gcd-based classification
    actual_class = _magic_number_class(a)
    if declared_class != actual_class:
        errors.append(
            f"NMO_4: a={a} declared magic_number_class={declared_class!r} "
            f"but actual class is {actual_class!r}"
        )
        return errors

    counts = _enumerate_routes(a)

    # Check counts match fixture
    if counts["total"] != fix["expected_total_routes"]:
        errors.append(
            f"NMO_1: a={a} total_routes={counts['total']}, expected={fix['expected_total_routes']}"
        )
    if counts["cosmos"] != fix["expected_cosmos"]:
        errors.append(
            f"NMO_1: a={a} cosmos={counts['cosmos']}, expected={fix['expected_cosmos']}"
        )
    if counts["satellite"] != fix["expected_satellite"]:
        errors.append(
            f"NMO_2: a={a} satellite={counts['satellite']}, expected={fix['expected_satellite']}"
        )
    if counts["singularity"] != fix["expected_singularity"]:
        errors.append(
            f"NMO_3: a={a} singularity={counts['singularity']}, expected={fix['expected_singularity']}"
        )

    # Gate-specific invariants
    if declared_class == "no_routes":
        # NMO_1
        if counts["total"] != 0:
            errors.append(f"NMO_1: no_routes a={a} must have total_routes=0; got {counts['total']}")
    elif declared_class == "coprime_to_3":
        # NMO_2
        if counts["satellite"] != 0:
            errors.append(f"NMO_2: coprime_to_3 a={a} must have satellite=0; got {counts['satellite']}")
        if counts["singularity"] != 0:
            errors.append(f"NMO_2: coprime_to_3 a={a} must have singularity=0; got {counts['singularity']}")
    elif declared_class == "mul_9":
        # NMO_3
        if counts["satellite"] == 0:
            errors.append(f"NMO_3: mul_9 a={a} must have satellite>0; got {counts['satellite']}")
        if counts["singularity"] == 0:
            errors.append(f"NMO_3: mul_9 a={a} must have singularity>0; got {counts['singularity']}")

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
        return errors + [f"F: FAIL fixture {expected!r} cannot evaluate: {schema_errs}"]

    a = fix["a_value"]
    counts = _enumerate_routes(a)

    if expected == "WRONG_SATELLITE_FOR_COPRIME":
        if fix.get("magic_number_class") != "coprime_to_3":
            errors.append("F: WRONG_SATELLITE_FOR_COPRIME must have magic_number_class=coprime_to_3")
        if fix["expected_satellite"] == 0:
            errors.append("F: WRONG_SATELLITE_FOR_COPRIME must claim expected_satellite>0")
        actual_sat = counts["satellite"]
        if actual_sat != 0:
            errors.append(f"F: coprime_to_3 a={a} actually has satellite={actual_sat} (cert claim broken)")
        if fix["expected_satellite"] == actual_sat:
            errors.append(f"F: WRONG_SATELLITE_FOR_COPRIME does not actually fail (both=0)")

    elif expected == "WRONG_COUNTS_FOR_MUL9":
        if fix.get("magic_number_class") != "mul_9":
            errors.append("F: WRONG_COUNTS_FOR_MUL9 must have magic_number_class=mul_9")
        actual_sat = counts["satellite"]
        actual_sing = counts["singularity"]
        actual_cos = counts["cosmos"]
        actual_total = counts["total"]
        # At least one count must differ from actual
        claimed_matches = (
            fix["expected_satellite"] == actual_sat
            and fix["expected_singularity"] == actual_sing
            and fix["expected_cosmos"] == actual_cos
            and fix["expected_total_routes"] == actual_total
        )
        if claimed_matches:
            errors.append(
                f"F: WRONG_COUNTS_FOR_MUL9 fixture claims correct counts for a={a} "
                f"(all match actual); does not actually fail"
            )
        if actual_sat == 0:
            errors.append(f"F: mul_9 a={a} should have actual satellite>0; got {actual_sat}")
        if actual_sing == 0:
            errors.append(f"F: mul_9 a={a} should have actual singularity>0; got {actual_sing}")

    elif expected == "SINGULARITY_FOR_COPRIME":
        if fix.get("magic_number_class") != "coprime_to_3":
            errors.append("F: SINGULARITY_FOR_COPRIME must have magic_number_class=coprime_to_3")
        if fix["expected_singularity"] == 0:
            errors.append("F: SINGULARITY_FOR_COPRIME must claim expected_singularity>0")
        actual_sing = counts["singularity"]
        if actual_sing != 0:
            errors.append(f"F: coprime_to_3 a={a} actually has singularity={actual_sing}")
        if fix["expected_singularity"] == actual_sing:
            errors.append("F: SINGULARITY_FOR_COPRIME does not actually fail (both=0)")

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
        errors.append("NMO_1: fixtures/ directory missing")
        return False, errors

    fix_files = sorted(fixtures_dir.glob("*.json"))
    if not fix_files:
        errors.append("NMO_1: no fixture files found")
        return False, errors

    pass_count = fail_count = 0
    for fix_path in fix_files:
        try:
            fix = _load_json(fix_path)
        except Exception as exc:
            errors.append(f"NMO_1: could not load {fix_path.name}: {exc}")
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
        errors.append("NMO_1: no PASS fixtures found")
    if fail_count == 0:
        errors.append("F: no FAIL fixtures found")

    return len(errors) == 0, errors


def self_test() -> list[str]:
    """Verify orbit counts for all canonical magic numbers."""
    expected = [
        (2, "no_routes", 0, 0, 0, 0),
        (8, "coprime_to_3", 3, 3, 0, 0),
        (20, "coprime_to_3", 9, 9, 0, 0),
        (28, "coprime_to_3", 13, 13, 0, 0),
        (50, "coprime_to_3", 24, 24, 0, 0),
        (82, "coprime_to_3", 40, 40, 0, 0),
        (126, "mul_9", 62, 42, 14, 6),
        (184, "coprime_to_3", 91, 91, 0, 0),
    ]
    errors = []
    for a, cls, n, cos, sat, sg in expected:
        actual_cls = _magic_number_class(a)
        if actual_cls != cls:
            errors.append(f"a={a}: expected class {cls!r}, got {actual_cls!r}")
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
        description="QA Nuclear Magic Orbit Cert validator [280]"
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
        print(f"PASS [{CERT_SLUG}]: NMO_1/NMO_2/NMO_3/NMO_4/SRC/F all ok")
        return 0
    print(f"FAIL [{CERT_SLUG}]: {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
