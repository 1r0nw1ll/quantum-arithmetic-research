#!/usr/bin/env python3
"""QA Fibonacci-Orbit Index Correspondence Cert validator.

Family [282]. Primary source:
  Wall, D. D. (1960). Fibonacci series modulo m.
  American Mathematical Monthly, 67(6), 525-532.
  DOI: 10.1080/00029890.1960.11989541.
Mechanism: cert [281] (Pisano-Orbit Correspondence) +
           cert [279] (Orbit Access Theorem).

Certifies the claim:

    For n >= 1, the mod-9 orbit class of the Fibonacci number F_n,
    when used as an a-value in mod-9 route enumeration (b+2e=a),
    is determined solely by n mod 12:

      mul_9        iff 12 | n   (equivalently, 9 | F_n)
      mul_3_not_9  iff n == 4 or 8 mod 12  (equivalently, 3|F_n, 9 not divides F_n)
      coprime_to_3 otherwise    (equivalently, 3 not divides F_n)

    Wall (1960): rank of apparition alpha(3)=4, so 3|F_n iff 4|n.
                 rank of apparition alpha(9)=12, so 9|F_n iff 12|n.

    Corollary of cert [281]: Pisano period of 9 is 24; within each period,
    mul_9 appears at positions == 0 mod 12, mul_3_not_9 at == 4 or 8 mod 12.

    Verified exhaustively for n=1..48 (two full Pisano periods of 9).

Checks:
  FOI_1 -- coprime_to_3: 3 not divides F_n.
  FOI_2 -- mul_3_not_9: 3|F_n but 9 not divides F_n.
  FOI_3 -- mul_9: 9|F_n.
  FOI_4 -- declared class matches n mod 12 rule AND actual F_n divisibility.
  SRC   -- mapping_protocol_ref.json present and well-formed.
  F     -- every FAIL fixture declares expected_fail_type and fires.
"""

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic; F_n via integer recurrence; "
    "orbit class from F_n mod 9 and n mod 12; no float feedback into QA layer"
)

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_FIBONACCI_ORBIT_INDEX_CERT.v1"
CERT_SLUG = "qa_fibonacci_orbit_index_cert_v1"
CANDIDATE_FAMILY_ID = 282
MODULUS = 9

PISANO_PERIOD_9 = 24
VERIFICATION_RANGE = 2 * PISANO_PERIOD_9  # 48 — two full periods


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fibonacci(n: int) -> int:
    """Compute F_n (F_1=1, F_2=1, F_3=2, ...) by integer recurrence."""
    if n <= 0:
        raise ValueError(f"Fibonacci index must be >= 1; got {n}")
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def _fibonacci_class_from_index(n: int) -> str:
    """Return orbit class of F_n determined solely by n mod 12 (Wall 1960)."""
    r = n % 12
    if r == 0:
        return "mul_9"
    if r in (4, 8):
        return "mul_3_not_9"
    return "coprime_to_3"


def _fibonacci_class_from_value(f: int) -> str:
    """Return orbit class of integer f by 3-adic divisibility."""
    if f % MODULUS == 0:
        return "mul_9"
    if f % 3 == 0:
        return "mul_3_not_9"
    return "coprime_to_3"


def _validate_schema(fix: dict) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_version", "fixture_kind",
        "fibonacci_index", "fibonacci_value", "expected_class",
    ]
    for field in required:
        if field not in fix:
            errors.append(f"FOI_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"FOI_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("FOI_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _check_pass_fixture(fix: dict) -> list[str]:
    errors = _validate_schema(fix)
    if errors:
        return errors

    n = fix["fibonacci_index"]
    declared_value = fix["fibonacci_value"]
    declared_class = fix["expected_class"]

    # Verify declared F_n value
    actual_value = _fibonacci(n)
    if declared_value != actual_value:
        errors.append(
            f"FOI_4: n={n} declared fibonacci_value={declared_value} "
            f"but computed F_{n}={actual_value}"
        )
        return errors

    # Verify class from index rule
    rule_class = _fibonacci_class_from_index(n)
    # Verify class from value divisibility
    value_class = _fibonacci_class_from_value(actual_value)

    if rule_class != value_class:
        errors.append(
            f"FOI_4: n={n} index-rule class={rule_class!r} != "
            f"value class={value_class!r} (Wall 1960 consistency broken)"
        )
        return errors

    if declared_class != rule_class:
        errors.append(
            f"FOI_4: n={n} declared class={declared_class!r} "
            f"but rule+value both give {rule_class!r}"
        )
        return errors

    # Gate-specific checks
    if declared_class == "coprime_to_3":
        if actual_value % 3 == 0:
            errors.append(
                f"FOI_1: n={n} declared coprime_to_3 but F_{n}={actual_value} "
                f"is divisible by 3"
            )
    elif declared_class == "mul_3_not_9":
        if actual_value % 3 != 0:
            errors.append(
                f"FOI_2: n={n} declared mul_3_not_9 but 3 does not divide F_{n}={actual_value}"
            )
        if actual_value % MODULUS == 0:
            errors.append(
                f"FOI_2: n={n} declared mul_3_not_9 but 9 divides F_{n}={actual_value}"
            )
    elif declared_class == "mul_9":
        if actual_value % MODULUS != 0:
            errors.append(
                f"FOI_3: n={n} declared mul_9 but 9 does not divide F_{n}={actual_value}"
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

    n = fix["fibonacci_index"]
    declared_class = fix["expected_class"]
    actual_value = _fibonacci(n)
    actual_class = _fibonacci_class_from_value(actual_value)

    if expected in ("WRONG_CLASS_COPRIME", "WRONG_CLASS_MUL3", "WRONG_CLASS_MUL9"):
        if declared_class == actual_class:
            errors.append(
                f"F: {expected}: declared class={declared_class!r} matches actual "
                f"class={actual_class!r} for n={n}; does not fail"
            )
        # Verify the expected_fail_type matches the declared (wrong) class
        if expected == "WRONG_CLASS_COPRIME" and declared_class != "coprime_to_3":
            errors.append(
                f"F: WRONG_CLASS_COPRIME: declared class should be coprime_to_3, got {declared_class!r}"
            )
        if expected == "WRONG_CLASS_MUL3" and declared_class != "mul_3_not_9":
            errors.append(
                f"F: WRONG_CLASS_MUL3: declared class should be mul_3_not_9, got {declared_class!r}"
            )
        if expected == "WRONG_CLASS_MUL9" and declared_class != "mul_9":
            errors.append(
                f"F: WRONG_CLASS_MUL9: declared class should be mul_9, got {declared_class!r}"
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
        errors.append("FOI_1: fixtures/ directory missing")
        return False, errors

    fix_files = sorted(fixtures_dir.glob("*.json"))
    if not fix_files:
        errors.append("FOI_1: no fixture files found")
        return False, errors

    pass_count = fail_count = 0
    for fix_path in fix_files:
        try:
            fix = _load_json(fix_path)
        except Exception as exc:
            errors.append(f"FOI_1: could not load {fix_path.name}: {exc}")
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
        errors.append("FOI_1: no PASS fixtures found")
    if fail_count == 0:
        errors.append("F: no FAIL fixtures found")

    return len(errors) == 0, errors


def self_test() -> list[str]:
    """Verify the index-class rule against actual F_n divisibility for n=1..48."""
    errors = []
    for n in range(1, VERIFICATION_RANGE + 1):
        f = _fibonacci(n)
        rule_class = _fibonacci_class_from_index(n)
        value_class = _fibonacci_class_from_value(f)
        if rule_class != value_class:
            errors.append(
                f"n={n}: F_{n}={f}: rule_class={rule_class!r} != value_class={value_class!r}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QA Fibonacci-Orbit Index Correspondence Cert validator [282]"
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
        print(f"PASS [{CERT_SLUG}]: FOI_1/FOI_2/FOI_3/FOI_4/SRC/F all ok")
        return 0
    print(f"FAIL [{CERT_SLUG}]: {len(errors)} error(s)")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
