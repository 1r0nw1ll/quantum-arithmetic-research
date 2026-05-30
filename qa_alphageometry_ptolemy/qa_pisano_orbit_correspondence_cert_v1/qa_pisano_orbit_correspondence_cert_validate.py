#!/usr/bin/env python3
"""QA Pisano-Orbit Correspondence Cert validator.

Family [281]. Primary source:
  Wall, D. D. (1960). Fibonacci series modulo m.
  American Mathematical Monthly, 67(6), 525-532.
  DOI: 10.1080/00029890.1960.11989541.
Mechanism: QA Orbit Access Theorem cert [279].

Certifies the claim:

    For m=9, under qa_step(b,e) = (e, ((b+e-1)%m)+1), the period of
    every (b,e) in {1,...,9}^2 is exactly:
      Singularity  -> period  1
      Satellite    -> period  8  = pi(3)  (Pisano period of 3)
      Cosmos       -> period 24  = pi(9)  (Pisano period of 9)

    Pisano period pi(k) defined as the smallest i>0 with F_i=0 and
    F_{i+1}=1 mod k (Wall 1960, Section 3). Verified: pi(3)=8, pi(9)=24.

    Cert does NOT claim the correspondence extends to arbitrary m.
    Cert does NOT certify the period structure of mod-24 QA.
    Cert does NOT prove the identity algebraically (empirical verification).

Checks:
  POC_1 -- singularity period = 1.
  POC_2 -- satellite period = 8 = pi(3).
  POC_3 -- cosmos period = 24 = pi(9).
  POC_4 -- Pisano period pi(3) = 8 (Fibonacci recurrence).
  POC_5 -- Pisano period pi(9) = 24 (Fibonacci recurrence).
  SRC   -- mapping_protocol_ref.json present and well-formed.
  F     -- every FAIL fixture declares expected_fail_type and fires.
"""

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic on (b, e); A1-reduction via "
    "((v-1)%9)+1; orbit_family and orbit_period on (b, e, 9) from "
    "qa_orbit_rules; Pisano period via Fibonacci recurrence mod k; "
    "no float feedback into QA layer"
)

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family, orbit_period  # noqa: E402

SCHEMA_VERSION = "QA_PISANO_ORBIT_CORRESPONDENCE_CERT.v1"
CERT_SLUG = "qa_pisano_orbit_correspondence_cert_v1"
CANDIDATE_FAMILY_ID = 281
MODULUS = 9

EXPECTED_SINGULARITY_PERIOD = 1
EXPECTED_SATELLITE_PERIOD = 8
EXPECTED_COSMOS_PERIOD = 24
EXPECTED_PISANO_3 = 8
EXPECTED_PISANO_9 = 24


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pisano_period(k: int) -> int:
    """Compute Pisano period pi(k) via Fibonacci recurrence mod k."""
    if k == 1:
        return 1
    a, b = 0, 1
    for i in range(1, 6 * k + 1):
        a, b = b, (a + b) % k
        if a == 0 and b == 1:
            return i
    raise ValueError(f"Pisano period not found for k={k} within 6k steps")


def _validate_schema(fix: dict) -> list[str]:
    errors: list[str] = []
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"POC_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("POC_1: fixture_kind must be 'pass' or 'fail'")
    ftype = fix.get("fixture_type")
    if ftype not in ("orbit_period", "pisano_period"):
        errors.append(
            f"POC_1: fixture_type must be 'orbit_period' or 'pisano_period'; got {ftype!r}"
        )
    return errors


def _check_orbit_period_pass(fix: dict) -> list[str]:
    errors: list[str] = []
    for field in ("b_value", "e_value", "expected_period", "expected_family"):
        if field not in fix:
            errors.append(f"POC_1: orbit_period fixture missing field {field!r}")
    if errors:
        return errors

    b = fix["b_value"]
    e = fix["e_value"]
    expected_period = fix["expected_period"]
    expected_fam = fix["expected_family"]

    actual_fam = orbit_family(b, e, MODULUS)
    actual_period = orbit_period(b, e, MODULUS)

    if actual_fam != expected_fam:
        errors.append(
            f"POC_1: ({b},{e}) declared family={expected_fam!r} "
            f"but actual={actual_fam!r}"
        )
        return errors

    if actual_period != expected_period:
        errors.append(
            f"POC_1: ({b},{e}) declared period={expected_period} "
            f"but actual={actual_period}"
        )

    if actual_fam == "singularity" and actual_period != EXPECTED_SINGULARITY_PERIOD:
        errors.append(
            f"POC_1: singularity ({b},{e}) period={actual_period} != 1"
        )
    elif actual_fam == "satellite" and actual_period != EXPECTED_SATELLITE_PERIOD:
        errors.append(
            f"POC_2: satellite ({b},{e}) period={actual_period} != 8"
        )
    elif actual_fam == "cosmos" and actual_period != EXPECTED_COSMOS_PERIOD:
        errors.append(
            f"POC_3: cosmos ({b},{e}) period={actual_period} != 24"
        )

    return errors


def _check_pisano_period_pass(fix: dict) -> list[str]:
    errors: list[str] = []
    for field in ("pisano_k", "expected_pisano_period"):
        if field not in fix:
            errors.append(f"POC_4: pisano_period fixture missing field {field!r}")
    if errors:
        return errors

    k = fix["pisano_k"]
    expected = fix["expected_pisano_period"]
    actual = _pisano_period(k)

    if actual != expected:
        errors.append(f"POC_4: pi({k}) claimed={expected} but computed={actual}")

    if k == 3 and actual != EXPECTED_PISANO_3:
        errors.append(f"POC_4: pi(3)={actual} != {EXPECTED_PISANO_3}")
    if k == 9 and actual != EXPECTED_PISANO_9:
        errors.append(f"POC_5: pi(9)={actual} != {EXPECTED_PISANO_9}")

    return errors


def _check_pass_fixture(fix: dict) -> list[str]:
    errors = _validate_schema(fix)
    if errors:
        return errors
    ftype = fix["fixture_type"]
    if ftype == "orbit_period":
        return _check_orbit_period_pass(fix)
    return _check_pisano_period_pass(fix)


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

    ftype = fix.get("fixture_type")

    if expected == "WRONG_SATELLITE_PERIOD":
        if ftype != "orbit_period":
            errors.append("F: WRONG_SATELLITE_PERIOD must be orbit_period fixture")
            return errors
        b = fix.get("b_value")
        e = fix.get("e_value")
        if b is None or e is None:
            errors.append("F: WRONG_SATELLITE_PERIOD missing b_value/e_value")
            return errors
        actual_fam = orbit_family(b, e, MODULUS)
        if actual_fam != "satellite":
            errors.append(
                f"F: WRONG_SATELLITE_PERIOD: ({b},{e}) is {actual_fam!r}, not satellite"
            )
        actual_period = orbit_period(b, e, MODULUS)
        claimed = fix.get("expected_period")
        if claimed == actual_period:
            errors.append(
                f"F: WRONG_SATELLITE_PERIOD: claimed period={claimed} matches actual; "
                "does not fail"
            )

    elif expected == "WRONG_COSMOS_PERIOD":
        if ftype != "orbit_period":
            errors.append("F: WRONG_COSMOS_PERIOD must be orbit_period fixture")
            return errors
        b = fix.get("b_value")
        e = fix.get("e_value")
        if b is None or e is None:
            errors.append("F: WRONG_COSMOS_PERIOD missing b_value/e_value")
            return errors
        actual_fam = orbit_family(b, e, MODULUS)
        if actual_fam != "cosmos":
            errors.append(
                f"F: WRONG_COSMOS_PERIOD: ({b},{e}) is {actual_fam!r}, not cosmos"
            )
        actual_period = orbit_period(b, e, MODULUS)
        claimed = fix.get("expected_period")
        if claimed == actual_period:
            errors.append(
                f"F: WRONG_COSMOS_PERIOD: claimed period={claimed} matches actual; "
                "does not fail"
            )

    elif expected == "WRONG_PISANO_VALUE":
        if ftype != "pisano_period":
            errors.append("F: WRONG_PISANO_VALUE must be pisano_period fixture")
            return errors
        k = fix.get("pisano_k")
        if k is None:
            errors.append("F: WRONG_PISANO_VALUE missing pisano_k")
            return errors
        actual = _pisano_period(k)
        claimed = fix.get("expected_pisano_period")
        if claimed == actual:
            errors.append(
                f"F: WRONG_PISANO_VALUE: claimed pi({k})={claimed} matches actual; "
                "does not fail"
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
        errors.append("POC_1: fixtures/ directory missing")
        return False, errors

    fix_files = sorted(fixtures_dir.glob("*.json"))
    if not fix_files:
        errors.append("POC_1: no fixture files found")
        return False, errors

    pass_count = fail_count = 0
    for fix_path in fix_files:
        try:
            fix = _load_json(fix_path)
        except Exception as exc:
            errors.append(f"POC_1: could not load {fix_path.name}: {exc}")
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
        errors.append("POC_1: no PASS fixtures found")
    if fail_count == 0:
        errors.append("F: no FAIL fixtures found")

    return len(errors) == 0, errors


def self_test() -> list[str]:
    """Verify orbit periods for all 81 pairs and Pisano periods pi(3), pi(9)."""
    errors = []
    for b in range(1, MODULUS + 1):
        for e in range(1, MODULUS + 1):
            fam = orbit_family(b, e, MODULUS)
            per = orbit_period(b, e, MODULUS)
            if fam == "singularity":
                if per != EXPECTED_SINGULARITY_PERIOD:
                    errors.append(
                        f"({b},{e}) singularity: period={per} != {EXPECTED_SINGULARITY_PERIOD}"
                    )
            elif fam == "satellite":
                if per != EXPECTED_SATELLITE_PERIOD:
                    errors.append(
                        f"({b},{e}) satellite: period={per} != {EXPECTED_SATELLITE_PERIOD}"
                    )
            elif fam == "cosmos":
                if per != EXPECTED_COSMOS_PERIOD:
                    errors.append(
                        f"({b},{e}) cosmos: period={per} != {EXPECTED_COSMOS_PERIOD}"
                    )
            else:
                errors.append(f"({b},{e}) unknown family: {fam!r}")

    actual_p3 = _pisano_period(3)
    if actual_p3 != EXPECTED_PISANO_3:
        errors.append(f"pi(3)={actual_p3} != {EXPECTED_PISANO_3}")

    actual_p9 = _pisano_period(9)
    if actual_p9 != EXPECTED_PISANO_9:
        errors.append(f"pi(9)={actual_p9} != {EXPECTED_PISANO_9}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QA Pisano-Orbit Correspondence Cert validator [281]"
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
        print(f"PASS [{CERT_SLUG}]: POC_1/POC_2/POC_3/POC_4/POC_5/SRC/F all ok")
        return 0
    print(f"FAIL [{CERT_SLUG}]: {len(errors)} error(s)")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
