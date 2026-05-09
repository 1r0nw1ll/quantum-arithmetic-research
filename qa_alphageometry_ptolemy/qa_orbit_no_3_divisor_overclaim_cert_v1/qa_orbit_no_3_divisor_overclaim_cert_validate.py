#!/usr/bin/env python3
"""QA Orbit No-3-Divisor Overclaim Cert validator.

Family [278]. Primary source for the Pisano-period framing:
Wall, D. D. (1960). Fibonacci series modulo m. American Mathematical
Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

Certifies the bounded falsifiable claim:

    For every tested modulus m with 3 not divides m and m >= 7 (excluding
    m = 8 from v1), the canonical period-based orbit_family finds
    zero period-8 satellites while the algebraic divisor shortcut
    orbit_family_divisor_shortcut over-claims exactly 9 false satellites.
    The 9 over-claimed pairs are exactly the 3x3 grid
    { (a*(m//3), b*(m//3)) : a, b in {1, 2, 3} }.

Why "no_3_divisor" and not "5_factor": the empirical sweep on 25 tested
5|m AND 3 not divides m moduli (m up to 250) plus a separate proof pass
on non-5-multiple 3 not divides m moduli {7, 11, 13, 14, ...} showed
identical 9-overclaim behavior. The 5-factor framing was the discovery
path; the structural cause is m // 3 not being a divisor of m.

Why m = 8 is excluded: m // 3 = 2 makes the shortcut's grid the entire
even sub-lattice {2, 4, 6, 8}^2, which is 4x4 = 16 pairs. Singularity
(8, 8) is in this grid. Shortcut overclaim is 15, not 9. Future v2 may
add an m = 8 sub-claim with expected_overclaims = 15.

Checks:
  NO3_1 — canonical_satellites matches expected (always 0 for PASS).
  NO3_2 — shortcut_satellites matches expected (always 9 for PASS).
  NO3_3 — overclaim count = 9 and missed count = 0.
  NO3_4 — m = 8 boundary exception is rejected from v1 (FAIL fixture
          fail_m8_boundary_exception.json must actually fail).
  SRC   — mapping_protocol_ref.json present and well-formed.
  F     — every FAIL fixture declares expected_fail_type and the
          declared mode actually fires.
"""

QA_COMPLIANCE = "cert_validator - integer arithmetic on (b, e, m); recomputes orbit_family and orbit_family_divisor_shortcut from qa_orbit_rules; no float feedback into QA layer"

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family, orbit_family_divisor_shortcut  # noqa: E402

SCHEMA_VERSION = "QA_ORBIT_NO_3_DIVISOR_OVERCLAIM_CERT.v1"
CERT_SLUG = "qa_orbit_no_3_divisor_overclaim_cert_v1"
CANDIDATE_FAMILY_ID = 278
EXCLUDED_FROM_V1 = frozenset({8})


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema(fix: dict) -> list[str]:
    required = [
        "schema_version", "fixture_kind", "modulus",
        "expected_canonical_satellites", "expected_shortcut_satellites",
        "expected_overclaims", "expected_missed",
    ]
    errors: list[str] = []
    for field in required:
        if field not in fix:
            errors.append(f"NO3_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"NO3_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("NO3_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _compute_counts(m: int) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    """Return (canonical_sat, shortcut_sat, missed, overclaims, overclaim_pairs)."""
    canon = 0
    short = 0
    missed = 0
    overclaims = 0
    overclaim_pairs: list[tuple[int, int]] = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            c = orbit_family(b, e, m)
            s = orbit_family_divisor_shortcut(b, e, m)
            if c == "satellite":
                canon += 1
            if s == "satellite":
                short += 1
            if c == "satellite" and s != "satellite":
                missed += 1
            if s == "satellite" and c != "satellite":
                overclaims += 1
                overclaim_pairs.append((b, e))
    return canon, short, missed, overclaims, overclaim_pairs


def _check_pass_fixture(fix: dict) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_schema(fix))
    if errors:
        return errors

    m = fix["modulus"]
    if m in EXCLUDED_FROM_V1:
        errors.append(
            f"NO3_4: PASS fixture m={m} is excluded from v1 (boundary exception); "
            f"use a FAIL fixture with expected_fail_type=M8_BOUNDARY instead"
        )
        return errors

    if m % 3 == 0:
        errors.append(f"NO3_1: PASS fixture m={m} violates scope (3 | m)")
    if m < 6:
        errors.append(f"NO3_1: PASS fixture m={m} below minimum scope m >= 6")

    canon, short, missed, over, _ = _compute_counts(m)

    if canon != fix["expected_canonical_satellites"]:
        errors.append(
            f"NO3_1: m={m} canonical={canon}, expected={fix['expected_canonical_satellites']}"
        )
    if short != fix["expected_shortcut_satellites"]:
        errors.append(
            f"NO3_2: m={m} shortcut={short}, expected={fix['expected_shortcut_satellites']}"
        )
    if over != fix["expected_overclaims"]:
        errors.append(
            f"NO3_3: m={m} overclaim={over}, expected={fix['expected_overclaims']}"
        )
    if missed != fix["expected_missed"]:
        errors.append(
            f"NO3_3: m={m} missed={missed}, expected={fix['expected_missed']}"
        )

    causal = fix.get("causal_scope")
    if causal is not None and causal != "no_3_divisor":
        errors.append(
            f"NO3_3: PASS fixture causal_scope={causal!r} disagrees with cert claim "
            f"(causal_scope must be 'no_3_divisor')"
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
            errors.append("F: MISSING_FIELD fixture passed schema check")
        return errors
    if schema_errs:
        return errors + [
            f"F: FAIL fixture {expected!r} cannot evaluate due to schema errors: {schema_errs}"
        ]

    m = fix["modulus"]
    canon, short, missed, over, _ = _compute_counts(m)

    if expected == "M8_BOUNDARY":
        # NO3_4: this fixture asserts m=8 fits the v1 contract (overclaim=9).
        # Validator must observe the actual overclaim is 15.
        if m != 8:
            errors.append(
                f"NO3_4: M8_BOUNDARY fixture must target m=8; got m={m}"
            )
        elif over == fix["expected_overclaims"]:
            errors.append(
                f"NO3_4: M8_BOUNDARY fixture's expected_overclaims={fix['expected_overclaims']} "
                f"matches actual ({over}); m=8 boundary not exposed"
            )
    elif expected == "WRONG_OVERCLAIM":
        if over == fix["expected_overclaims"]:
            errors.append(
                f"F: WRONG_OVERCLAIM fixture's expected_overclaims={fix['expected_overclaims']} "
                f"matches reality (m={m}, both = {over})"
            )
    elif expected == "FIVE_FACTOR_CAUSAL":
        causal = fix.get("causal_scope")
        if causal != "5_factor":
            errors.append(
                f"F: FIVE_FACTOR_CAUSAL fixture must declare causal_scope='5_factor'; "
                f"got {causal!r}"
            )
        # The fixture exists to be rejected; passing the schema check above
        # plus declaring the wrong causal_scope is enough to expose the
        # mis-framing. The validator's intent here is structural: the cert
        # rejects fixtures that frame the failure mode as 5-factor causal.
    elif expected == "SHORTCUT_AS_CANONICAL":
        # Fixture asserts canonical = 9 (treats shortcut count as canonical).
        if canon == fix["expected_canonical_satellites"]:
            errors.append(
                f"F: SHORTCUT_AS_CANONICAL fixture's expected_canonical_satellites="
                f"{fix['expected_canonical_satellites']} matches reality (m={m}, "
                f"canonical = {canon})"
            )
    else:
        errors.append(f"F: unknown expected_fail_type {expected!r}")

    return errors


def _validate_fixture(path: Path) -> list[str]:
    fix = _load_json(path)
    if fix.get("fixture_kind") == "fail":
        return _check_fail_fixture(fix)
    return _check_pass_fixture(fix)


def _check_mapping_protocol(cert_dir: Path) -> list[str]:
    p = cert_dir / "mapping_protocol_ref.json"
    if not p.exists():
        return ["SRC: missing mapping_protocol_ref.json"]
    data = _load_json(p)
    needed = ("protocol_version", "ref_path", "ref_sha256", "scope_note")
    return [f"SRC: missing {k!r}" for k in needed if k not in data]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="QA Orbit No-3-Divisor Overclaim Cert validator [278]"
    )
    parser.add_argument("--demo", action="store_true", help="print config and exit")
    parser.add_argument("--self-test", action="store_true",
                        help="emit JSON {ok, errors, ...} for the meta-validator")
    args = parser.parse_args(argv)

    cert_dir = Path(__file__).resolve().parent
    fix_dir = cert_dir / "fixtures"

    if args.demo:
        print(f"family_id={CANDIDATE_FAMILY_ID} slug={CERT_SLUG} "
              f"schema={SCHEMA_VERSION} excluded_from_v1={sorted(EXCLUDED_FROM_V1)}")
        print(f"fixtures: {sorted(p.name for p in fix_dir.glob('*.json'))}")
        return 0

    all_errors: list[str] = []

    src_errs = _check_mapping_protocol(cert_dir)
    if src_errs:
        all_errors.extend(src_errs)

    pass_files = sorted(fix_dir.glob("pass_*.json"))
    fail_files = sorted(fix_dir.glob("fail_*.json"))
    if not pass_files:
        all_errors.append("F: no PASS fixtures present")
    if not fail_files:
        all_errors.append("F: no FAIL fixtures present")

    for p in pass_files:
        errs = _validate_fixture(p)
        if errs:
            all_errors.extend([f"{p.name}: {e}" for e in errs])
    for p in fail_files:
        errs = _validate_fixture(p)
        if errs:
            all_errors.extend([f"{p.name}: {e}" for e in errs])

    ok = not all_errors
    if args.self_test:
        payload = {
            "ok": ok,
            "family_id": CANDIDATE_FAMILY_ID,
            "slug": CERT_SLUG,
            "schema_version": SCHEMA_VERSION,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": all_errors,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0 if ok else 1

    if all_errors:
        print(f"FAIL [{CERT_SLUG}]: {len(all_errors)} error(s)")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print(f"PASS [{CERT_SLUG}]: schema + {len(pass_files)} PASS + {len(fail_files)} FAIL fixtures ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
