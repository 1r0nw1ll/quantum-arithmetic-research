#!/usr/bin/env python3
"""QA Orbit Pisano 5-Factor Boundary Cert validator.

Family [277]. Primary source for the Pisano-period framing:
Wall, D. D. (1960). Fibonacci series modulo m. American Mathematical
Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

Certifies the bounded falsifiable claim:

    For m = 15k with k in K_verified = {1..12, 15, 20} (14 verified values),
    the divisor shortcut orbit_family_divisor_shortcut(b, e, m) under-counts
    the canonical period-based orbit_family(b, e, m) satellite class by
    exactly 32 pairs and never over-claims; the 32 missed pairs partition
    by (gcd(b, m), gcd(e, m)) into:

        (k,  3k):  8 pairs
        (k,  k ): 16 pairs
        (3k, k ):  8 pairs

The validator recomputes both classifiers on each fixture's modulus,
counts undercount and overclaim, computes the gcd-signature distribution,
and compares against the fixture's expected_* fields.

Checks:
  PISANO_1 — undercount matches expected
  PISANO_2 — overclaim count matches expected (always 0 for PASS)
  PISANO_3 — gcd-signature decomposition matches expected (PASS only)
  SRC      — mapping_protocol_ref.json present and well-formed
  F        — every FAIL fixture declares expected_fail_type and actually
             fails the structural check it declares.
"""

QA_COMPLIANCE = "cert_validator - integer arithmetic on (b, e, m); recomputes orbit_family and orbit_family_divisor_shortcut from qa_orbit_rules; no float feedback into QA layer"

import argparse
import json
import sys
from collections import Counter
from math import gcd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family, orbit_family_divisor_shortcut  # noqa: E402

SCHEMA_VERSION = "QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT.v1"
CERT_SLUG = "qa_orbit_pisano_5_factor_boundary_cert_v1"
CANDIDATE_FAMILY_ID = 277


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema(fix: dict) -> list[str]:
    required = [
        "schema_version", "fixture_kind", "modulus",
        "expected_shortcut_undercount", "expected_overclaims",
    ]
    errors: list[str] = []
    for field in required:
        if field not in fix:
            errors.append(f"PISANO_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"PISANO_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("PISANO_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _compute_misses(m: int) -> tuple[list[tuple[int, int]], int]:
    """Return (missed_pairs, overclaim_count) by recomputing both classifiers."""
    misses: list[tuple[int, int]] = []
    overclaims = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            canonical = orbit_family(b, e, m)
            shortcut = orbit_family_divisor_shortcut(b, e, m)
            if canonical == "satellite" and shortcut != "satellite":
                misses.append((b, e))
            if shortcut == "satellite" and canonical != "satellite":
                overclaims += 1
    return misses, overclaims


def _check_pass_fixture(fix: dict) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_schema(fix))
    if errors:
        return errors

    m = fix["modulus"]
    expected_undercount = fix["expected_shortcut_undercount"]
    expected_overclaims = fix["expected_overclaims"]

    misses, overclaims = _compute_misses(m)

    if len(misses) != expected_undercount:
        errors.append(
            f"PISANO_1: m={m} undercount={len(misses)}, expected={expected_undercount}"
        )
    if overclaims != expected_overclaims:
        errors.append(
            f"PISANO_2: m={m} overclaims={overclaims}, expected={expected_overclaims}"
        )

    expected_sig = fix.get("expected_signatures")
    if expected_sig is not None:
        observed = Counter((gcd(b, m), gcd(e, m)) for b, e in misses)
        observed_sig = {f"{a},{b}": n for (a, b), n in observed.items()}
        if observed_sig != expected_sig:
            errors.append(
                f"PISANO_3: m={m} signatures {observed_sig} != expected {expected_sig}"
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
        # Other expected_fail_type relies on the validator running, so the
        # schema must be sound first.
        return errors + [
            f"F: FAIL fixture {expected!r} cannot be evaluated due to schema errors: {schema_errs}"
        ]

    m = fix["modulus"]
    expected_undercount = fix["expected_shortcut_undercount"]
    expected_overclaims = fix["expected_overclaims"]
    expected_sig = fix.get("expected_signatures")

    misses, overclaims = _compute_misses(m)
    observed = Counter((gcd(b, m), gcd(e, m)) for b, e in misses)
    observed_sig = {f"{a},{b}": n for (a, b), n in observed.items()}

    if expected == "WRONG_UNDERCOUNT":
        if len(misses) == expected_undercount:
            errors.append(
                f"F: WRONG_UNDERCOUNT fixture's expected_shortcut_undercount "
                f"matches reality (m={m}, both = {len(misses)})"
            )
    elif expected == "OVERCLAIM_DECLARED":
        if overclaims == expected_overclaims:
            errors.append(
                f"F: OVERCLAIM_DECLARED fixture's expected_overclaims matches reality "
                f"(m={m}, both = {overclaims})"
            )
    elif expected == "WRONG_SIGNATURES":
        if expected_sig is None:
            errors.append("F: WRONG_SIGNATURES fixture must declare expected_signatures")
        elif observed_sig == expected_sig:
            errors.append(
                f"F: WRONG_SIGNATURES fixture's expected_signatures matches reality "
                f"(m={m}, both = {expected_sig})"
            )
    elif expected == "TREATS_SHORTCUT_AS_CANONICAL":
        # Fixture asserts (via expected_undercount = 0 and shortcut count =
        # canonical count) that the shortcut IS the canonical. For 5|m with
        # k in K_verified, the canonical has 32 more satellites than shortcut.
        if len(misses) == 0:
            errors.append(
                f"F: TREATS_SHORTCUT_AS_CANONICAL fixture must target an m where "
                f"shortcut diverges from canonical; m={m} has 0 missed."
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
        description="QA Orbit Pisano 5-Factor Boundary Cert validator [277]"
    )
    parser.add_argument("--demo", action="store_true", help="print config and exit")
    parser.add_argument("--self-test", action="store_true",
                        help="emit JSON {ok, errors, ...} for the meta-validator")
    args = parser.parse_args(argv)

    cert_dir = Path(__file__).resolve().parent
    fix_dir = cert_dir / "fixtures"

    if args.demo:
        print(f"family_id={CANDIDATE_FAMILY_ID} slug={CERT_SLUG} "
              f"schema={SCHEMA_VERSION}")
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
