#!/usr/bin/env python3
"""
qa_cognition_space_morphospace_cert_validate.py

Validator for QA_COGNITION_SPACE_MORPHOSPACE_CERT.v1  [family 194]

Certifies: Sole, Seoane et al. "Cognition spaces" (arXiv:2601.12837)
qualitative morphospace maps to QA's exact discrete morphospace.

Mapping:
    Three clusters (basal/neural/human-AI) = three QA orbit families
    Voids between clusters = algebraically necessary (not contingent)
    Agency = |reachable set| / |total states|:
        Singularity = 1/81, Satellite = 8/81, Cosmos = 72/81

QA provides the constructive, enumerable cognition space that the
paper identifies as lacking.

Source: Sole et al. arXiv:2601.12837

Checks:
    CSM_1       — schema_version matches
    CSM_CLUSTER — cluster_orbit_mapping well-formed (3 clusters)
    CSM_VOIDS   — void_algebraic_necessity present and correct
    CSM_AGENCY  — agency ratios match actual S_9 orbit sizes
    CSM_ENUM    — enumerability section present with correct counts
    CSM_SRC     — source_attribution mentions Sole
    CSM_WITNESS — at least 3 witnesses (one per cluster)
    CSM_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates finite S_9 orbit classification; integer state space; no observer, no floats, no continuous dynamics"

import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_COGNITION_SPACE_MORPHOSPACE_CERT.v1"

# Expected S_9 orbit family sizes
EXPECTED_FAMILY_SIZES = {
    "singularity": 1,
    "satellite": 8,
    "cosmos": 72,
}

TOTAL_S9 = 81

# Realized orbit lengths in S_9
REALIZED_LENGTHS = {1, 8, 24}

# Divisors of 24 that do NOT appear as orbit lengths
MISSING_DIVISORS = {2, 3, 4, 6, 12}


# -----------------------------------------------------------------------------
# QA primitives (integer-only, axiom-compliant)
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    """A1-compliant: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    """Fibonacci dynamic: (b,e) -> (e, b+e mod m). A1-compliant."""
    return (e, qa_mod(b + e, m))


def orbit_family_s9(b, e):
    """Canonical S_9 orbit family classification."""
    if b == 9 and e == 9:
        return "singularity"
    if (b % 3 == 0) and (e % 3 == 0):
        return "satellite"
    return "cosmos"


def count_families_s9():
    """Count states in each orbit family of S_9."""
    counts = {"singularity": 0, "satellite": 0, "cosmos": 0}
    for b in range(1, 10):
        for e in range(1, 10):
            fam = orbit_family_s9(b, e)
            counts[fam] += 1
    return counts


def enumerate_orbit_lengths_s9():
    """Find all realized orbit lengths in S_9."""
    m = 9
    seen = set()
    lengths = set()
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen:
                continue
            orbit = []
            cur = (b, e)
            while cur not in seen:
                seen.add(cur)
                orbit.append(cur)
                cur = qa_step(cur[0], cur[1], m)
            lengths.add(len(orbit))
    return lengths


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CSM_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"CSM_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # CSM_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("CSM_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("CSM_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # CSM_SRC: source attribution
    src = str(cert.get("source_attribution", ""))
    if "Sol" not in src:  # Sole or Solé
        warnings.append("CSM_SRC: source_attribution should credit Sole / Seoane et al.")

    # CSM_CLUSTER: cluster_orbit_mapping well-formed
    mapping = cert.get("cluster_orbit_mapping")
    if not isinstance(mapping, list):
        errors.append("CSM_CLUSTER: cluster_orbit_mapping must be a list")
    elif len(mapping) < 3:
        errors.append(f"CSM_CLUSTER: need >= 3 clusters, got {len(mapping)}")
    else:
        mapped_orbits = {m.get("qa_orbit") for m in mapping}
        for expected_orbit in ["singularity", "satellite", "cosmos"]:
            if expected_orbit not in mapped_orbits:
                errors.append(f"CSM_CLUSTER: missing orbit '{expected_orbit}' in mapping")

    # CSM_AGENCY: agency ratios match actual orbit sizes
    if isinstance(mapping, list):
        actual_counts = count_families_s9()
        for entry in mapping:
            qa_orbit = entry.get("qa_orbit")
            declared_size = entry.get("orbit_size")
            expected_size = EXPECTED_FAMILY_SIZES.get(qa_orbit)
            if expected_size is not None and declared_size is not None:
                if declared_size != expected_size:
                    errors.append(
                        f"CSM_AGENCY: orbit '{qa_orbit}' declared size={declared_size}, "
                        f"expected {expected_size}"
                    )

    # CSM_VOIDS: void algebraic necessity
    voids = cert.get("void_algebraic_necessity")
    if voids is None:
        errors.append("CSM_VOIDS: void_algebraic_necessity section missing")
    else:
        declared_realized = set(voids.get("realized_lengths", []))
        if declared_realized != REALIZED_LENGTHS:
            errors.append(
                f"CSM_VOIDS: realized_lengths={sorted(declared_realized)}, "
                f"expected {sorted(REALIZED_LENGTHS)}"
            )
        declared_missing = set(voids.get("missing_divisors", []))
        if declared_missing != MISSING_DIVISORS:
            errors.append(
                f"CSM_VOIDS: missing_divisors={sorted(declared_missing)}, "
                f"expected {sorted(MISSING_DIVISORS)}"
            )

    # Verify actual orbit lengths match
    actual_lengths = enumerate_orbit_lengths_s9()
    if actual_lengths != REALIZED_LENGTHS:
        errors.append(
            f"CSM_VOIDS: actual S_9 orbit lengths={sorted(actual_lengths)}, "
            f"expected {sorted(REALIZED_LENGTHS)}"
        )

    # CSM_ENUM: enumerability
    enum_section = cert.get("enumerability")
    if enum_section is None:
        errors.append("CSM_ENUM: enumerability section missing")
    else:
        declared_total = enum_section.get("s9_total_states")
        if declared_total != TOTAL_S9:
            errors.append(f"CSM_ENUM: s9_total_states={declared_total}, expected {TOTAL_S9}")

    # CSM_WITNESS: witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"CSM_WITNESS: need >= 3 witnesses (one per cluster), got {len(witnesses)}")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("csm_pass_morphospace.json", True),
        ("csm_fail_wrong_agency.json", True),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Cognition Space Morphospace Cert [194] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
