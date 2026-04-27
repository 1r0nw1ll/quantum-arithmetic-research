#!/usr/bin/env python3
"""
qa_levin_cognitive_lightcone_cert_validate.py

Validator for QA_LEVIN_COGNITIVE_LIGHTCONE_CERT.v1  [family 193]

Certifies: Michael Levin's Cognitive Light Cone (CLC) mapped to QA orbit
radius. CLC = spatiotemporal scale of largest goal an agent can pursue.

Mapping:
    Singularity = radius 0  (fixed point, no goals, CLC collapsed)
    Satellite   = radius 8  (8-cycle, local goals)
    Cosmos      = radius 24 (24-cycle, far-reaching goals)

Cancer = CLC shrinkage = Cosmos->Satellite orbit transition (L_2a demotion).
Tiered Reachability [191]: 26% L1-reachable = structural CLC ceiling.

Source: Levin & Resnik "Mind Everywhere" (Biological Theory 2026);
        Lyons/Pio-Lopez/Levin "Cancer to AI Alignment" (Preprints 2026).

Checks:
    CLC_1       — schema_version matches
    CLC_ORBIT   — clc_orbit_mapping well-formed (3 tiers)
    CLC_RADIUS  — orbit radii match actual S_9 cycle lengths {1,8,24}
    CLC_CANCER  — cancer_as_orbit_demotion present and structurally sound
    CLC_CEIL    — structural_ceiling references [191] and correct fraction
    CLC_SRC     — source_attribution mentions Levin
    CLC_WITNESS — at least 3 witnesses (one per orbit family)
    CLC_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates finite S_9 orbit classification; integer state space; no observer, no floats, no continuous dynamics"

import json
import os
import sys
from pathlib import Path

# Make the repo root importable so tools/qa_kg/orbit_failure_enumeration.py
# is reachable regardless of CWD when meta_validator runs us. Mirrors the
# pattern used by cert [263] qa_failure_density_enumeration_cert_v1.
# Source attribution unchanged: (Levin, 2026) Mind Everywhere; (Lyons, 2026)
# Cancer to AI Alignment; (Dale, 2026) QA formalization.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Shared QA primitives + orbit-family classifier from cert [263]'s utility
# module. Refactor 2026-04-27: replaces the previous local copies that
# duplicated cert [194] qa_cognition_space_morphospace_cert_v1's primitives.
from tools.qa_kg.orbit_failure_enumeration import (  # noqa: E402
    qa_mod,
    qa_step,
    orbit_family_s9,
)

SCHEMA_VERSION = "QA_LEVIN_COGNITIVE_LIGHTCONE_CERT.v1"

# Expected orbit cycle lengths in S_9
EXPECTED_CYCLE_LENGTHS = {
    "singularity": 1,
    "satellite": 8,
    "cosmos": 24,
}

# S_9 state counts by family
EXPECTED_FAMILY_SIZES = {
    "singularity": 1,
    "satellite": 8,
    "cosmos": 72,
}


# -----------------------------------------------------------------------------
# QA primitives — qa_mod / qa_step / orbit_family_s9 imported from the
# shared tools/qa_kg/orbit_failure_enumeration.py utility (cert [263] is the
# anchor). orbit_length below is a Levin-cone-specific cycle-length helper
# that the utility does not currently expose.
# -----------------------------------------------------------------------------

def orbit_length(b, e, m=9):
    """Compute the orbit length of (b,e) under T in S_m."""
    start = (b, e)
    cur = qa_step(b, e, m)
    length = 1
    while cur != start:
        cur = qa_step(cur[0], cur[1], m)
        length += 1
    return length


def verify_cycle_lengths():
    """Exhaustively verify S_9 cycle lengths by family."""
    m = 9
    lengths = {"singularity": set(), "satellite": set(), "cosmos": set()}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            fam = orbit_family_s9(b, e)
            ol = orbit_length(b, e, m)
            lengths[fam].add(ol)
    return {fam: sorted(ls) for fam, ls in lengths.items()}


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CLC_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"CLC_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # CLC_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("CLC_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("CLC_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # CLC_SRC: source attribution
    src = str(cert.get("source_attribution", ""))
    if "Levin" not in src:
        warnings.append("CLC_SRC: source_attribution should credit Michael Levin")

    # CLC_ORBIT: clc_orbit_mapping well-formed
    mapping = cert.get("clc_orbit_mapping")
    if not isinstance(mapping, list):
        errors.append("CLC_ORBIT: clc_orbit_mapping must be a list")
    elif len(mapping) < 3:
        errors.append(f"CLC_ORBIT: need >= 3 CLC tiers, got {len(mapping)}")
    else:
        mapped_orbits = {m.get("qa_orbit") for m in mapping}
        for expected_orbit in ["singularity", "satellite", "cosmos"]:
            if expected_orbit not in mapped_orbits:
                errors.append(f"CLC_ORBIT: missing orbit '{expected_orbit}' in mapping")

    # CLC_RADIUS: orbit radii match actual cycle lengths
    if isinstance(mapping, list):
        for entry in mapping:
            qa_orbit = entry.get("qa_orbit")
            declared_radius = entry.get("orbit_radius")
            expected_length = EXPECTED_CYCLE_LENGTHS.get(qa_orbit)
            if expected_length is not None and declared_radius is not None:
                # Radius should equal cycle length (or 0 for singularity)
                expected_radius = 0 if qa_orbit == "singularity" else expected_length
                if declared_radius != expected_radius:
                    errors.append(
                        f"CLC_RADIUS: orbit '{qa_orbit}' declared radius={declared_radius}, "
                        f"expected {expected_radius}"
                    )

    # CLC_CANCER: cancer as orbit demotion
    cancer = cert.get("cancer_as_orbit_demotion")
    if cancer is None:
        errors.append("CLC_CANCER: cancer_as_orbit_demotion section missing")
    else:
        op_class = cancer.get("qa_operator_class", "")
        if "L_2" not in op_class:
            errors.append(f"CLC_CANCER: operator class should be L_2a or L_2b, got {op_class!r}")

    # CLC_CEIL: structural ceiling
    ceiling = cert.get("structural_ceiling")
    if ceiling is None:
        errors.append("CLC_CEIL: structural_ceiling section missing")
    else:
        frac = ceiling.get("level_i_reachable_fraction")
        if frac is not None and abs(frac - 0.2609) > 0.01:
            errors.append(f"CLC_CEIL: level_i_reachable_fraction={frac}, expected ~0.2609")

    # CLC_WITNESS: witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"CLC_WITNESS: need >= 3 witnesses (one per orbit family), got {len(witnesses)}")

    # Verify actual S_9 cycle lengths match expectations
    actual_lengths = verify_cycle_lengths()
    for fam, expected_len in EXPECTED_CYCLE_LENGTHS.items():
        actual = actual_lengths.get(fam, [])
        if expected_len not in actual:
            errors.append(
                f"CLC_RADIUS: family '{fam}' expected cycle length {expected_len} "
                f"but found {actual} in S_9"
            )

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("clc_pass_orbit_radius.json", True),
        ("clc_fail_wrong_radius.json", True),
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
        description="QA Levin Cognitive Lightcone Cert [193] validator")
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
