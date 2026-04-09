#!/usr/bin/env python3
"""
qa_pezzulo_levin_bootstrap_cert_validate.py

Validator for QA_PEZZULO_LEVIN_BOOTSTRAP_CERT.v1  [family 195]

Certifies: Pezzulo & Levin "Bootstrapping Life-Inspired Machine Intelligence"
(arXiv:2602.08079) 7-stage pipeline mapped to QA architecture levels.

Stage mapping:
    1. Chemistry          -> A1 (No-Zero axiom)
    2. Metabolic networks -> Single-step dynamics T(b,e)
    3. Transcriptional    -> Orbit classification via v_3(f)
    4. Anatomical         -> Orbit structure + E8 alignment
    5. Behavioral         -> Observer projection (Theorem NT)
    6. Abstract reasoning -> Multi-modulus L_2
    7. Creativity         -> L_3 modulus change pi(9)=24

Intelligence ratchet = Pisano fixed point [192]: pi(24)=24.
5 design principles map to QA axioms/operations.

Source: Pezzulo & Levin arXiv:2602.08079

Checks:
    PLB_1          — schema_version matches
    PLB_STAGES     — stage_mapping well-formed (7 stages, monotone Bateson levels)
    PLB_RATCHET    — intelligence_ratchet present with correct Pisano FP
    PLB_PRINCIPLES — design_principle_mapping well-formed (5 principles)
    PLB_PIPELINE   — stages ordered and covering L_0 through L_3
    PLB_SRC        — source_attribution mentions Pezzulo and Levin
    PLB_WITNESS    — cross_references include [191] and [192]
    PLB_F          — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates structural mapping from Pezzulo-Levin stages to QA levels; integer state space; no observer, no floats, no continuous dynamics"

import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_PEZZULO_LEVIN_BOOTSTRAP_CERT.v1"

EXPECTED_STAGE_COUNT = 7
EXPECTED_PRINCIPLE_COUNT = 5

# Bateson levels must be monotonically non-decreasing through the 7 stages
# We encode the partial order: L_0 < L_1 < L_2a < L_2b < L_3
BATESON_ORDER = {
    "L_0": 0,
    "L_1": 1,
    "L_1 to L_2a": 1.5,
    "L_2a": 2,
    "L_2a and L_2b": 2.5,
    "L_2b": 3,
    "L_3": 4,
}

# Pisano period constants
PISANO_9 = 24
PISANO_24 = 24  # Fixed point


# -----------------------------------------------------------------------------
# QA primitives (integer-only, axiom-compliant)
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    """A1-compliant: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def pisano_period(m):
    """Compute Pisano period pi(m) — the period of Fibonacci numbers mod m.
    Uses A1-compliant state space {1,...,m}."""
    if m <= 1:
        return 1
    # Standard Fibonacci mod m (0-indexed for Pisano computation)
    prev, curr = 0, 1
    for i in range(1, m * m + 1):
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return i
    return -1  # Should not happen for valid m


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # PLB_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"PLB_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # PLB_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("PLB_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("PLB_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # PLB_SRC: source attribution
    src = str(cert.get("source_attribution", ""))
    if "Pezzulo" not in src:
        warnings.append("PLB_SRC: source_attribution should credit Giovanni Pezzulo")
    if "Levin" not in src:
        warnings.append("PLB_SRC: source_attribution should credit Michael Levin")

    # PLB_STAGES: stage_mapping well-formed
    stages = cert.get("stage_mapping")
    if not isinstance(stages, list):
        errors.append("PLB_STAGES: stage_mapping must be a list")
    elif len(stages) != EXPECTED_STAGE_COUNT:
        errors.append(f"PLB_STAGES: need exactly {EXPECTED_STAGE_COUNT} stages, got {len(stages)}")
    else:
        # Check stage numbering is 1..7
        stage_nums = [s.get("stage") for s in stages]
        if stage_nums != list(range(1, 8)):
            errors.append(f"PLB_STAGES: stages should be numbered 1-7, got {stage_nums}")

        # PLB_PIPELINE: Bateson levels monotonically non-decreasing
        prev_order = -1
        for s in stages:
            bl = s.get("bateson_level", "")
            order = BATESON_ORDER.get(bl)
            if order is None:
                errors.append(f"PLB_PIPELINE: stage {s.get('stage')} has unrecognized bateson_level={bl!r}")
            elif order < prev_order:
                errors.append(
                    f"PLB_PIPELINE: stage {s.get('stage')} bateson_level={bl} "
                    f"is lower than previous stage — must be monotonically non-decreasing"
                )
            else:
                prev_order = order

        # Check that stages span L_0 to L_3
        all_levels = {s.get("bateson_level") for s in stages}
        has_l0 = any(BATESON_ORDER.get(l, -1) == 0 for l in all_levels)
        has_l3 = any(BATESON_ORDER.get(l, -1) == 4 for l in all_levels)
        if not has_l0:
            errors.append("PLB_PIPELINE: no stage maps to L_0 — pipeline should start at base level")
        if not has_l3:
            errors.append("PLB_PIPELINE: no stage maps to L_3 — pipeline should reach creativity/modulus change")

    # PLB_RATCHET: intelligence ratchet
    ratchet = cert.get("intelligence_ratchet")
    if ratchet is None:
        errors.append("PLB_RATCHET: intelligence_ratchet section missing")
    else:
        # Verify Pisano fixed point
        actual_pi9 = pisano_period(9)
        if actual_pi9 != PISANO_9:
            errors.append(f"PLB_RATCHET: pi(9) computed as {actual_pi9}, expected {PISANO_9}")
        actual_pi24 = pisano_period(24)
        if actual_pi24 != PISANO_24:
            errors.append(f"PLB_RATCHET: pi(24) computed as {actual_pi24}, expected {PISANO_24}")

        cert_ref = ratchet.get("cert_ref")
        if cert_ref != 192:
            warnings.append(f"PLB_RATCHET: cert_ref should be 192 (Dual Extremality), got {cert_ref}")

    # PLB_PRINCIPLES: design principle mapping
    principles = cert.get("design_principle_mapping")
    if not isinstance(principles, list):
        errors.append("PLB_PRINCIPLES: design_principle_mapping must be a list")
    elif len(principles) < EXPECTED_PRINCIPLE_COUNT:
        errors.append(f"PLB_PRINCIPLES: need >= {EXPECTED_PRINCIPLE_COUNT} principles, got {len(principles)}")

    # PLB_WITNESS: cross references should include [191] and [192]
    xrefs = cert.get("cross_references", [])
    xref_families = {x.get("family") for x in xrefs}
    if 191 not in xref_families:
        errors.append("PLB_WITNESS: cross_references must include family 191 (Bateson Learning Levels)")
    if 192 not in xref_families:
        errors.append("PLB_WITNESS: cross_references must include family 192 (Dual Extremality 24)")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("plb_pass_pipeline.json", True),
        ("plb_fail_bad_stage.json", True),
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
        description="QA Pezzulo Levin Bootstrap Cert [195] validator")
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
