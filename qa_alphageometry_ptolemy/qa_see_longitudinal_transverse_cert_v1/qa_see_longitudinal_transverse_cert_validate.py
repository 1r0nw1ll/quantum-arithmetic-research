#!/usr/bin/env python3
"""
qa_see_longitudinal_transverse_cert_validate.py

Validator for QA_SEE_LONGITUDINAL_TRANSVERSE_CERT.v1  [family 197]

Certifies: T.J.J. See's wave duality (1917) — longitudinal (compression)
waves = gravity/magnetism; transverse (shear) waves = light/heat — mapped
to QA's generator/observer duality.

See's claim (Electrodynamic Wave-Theory of Physical Forces, 1917):
    The same aether medium supports two orthogonal wave modes:
    (1) Longitudinal (compression) — gravity, magnetism, electrostatic fields
    (2) Transverse (shear) — light, heat, radio waves
    These modes do not mix: compression cannot produce shear, and vice versa.

QA mapping:
    Same medium         = same tuple space (b, e, d, a)
    Longitudinal mode   = generator action (T-operator along orbit path)
                          Discrete, causal, drives state transitions.
    Transverse mode     = observer projection (continuous functions measuring
                          QA state). Non-causal, measurement-only.
    "Modes don't mix"   = Theorem NT (Observer Projection Firewall):
                          observer projections NEVER enter QA logic as
                          causal inputs.

Physical grounding for Theorem NT:
    In a linear elastic medium, longitudinal and transverse modes are
    orthogonal eigenmodes of the wave equation. A pure compression wave
    cannot generate shear, and vice versa, because they belong to
    different eigenspaces of the stress tensor. This is not a convention
    but a structural impossibility.

    See's insight: the same medium can carry both modes, but they propagate
    independently. QA's insight: the same tuple space supports both
    generator dynamics and observer projections, but they operate in
    orthogonal 'directions' (discrete vs continuous).

Complementarity with Keely/SVP [153]:
    Keely's triune (enharmonic/harmonic/dominant) maps to THREE orbit types
    within the longitudinal (generator) mode. See's duality is a BINARY
    decomposition (generator vs observer) that is orthogonal to Keely's
    triune. They are complementary, not redundant:
    - Keely: structure WITHIN the generator mode (3 orbit families)
    - See: structure BETWEEN the two access modes (generator vs observer)

Source: T.J.J. See, "Electrodynamic Wave-Theory of Physical Forces" (1917),
Vols I-II; "New Theory of the Aether", Astronomische Nachrichten 211-226
(1920-1926). Tesla endorsement noted.

Checks:
    SLT_1       — schema_version matches
    SLT_LONG    — longitudinal mode mapping well-formed
    SLT_TRANS   — transverse mode mapping well-formed
    SLT_ORTH    — orthogonality claim articulated
    SLT_NT      — link to Theorem NT explicit
    SLT_KEELY   — complementarity with [153] Keely triune stated
    SLT_SRC     — source attribution to See present
    SLT_WITNESS — at least 2 witnesses (one per mode)
    SLT_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates structural mapping claims; no floats, no observer projections, no continuous dynamics in validation logic"

import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SEE_LONGITUDINAL_TRANSVERSE_CERT.v1"


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def _run_checks(fixture):
    results = {}

    # SLT_1: schema version
    results["SLT_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    # SLT_LONG: longitudinal mode mapping
    long_mode = fixture.get("longitudinal_mode", {})
    results["SLT_LONG"] = (
        "see_description" in long_mode
        and "qa_mapping" in long_mode
        and "causal" in str(long_mode.get("qa_mapping", "")).lower()
        or "generator" in str(long_mode.get("qa_mapping", "")).lower()
        or "t-operator" in str(long_mode.get("qa_mapping", "")).lower()
        or "discrete" in str(long_mode.get("qa_mapping", "")).lower()
    )

    # SLT_TRANS: transverse mode mapping
    trans_mode = fixture.get("transverse_mode", {})
    results["SLT_TRANS"] = (
        "see_description" in trans_mode
        and "qa_mapping" in trans_mode
        and ("observer" in str(trans_mode.get("qa_mapping", "")).lower()
             or "projection" in str(trans_mode.get("qa_mapping", "")).lower()
             or "continuous" in str(trans_mode.get("qa_mapping", "")).lower()
             or "measurement" in str(trans_mode.get("qa_mapping", "")).lower())
    )

    # SLT_ORTH: orthogonality
    orth = fixture.get("orthogonality", {})
    results["SLT_ORTH"] = (
        "see_claim" in orth
        and "qa_claim" in orth
        and "physical_basis" in orth
    )

    # SLT_NT: Theorem NT link
    nt = fixture.get("theorem_nt_link", {})
    results["SLT_NT"] = (
        "statement" in nt
        and ("firewall" in str(nt.get("statement", "")).lower()
             or "observer" in str(nt.get("statement", "")).lower()
             or "never" in str(nt.get("statement", "")).lower())
    )

    # SLT_KEELY: complementarity with [153]
    keely = fixture.get("keely_complementarity", {})
    results["SLT_KEELY"] = (
        "keely_scope" in keely
        and "see_scope" in keely
        and "relation" in keely
    )

    # SLT_SRC: source attribution
    src = fixture.get("source_attribution", "")
    results["SLT_SRC"] = "See" in src and ("1917" in src or "Wave" in src)

    # SLT_WITNESS: at least 2 witnesses
    witnesses = fixture.get("witnesses", [])
    results["SLT_WITNESS"] = len(witnesses) >= 2
    if results["SLT_WITNESS"]:
        modes = {w.get("mode") for w in witnesses}
        results["SLT_WITNESS"] = "longitudinal" in modes and "transverse" in modes

    # SLT_F: fail_ledger
    fl = fixture.get("fail_ledger")
    results["SLT_F"] = isinstance(fl, list)

    return results


def validate_fixture(path):
    with open(path) as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    all_pass = all(checks.values())
    actual = "PASS" if all_pass else "FAIL"
    ok = actual == expected
    return {"ok": ok, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    """Run validator against bundled fixtures."""
    fdir = Path(__file__).parent / "fixtures"
    results = {}
    for fp in sorted(fdir.glob("*.json")):
        results[fp.name] = validate_fixture(fp)
    all_ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    elif len(sys.argv) > 1:
        r = validate_fixture(sys.argv[1])
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    else:
        print("Usage: python qa_see_longitudinal_transverse_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
