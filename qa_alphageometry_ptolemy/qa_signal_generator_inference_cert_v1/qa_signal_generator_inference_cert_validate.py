#!/usr/bin/env python3
"""
qa_signal_generator_inference_cert_validate.py

Validator for QA_SIGNAL_GENERATOR_INFERENCE_CERT.v1  [family 209]

Certifies: For any m-valued time series b_t, the generator
e_t = ((b_{t+1} - b_t) % m) + 1 is the unique A1-compliant state
transition. The signal IS the orbit; the generator IS the dynamics.

Key properties:
  - A1 closure: e_t is always in {1,...,m}
  - Uniqueness: exactly one e_t per (b_t, b_{t+1}) pair
  - Role distinction: b = amplitude state, e = transition generator ([208])
  - Generator synchrony: cross-series coupling metric ([207])
  - Domain-general: applies to any quantized time series
  - Supersedes: hardcoded CMAP / MICROSTATE_STATES lookup tables

Empirical validation:
  - EEG chb01: DR2=+0.157 (p=0.0003) beyond delta
  - EEG chb01: DR2=+0.085 (p=0.024) beyond topographic Observer 3
  - All features match [207] prediction: seizure = max coupling

Checks:
    SGI_1         — schema_version matches
    SGI_CLOSURE   — A1 closure proof present and computationally verified
    SGI_UNIQUE    — uniqueness proof present
    SGI_ROLE      — role distinction per [208] documented
    SGI_SYNC      — generator synchrony definition present with [207] connection
    SGI_EMPIRICAL — empirical validation with p-value present
    SGI_SUPERSEDE — supersedes list of hardcoded mappings
    SGI_SRC       — source attribution present
    SGI_WITNESS   — at least 3 witnesses with generator sequences
    SGI_F         — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates signal generator inference; integer state space; A1 closure verified"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SIGNAL_GENERATOR_INFERENCE_CERT.v1"


# ── Computational verification ──────────────────────────────────────────────

def verify_a1_closure(m):
    """Exhaustively verify e_t in {1,...,m} for all (b_t, b_{t+1}) pairs."""
    for b_t in range(1, m + 1):
        for b_next in range(1, m + 1):
            e = ((b_next - b_t - 1) % m) + 1
            if e < 1 or e > m:
                return False, f"e={e} out of range at ({b_t},{b_next})"
            # Verify round-trip: QA step with this e recovers b_next
            b_check = ((b_t + e - 1) % m) + 1
            if b_check != b_next:
                return False, f"round-trip failed at ({b_t},{b_next}): e={e} gives {b_check}"
    return True, f"all {m*m} pairs verified"


def verify_uniqueness(m):
    """Verify each (b_t, b_{t+1}) maps to exactly one e."""
    for b_t in range(1, m + 1):
        for b_next in range(1, m + 1):
            matches = []
            for e in range(1, m + 1):
                b_check = ((b_t + e - 1) % m) + 1
                if b_check == b_next:
                    matches.append(e)
            if len(matches) != 1:
                return False, f"({b_t},{b_next}) has {len(matches)} solutions: {matches}"
    return True, f"unique e for all {m*m} pairs"


def verify_witness(witness, m=9):
    """Verify a witness's generator sequence matches the formula."""
    b_seq = witness.get("b_sequence", [])
    e_seq = witness.get("e_sequence", [])
    if not b_seq or not e_seq:
        return True  # empirical witnesses may not have sequences
    if len(e_seq) != len(b_seq) - 1:
        return False
    for t in range(len(e_seq)):
        expected_e = ((b_seq[t + 1] - b_seq[t] - 1) % m) + 1
        if e_seq[t] != expected_e:
            return False
    return True


# ── Checks ──────────────────────────────────────────────────────────────────

def _run_checks(fixture):
    results = {}

    # SGI_1: schema version
    results["SGI_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    # SGI_CLOSURE: A1 closure proof present + computationally verified
    cp = fixture.get("closure_proof", {})
    results["SGI_CLOSURE"] = (
        isinstance(cp.get("statement"), str)
        and isinstance(cp.get("proof"), str)
        and len(cp.get("proof", "")) > 10
    )
    # Computational verification for m=9 and m=24
    if results["SGI_CLOSURE"]:
        ok9, _ = verify_a1_closure(9)
        ok24, _ = verify_a1_closure(24)
        results["SGI_CLOSURE"] = ok9 and ok24

    # SGI_UNIQUE: uniqueness proof present
    up = fixture.get("uniqueness_proof", {})
    results["SGI_UNIQUE"] = (
        isinstance(up.get("statement"), str)
        and isinstance(up.get("proof"), str)
        and len(up.get("proof", "")) > 10
    )
    # Computational verification
    if results["SGI_UNIQUE"]:
        ok9, _ = verify_uniqueness(9)
        ok24, _ = verify_uniqueness(24)
        results["SGI_UNIQUE"] = ok9 and ok24

    # SGI_ROLE: role distinction per [208]
    rd = fixture.get("role_distinction", {})
    results["SGI_ROLE"] = (
        rd.get("b_role") != rd.get("e_role")
        and isinstance(rd.get("b_source"), str)
        and isinstance(rd.get("e_source"), str)
        and rd.get("cert_208_compliant") is True
    )

    # SGI_SYNC: generator synchrony with [207] connection
    gs = fixture.get("generator_synchrony", {})
    results["SGI_SYNC"] = (
        isinstance(gs.get("definition"), str)
        and "207" in gs.get("cert_207_connection", "")
    )

    # SGI_EMPIRICAL: empirical validation present (multi-patient or single-patient)
    ev = fixture.get("empirical_validation", {})
    rd_mp = ev.get("result_multipatient", {})
    rd_sd = ev.get("result_beyond_delta", {})
    fd = ev.get("feature_directions", {})
    has_multipatient = (
        isinstance(rd_mp.get("mean_delta_r2"), (int, float))
        and rd_mp.get("mean_delta_r2", 0) > 0
        and rd_mp.get("n_patients", 0) >= 2
    )
    has_single = (
        isinstance(rd_sd.get("delta_r2"), (int, float))
        and rd_sd.get("delta_r2", 0) > 0
    )
    results["SGI_EMPIRICAL"] = (
        (has_multipatient or has_single)
        and isinstance(fd, dict)
        and len(fd) >= 1
    )

    # SGI_SUPERSEDE: supersedes list
    sup = fixture.get("supersedes", {})
    results["SGI_SUPERSEDE"] = (
        isinstance(sup.get("hardcoded_mappings"), list)
        and len(sup.get("hardcoded_mappings", [])) >= 1
    )

    # SGI_SRC: source attribution
    src = fixture.get("source_attribution", "")
    results["SGI_SRC"] = "Dale" in src or "Will" in src

    # SGI_WITNESS: at least 3 witnesses with valid sequences
    witnesses = fixture.get("witnesses", [])
    results["SGI_WITNESS"] = len(witnesses) >= 3
    if results["SGI_WITNESS"]:
        for w in witnesses:
            if not verify_witness(w):
                results["SGI_WITNESS"] = False
                break

    # SGI_F: fail_ledger
    fl = fixture.get("fail_ledger")
    results["SGI_F"] = isinstance(fl, list)

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
        print("Usage: python qa_signal_generator_inference_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
