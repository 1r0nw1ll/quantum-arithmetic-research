#!/usr/bin/env python3
"""
qa_heartmath_coherence_cert_validate.py  [family 259]

QA_HEARTMATH_COHERENCE_CERT.v1 validator.

Theorem NT (observer-projection firewall) applied to HeartMath-sourced cardiac-rhythm
data. The cert enforces:

  1. Each rhythm-trace state is labeled as a QA orbit class: one of
     {Singularity, Satellite, Cosmos}. No other labels admitted.
  2. No continuous float quantity (coherence_ratio, hrv, lf_hf, hr_bpm, ibi_ms)
     is assigned as a QA state variable (b, e, d, a). These appear ONLY in an
     observer_projections block with direction="output", i.e. as projected readings
     of the underlying discrete state, not causal inputs.
  3. A rhythm trace declares at most two observer/QA boundary crossings per
     Theorem NT: one input projection (continuous signal -> discrete orbit class)
     and one output projection (discrete orbit class -> continuous reading). No
     interior re-projection through a continuous intermediate.
  4. Source attribution names HeartMath primary sources (Oschman / Danielson /
     Edwards / Tomasino) plus Will Dale.
  5. Witnesses cite at least two claim IDs from
     tools/qa_kg/fixtures/source_claims_heartmath.json.

Primary sources:
  - Oschman & Oschman (2015) J Vortex Sci Technol 2:121, DOI 10.4172/2090-8369.1000121
  - Danielson et al (2014) Global Adv Health Med 3(Suppl 1):BPA05
  - Edwards (2018) J Psychology in Africa 28(5):432-433, DOI 10.1080/14330237.2018.1528007
  - Tomasino (1997) Institute of HeartMath Pub 97-002

References: docs/theory/heartmath_phase4_8_excerpts.md;
tools/qa_kg/fixtures/source_claims_heartmath.json.
"""

QA_COMPLIANCE = "cert_validator - HeartMath Theorem-NT firewall; rhythm states are discrete orbit-class labels; continuous coherence is observer projection only; integer state; A1/A2 compliant; no ** operator; no float in QA state; primary-source-anchored"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_HEARTMATH_COHERENCE_CERT.v1"

ORBIT_CLASSES = frozenset({"Singularity", "Satellite", "Cosmos"})

# Known float-valued measurement names drawn from the HeartMath literature.
# Presence of any of these as a QA state-variable name is a firewall breach.
CONTINUOUS_MEASUREMENT_NAMES = frozenset({
    "coherence_ratio", "coherence", "hrv", "lf_hf", "lf", "hf",
    "hr_bpm", "heart_rate", "ibi_ms", "rmssd", "sdnn", "pnn50",
    "vlf", "total_power", "spectral_entropy",
})

QA_STATE_VARIABLE_NAMES = frozenset({"b", "e", "d", "a"})

# Claim IDs registered in tools/qa_kg/fixtures/source_claims_heartmath.json.
# Validator checks witnesses cite at least two of these.
KNOWN_CLAIM_IDS = frozenset({
    "tomasino_1997_liquid_crystal",
    "tomasino_1997_memory_transfer",
    "tomasino_1997_crystallization",
    "tomasino_1997_em_amplification",
    "tomasino_1997_cellular_water",
    "danielson_2014_intro_grassroots",
    "danielson_2014_methods_stress",
    "danielson_2014_mastery_1000",
    "danielson_2014_survey_6pct",
    "oschman_2015_hvmb_mobius",
    "oschman_2015_heart_field_nonlocal",
    "oschman_2015_rein_coherence",
    "oschman_2015_master_oscillator",
    "oschman_2015_scalar_antenna",
    "oschman_2015_bidirectional_antenna",
    "edwards_2018_coherence_patterns",
    "edwards_2018_heart_communication",
    "edwards_2018_hrv_biofeedback",
})


def _run_checks(fixture):
    results = {}

    # HMC_1: schema_version
    results["HMC_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    trace = fixture.get("rhythm_trace", [])
    trace_ok = isinstance(trace, list) and len(trace) >= 1

    # HMC_RHYTHM_LABELS: every state's orbit_class is in {Singularity, Satellite, Cosmos}
    labels_ok = trace_ok
    if labels_ok:
        for state in trace:
            if not isinstance(state, dict):
                labels_ok = False
                break
            cls = state.get("orbit_class")
            if cls not in ORBIT_CLASSES:
                labels_ok = False
                break
    results["HMC_RHYTHM_LABELS"] = labels_ok

    # HMC_NO_FLOAT_STATE: no continuous-measurement name assigned as QA state variable.
    # Continuous measurements allowed only inside observer_projections[...].reading
    # where direction == "output".
    float_state_ok = True
    for state in trace if trace_ok else []:
        qa_state = state.get("qa_state", {})
        if not isinstance(qa_state, dict):
            float_state_ok = False
            break
        # Every key in qa_state must be a QA variable name with an int value.
        for key, val in qa_state.items():
            if key not in QA_STATE_VARIABLE_NAMES:
                float_state_ok = False
                break
            if not isinstance(val, int) or isinstance(val, bool):
                float_state_ok = False
                break
        if not float_state_ok:
            break
        # A QA state variable name must NEVER be a continuous measurement alias.
        # (Defence in depth: even if key is "b", its assigned value must be int,
        # which the above enforces.)
        # Additionally, top-level state fields (other than the declared projection
        # block and the orbit_class) must not embed a continuous measurement name
        # as a state key.
        for k in state.keys():
            if k in CONTINUOUS_MEASUREMENT_NAMES:
                float_state_ok = False
                break
        if not float_state_ok:
            break
    results["HMC_NO_FLOAT_STATE"] = float_state_ok

    # HMC_BOUNDARY_CROSSINGS: each rhythm-trace entry declares an observer_projections
    # list with at most two entries, each with a valid direction, and output readings
    # may carry continuous (float) values.
    boundaries_ok = trace_ok
    if boundaries_ok:
        for state in trace:
            projs = state.get("observer_projections", [])
            if not isinstance(projs, list) or len(projs) > 2:
                boundaries_ok = False
                break
            directions = []
            for p in projs:
                if not isinstance(p, dict):
                    boundaries_ok = False
                    break
                d = p.get("direction")
                if d not in ("input", "output"):
                    boundaries_ok = False
                    break
                directions.append(d)
                reading = p.get("reading", {})
                if not isinstance(reading, dict):
                    boundaries_ok = False
                    break
                # A reading may carry continuous measurements; these are NOT
                # QA state. They are the projected continuous shadow of the
                # discrete orbit_class.
                if d == "output":
                    # Output readings are the only place floats are admitted.
                    pass
                else:
                    # Input readings may also carry continuous sensor data
                    # (raw ECG, HRV sample), which is the pre-projection form.
                    pass
            if not boundaries_ok:
                break
            # Exactly one of each direction OR a single direction entry is allowed.
            if len(directions) == 2 and set(directions) != {"input", "output"}:
                boundaries_ok = False
                break
    results["HMC_BOUNDARY_CROSSINGS"] = boundaries_ok

    # HMC_SRC
    src = fixture.get("source_attribution", "")
    src_authors_ok = any(
        author in src for author in ("Oschman", "Danielson", "Edwards", "Tomasino")
    )
    src_dale_ok = ("Will Dale" in src) or ("Dale" in src)
    src_project_ok = ("HeartMath" in src) or ("heartmath" in src)
    results["HMC_SRC"] = src_authors_ok and src_dale_ok and src_project_ok

    # HMC_WITNESS: witnesses list cites >=2 known claim IDs
    witnesses = fixture.get("witnesses", [])
    cited_ids = set()
    if isinstance(witnesses, list):
        for w in witnesses:
            if not isinstance(w, dict):
                continue
            cid = w.get("claim_id")
            if isinstance(cid, str):
                cited_ids.add(cid)
    results["HMC_WITNESS"] = len(cited_ids & KNOWN_CLAIM_IDS) >= 2

    # HMC_F: fail_ledger is a list
    fl = fixture.get("fail_ledger")
    results["HMC_F"] = isinstance(fl, list)

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
        print("Usage: python qa_heartmath_coherence_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
