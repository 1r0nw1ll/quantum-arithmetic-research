#!/usr/bin/env python3
"""QA Wave Co-processor Boundary Cert validator.

Family ID: [284].

This validator certifies a narrow hybrid-computation boundary contract:
exact QA phase packets may be encoded into a continuous physical-wave
co-processor, continuous amplitudes may interfere only inside that declared
co-processor boundary, and readout must return to finite declared bins.

Claim scope: boundary discipline and exact phase bookkeeping only. This cert
does not prove optical/neural/analog speedup, Maxwell physics, reservoir
universality, or any physical implementation claim.
"""

QA_COMPLIANCE = "cert_validator - exact Fraction phase bookkeeping; continuous state allowed only in declared wave co-processor boundary; no pow operator"

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

SCHEMA_VERSION = "QA_WAVE_COPROCESSOR_BOUNDARY_CERT.v1"
CERT_SLUG = "qa_wave_coprocessor_boundary_cert_v1"
FAMILY_ID = 284

ALLOWED_BOUNDARY_KIND = "physical_wave_interference_coprocessor"
ALLOWED_INPUT_DIRECTION = "integer_phase_packet_to_continuous_wave"
ALLOWED_OUTPUT_DIRECTION = "continuous_wave_readout_to_integer_bin"
ALLOWED_RELATIONS = {"support", "oppose", "neutral"}


def _err(errors, code, msg):
    errors.append(f"{code}: {msg}")


def _check_rational_object(value, errors, code, field_name):
    if not isinstance(value, dict):
        _err(errors, code, f"{field_name} must be rational object")
        return None
    if set(value) != {"num", "den"}:
        _err(errors, code, f"{field_name} must have exactly num and den")
        return None
    if not isinstance(value.get("num"), int):
        _err(errors, code, f"{field_name}.num must be integer")
        return None
    if not isinstance(value.get("den"), int) or value.get("den") <= 0:
        _err(errors, code, f"{field_name}.den must be positive integer")
        return None
    frac = Fraction(value["num"], value["den"])
    if frac.numerator != value["num"] or frac.denominator != value["den"]:
        _err(errors, code, f"{field_name} must be reduced with positive denominator")
    return frac


def _fraction_witness(value):
    return {"num": value.numerator, "den": value.denominator}


def _phase_relation(delta):
    delta = delta % 1
    if delta == 0:
        return "support"
    if delta == Fraction(1, 2):
        return "oppose"
    return "neutral"


def _packet_phase(packet, errors, code, field_name):
    m = packet.get("modulus")
    q = packet.get("phase_index")
    if not isinstance(m, int) or m <= 1:
        _err(errors, code, f"{field_name}.modulus must be integer > 1")
        return None
    if not isinstance(q, int) or not (0 <= q < m):
        _err(errors, code, f"{field_name}.phase_index must be integer in [0, modulus)")
        return None
    declared = _check_rational_object(packet.get("phase_turns"), errors, code, f"{field_name}.phase_turns")
    actual = Fraction(q, m)
    if declared is not None and declared != actual:
        _err(errors, code, f"{field_name}.phase_turns mismatch: expected {_fraction_witness(actual)}")
    amp = _check_rational_object(packet.get("amplitude"), errors, code, f"{field_name}.amplitude")
    if amp is not None and amp < 0:
        _err(errors, code, f"{field_name}.amplitude must be nonnegative")
    freq = _check_rational_object(packet.get("frequency"), errors, code, f"{field_name}.frequency")
    if freq is not None and freq <= 0:
        _err(errors, code, f"{field_name}.frequency must be positive")
    return actual


def _check_non_claims(data, errors):
    non_claims = data.get("non_claims", [])
    if not isinstance(non_claims, list):
        _err(errors, "WCB_7", "non_claims must be list")
        return
    blob = " | ".join(str(item) for item in non_claims)
    required = [
        "unlimited parallel computation",
        "computational complexity bypass",
        "optical speedup",
        "neural mechanism proof",
        "Maxwell",
        "reservoir universality",
        "physical implementation",
    ]
    for term in required:
        if term not in blob:
            _err(errors, "WCB_7", f"non_claims missing {term!r}")


def _check_claim_policy(data, errors):
    policy = data.get("claim_policy", {})
    if not isinstance(policy, dict):
        _err(errors, "WCB_7", "claim_policy must be object")
        return
    for key in (
        "claims_unlimited_parallelism",
        "claims_complexity_bypass",
        "claims_physical_speedup",
        "claims_neural_mechanism",
        "claims_maxwell_physics",
        "claims_reservoir_universality",
    ):
        if policy.get(key) is not False:
            _err(errors, "WCB_7", f"claim_policy.{key} must be false")


def _check_source_and_mapping(data, errors):
    src = data.get("source_attribution", "")
    if not (
        isinstance(src, str)
        and "Iverson" in src
        and "Synchronous Harmonics" in src
        and "boundary" in src
    ):
        _err(errors, "WCB_6", "source_attribution must mention Iverson, Synchronous Harmonics, and boundary")
    mapping_ref = Path(__file__).resolve().parent / "mapping_protocol_ref.json"
    if not mapping_ref.exists():
        _err(errors, "WCB_6", "mapping_protocol_ref.json missing")
        return
    try:
        mapping = json.loads(mapping_ref.read_text(encoding="utf-8"))
    except Exception as exc:
        _err(errors, "WCB_6", f"mapping_protocol_ref.json invalid JSON: {exc}")
        return
    if mapping.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
        _err(errors, "WCB_6", "mapping_protocol_ref.protocol_version must be QA_MAPPING_PROTOCOL_REF.v1")
    if not isinstance(mapping.get("ref_path"), str) or not mapping.get("ref_path"):
        _err(errors, "WCB_6", "mapping_protocol_ref.ref_path must be nonempty string")
    ref_sha = mapping.get("ref_sha256")
    if not isinstance(ref_sha, str) or len(ref_sha) != 64 or any(ch not in "0123456789abcdef" for ch in ref_sha):
        _err(errors, "WCB_6", "mapping_protocol_ref.ref_sha256 must be 64 lowercase hex chars")


def _check_boundary(data, errors):
    boundary = data.get("coprocessor_boundary", {})
    if not isinstance(boundary, dict):
        _err(errors, "WCB_2", "coprocessor_boundary must be object")
        return
    if boundary.get("kind") != ALLOWED_BOUNDARY_KIND:
        _err(errors, "WCB_2", f"coprocessor_boundary.kind must be {ALLOWED_BOUNDARY_KIND!r}")
    if boundary.get("role") != "observer_coprocessor":
        _err(errors, "WCB_2", "coprocessor_boundary.role must be observer_coprocessor")
    if boundary.get("continuous_state_allowed") is not True:
        _err(errors, "WCB_2", "continuous_state_allowed must be true inside boundary")
    if boundary.get("continuous_state_is_qa_core") is not False:
        _err(errors, "WCB_2", "continuous_state_is_qa_core must be false")
    if boundary.get("input_direction") != ALLOWED_INPUT_DIRECTION:
        _err(errors, "WCB_2", f"input_direction must be {ALLOWED_INPUT_DIRECTION!r}")
    if boundary.get("output_direction") != ALLOWED_OUTPUT_DIRECTION:
        _err(errors, "WCB_2", f"output_direction must be {ALLOWED_OUTPUT_DIRECTION!r}")


def _check_pipeline(data, errors):
    stages = data.get("pipeline_stages", [])
    if not isinstance(stages, list) or not stages:
        _err(errors, "WCB_3", "pipeline_stages must be nonempty list")
        return
    inside_count = 0
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            _err(errors, "WCB_3", f"pipeline_stages[{idx}] must be object")
            continue
        stype = stage.get("state_type")
        loc = stage.get("location")
        if stype == "continuous_wave":
            if loc != "coprocessor_boundary":
                _err(errors, "WCB_3", f"continuous_wave stage {idx} must be inside coprocessor_boundary")
            inside_count += 1
        elif stype not in {"integer_phase_packet", "integer_bin", "declared_measurement_bin"}:
            _err(errors, "WCB_3", f"pipeline_stages[{idx}] has unsupported state_type {stype!r}")
    if inside_count == 0:
        _err(errors, "WCB_3", "at least one continuous_wave stage must be declared inside boundary")


def _check_packets(data, errors):
    packets = data.get("wave_packets", [])
    if not isinstance(packets, list) or not packets:
        _err(errors, "WCB_1", "wave_packets must be nonempty list")
        return {}
    phases = {}
    for idx, packet in enumerate(packets):
        if not isinstance(packet, dict):
            _err(errors, "WCB_1", f"wave_packets[{idx}] must be object")
            continue
        pid = packet.get("packet_id")
        if not isinstance(pid, str) or not pid:
            _err(errors, "WCB_1", f"wave_packets[{idx}].packet_id must be nonempty string")
            continue
        if pid in phases:
            _err(errors, "WCB_1", f"duplicate packet_id {pid!r}")
            continue
        phases[pid] = _packet_phase(packet, errors, "WCB_1", f"wave_packets[{idx}]")
    return phases


def _check_interference(data, phases, errors):
    witnesses = data.get("interference_witnesses", [])
    if not isinstance(witnesses, list) or not witnesses:
        _err(errors, "WCB_4", "interference_witnesses must be nonempty list")
        return
    for idx, witness in enumerate(witnesses):
        if not isinstance(witness, dict):
            _err(errors, "WCB_4", f"interference_witnesses[{idx}] must be object")
            continue
        left = witness.get("left_packet_id")
        right = witness.get("right_packet_id")
        if left not in phases:
            _err(errors, "WCB_4", f"witness {idx} references unknown left packet {left!r}")
            continue
        if right not in phases:
            _err(errors, "WCB_4", f"witness {idx} references unknown right packet {right!r}")
            continue
        if phases[left] is None or phases[right] is None:
            continue
        declared_delta = _check_rational_object(
            witness.get("phase_delta_turns"), errors, "WCB_4", f"interference_witnesses[{idx}].phase_delta_turns"
        )
        actual_delta = (phases[left] - phases[right]) % 1
        if declared_delta is not None and declared_delta != actual_delta:
            _err(errors, "WCB_4", f"witness {idx} phase_delta_turns mismatch: expected {_fraction_witness(actual_delta)}")
        relation = witness.get("expected_relation")
        actual_relation = _phase_relation(actual_delta)
        if relation not in ALLOWED_RELATIONS:
            _err(errors, "WCB_4", f"witness {idx} expected_relation must be support/oppose/neutral")
        elif relation != actual_relation:
            _err(errors, "WCB_4", f"witness {idx} relation mismatch: expected {actual_relation!r}")


def _check_readout(data, errors):
    readout = data.get("readout_policy", {})
    if not isinstance(readout, dict):
        _err(errors, "WCB_5", "readout_policy must be object")
        return
    if readout.get("measurement_kind") not in {"intensity_bins", "phase_bins", "correlation_bins"}:
        _err(errors, "WCB_5", "measurement_kind must be intensity_bins, phase_bins, or correlation_bins")
    if readout.get("decode_rule") != "declared_threshold_bins":
        _err(errors, "WCB_5", "decode_rule must be declared_threshold_bins")
    if readout.get("rejects_ambiguous_readout") is not True:
        _err(errors, "WCB_5", "rejects_ambiguous_readout must be true")
    noise = _check_rational_object(readout.get("sensor_noise_bound"), errors, "WCB_5", "sensor_noise_bound")
    if noise is not None and noise < 0:
        _err(errors, "WCB_5", "sensor_noise_bound must be nonnegative")
    bins = readout.get("declared_bins", [])
    if not isinstance(bins, list) or len(bins) < 2:
        _err(errors, "WCB_5", "declared_bins must contain at least two bins")


def validate(data):
    errors = []
    if data.get("schema_version") != SCHEMA_VERSION:
        _err(errors, "WCB_0", f"schema_version must be {SCHEMA_VERSION!r}")
    if data.get("cert_slug") != CERT_SLUG:
        _err(errors, "WCB_0", f"cert_slug must be {CERT_SLUG!r}")
    if data.get("family_id") != FAMILY_ID:
        _err(errors, "WCB_0", f"family_id must be {FAMILY_ID}")
    _check_source_and_mapping(data, errors)
    _check_non_claims(data, errors)
    _check_claim_policy(data, errors)
    _check_boundary(data, errors)
    _check_pipeline(data, errors)
    phases = _check_packets(data, errors)
    _check_interference(data, phases, errors)
    _check_readout(data, errors)
    return {"ok": not errors, "errors": errors}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def self_test():
    here = Path(__file__).resolve().parent
    results = []
    all_ok = True
    for fixture in sorted((here / "fixtures").glob("*.json")):
        data = _load_json(fixture)
        expected = data.get("expected_result")
        result = validate(data)
        ok = (expected == "PASS" and result["ok"]) or (expected == "FAIL" and not result["ok"])
        expected_fail_types = data.get("expected_fail_type", [])
        if expected == "FAIL":
            prefixes = {err.split(":", 1)[0] for err in result["errors"]}
            if not isinstance(expected_fail_types, list) or not expected_fail_types:
                ok = False
                result["errors"].append("SELF_TEST: FAIL fixture must declare expected_fail_type list")
            elif not set(expected_fail_types).issubset(prefixes):
                ok = False
                result["errors"].append(
                    "SELF_TEST: expected_fail_type not covered: "
                    + ",".join(sorted(set(expected_fail_types) - prefixes))
                )
        all_ok = all_ok and ok
        results.append({
            "fixture": fixture.name,
            "expected": expected,
            "ok": ok,
            "errors": result["errors"],
        })
    return {
        "ok": all_ok,
        "schema_version": SCHEMA_VERSION,
        "cert_slug": CERT_SLUG,
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result["ok"] else 1
    if not args.path:
        parser.error("path required unless --self-test is used")
    result = validate(_load_json(args.path))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
