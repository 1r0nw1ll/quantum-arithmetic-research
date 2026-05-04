#!/usr/bin/env python3
"""QA Whittaker Phase-Packet Algebra Cert validator.

Candidate family ID: [275], standalone unregistered artifact.

Layer 3.1 v1 of the Whittaker -> QA development ladder. This validator
certifies exact finite phase-packet algebra over the registered [273] S2
rational direction substrate.

Claim scope: exact rational phase algebra only. This cert does not evaluate
trigonometric functions and does not prove Whittaker 1903, Maxwell/EM,
scalar-potential physics, or any physical field reconstruction.
"""

QA_COMPLIANCE = "cert_validator - exact Fraction phase-packet algebra over [273] S2 packets; no trig evaluation; no pow operator"

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_WHITTAKER_PHASE_PACKET_ALGEBRA_CERT.v1"
CERT_SLUG = "qa_whittaker_phase_packet_algebra_cert_v1"
CANDIDATE_FAMILY_ID = 275
DEPENDENCY_FAMILY_ID = 273
DEPENDENCY_SLUG = "qa_whittaker_rational_direction_s2_cert_v1"
DEPENDENCY_CHART = "inverse_stereographic_excluding_south_pole"
LINEAGE_CONTEXT_FAMILY_ID = 274
LINEAGE_CONTEXT_SLUG = "qa_whittaker_scalar_angular_kernel_sampling_cert_v1"
ALLOWED_M = {3, 5, 9}
ALLOWED_PACKET_FAMILIES = {
    "phase_arg",
    "phase_pair",
    "formal_cos_symbol",
    "formal_sin_symbol",
}


def _err(errors, code, msg):
    errors.append(f"{code}: {msg}")


def _load_s2_validator():
    here = Path(__file__).resolve()
    dep = (
        here.parents[1]
        / DEPENDENCY_SLUG
        / "qa_whittaker_rational_direction_s2_cert_validate.py"
    )
    if not dep.exists():
        raise RuntimeError(f"missing [273] validator at {dep}")
    spec = importlib.util.spec_from_file_location("qa_wrd_s2_dep", dep)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_S2 = None


def _s2():
    global _S2
    if _S2 is None:
        _S2 = _load_s2_validator()
    return _S2


def _canonical_fraction(value):
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, list) and len(value) == 2:
        return Fraction(value[0], value[1])
    if isinstance(value, dict):
        if set(value) != {"num", "den"}:
            raise ValueError("rational object must have num and den only")
        return Fraction(value["num"], value["den"])
    raise ValueError(f"cannot parse rational value {value!r}")


def _fraction_witness(value):
    return {
        "num": value.numerator,
        "den": value.denominator,
    }


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
    if value["num"] != frac.numerator or value["den"] != frac.denominator:
        _err(errors, code, f"{field_name} must be reduced with positive denominator")
    return frac


def _check_rational_vector(value, errors, code, field_name):
    if not isinstance(value, list) or len(value) != 3:
        _err(errors, code, f"{field_name} must be a 3-vector")
        return None
    out = []
    for idx, item in enumerate(value):
        frac = _check_rational_object(item, errors, code, f"{field_name}[{idx}]")
        if frac is None:
            return None
        out.append(frac)
    return out


def _packet_to_fracs(packet, errors, code, field_name):
    if not isinstance(packet, list) or len(packet) != 4:
        _err(errors, code, f"{field_name} must be [x_num,y_num,z_num,den]")
        return None
    if not all(isinstance(item, int) for item in packet):
        _err(errors, code, f"{field_name} entries must be integers")
        return None
    x_num, y_num, z_num, den = packet
    if den <= 0:
        _err(errors, code, f"{field_name}.den must be positive")
        return None
    common = den
    for num in (x_num, y_num, z_num):
        common = gcd(common, abs(num))
    if common != 1:
        _err(errors, code, f"{field_name} must be canonical")
    lhs = x_num * x_num + y_num * y_num + z_num * z_num
    rhs = den * den
    if lhs != rhs:
        _err(errors, code, f"{field_name} does not satisfy S2 identity")
        return None
    return [
        Fraction(x_num, den),
        Fraction(y_num, den),
        Fraction(z_num, den),
    ]


def _build_dependency_packets(m):
    model = _s2().build_model(m)
    return {tuple(packet) for packet in model["points"]}


def _dot(lhs, rhs):
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _phase_arg(packet, point):
    omega = packet["omega"]
    x = point["x"]
    t = point["t"]
    k = packet["k"]
    v = packet["v"]
    return k * (_dot(omega, x) - v * t)


def _check_source(data, errors):
    src = data.get("source_attribution", "")
    if not (
        isinstance(src, str)
        and "Whittaker" in src
        and "1903" in src
        and "10.1007/BF01444290" in src
    ):
        _err(
            errors,
            "WPPA_SRC",
            "source_attribution must mention Whittaker, 1903, and DOI 10.1007/BF01444290",
        )


def _check_non_claims(data, errors):
    non_claims = data.get("non_claims", [])
    if not isinstance(non_claims, list):
        _err(errors, "WPPA_8", "non_claims must be list")
        return
    blob = " | ".join(str(item) for item in non_claims)
    required = [
        "trigonometric evaluation",
        "numerical approximation",
        "Whittaker 1903 theorem",
        "Maxwell",
        "electromagnetism",
        "scalar-potential physics",
        "physical field",
    ]
    for term in required:
        if term not in blob:
            _err(errors, "WPPA_8", f"non_claims missing {term!r}")


def _check_claim_policy(data, errors):
    policy = data.get("claim_policy", {})
    if not isinstance(policy, dict):
        _err(errors, "WPPA_8", "claim_policy must be object")
        return
    for key in (
        "claims_full_whittaker_theorem",
        "claims_maxwell_em",
        "claims_scalar_potential_physics",
        "claims_physical_field_reconstruction",
        "claims_numerical_wave_kernel_approximation",
    ):
        if policy.get(key) is not False:
            _err(errors, "WPPA_8", f"claim_policy.{key} must be false")


def _check_numerical_firewall(data, errors):
    policy = data.get("numerical_policy", {})
    if not isinstance(policy, dict):
        _err(errors, "WPPA_7", "numerical_policy must be object")
        return
    for key in (
        "uses_trig_evaluation",
        "uses_float_pass_fail",
        "uses_numerical_approximation",
        "uses_fitted_coefficients",
    ):
        if policy.get(key) is not False:
            _err(errors, "WPPA_7", f"numerical_policy.{key} must be false")
    forbidden = data.get("forbidden_operations", [])
    if not isinstance(forbidden, list):
        _err(errors, "WPPA_7", "forbidden_operations must be list")
        return
    if forbidden:
        _err(errors, "WPPA_7", "v1 must not declare forbidden numerical operations")
    if data.get("coefficient_source") != "declared":
        _err(errors, "WPPA_7", "coefficient_source must be declared")
    for forbidden_key in ("fit_method", "training_points", "fitted_coefficients"):
        if forbidden_key in data:
            _err(errors, "WPPA_7", f"{forbidden_key} is not allowed in v1")


def _check_dependency(data, errors):
    dep = data.get("dependency", {})
    if not isinstance(dep, dict):
        _err(errors, "WPPA_1", "dependency must be object")
        return
    if dep.get("family_id") != DEPENDENCY_FAMILY_ID:
        _err(errors, "WPPA_1", "dependency.family_id must be 273")
    if dep.get("slug") != DEPENDENCY_SLUG:
        _err(errors, "WPPA_1", f"dependency.slug must be {DEPENDENCY_SLUG!r}")
    if dep.get("chart") != DEPENDENCY_CHART:
        _err(errors, "WPPA_1", f"dependency.chart must be {DEPENDENCY_CHART!r}")
    if dep.get("registered") is not True:
        _err(errors, "WPPA_1", "dependency.registered must be true")

    lineage = data.get("lineage_context", {})
    if not isinstance(lineage, dict):
        _err(errors, "WPPA_1", "lineage_context must be object")
        return
    if lineage.get("family_id") != LINEAGE_CONTEXT_FAMILY_ID:
        _err(errors, "WPPA_1", "lineage_context.family_id must be 274")
    if lineage.get("slug") != LINEAGE_CONTEXT_SLUG:
        _err(errors, "WPPA_1", f"lineage_context.slug must be {LINEAGE_CONTEXT_SLUG!r}")
    if lineage.get("hard_dependency") is not False:
        _err(errors, "WPPA_1", "lineage_context.hard_dependency must be false")


def _check_packets(data, errors, dependency_packets):
    raw_packets = data.get("packets", [])
    if not isinstance(raw_packets, list) or not raw_packets:
        _err(errors, "WPPA_2", "packets must be non-empty list")
        return {}

    packets = {}
    for raw in raw_packets:
        if not isinstance(raw, dict):
            _err(errors, "WPPA_2", "packet must be object")
            continue
        packet_id = raw.get("packet_id")
        if not isinstance(packet_id, str) or not packet_id:
            _err(errors, "WPPA_2", "packet_id must be non-empty string")
            continue
        if packet_id in packets:
            _err(errors, "WPPA_2", f"duplicate packet_id {packet_id!r}")
            continue
        family = raw.get("packet_family")
        if family not in ALLOWED_PACKET_FAMILIES:
            _err(errors, "WPPA_2", f"invalid packet_family for {packet_id}")

        omega_packet = raw.get("omega_packet")
        if tuple(omega_packet or []) not in dependency_packets:
            _err(errors, "WPPA_1", f"omega_packet for {packet_id} not in [273] D_m^(2)")
        omega = _packet_to_fracs(omega_packet, errors, "WPPA_1", f"{packet_id}.omega_packet")

        k = _check_rational_object(raw.get("k"), errors, "WPPA_2", f"{packet_id}.k")
        v = _check_rational_object(raw.get("v"), errors, "WPPA_2", f"{packet_id}.v")
        weight = _check_rational_object(raw.get("weight"), errors, "WPPA_4", f"{packet_id}.weight")

        if family == "phase_pair":
            refs = raw.get("component_packet_ids")
            if not isinstance(refs, list) or len(refs) != 2:
                _err(errors, "WPPA_2", f"{packet_id}.component_packet_ids must have two entries")

        packets[packet_id] = {
            "packet_id": packet_id,
            "family": family,
            "omega": omega,
            "k": k,
            "v": v,
            "weight": weight,
            "raw": raw,
        }

    for packet in packets.values():
        refs = packet["raw"].get("component_packet_ids", [])
        if refs:
            for ref in refs:
                if ref not in packets:
                    _err(errors, "WPPA_5", f"phase_pair references unknown packet {ref!r}")
    return packets


def _check_target_composition(data, errors, packets):
    terms = data.get("target_composition", [])
    if not isinstance(terms, list) or not terms:
        _err(errors, "WPPA_5", "target_composition must be non-empty list")
        return
    for idx, term in enumerate(terms):
        if not isinstance(term, dict):
            _err(errors, "WPPA_5", f"target_composition[{idx}] must be object")
            continue
        packet_id = term.get("packet_id")
        if packet_id not in packets:
            _err(errors, "WPPA_5", f"target_composition references unknown packet {packet_id!r}")
        _check_rational_object(term.get("weight"), errors, "WPPA_4", f"target_composition[{idx}].weight")


def _point_records(data):
    out = []
    for key, expected_label in (
        ("evaluation_points", "evaluation"),
        ("heldout_points", "heldout"),
    ):
        points = data.get(key, [])
        yield key, expected_label, points


def _check_points(data, errors, packets):
    saw_heldout = False
    for key, expected_label, points in _point_records(data):
        if not isinstance(points, list) or not points:
            _err(errors, "WPPA_6", f"{key} must be non-empty list")
            continue
        for idx, point in enumerate(points):
            if not isinstance(point, dict):
                _err(errors, "WPPA_6", f"{key}[{idx}] must be object")
                continue
            if point.get("split_label") != expected_label:
                _err(errors, "WPPA_6", f"{key}[{idx}].split_label must be {expected_label}")
            if expected_label == "heldout":
                saw_heldout = True
                for leak_key in (
                    "used_to_choose_target",
                    "used_to_choose_packet",
                    "used_to_choose_weight",
                ):
                    if point.get(leak_key) is True:
                        _err(errors, "WPPA_6", f"{key}[{idx}] declares held-out leakage via {leak_key}")
            x = _check_rational_vector(point.get("x"), errors, "WPPA_3", f"{key}[{idx}].x")
            t = _check_rational_object(point.get("t"), errors, "WPPA_3", f"{key}[{idx}].t")
            target_ids = point.get("target_packet_ids", [])
            if not isinstance(target_ids, list) or not target_ids:
                _err(errors, "WPPA_6", f"{key}[{idx}].target_packet_ids must be non-empty list")
                continue
            witnesses = point.get("phase_witnesses", {})
            if not isinstance(witnesses, dict):
                _err(errors, "WPPA_3", f"{key}[{idx}].phase_witnesses must be object")
                continue
            for packet_id in target_ids:
                if packet_id not in packets:
                    _err(errors, "WPPA_5", f"{key}[{idx}] references unknown packet {packet_id!r}")
                    continue
                if x is None or t is None:
                    continue
                expected = _phase_arg(packets[packet_id], {"x": x, "t": t})
                witness = witnesses.get(packet_id)
                if witness is None:
                    _err(errors, "WPPA_3", f"{key}[{idx}] missing phase witness for {packet_id}")
                    continue
                actual = _check_rational_object(witness, errors, "WPPA_3", f"{key}[{idx}].phase_witnesses[{packet_id}]")
                if actual is not None and actual != expected:
                    _err(
                        errors,
                        "WPPA_3",
                        f"{key}[{idx}] phase_arg mismatch for {packet_id}: expected {_fraction_witness(expected)}",
                    )
    if not saw_heldout:
        _err(errors, "WPPA_6", "at least one heldout point is required")


def validate_fixture(data):
    errors = []
    if data.get("schema_version") != SCHEMA_VERSION:
        _err(errors, "WPPA_SCHEMA", f"schema_version must be {SCHEMA_VERSION}")
    if data.get("candidate_family_id") != CANDIDATE_FAMILY_ID:
        _err(errors, "WPPA_SCHEMA", "candidate_family_id must be 275")
    if data.get("cert_slug") != CERT_SLUG:
        _err(errors, "WPPA_SCHEMA", f"cert_slug must be {CERT_SLUG!r}")

    _check_source(data, errors)
    _check_non_claims(data, errors)
    _check_claim_policy(data, errors)
    _check_numerical_firewall(data, errors)
    _check_dependency(data, errors)

    m = data.get("m")
    if m not in ALLOWED_M:
        _err(errors, "WPPA_1", "m must be one of {3,5,9}")
        return errors

    dependency_packets = _build_dependency_packets(m)
    packets = _check_packets(data, errors, dependency_packets)
    _check_target_composition(data, errors, packets)
    _check_points(data, errors, packets)
    return errors


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_path(path):
    return validate_fixture(load_json(path))


def fixture_paths():
    return sorted((Path(__file__).resolve().parent / "fixtures").glob("*.json"))


def self_test():
    results = []
    ok = True
    for path in fixture_paths():
        data = load_json(path)
        expected = data.get("expected_result")
        expected_code = data.get("expected_error_code")
        errors = validate_fixture(data)
        if expected == "PASS":
            passed = not errors
        elif expected == "FAIL":
            passed = bool(errors) and (
                expected_code is None
                or any(str(error).startswith(str(expected_code) + ":") for error in errors)
            )
        else:
            passed = False
            errors.append("WPPA_F: expected_result must be PASS or FAIL")
        ok = ok and passed
        results.append({
            "fixture": path.name,
            "expected": expected,
            "ok": passed,
            "errors": errors,
        })
    return {
        "ok": ok,
        "schema_version": SCHEMA_VERSION,
        "candidate_family_id": CANDIDATE_FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        payload = self_test()
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0 if payload["ok"] else 1
    if not args.fixture:
        parser.error("provide a fixture path or --self-test")
    errors = validate_path(args.fixture)
    payload = {
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
