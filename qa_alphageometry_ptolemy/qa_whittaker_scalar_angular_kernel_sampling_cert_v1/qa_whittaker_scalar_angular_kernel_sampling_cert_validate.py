#!/usr/bin/env python3
"""QA Whittaker Scalar Angular-Kernel Sampling Cert validator.

Candidate family ID: [274], standalone artifact only until hostile review.

Layer 3 v1 of the Whittaker -> QA development ladder. This validator certifies
only exact finite scalar-profile sampling over the registered [273] S2 rational
direction substrate.

Claim scope: deterministic scalar sampling and exact finite-set averaging.
This cert does not prove Whittaker 1903, a Whittaker wave kernel, spherical
quadrature, convergence, Maxwell/EM, scalar-potential physics, or any physical
field reconstruction.
"""

QA_COMPLIANCE = "cert_validator - exact Fraction scalar sampling over [273] S2 packets; d*d style inherited from [273]; hashes use canonical ASCII num/den; floats observer display only"

import argparse
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

SCHEMA_VERSION = "QA_WHITTAKER_SCALAR_ANGULAR_KERNEL_SAMPLING_CERT.v1"
CERT_SLUG = "qa_whittaker_scalar_angular_kernel_sampling_cert_v1"
CANDIDATE_FAMILY_ID = 274
DEPENDENCY_FAMILY_ID = 273
DEPENDENCY_SLUG = "qa_whittaker_rational_direction_s2_cert_v1"
ALLOWED_M = {3, 5, 9}
ALLOWED_PROFILES = {"const", "z", "z2"}
WEIGHT_RULE = "uniform_points"
CHART = "inverse_stereographic_excluding_south_pole"


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


def build_s2_points(m):
    """Recompute [273] D_m^(2) packets using the registered S2 construction."""
    module = _s2()
    _seeds, ratio_channels, _raw_ratio_count = module.enumerate_seed_ratios(m)
    point_by_direction = {}
    for r in sorted(ratio_channels):
        for s in sorted(ratio_channels):
            packet, direction = module.s2_point_from_ratios(r, s)
            point_by_direction.setdefault(direction, packet)
    return [point_by_direction[key] for key in sorted(point_by_direction)]


def profile_value(packet, profile_name):
    x_num, y_num, z_num, den = packet
    del x_num, y_num
    z = Fraction(z_num, den)
    if profile_name == "const":
        return Fraction(1, 1)
    if profile_name == "z":
        return z
    if profile_name == "z2":
        return z * z
    raise ValueError(f"unknown profile {profile_name!r}")


def compute_discrete_uniform_average(m, profile_name):
    points = build_s2_points(m)
    total = Fraction(0, 1)
    for packet in points:
        total += profile_value(packet, profile_name)
    average = total / len(points)
    return points, average


def canonical_fraction_string(value):
    return f"{value.numerator}/{value.denominator}"


def canonical_fraction_sha256(value):
    payload = canonical_fraction_string(value).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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
            "WKB_SRC",
            "source_attribution must mention Whittaker, 1903, and DOI 10.1007/BF01444290",
        )


def _check_non_claims(data, errors):
    non_claims = data.get("non_claims", [])
    if not isinstance(non_claims, list):
        _err(errors, "WKB_DECL", "non_claims must be a list")
        return
    blob = " | ".join(str(item) for item in non_claims)
    required = [
        "Whittaker kernel",
        "spherical quadrature",
        "density",
        "convergence",
        "Maxwell",
        "electromagnetism",
        "scalar-potential physics",
        "physical field",
    ]
    for term in required:
        if term not in blob:
            _err(errors, "WKB_DECL", f"non_claims missing {term!r}")


def _check_dependency(data, errors):
    dep = data.get("dependency", {})
    if not isinstance(dep, dict):
        _err(errors, "WKB_1", "dependency must be an object")
        return
    if dep.get("family_id") != DEPENDENCY_FAMILY_ID:
        _err(errors, "WKB_1", "dependency.family_id must be 273")
    if dep.get("slug") != DEPENDENCY_SLUG:
        _err(errors, "WKB_1", f"dependency.slug must be {DEPENDENCY_SLUG!r}")
    if dep.get("chart") != CHART:
        _err(errors, "WKB_1", f"dependency.chart must be {CHART!r}")
    if dep.get("registered") is not True:
        _err(errors, "WKB_1", "dependency.registered must be true")


def _check_observer_policy(data, errors):
    policy = data.get("observer_diagnostics", {})
    if not isinstance(policy, dict):
        _err(errors, "WKB_5", "observer_diagnostics must be an object")
        return
    if policy.get("display_only") is not True:
        _err(errors, "WKB_5", "observer diagnostics must be display_only=true")
    if policy.get("uses_observer_float_for_pass_fail") is not False:
        _err(errors, "WKB_5", "observer floats must not be used for pass/fail")
    overclaim_keys = (
        "claims_spherical_quadrature",
        "claims_whittaker_kernel_error",
        "claims_physics",
        "claims_maxwell_em",
        "claims_scalar_potential",
        "claims_density",
        "claims_convergence",
    )
    for key in overclaim_keys:
        if policy.get(key) is not False:
            _err(errors, "WKB_5", f"observer_diagnostics.{key} must be false")


def _check_average_witness(witness, average, errors):
    if not isinstance(witness, dict):
        _err(errors, "WKB_4", "discrete_uniform_average_witness must be object")
        return
    mode = witness.get("mode")
    actual_num = str(average.numerator)
    actual_den = str(average.denominator)
    actual_hash = canonical_fraction_sha256(average)
    actual_num_digits = len(actual_num)
    actual_den_digits = len(actual_den)

    if witness.get("numerator_digit_count") != actual_num_digits:
        _err(errors, "WKB_4_DIGITS", "numerator_digit_count mismatch")
    if witness.get("denominator_digit_count") != actual_den_digits:
        _err(errors, "WKB_4_DIGITS", "denominator_digit_count mismatch")

    declared_hash = witness.get("canonical_fraction_sha256")
    if not (isinstance(declared_hash, str) and len(declared_hash) == 64):
        _err(errors, "WKB_4_HASH", "canonical_fraction_sha256 must be full 64 hex chars")
    elif declared_hash != actual_hash:
        _err(errors, "WKB_4_HASH", "canonical_fraction_sha256 mismatch")

    if "canonical_fraction_sha256_16" in witness:
        if witness.get("canonical_fraction_sha256_16") != actual_hash[:16]:
            _err(errors, "WKB_4_HASH", "canonical_fraction_sha256_16 display prefix mismatch")

    if mode == "direct_fraction":
        if str(witness.get("numerator")) != actual_num:
            _err(errors, "WKB_4", "declared numerator mismatch")
        if str(witness.get("denominator")) != actual_den:
            _err(errors, "WKB_4", "declared denominator mismatch")
    elif mode == "canonical_hash":
        if "numerator" in witness or "denominator" in witness:
            _err(errors, "WKB_4", "canonical_hash mode must not declare numerator/denominator")
    else:
        _err(errors, "WKB_4", "witness.mode must be direct_fraction or canonical_hash")

    if "observer_float_display" in witness:
        try:
            float(witness["observer_float_display"])
        except Exception:
            _err(errors, "WKB_5", "observer_float_display must parse as float when present")


def validate_fixture(data):
    errors = []
    if data.get("schema_version") != SCHEMA_VERSION:
        _err(errors, "WKB_SCHEMA", f"schema_version must be {SCHEMA_VERSION}")
    if data.get("candidate_family_id") != CANDIDATE_FAMILY_ID:
        _err(errors, "WKB_SCHEMA", "candidate_family_id must be 274")
    if data.get("cert_slug") != CERT_SLUG:
        _err(errors, "WKB_SCHEMA", f"cert_slug must be {CERT_SLUG!r}")

    _check_source(data, errors)
    _check_non_claims(data, errors)
    _check_dependency(data, errors)
    _check_observer_policy(data, errors)

    m = data.get("m")
    if m not in ALLOWED_M:
        _err(errors, "WKB_1", "m must be one of {3,5,9}")
        return errors

    profile_name = data.get("profile_name")
    if profile_name not in ALLOWED_PROFILES:
        _err(errors, "WKB_2", "profile_name must be one of {const,z,z2}")
        return errors

    if data.get("weight_rule") != WEIGHT_RULE:
        _err(errors, "WKB_3", "weight_rule must be uniform_points")

    if data.get("pass_fail_basis") != "exact_fraction_or_hash_witness":
        _err(errors, "WKB_5", "pass_fail_basis must be exact_fraction_or_hash_witness")

    points, average = compute_discrete_uniform_average(m, profile_name)
    if data.get("unique_S2_direction_count") != len(points):
        _err(errors, "WKB_1", "unique_S2_direction_count mismatch with [273] D_m^(2)")

    _check_average_witness(data.get("discrete_uniform_average_witness"), average, errors)
    return errors


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_path(path):
    data = load_json(path)
    return validate_fixture(data)


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
            errors.append("WKB_F: expected_result must be PASS or FAIL")
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
