#!/usr/bin/env python3
"""QA Whittaker Two-Scalar-Potential Bridge Cert validator.

Candidate family ID: [507], standalone unregistered artifact until reviewed.

Primary source: E. T. Whittaker (1904), "On an expression of the
electromagnetic field due to electrons by means of two scalar potential
functions," Proc. London Math. Soc. s2-1:367-372. DOI: 10.1112/plms/s2-1.1.367
(on disk: Documents/whittaker_corpus/whittaker_1904_two_scalar_potential_functions_electromagnetic_field.pdf,
verbatim formulas transcribed from p.370 sec.3, cross-checked against the PDF
in-session before this validator was written).

Layer 4 v1 of the Whittaker -> QA development ladder
(docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md sec. 8).

Naming guardrail: Whittaker's two potentials are renamed Phi, Psi throughout
(never F, G) because F = a*b = d^2-e^2 and G = d^2+e^2 are QA-reserved
chromogeometric invariants; reusing his letters would silently collide with
them.

CLAIM (narrow). For a QA-rational plane-wave packet built from a registered
[273] S2 direction omega=(x,y,z)/den and declared QA-rational k, v, c, the
twelve raw differential-operator coefficients of Whittaker's 1904 six-component
map (Phi,Psi) -> (dx,dy,dz,hx,hy,hz) -- reproduced verbatim from the primary
source, NOT reconstructed by symmetry-guessing -- are exact fractions.Fraction
values, and:

  (a) div(h) [both Phi-channel and Psi-channel] vanishes exactly and
      unconditionally (no dispersion relation required);
  (b) div(d) Psi-channel vanishes exactly and unconditionally;
  (c) div(d) Phi-channel vanishes exactly whenever the packet satisfies the
      vacuum dispersion relation v*v = c*c (sufficient, for any direction),
      OR whenever the propagation direction is degenerate with Kz = 0
      (omega_z = 0), regardless of dispersion. For non-degenerate packets
      (Kz != 0) that do NOT satisfy the dispersion relation, div(d)
      Phi-channel is exactly nonzero -- i.e. among directions with a nonzero
      z-component, the dispersion relation is necessary and sufficient.
      (An earlier draft of this claim stated a blanket "iff v*v=c*c" without
      the Kz=0 exception; that overreach was caught by Codex hostile review
      before this cert was committed and is fixed here.)

This is a linear-coefficient-algebra / exact-identity claim about Whittaker's
own published operators instantiated at QA-rational directions. It is NOT a
derivation of Maxwell's equations (which are the external definition being
checked against, not proved), NOT a claim that QA computes electromagnetism,
and NOT a physical field reconstruction. See non_claims / claim_policy below.

Note on drafting provenance: an earlier hand-derived draft of this operator
(before the primary-source PDF was consulted) reproduced hz by false symmetry
with dz as ``Gzz - (1/c^2)Gtt``. That form breaks div(h)=0 identically. The
primary source (p.370) gives hz = Gxx + Gyy, which is what is implemented
below and is the form that makes div(h)=0 an unconditional identity.
"""

QA_COMPLIANCE = (
    "cert_validator - exact Fraction two-scalar-potential coefficient algebra "
    "over [273] S2 packets; primary-source-verified 1904 operator identities; "
    "no trig evaluation; no pow operator"
)

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_WHITTAKER_TWO_SCALAR_POTENTIAL_BRIDGE_CERT.v1"
CERT_SLUG = "qa_whittaker_two_scalar_potential_bridge_cert_v1"
CANDIDATE_FAMILY_ID = 507
DEPENDENCY_FAMILY_ID = 273
DEPENDENCY_SLUG = "qa_whittaker_rational_direction_s2_cert_v1"
DEPENDENCY_CHART = "inverse_stereographic_excluding_south_pole"
LINEAGE_CONTEXT_FAMILY_ID = 498
LINEAGE_CONTEXT_SLUG = "qa_whittaker_phase_packet_algebra_cert_v1"
ALLOWED_M = {3, 5, 9}
COEFFICIENT_KEYS = (
    "dx_phi", "dx_psi",
    "dy_phi", "dy_psi",
    "dz_phi", "dz_psi",
    "hx_phi", "hx_psi",
    "hy_phi", "hy_psi",
    "hz_phi", "hz_psi",
)
PHI_ONLY_ZERO_KEYS = ("dz_psi", "hz_phi")


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
    spec = importlib.util.spec_from_file_location("qa_wrd_s2_dep_wspb", dep)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_S2 = None


def _s2():
    global _S2
    if _S2 is None:
        _S2 = _load_s2_validator()
    return _S2


def _build_dependency_packets(m):
    model = _s2().build_model(m)
    return {tuple(packet) for packet in model["points"]}


def _fraction_witness(value):
    return {"num": value.numerator, "den": value.denominator}


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


def _packet_omega_to_fracs(packet, errors, code, field_name):
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
    if x_num * x_num + y_num * y_num + z_num * z_num != den * den:
        _err(errors, code, f"{field_name} does not satisfy S2 identity")
        return None
    return (
        Fraction(x_num, den),
        Fraction(y_num, den),
        Fraction(z_num, den),
    )


def _wavevector(omega, k, v):
    ox, oy, oz = omega
    return (k * ox, k * oy, k * oz, k * v)


def _coefficients(K, c):
    """Whittaker 1904 (Sec. 3, p.370) six-component operator, reproduced
    verbatim (Phi<-F, Psi<-G):
        dx = Fxz + (1/c)Gyt        dy = Fyz - (1/c)Gxt      dz = Fzz - (1/c^2)Ftt
        hx = (1/c)Fyt - Gxz        hy = -(1/c)Fxt - Gyz     hz = Gxx + Gyy
    Second-mixed-partial rule for a formal plane-wave trig(theta) ansatz with
    theta = Kx*x + Ky*y + Kz*z - Kt*t:
        d^2/du dw [trig(theta)] = -(Ku * Kw) * trig(theta)   (label preserved)
        d^2/du dt [trig(theta)] =  (Ku * Kt) * trig(theta)   (label preserved)
        d^2/dt dt [trig(theta)] = -(Kt * Kt) * trig(theta)
    """
    Kx, Ky, Kz, Kt = K
    return {
        "dx_phi": -Kx * Kz,
        "dx_psi": (Ky * Kt) / c,
        "dy_phi": -Ky * Kz,
        "dy_psi": -(Kx * Kt) / c,
        "dz_phi": (Kt * Kt) / (c * c) - Kz * Kz,
        "dz_psi": Fraction(0),
        "hx_phi": (Ky * Kt) / c,
        "hx_psi": Kx * Kz,
        "hy_phi": -(Kx * Kt) / c,
        "hy_psi": Ky * Kz,
        "hz_phi": Fraction(0),
        "hz_psi": -(Kx * Kx + Ky * Ky),
    }


def _divergences(K, coeff):
    Kx, Ky, Kz, _ = K
    div_d_phi = Kx * coeff["dx_phi"] + Ky * coeff["dy_phi"] + Kz * coeff["dz_phi"]
    div_d_psi = Kx * coeff["dx_psi"] + Ky * coeff["dy_psi"] + Kz * coeff["dz_psi"]
    div_h_phi = Kx * coeff["hx_phi"] + Ky * coeff["hy_phi"] + Kz * coeff["hz_phi"]
    div_h_psi = Kx * coeff["hx_psi"] + Ky * coeff["hy_psi"] + Kz * coeff["hz_psi"]
    return {
        "div_d_phi": div_d_phi,
        "div_d_psi": div_d_psi,
        "div_h_phi": div_h_phi,
        "div_h_psi": div_h_psi,
    }


def _check_source(data, errors):
    src = data.get("source_attribution", "")
    ok = (
        isinstance(src, str)
        and "Whittaker" in src
        and "1904" in src
        and "10.1112/plms/s2-1.1.367" in src
        and "[273]" in src
    )
    if not ok:
        _err(
            errors,
            "WSPB_6",
            "source_attribution must mention Whittaker, 1904, DOI "
            "10.1112/plms/s2-1.1.367, and dependency [273]",
        )


def _check_non_claims(data, errors):
    non_claims = data.get("non_claims", [])
    if not isinstance(non_claims, list):
        _err(errors, "WSPB_8", "non_claims must be list")
        return
    blob = " | ".join(str(item) for item in non_claims)
    required = [
        "trigonometric evaluation",
        "numerical approximation",
        "Maxwell",
        "electromagnetism",
        "physical field",
        "Mie scattering",
        "scalar wave energy",
    ]
    for term in required:
        if term not in blob:
            _err(errors, "WSPB_8", f"non_claims missing {term!r}")


def _check_claim_policy(data, errors):
    policy = data.get("claim_policy", {})
    if not isinstance(policy, dict):
        _err(errors, "WSPB_8", "claim_policy must be object")
        return
    for key in (
        "claims_maxwell_derivation",
        "claims_electromagnetism",
        "claims_physical_field_reconstruction",
        "claims_scalar_wave_energy_physics",
        "claims_mie_scattering",
        "claims_layer5_or_beyond",
    ):
        if policy.get(key) is not False:
            _err(errors, "WSPB_8", f"claim_policy.{key} must be false")


def _check_numerical_firewall(data, errors):
    policy = data.get("numerical_policy", {})
    if not isinstance(policy, dict):
        _err(errors, "WSPB_7", "numerical_policy must be object")
        return
    for key in (
        "uses_trig_evaluation",
        "uses_float_pass_fail",
        "uses_numerical_approximation",
        "uses_fitted_coefficients",
    ):
        if policy.get(key) is not False:
            _err(errors, "WSPB_7", f"numerical_policy.{key} must be false")
    if data.get("coefficient_source") != "declared":
        _err(errors, "WSPB_7", "coefficient_source must be declared")
    forbidden = data.get("forbidden_operations", [])
    if not isinstance(forbidden, list):
        _err(errors, "WSPB_7", "forbidden_operations must be list")
        return
    if forbidden:
        _err(errors, "WSPB_7", "v1 must not declare forbidden numerical operations")
    for forbidden_key in ("fit_method", "training_points", "fitted_coefficients"):
        if forbidden_key in data:
            _err(errors, "WSPB_7", f"{forbidden_key} is not allowed in v1")


def _check_dependency(data, errors):
    dep = data.get("dependency", {})
    if not isinstance(dep, dict):
        _err(errors, "WSPB_1", "dependency must be object")
        return
    if dep.get("family_id") != DEPENDENCY_FAMILY_ID:
        _err(errors, "WSPB_1", "dependency.family_id must be 273")
    if dep.get("slug") != DEPENDENCY_SLUG:
        _err(errors, "WSPB_1", f"dependency.slug must be {DEPENDENCY_SLUG!r}")
    if dep.get("chart") != DEPENDENCY_CHART:
        _err(errors, "WSPB_1", f"dependency.chart must be {DEPENDENCY_CHART!r}")
    if dep.get("registered") is not True:
        _err(errors, "WSPB_1", "dependency.registered must be true")

    lineage = data.get("lineage_context", {})
    if not isinstance(lineage, dict):
        _err(errors, "WSPB_1", "lineage_context must be object")
        return
    if lineage.get("family_id") != LINEAGE_CONTEXT_FAMILY_ID:
        _err(errors, "WSPB_1", "lineage_context.family_id must be 498")
    if lineage.get("slug") != LINEAGE_CONTEXT_SLUG:
        _err(errors, "WSPB_1", f"lineage_context.slug must be {LINEAGE_CONTEXT_SLUG!r}")
    if lineage.get("hard_dependency") is not False:
        _err(errors, "WSPB_1", "lineage_context.hard_dependency must be false")


def _check_packets(data, errors, dependency_packets):
    raw_packets = data.get("packets", [])
    if not isinstance(raw_packets, list) or not raw_packets:
        _err(errors, "WSPB_2", "packets must be non-empty list")
        return {}

    packets = {}
    for raw in raw_packets:
        if not isinstance(raw, dict):
            _err(errors, "WSPB_2", "packet must be object")
            continue
        packet_id = raw.get("packet_id")
        if not isinstance(packet_id, str) or not packet_id:
            _err(errors, "WSPB_2", "packet_id must be non-empty string")
            continue
        if packet_id in packets:
            _err(errors, "WSPB_2", f"duplicate packet_id {packet_id!r}")
            continue

        omega_packet = raw.get("omega_packet")
        if tuple(omega_packet or []) not in dependency_packets:
            _err(errors, "WSPB_1", f"omega_packet for {packet_id} not in [273] D_m^(2)")
        omega = _packet_omega_to_fracs(omega_packet, errors, "WSPB_1", f"{packet_id}.omega_packet")

        k = _check_rational_object(raw.get("k"), errors, "WSPB_2", f"{packet_id}.k")
        v = _check_rational_object(raw.get("v"), errors, "WSPB_2", f"{packet_id}.v")
        c = _check_rational_object(raw.get("c"), errors, "WSPB_2", f"{packet_id}.c")
        wave_eq_claimed = raw.get("wave_equation_satisfied")
        if not isinstance(wave_eq_claimed, bool):
            _err(errors, "WSPB_2", f"{packet_id}.wave_equation_satisfied must be boolean")
            wave_eq_claimed = None

        if None in (omega, k, v, c) or wave_eq_claimed is None:
            continue
        if c == 0:
            _err(errors, "WSPB_2", f"{packet_id}.c must be nonzero")
            continue

        actual_wave_eq_holds = (v * v == c * c)
        if wave_eq_claimed != actual_wave_eq_holds:
            _err(
                errors,
                "WSPB_4",
                f"{packet_id}.wave_equation_satisfied={wave_eq_claimed} but v*v==c*c is {actual_wave_eq_holds}",
            )

        K = _wavevector(omega, k, v)
        expected_coeff = _coefficients(K, c)
        expected_div = _divergences(K, expected_coeff)

        declared_coeff = raw.get("coefficients", {})
        if not isinstance(declared_coeff, dict):
            _err(errors, "WSPB_3", f"{packet_id}.coefficients must be object")
        else:
            for key in COEFFICIENT_KEYS:
                witness = declared_coeff.get(key)
                actual = _check_rational_object(witness, errors, "WSPB_3", f"{packet_id}.coefficients.{key}")
                if actual is not None and actual != expected_coeff[key]:
                    _err(
                        errors,
                        "WSPB_3",
                        f"{packet_id}.coefficients.{key} mismatch: expected {_fraction_witness(expected_coeff[key])}",
                    )
            for key in PHI_ONLY_ZERO_KEYS:
                witness = declared_coeff.get(key)
                if isinstance(witness, dict) and witness.get("num") != 0:
                    _err(errors, "WSPB_3", f"{packet_id}.coefficients.{key} must be zero")

        declared_div = raw.get("divergences", {})
        if not isinstance(declared_div, dict):
            _err(errors, "WSPB_5", f"{packet_id}.divergences must be object")
        else:
            for key in ("div_d_phi", "div_d_psi", "div_h_phi", "div_h_psi"):
                witness = declared_div.get(key)
                actual = _check_rational_object(witness, errors, "WSPB_5", f"{packet_id}.divergences.{key}")
                if actual is not None and actual != expected_div[key]:
                    _err(
                        errors,
                        "WSPB_5",
                        f"{packet_id}.divergences.{key} mismatch: expected {_fraction_witness(expected_div[key])}",
                    )
            # Unconditional identities: these three must vanish regardless of
            # the dispersion relation.
            for key in ("div_d_psi", "div_h_phi", "div_h_psi"):
                if expected_div[key] != 0:
                    _err(errors, "WSPB_5", f"{packet_id}.{key} failed to vanish unconditionally")
            # Conditional identity: div(d)_phi vanishes under dispersion, or
            # under the degenerate Kz == 0 direction. For Kz != 0 and no
            # dispersion, it must be nonzero; this is the Codex hostile-review
            # guard against the earlier blanket "iff v*v == c*c" overclaim.
            kz = K[2]
            div_d_phi_zero = (expected_div["div_d_phi"] == 0)
            if actual_wave_eq_holds or kz == 0:
                if not div_d_phi_zero:
                    _err(
                        errors,
                        "WSPB_5",
                        f"{packet_id}.div_d_phi must vanish when dispersion holds or Kz == 0",
                    )
            elif div_d_phi_zero:
                _err(
                    errors,
                    "WSPB_5",
                    f"{packet_id}.div_d_phi must be nonzero when Kz != 0 and v*v != c*c",
                )

        packets[packet_id] = raw

    return packets


def validate_fixture(data):
    errors = []
    if data.get("schema_version") != SCHEMA_VERSION:
        _err(errors, "WSPB_SCHEMA", f"schema_version must be {SCHEMA_VERSION}")
    if data.get("candidate_family_id") != CANDIDATE_FAMILY_ID:
        _err(errors, "WSPB_SCHEMA", "candidate_family_id must be 507")
    if data.get("cert_slug") != CERT_SLUG:
        _err(errors, "WSPB_SCHEMA", f"cert_slug must be {CERT_SLUG!r}")

    _check_source(data, errors)
    _check_non_claims(data, errors)
    _check_claim_policy(data, errors)
    _check_numerical_firewall(data, errors)
    _check_dependency(data, errors)

    m = data.get("m")
    if m not in ALLOWED_M:
        _err(errors, "WSPB_1", "m must be one of {3,5,9}")
        return errors

    dependency_packets = _build_dependency_packets(m)
    _check_packets(data, errors, dependency_packets)
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
            errors.append("WSPB_F: expected_result must be PASS or FAIL")
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
