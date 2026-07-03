#!/usr/bin/env python3
"""QA Longitudinal Scalar EM Boundary Cert validator.

Candidate family ID: [514].

Primary source: James Clerk Maxwell (1865), "A Dynamical Theory of the
Electromagnetic Field," Phil. Trans. R. Soc. 155:459-512. Mathematical
substrate anchors: Allen Hatcher, Algebraic Topology (2002), Ch. 2, ISBN
978-0-521-79540-1; Alain Bossavit, Computational Electromagnetism (1998),
ISBN 978-0-12-118710-1.

Claim: after bounded Maxwell assembly [513], scalar/longitudinal EM language is
admissible only as representation-bound structure: scalar potentials, gauge
pieces, constrained/source/boundary/media components, or observer projections.
The cert explicitly rejects treating Heaviside/Hertz/Gibbs vector reductions as
premises for the scalar/longitudinal question. It rejects extra scalar/free
energy assertions only as uncertified claims inside this cert; it does not
globally disprove them.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


SCHEMA_VERSION = "QA_LONGITUDINAL_SCALAR_EM_BOUNDARY_CERT.v1"
CERT_SLUG = "qa_longitudinal_scalar_em_boundary_cert_v1"
FAMILY_ID = 514
REQUIRED_PHRASE = (
    "QA does not use Heaviside-Hertz-Gibbs vector reduction as a premise for "
    "the longitudinal/scalar EM question; scalar and longitudinal terms are "
    "admitted only as certified carrier, source, boundary, media, gauge, or "
    "observer-projection structure."
)
REQUIRED_DISPROOF_PHRASE = (
    "This cert rejects unsupported scalar/longitudinal claims only as claims "
    "inside this cert; it does not globally disprove source-free scalar modes, "
    "free-energy claims, Bearden/Pond/SVP claims, or scalar-potential physics."
)


class ValidationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ValidationError(code, message)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    _require(isinstance(payload, dict), "SCHEMA", "top-level JSON must be an object")
    return payload


def _reject_float(value: Any, code: str, path: str = "$") -> None:
    _require(not isinstance(value, float), code, f"float value forbidden at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_float(child, code, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _reject_float(child, code, f"{path}[{i}]")


def _require_nonempty_string(payload: dict[str, Any], key: str, code: str) -> str:
    value = payload.get(key)
    _require(isinstance(value, str) and value, code, f"{key} must be a non-empty string")
    return value


def _validate_dependencies(payload: dict[str, Any]) -> None:
    deps = payload.get("dependencies")
    _require(isinstance(deps, dict), "LSE_1", "dependencies must be present")
    _require(deps.get("bounded_maxwell_family_id") == 513, "LSE_1", "must depend on bounded Maxwell assembly [513]")
    _require(deps.get("hodge_boundary_family_id") == 510, "LSE_1", "must reference Hodge boundary gate [510]")
    _require(deps.get("source_continuity_family_id") == 511, "LSE_1", "must reference source continuity [511]")
    _require(deps.get("whittaker_two_scalar_family_id") == 507, "LSE_1", "must reference Whittaker two-scalar bridge [507] as compatibility context")
    _require(deps.get("uses_whittaker_as_derivation_premise") is False, "LSE_1", "Whittaker cannot be used as derivation premise")
    _require(deps.get("bounded_maxwell_statement") == deps.get("required_bounded_maxwell_statement"), "LSE_1", "bounded Maxwell phrase must be carried through exactly")


def _validate_source_lineage(payload: dict[str, Any]) -> None:
    source = _require_nonempty_string(payload, "source_attribution", "LSE_2")
    for token in ("Maxwell", "1865", "Hatcher", "Bossavit"):
        _require(token in source, "LSE_2", f"source_attribution must cite {token}")
    lineage = payload.get("source_lineage")
    _require(isinstance(lineage, dict), "LSE_2", "source_lineage must be present")
    _require(lineage.get("primary_target") == "Maxwell_1865", "LSE_2", "primary target must be Maxwell_1865")
    _require(lineage.get("qa_substrate") == "finite_cochain_exterior_calculus", "LSE_2", "QA substrate must be finite cochain exterior calculus")
    for key in ("heaviside_vector_reduction_is_premise", "hertz_transverse_only_assumption_is_premise", "gibbs_vector_formalism_is_premise"):
        _require(lineage.get(key) is False, "LSE_2", f"{key} must be false")
    _require(lineage.get("vector_forms_allowed_only_as_observer_notation") is True, "LSE_2", "vector forms may only be observer notation")


def _validate_boundary_statement(payload: dict[str, Any]) -> None:
    _require(payload.get("representation_boundary_statement") == REQUIRED_PHRASE, "LSE_3", "representation boundary statement must match required phrase exactly")


def _validate_longitudinal_policy(payload: dict[str, Any]) -> None:
    policy = payload.get("longitudinal_scalar_policy")
    _require(isinstance(policy, dict), "LSE_4", "longitudinal_scalar_policy must be present")
    allowed_true = [
        "allows_scalar_potential_carriers",
        "allows_longitudinal_source_components",
        "allows_boundary_constrained_longitudinal_components",
        "allows_media_or_constitutive_longitudinal_components",
        "allows_gauge_or_projection_longitudinal_terms",
    ]
    for key in allowed_true:
        _require(policy.get(key) is True, "LSE_4", f"allowed policy must be true: {key}")
    forbidden_false = [
        "claims_extra_source_free_vacuum_scalar_mode",
        "claims_longitudinal_free_energy",
        "claims_bearden_pond_svp_scalar_energy",
        "claims_scalar_potential_equals_physical_field",
        "claims_transverse_only_maxwell_premise",
    ]
    for key in forbidden_false:
        _require(policy.get(key) is False, "LSE_4", f"forbidden policy must be false: {key}")


def _validate_claim_policy(payload: dict[str, Any]) -> None:
    policy = payload.get("claim_policy")
    _require(isinstance(policy, dict), "LSE_5", "claim_policy must be present")
    _require(policy.get("claims_representation_boundary") is True, "LSE_5", "must claim representation boundary")
    _require(policy.get("claims_scalar_longitudinal_certified_when_constrained") is True, "LSE_5", "must claim constrained scalar/longitudinal admissibility")
    forbidden = [
        "claims_new_physical_radiation_mode",
        "claims_physical_electromagnetism_beyond_513",
        "claims_numeric_field_values",
        "claims_unbounded_maxwell",
        "claims_heaviside_hertz_gibbs_as_foundation",
    ]
    for key in forbidden:
        _require(policy.get(key) is False, "LSE_5", f"forbidden claim must be false: {key}")


def _validate_disproof_boundary(payload: dict[str, Any]) -> None:
    boundary = payload.get("disproof_boundary")
    _require(isinstance(boundary, dict), "LSE_6", "disproof_boundary must be present")
    _require(boundary.get("statement") == REQUIRED_DISPROOF_PHRASE, "LSE_6", "disproof boundary statement must match required phrase exactly")
    _require(boundary.get("rejects_meaning") == "unsupported_inside_this_cert_not_global_disproof", "LSE_6", "rejects_meaning must preserve scope")
    _require(boundary.get("does_not_disprove_global_claims") is True, "LSE_6", "must explicitly avoid global-disproof claim")
    _require(boundary.get("requires_separate_cert_for_positive_claims") is True, "LSE_6", "positive claims require separate cert")
    forbidden = [
        "claims_disproof_of_source_free_scalar_modes",
        "claims_disproof_of_free_energy",
        "claims_disproof_of_bearden_pond_svp",
        "claims_disproof_of_scalar_potential_physics",
    ]
    for key in forbidden:
        _require(boundary.get(key) is False, "LSE_6", f"disproof overclaim must be false: {key}")


def _validate_negative_evidence(payload: dict[str, Any]) -> None:
    evidence = payload.get("negative_evidence")
    _require(isinstance(evidence, dict), "LSE_7", "negative_evidence must be present")
    required_true = [
        "no_transverse_only_assumption",
        "no_heaviside_vector_premise",
        "no_hertz_transverse_premise",
        "no_gibbs_vector_premise",
        "no_hidden_physical_medium",
        "no_free_scalar_energy_channel",
        "no_potential_field_equivalence",
    ]
    for key in required_true:
        _require(evidence.get(key) is True, "LSE_7", f"negative evidence must be true: {key}")


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _reject_float(payload, "LSE_8")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "SCHEMA", "wrong schema_version")
    _require(payload.get("candidate_family_id") == FAMILY_ID, "SCHEMA", "wrong candidate_family_id")
    _require(payload.get("cert_slug") == CERT_SLUG, "SCHEMA", "wrong cert_slug")
    _validate_dependencies(payload)
    _validate_source_lineage(payload)
    _validate_boundary_statement(payload)
    _validate_longitudinal_policy(payload)
    _validate_claim_policy(payload)
    _validate_disproof_boundary(payload)
    _validate_negative_evidence(payload)
    return {
        "ok": True,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "representation_boundary_statement": REQUIRED_PHRASE,
        "disproof_boundary_statement": REQUIRED_DISPROOF_PHRASE,
        "checks": ["LSE_1", "LSE_2", "LSE_3", "LSE_4", "LSE_5", "LSE_6", "LSE_7", "LSE_8"],
    }


def validate_file(path: str) -> dict[str, Any]:
    return validate_payload(_load_json(path))


def _fixture_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.join(here, "fixtures")
    return [os.path.join(fixtures, name) for name in sorted(os.listdir(fixtures)) if name.endswith(".json")]


def self_test() -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    n_pass = 0
    n_fail = 0
    for path in _fixture_paths():
        payload = _load_json(path)
        expected = payload.get("expected_result")
        try:
            result = validate_payload(payload)
            if expected != "PASS":
                failures.append({"fixture": os.path.basename(path), "reason": "expected failure but passed", "result": json.dumps(result, sort_keys=True)})
            else:
                n_pass += 1
        except ValidationError as exc:
            if expected == "FAIL" and payload.get("expected_error_code") == exc.code:
                n_fail += 1
            else:
                failures.append({"fixture": os.path.basename(path), "code": exc.code, "message": exc.message})
    return {"ok": not failures, "schema_version": SCHEMA_VERSION, "candidate_family_id": FAMILY_ID, "cert_slug": CERT_SLUG, "n_pass": n_pass, "n_fail": n_fail, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA longitudinal/scalar EM boundary cert fixtures")
    parser.add_argument("file", nargs="?", help="Fixture JSON to validate")
    parser.add_argument("--self-test", action="store_true", help="Run bundled PASS/FAIL fixture checks")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0 if result.get("ok") else 1
        if not args.file:
            parser.error("provide a fixture path or --self-test")
        print(json.dumps(validate_file(args.file), sort_keys=True, separators=(",", ":")))
        return 0
    except ValidationError as exc:
        print(json.dumps({"ok": False, "error_code": exc.code, "message": exc.message}, sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
