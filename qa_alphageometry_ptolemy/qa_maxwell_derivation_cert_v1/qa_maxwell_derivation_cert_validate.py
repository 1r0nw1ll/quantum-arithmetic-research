#!/usr/bin/env python3
"""QA Maxwell Derivation Cert validator.

Candidate family ID: [513].

Primary mathematical anchors: Allen Hatcher, Algebraic Topology (2002), Ch. 2,
ISBN 978-0-521-79540-1; Alain Bossavit, Computational Electromagnetism (1998),
ISBN 978-0-12-118710-1; James Clerk Maxwell (1865), "A Dynamical Theory of the
Electromagnetic Field," Phil. Trans. R. Soc. 155:459-512.

Claim: assemble the certified homogeneous half [509] and inhomogeneous half
[512] into a bounded full-Maxwell claim only within the stated carrier,
boundary, source, metric, unit, and observer-projection conventions.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


SCHEMA_VERSION = "QA_MAXWELL_DERIVATION_CERT.v1"
CERT_SLUG = "qa_maxwell_derivation_cert_v1"
FAMILY_ID = 513
REQUIRED_PHRASE = "QA derives Maxwell only within the stated carrier, boundary, source, metric, unit, and observer-projection conventions certified here."


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


def _as_int(value: Any, code: str, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code, f"{name} must be an integer")
    return value


def _validate_dependencies(payload: dict[str, Any]) -> dict[str, Any]:
    deps = payload.get("dependencies")
    _require(isinstance(deps, dict), "MXD_1", "dependencies must be present")
    expected = {
        "field_bianchi_family_id": 509,
        "hodge_boundary_family_id": 510,
        "source_continuity_family_id": 511,
        "inhomogeneous_recovery_family_id": 512,
    }
    for key, value in expected.items():
        _require(deps.get(key) == value, "MXD_1", f"{key} must be {value}")
    _require(deps.get("hodge_boundary_verdict") == "QA_NATIVE", "MXD_1", "M5 derivation requires [510] QA_NATIVE Hodge branch")
    _require(deps.get("inhomogeneous_recovery_branch") == "QA_NATIVE", "MXD_1", "M5 derivation requires [512] QA_NATIVE branch")
    source = payload.get("source_attribution")
    _require(isinstance(source, str) and "Hatcher" in source and "Bossavit" in source and "Maxwell" in source, "MXD_1", "source_attribution must cite Hatcher, Bossavit, and Maxwell")
    return deps


def _validate_claim_policy(payload: dict[str, Any]) -> None:
    policy = payload.get("claim_policy")
    _require(isinstance(policy, dict), "MXD_2", "claim_policy must be present")
    _require(policy.get("claims_bounded_full_maxwell") is True, "MXD_2", "bounded full-Maxwell claim must be true")
    forbidden = [
        "claims_unbounded_maxwell",
        "claims_physical_electromagnetism",
        "claims_physical_fields",
        "claims_physical_source_generation",
        "claims_whittaker_derives_maxwell",
        "claims_imported_maxwell_equations_as_premise",
        "claims_free_energy_or_scalar_wave_energy",
        "claims_bearden_pond_svp_longitudinal_energy",
    ]
    for key in forbidden:
        _require(policy.get(key) is False, "MXD_2", f"forbidden claim must be false: {key}")


def _validate_statement(payload: dict[str, Any]) -> None:
    statement = payload.get("bounded_derivation_statement")
    _require(statement == REQUIRED_PHRASE, "MXD_3", "bounded derivation statement must match required phrase exactly")


def _validate_component_claims(payload: dict[str, Any]) -> None:
    claims = payload.get("component_claims")
    _require(isinstance(claims, dict), "MXD_4", "component_claims must be present")
    required_true = [
        "homogeneous_half_from_509",
        "qa_native_hodge_from_510",
        "source_continuity_from_511",
        "inhomogeneous_half_from_512",
        "same_declared_carrier_conventions",
    ]
    for key in required_true:
        _require(claims.get(key) is True, "MXD_4", f"component claim must be true: {key}")


def _validate_conventions(payload: dict[str, Any]) -> None:
    conventions = payload.get("certified_conventions")
    _require(isinstance(conventions, dict), "MXD_5", "certified_conventions must be present")
    for key in ["carrier", "boundary", "source", "metric", "unit", "observer_projection"]:
        value = conventions.get(key)
        _require(isinstance(value, str) and value, "MXD_5", f"convention {key} must be a non-empty string")
    _require(_as_int(conventions.get("theorem_nt_crossing_count"), "MXD_5", "theorem_nt_crossing_count") <= 2, "MXD_5", "Theorem NT crossing count exceeds budget")
    _require(conventions.get("uses_numeric_field_values") is False, "MXD_5", "M5 assembly must not evaluate numeric physical fields")
    _require(conventions.get("observer_projection_status") in {"symbolic_projection_only", "bounded_certified_projection"}, "MXD_5", "observer projection status unsupported")


def _validate_negative_evidence(payload: dict[str, Any]) -> None:
    evidence = payload.get("negative_evidence")
    _require(isinstance(evidence, dict), "MXD_6", "negative_evidence must be present")
    required_true = [
        "no_maxwell_equations_assumed",
        "no_whittaker_operator_premise",
        "no_hidden_hodge_star",
        "no_observer_float_or_trig_before_projection",
        "no_physical_claim_beyond_projection",
        "no_scalar_wave_energy_claim",
    ]
    for key in required_true:
        _require(evidence.get(key) is True, "MXD_6", f"negative evidence must be true: {key}")


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _reject_float(payload, "MXD_7")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "SCHEMA", "wrong schema_version")
    _require(payload.get("candidate_family_id") == FAMILY_ID, "SCHEMA", "wrong candidate_family_id")
    _require(payload.get("cert_slug") == CERT_SLUG, "SCHEMA", "wrong cert_slug")
    _validate_dependencies(payload)
    _validate_claim_policy(payload)
    _validate_statement(payload)
    _validate_component_claims(payload)
    _validate_conventions(payload)
    _validate_negative_evidence(payload)
    return {
        "ok": True,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "bounded_derivation_statement": REQUIRED_PHRASE,
        "checks": ["MXD_1", "MXD_2", "MXD_3", "MXD_4", "MXD_5", "MXD_6", "MXD_7"],
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
    parser = argparse.ArgumentParser(description="Validate QA Maxwell Derivation cert fixtures")
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
