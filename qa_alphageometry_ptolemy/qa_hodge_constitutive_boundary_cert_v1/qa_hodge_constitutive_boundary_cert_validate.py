#!/usr/bin/env python3
"""QA Hodge Constitutive Boundary Cert validator.

Candidate family ID: [510].

Primary mathematical anchors: Allen Hatcher, Algebraic Topology (2002), Ch. 2,
ISBN 978-0-521-79540-1, for cochain/coboundary structure; Alain Bossavit,
Computational Electromagnetism (1998), ISBN 978-0-12-118710-1, for the Hodge
operator / constitutive relation as metric- and medium-dependent extra
structure in finite/discrete electromagnetism. QA project context:
docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md M2, building on certs [508] and
[509].

Claim: classify a declared Hodge/constitutive operator star_QA as one of:

* OBSERVER_BOUNDARY: metric/orientation/units/medium are explicitly imported;
* QA_NATIVE: all required evidence is exact QA-native integer/rational data;
* INVALID: hidden imports, floats, missing provenance, or overclaims.

This cert is a boundary gate. It does NOT derive full Maxwell, does NOT prove
the inhomogeneous equations, and does NOT turn an imported Hodge star into a
QA-native object.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


SCHEMA_VERSION = "QA_HODGE_CONSTITUTIVE_BOUNDARY_CERT.v1"
CERT_SLUG = "qa_hodge_constitutive_boundary_cert_v1"
FAMILY_ID = 510

ALLOWED_CLASSIFICATIONS = {"OBSERVER_BOUNDARY", "QA_NATIVE", "INVALID"}
REQUIRED_OBSERVER_IMPORTS = {"metric_signature", "orientation", "units", "medium_parameters"}
REQUIRED_NATIVE_EVIDENCE = {
    "integer_cell_pairing",
    "qa_invariant_metric_source",
    "exact_orientation_witness",
    "no_observer_imports",
}


class ValidationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RationalEntry:
    row: str
    col: str
    value: Fraction


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ValidationError(code, message)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    _require(isinstance(payload, dict), "SCHEMA", "top-level JSON must be an object")
    return payload


def _as_int(value: Any, code: str, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code, f"{name} must be an integer")
    return value


def _as_str(value: Any, code: str, name: str) -> str:
    _require(isinstance(value, str) and value, code, f"{name} must be a non-empty string")
    return value


def _fraction_from_payload(raw: Any, code: str, name: str) -> Fraction:
    _require(isinstance(raw, dict), code, f"{name} must be an object")
    num = _as_int(raw.get("num"), code, f"{name}.num")
    den = _as_int(raw.get("den"), code, f"{name}.den")
    _require(den > 0, code, f"{name}.den must be positive")
    return Fraction(num, den)


def _validate_claim_policy(payload: dict[str, Any]) -> None:
    policy = payload.get("claim_policy")
    _require(isinstance(policy, dict), "HCB_1", "claim_policy must be present")
    _require(policy.get("claims_hodge_boundary_classification") is True, "HCB_1", "boundary classification claim must be true")
    _require(policy.get("claims_full_maxwell_derivation") is False, "HCB_1", "full Maxwell derivation claim must be false")
    _require(policy.get("claims_inhomogeneous_maxwell") is False, "HCB_1", "inhomogeneous Maxwell claim must be false")
    _require(policy.get("claims_hodge_constructed") is False, "HCB_1", "constructed-Hodge claim must be false in this boundary cert")
    _require(policy.get("claims_electromagnetism") is False, "HCB_1", "electromagnetism claim must be false")
    _require(policy.get("claims_physical_field") is False, "HCB_1", "physical-field claim must be false")
    _require(policy.get("claims_observer_boundary_when_imported") is True, "HCB_1", "observer-boundary policy must be true")


def _validate_dependencies(payload: dict[str, Any]) -> None:
    deps = payload.get("dependencies")
    _require(isinstance(deps, dict), "HCB_2", "dependencies must be present")
    required = {
        "nilpotency_family_id": 508,
        "bianchi_family_id": 509,
    }
    for key, expected in required.items():
        _require(deps.get(key) == expected, "HCB_2", f"{key} must be {expected}")
    source = payload.get("source_attribution")
    _require(isinstance(source, str) and "Hatcher" in source and "Bossavit" in source, "HCB_2", "source_attribution must cite Hatcher and Bossavit")


def _validate_basis(payload: dict[str, Any]) -> tuple[set[str], set[str]]:
    basis = payload.get("cell_basis")
    _require(isinstance(basis, dict), "HCB_3", "cell_basis must be present")
    primal_faces_raw = basis.get("primal_faces")
    dual_faces_raw = basis.get("dual_faces")
    _require(isinstance(primal_faces_raw, list) and primal_faces_raw, "HCB_3", "primal_faces must be non-empty")
    _require(isinstance(dual_faces_raw, list) and dual_faces_raw, "HCB_3", "dual_faces must be non-empty")
    primal_faces = {_as_str(x, "HCB_3", "primal_faces[]") for x in primal_faces_raw}
    dual_faces = {_as_str(x, "HCB_3", "dual_faces[]") for x in dual_faces_raw}
    _require(len(primal_faces) == len(primal_faces_raw), "HCB_3", "duplicate primal face label")
    _require(len(dual_faces) == len(dual_faces_raw), "HCB_3", "duplicate dual face label")
    return primal_faces, dual_faces


def _validate_star_matrix(payload: dict[str, Any], primal_faces: set[str], dual_faces: set[str]) -> list[RationalEntry]:
    star = payload.get("star_operator")
    _require(isinstance(star, dict), "HCB_4", "star_operator must be present")
    entries_raw = star.get("matrix_entries")
    _require(isinstance(entries_raw, list) and entries_raw, "HCB_4", "star matrix_entries must be non-empty")
    entries: list[RationalEntry] = []
    seen: set[tuple[str, str]] = set()
    for i, raw in enumerate(entries_raw):
        _require(isinstance(raw, dict), "HCB_4", f"matrix_entries[{i}] must be an object")
        row = _as_str(raw.get("row"), "HCB_4", f"matrix_entries[{i}].row")
        col = _as_str(raw.get("col"), "HCB_4", f"matrix_entries[{i}].col")
        _require(row in dual_faces, "HCB_4", f"matrix row {row} is not a declared dual face")
        _require(col in primal_faces, "HCB_4", f"matrix col {col} is not a declared primal face")
        key = (row, col)
        _require(key not in seen, "HCB_4", f"duplicate star matrix entry {row},{col}")
        seen.add(key)
        entries.append(RationalEntry(row=row, col=col, value=_fraction_from_payload(raw.get("value"), "HCB_4", f"matrix_entries[{i}].value")))
    _require(any(entry.value != 0 for entry in entries), "HCB_4", "star matrix must not be identically zero")
    return entries


def _validate_classification(payload: dict[str, Any]) -> str:
    star = payload.get("star_operator")
    _require(isinstance(star, dict), "HCB_5", "star_operator must be present")
    classification = _as_str(star.get("classification"), "HCB_5", "star_operator.classification")
    _require(classification in ALLOWED_CLASSIFICATIONS, "HCB_5", f"classification must be one of {sorted(ALLOWED_CLASSIFICATIONS)}")
    declared = _as_str(payload.get("expected_boundary_verdict"), "HCB_5", "expected_boundary_verdict")
    _require(declared == classification, "HCB_5", "expected_boundary_verdict must match star_operator.classification")
    return classification


def _validate_observer_boundary(payload: dict[str, Any]) -> None:
    star = payload.get("star_operator")
    _require(isinstance(star, dict), "HCB_6", "star_operator must be present")
    imports = star.get("observer_imports")
    _require(isinstance(imports, dict), "HCB_6", "observer_imports must be present for OBSERVER_BOUNDARY")
    missing = sorted(key for key in REQUIRED_OBSERVER_IMPORTS if not imports.get(key))
    _require(not missing, "HCB_6", f"observer imports missing: {missing}")
    _require(star.get("qa_native_evidence") in ({}, None), "HCB_6", "observer-boundary star must not also claim QA-native evidence")
    downstream = payload.get("downstream_claim_policy")
    _require(isinstance(downstream, dict), "HCB_6", "downstream_claim_policy must be present")
    _require(downstream.get("allows_conditional_inhomogeneous_recovery") is True, "HCB_6", "conditional recovery must be allowed")
    _require(downstream.get("allows_full_maxwell_derivation") is False, "HCB_6", "full Maxwell derivation must remain blocked")
    _require(downstream.get("requires_m4_conditional_scope") is True, "HCB_6", "M4 conditional scope must be required")


def _validate_qa_native(payload: dict[str, Any]) -> None:
    star = payload.get("star_operator")
    _require(isinstance(star, dict), "HCB_7", "star_operator must be present")
    evidence = star.get("qa_native_evidence")
    _require(isinstance(evidence, dict), "HCB_7", "qa_native_evidence must be present for QA_NATIVE")
    missing = sorted(key for key in REQUIRED_NATIVE_EVIDENCE if evidence.get(key) is not True)
    _require(not missing, "HCB_7", f"QA-native evidence missing: {missing}")
    imports = star.get("observer_imports")
    _require(imports in ({}, None), "HCB_7", "QA_NATIVE star must not declare observer imports")
    downstream = payload.get("downstream_claim_policy")
    _require(isinstance(downstream, dict), "HCB_7", "downstream_claim_policy must be present")
    _require(downstream.get("allows_conditional_inhomogeneous_recovery") is False, "HCB_7", "QA-native Hodge path must not be labeled conditional observer recovery")
    _require(downstream.get("allows_qa_native_inhomogeneous_path") is True, "HCB_7", "QA-native Hodge path must be explicitly opened")
    _require(downstream.get("requires_source_evidence") is True, "HCB_7", "QA-native Hodge still requires source evidence before inhomogeneous Maxwell")
    _require(downstream.get("allows_full_maxwell_derivation") is False, "HCB_7", "full Maxwell remains blocked until source/M4/M5 evidence is present")
    _require(downstream.get("requires_m4_conditional_scope") is False, "HCB_7", "QA-native Hodge path is not the observer-boundary M4 conditional branch")


def _validate_invalid(payload: dict[str, Any]) -> None:
    reasons = payload.get("invalid_reasons")
    _require(isinstance(reasons, list) and reasons, "HCB_8", "INVALID classification requires invalid_reasons")
    for reason in reasons:
        _as_str(reason, "HCB_8", "invalid_reasons[]")


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "SCHEMA", "wrong schema_version")
    _require(payload.get("candidate_family_id") == FAMILY_ID, "SCHEMA", "wrong candidate_family_id")
    _require(payload.get("cert_slug") == CERT_SLUG, "SCHEMA", "wrong cert_slug")
    _validate_claim_policy(payload)
    _validate_dependencies(payload)
    primal_faces, dual_faces = _validate_basis(payload)
    entries = _validate_star_matrix(payload, primal_faces, dual_faces)
    classification = _validate_classification(payload)
    if classification == "OBSERVER_BOUNDARY":
        _validate_observer_boundary(payload)
    elif classification == "QA_NATIVE":
        _validate_qa_native(payload)
    elif classification == "INVALID":
        _validate_invalid(payload)
    return {
        "ok": True,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "classification": classification,
        "n_matrix_entries": len(entries),
        "checks": ["HCB_1", "HCB_2", "HCB_3", "HCB_4", "HCB_5", "HCB_6", "HCB_7", "HCB_8"],
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
    return {
        "ok": not failures,
        "schema_version": SCHEMA_VERSION,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA Hodge Constitutive Boundary cert fixtures")
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
