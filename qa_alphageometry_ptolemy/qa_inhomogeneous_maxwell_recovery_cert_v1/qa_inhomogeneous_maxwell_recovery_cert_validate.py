#!/usr/bin/env python3
"""QA Inhomogeneous Maxwell Recovery Cert validator.

Candidate family ID: [512].

Primary mathematical anchors: Allen Hatcher, Algebraic Topology (2002), Ch. 2,
ISBN 978-0-521-79540-1, for cochains/coboundary; Alain Bossavit,
Computational Electromagnetism (1998), ISBN 978-0-12-118710-1, for discrete
Hodge/constitutive structure; James Clerk Maxwell (1865), "A Dynamical Theory
of the Electromagnetic Field," Phil. Trans. R. Soc. 155:459-512, for the
classical inhomogeneous Maxwell target being recovered under declared
projection conventions.

Claim: under a declared [510] Hodge verdict, exact field 2-cochain F, exact
star_QA matrix, and explicit sign/unit/projection conventions, recompute
starF = star_QA(F) and J = delta(starF) exactly. This is M4 of
docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md.

This cert does NOT derive full Maxwell, does NOT claim physical
electromagnetism, and does NOT claim physical source generation.
"""

from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from typing import Any


SCHEMA_VERSION = "QA_INHOMOGENEOUS_MAXWELL_RECOVERY_CERT.v1"
CERT_SLUG = "qa_inhomogeneous_maxwell_recovery_cert_v1"
FAMILY_ID = 512


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


def _as_str(value: Any, code: str, name: str) -> str:
    _require(isinstance(value, str) and value, code, f"{name} must be a non-empty string")
    return value


def _fraction(raw: Any, code: str, name: str) -> Fraction:
    _require(isinstance(raw, dict), code, f"{name} must be an object")
    num = _as_int(raw.get("num"), code, f"{name}.num")
    den = _as_int(raw.get("den"), code, f"{name}.den")
    _require(den > 0, code, f"{name}.den must be positive")
    return Fraction(num, den)


def _fraction_values(raw: Any, expected_basis: set[str], code: str, name: str) -> dict[str, Fraction]:
    _require(isinstance(raw, dict), code, f"{name} must be an object")
    values = raw.get("values")
    _require(isinstance(values, dict), code, f"{name}.values must be an object")
    parsed = {str(key): _fraction(value, code, f"{name}[{key}]") for key, value in values.items()}
    _require(set(parsed) == expected_basis, code, f"{name} must assign exactly {sorted(expected_basis)}")
    return parsed


def _dependencies(payload: dict[str, Any]) -> dict[str, Any]:
    deps = payload.get("dependencies")
    _require(isinstance(deps, dict), "IMR_2", "dependencies must be present")
    expected = {
        "nilpotency_family_id": 508,
        "bianchi_family_id": 509,
        "hodge_boundary_family_id": 510,
        "source_continuity_family_id": 511,
    }
    for key, value in expected.items():
        _require(deps.get(key) == value, "IMR_2", f"{key} must be {value}")
    _require(deps.get("hodge_boundary_verdict") in {"OBSERVER_BOUNDARY", "QA_NATIVE"}, "IMR_2", "hodge_boundary_verdict must be recognized")
    source = payload.get("source_attribution")
    _require(isinstance(source, str) and "Hatcher" in source and "Bossavit" in source and "Maxwell" in source, "IMR_2", "source_attribution must cite Hatcher, Bossavit, and Maxwell")
    return deps


def _validate_claim_policy(payload: dict[str, Any], deps: dict[str, Any]) -> None:
    policy = payload.get("claim_policy")
    _require(isinstance(policy, dict), "IMR_1", "claim_policy must be present")
    _require(policy.get("claims_inhomogeneous_recovery") is True, "IMR_1", "inhomogeneous recovery claim must be true")
    forbidden = [
        "claims_full_maxwell_derivation",
        "claims_electromagnetism",
        "claims_physical_field",
        "claims_physical_source_generation",
        "claims_free_energy_or_scalar_wave_energy",
    ]
    for key in forbidden:
        _require(policy.get(key) is False, "IMR_1", f"forbidden claim must be false: {key}")
    if deps.get("hodge_boundary_verdict") == "OBSERVER_BOUNDARY":
        _require(policy.get("claims_conditional_recovery") is True, "IMR_1", "observer-boundary branch must be conditional")
        _require(policy.get("claims_qa_native_inhomogeneous") is False, "IMR_1", "observer-boundary branch must not claim QA-native inhomogeneous derivation")
    else:
        _require(policy.get("claims_conditional_recovery") is False, "IMR_1", "QA-native branch must not be labeled observer-conditional")
        _require(policy.get("claims_qa_native_inhomogeneous") is True, "IMR_1", "QA-native branch must claim native inhomogeneous recovery only")


def _parse_basis(payload: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    basis = payload.get("cell_basis")
    _require(isinstance(basis, dict), "IMR_3", "cell_basis must be present")
    result: list[list[str]] = []
    for key in ("two_cells", "three_cells", "four_cells"):
        raw = basis.get(key)
        _require(isinstance(raw, list) and raw, "IMR_3", f"{key} must be non-empty")
        labels = [_as_str(item, "IMR_3", f"{key}[]") for item in raw]
        _require(len(set(labels)) == len(labels), "IMR_3", f"{key} labels must be unique")
        result.append(labels)
    return result[0], result[1], result[2]


def _star_matrix(payload: dict[str, Any], two_cells: set[str], three_cells: set[str]) -> list[tuple[str, str, Fraction]]:
    star = payload.get("star_operator")
    _require(isinstance(star, dict), "IMR_4", "star_operator must be present")
    entries = star.get("matrix_entries")
    _require(isinstance(entries, list) and entries, "IMR_4", "matrix_entries must be non-empty")
    seen: set[tuple[str, str]] = set()
    parsed: list[tuple[str, str, Fraction]] = []
    for i, raw in enumerate(entries):
        _require(isinstance(raw, dict), "IMR_4", f"matrix_entries[{i}] must be an object")
        row = _as_str(raw.get("row"), "IMR_4", f"matrix_entries[{i}].row")
        col = _as_str(raw.get("col"), "IMR_4", f"matrix_entries[{i}].col")
        _require(row in three_cells, "IMR_4", f"row {row} is not a declared three-cell")
        _require(col in two_cells, "IMR_4", f"col {col} is not a declared two-cell")
        _require((row, col) not in seen, "IMR_4", f"duplicate matrix entry {row},{col}")
        seen.add((row, col))
        parsed.append((row, col, _fraction(raw.get("value"), "IMR_4", f"matrix_entries[{i}].value")))
    _require(any(value != 0 for _, _, value in parsed), "IMR_4", "star matrix must not be zero")
    return parsed


def _parse_four_boundaries(payload: dict[str, Any], three_cells: set[str], four_cells: set[str]) -> dict[str, list[tuple[str, int]]]:
    raw_boundaries = payload.get("four_cell_boundaries")
    _require(isinstance(raw_boundaries, list) and raw_boundaries, "IMR_5", "four_cell_boundaries must be present")
    parsed: dict[str, list[tuple[str, int]]] = {}
    for i, raw in enumerate(raw_boundaries):
        _require(isinstance(raw, dict), "IMR_5", f"four_cell_boundaries[{i}] must be an object")
        cell_id = _as_str(raw.get("id"), "IMR_5", f"four_cell_boundaries[{i}].id")
        _require(cell_id in four_cells, "IMR_5", f"undeclared four-cell {cell_id}")
        _require(cell_id not in parsed, "IMR_5", f"duplicate four-cell boundary {cell_id}")
        boundary = raw.get("boundary")
        _require(isinstance(boundary, list) and boundary, "IMR_5", f"four-cell {cell_id} boundary must be non-empty")
        terms: list[tuple[str, int]] = []
        for j, term in enumerate(boundary):
            _require(isinstance(term, dict), "IMR_5", f"boundary term {j} must be an object")
            subcell = _as_str(term.get("three_cell"), "IMR_5", f"boundary[{j}].three_cell")
            sign = _as_int(term.get("sign"), "IMR_5", f"boundary[{j}].sign")
            _require(subcell in three_cells, "IMR_5", f"undeclared three-cell {subcell}")
            _require(sign in (-1, 1), "IMR_5", "boundary signs must be +/-1")
            terms.append((subcell, sign))
        parsed[cell_id] = terms
    _require(set(parsed) == four_cells, "IMR_5", "every four-cell needs exactly one boundary")
    return parsed


def _compute_star_f(entries: list[tuple[str, str, Fraction]], field_f: dict[str, Fraction], three_cells: set[str]) -> dict[str, Fraction]:
    out = {cell: Fraction(0) for cell in three_cells}
    for row, col, value in entries:
        out[row] += value * field_f[col]
    return out


def _delta(values: dict[str, Fraction], boundaries: dict[str, list[tuple[str, int]]]) -> dict[str, Fraction]:
    return {cell_id: sum(Fraction(sign) * values[subcell] for subcell, sign in boundary) for cell_id, boundary in boundaries.items()}


def _validate_conventions(payload: dict[str, Any], deps: dict[str, Any]) -> None:
    conventions = payload.get("recovery_conventions")
    _require(isinstance(conventions, dict), "IMR_8", "recovery_conventions must be present")
    _require(conventions.get("exterior_equation") == "delta(starF)=J", "IMR_8", "exterior equation must be delta(starF)=J")
    _require(conventions.get("sign_convention") in {"right_coboundary_positive", "declared_boundary_orientation"}, "IMR_8", "unsupported sign convention")
    _require(_fraction(conventions.get("unit_scale"), "IMR_8", "unit_scale") > 0, "IMR_8", "unit_scale must be positive")
    projection = conventions.get("observer_projection")
    _require(isinstance(projection, dict), "IMR_8", "observer_projection must be present")
    _require(_as_int(projection.get("theorem_nt_crossing_count"), "IMR_8", "theorem_nt_crossing_count") <= 2, "IMR_8", "Theorem NT crossing count exceeds budget")
    _require(projection.get("uses_numeric_field_values") is False, "IMR_8", "M4 must not evaluate numeric physical fields")
    if deps.get("hodge_boundary_verdict") == "OBSERVER_BOUNDARY":
        _require(projection.get("projection_status") == "conditional_observer_recovery", "IMR_8", "observer-boundary branch projection_status mismatch")
    else:
        _require(projection.get("projection_status") == "qa_native_symbolic_recovery", "IMR_8", "QA-native branch projection_status mismatch")


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _reject_float(payload, "IMR_9")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "SCHEMA", "wrong schema_version")
    _require(payload.get("candidate_family_id") == FAMILY_ID, "SCHEMA", "wrong candidate_family_id")
    _require(payload.get("cert_slug") == CERT_SLUG, "SCHEMA", "wrong cert_slug")
    deps = _dependencies(payload)
    _validate_claim_policy(payload, deps)
    two_cells, three_cells, four_cells = _parse_basis(payload)
    two_set, three_set, four_set = set(two_cells), set(three_cells), set(four_cells)
    entries = _star_matrix(payload, two_set, three_set)
    field_f = _fraction_values(payload.get("field_F_2cochain"), two_set, "IMR_6", "field_F_2cochain")
    declared_star_f = _fraction_values(payload.get("declared_starF_3cochain"), three_set, "IMR_6", "declared_starF_3cochain")
    computed_star_f = _compute_star_f(entries, field_f, three_set)
    _require(declared_star_f == computed_star_f, "IMR_6", f"declared starF mismatch: {declared_star_f} != {computed_star_f}")
    boundaries = _parse_four_boundaries(payload, three_set, four_set)
    declared_j = _fraction_values(payload.get("declared_source_J"), four_set, "IMR_7", "declared_source_J")
    computed_j = _delta(computed_star_f, boundaries)
    _require(declared_j == computed_j, "IMR_7", f"declared J mismatch: {declared_j} != {computed_j}")
    _validate_conventions(payload, deps)
    return {
        "ok": True,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "hodge_boundary_verdict": deps.get("hodge_boundary_verdict"),
        "n_two_cells": len(two_cells),
        "n_four_cells": len(four_cells),
        "checks": ["IMR_1", "IMR_2", "IMR_3", "IMR_4", "IMR_5", "IMR_6", "IMR_7", "IMR_8", "IMR_9"],
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
    parser = argparse.ArgumentParser(description="Validate QA Inhomogeneous Maxwell Recovery cert fixtures")
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
