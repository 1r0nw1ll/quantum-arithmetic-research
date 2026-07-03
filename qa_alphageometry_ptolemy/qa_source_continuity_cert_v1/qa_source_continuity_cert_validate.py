#!/usr/bin/env python3
"""QA Source Continuity Cert validator.

Candidate family ID: [511].

Primary mathematical anchors: Allen Hatcher, Algebraic Topology (2002), Ch. 2,
ISBN 978-0-521-79540-1, for d(d)=0 / boundary(boundary)=0; Alain Bossavit,
Computational Electromagnetism (1998), ISBN 978-0-12-118710-1, for the discrete
Hodge/constitutive boundary context. QA project context:
docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md M3, building on certs [508],
[509], and [510].

Claim: for a declared finite exact cochain stack, if starF is a declared
integer/rational 3-cochain and J = delta(starF) is recomputed as a 4-cochain,
then delta(J) vanishes exactly on every declared 5-cell.

This cert proves source continuity as a nilpotency consequence only. It does
NOT derive sources, NOT derive inhomogeneous Maxwell, NOT derive full Maxwell,
NOT prove electromagnetism, and NOT claim physical charge/current generation.
"""

from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from typing import Any


SCHEMA_VERSION = "QA_SOURCE_CONTINUITY_CERT.v1"
CERT_SLUG = "qa_source_continuity_cert_v1"
FAMILY_ID = 511


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


def _as_int(value: Any, code: str, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code, f"{name} must be an integer")
    return value


def _fraction(raw: Any, code: str, name: str) -> Fraction:
    _require(isinstance(raw, dict), code, f"{name} must be an object")
    num = _as_int(raw.get("num"), code, f"{name}.num")
    den = _as_int(raw.get("den"), code, f"{name}.den")
    _require(den > 0, code, f"{name}.den must be positive")
    return Fraction(num, den)


def _validate_claim_policy(payload: dict[str, Any]) -> None:
    policy = payload.get("claim_policy")
    _require(isinstance(policy, dict), "SRC_1", "claim_policy must be present")
    _require(policy.get("claims_source_continuity") is True, "SRC_1", "source-continuity claim must be true")
    forbidden = [
        "claims_source_generation",
        "claims_inhomogeneous_maxwell",
        "claims_full_maxwell_derivation",
        "claims_electromagnetism",
        "claims_physical_charge_current",
        "claims_physical_field",
        "claims_qa_native_hodge",
    ]
    for key in forbidden:
        _require(policy.get(key) is False, "SRC_1", f"forbidden claim must be false: {key}")


def _validate_dependencies(payload: dict[str, Any]) -> None:
    deps = payload.get("dependencies")
    _require(isinstance(deps, dict), "SRC_2", "dependencies must be present")
    expected = {
        "nilpotency_family_id": 508,
        "bianchi_family_id": 509,
        "hodge_boundary_family_id": 510,
    }
    for key, value in expected.items():
        _require(deps.get(key) == value, "SRC_2", f"{key} must be {value}")
    _require(deps.get("hodge_boundary_verdict") == "OBSERVER_BOUNDARY", "SRC_2", "v1 source continuity must consume observer-boundary Hodge verdict")
    source = payload.get("source_attribution")
    _require(isinstance(source, str) and "Hatcher" in source and "Bossavit" in source, "SRC_2", "source_attribution must cite Hatcher and Bossavit")


def _parse_complex(payload: dict[str, Any]) -> tuple[set[int], dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    complex_payload = payload.get("cell_complex")
    _require(isinstance(complex_payload, dict), "SRC_3", "cell_complex must be present")
    raw_3cells = complex_payload.get("three_cells")
    raw_4cells = complex_payload.get("four_cells")
    raw_5cells = complex_payload.get("five_cells")
    _require(isinstance(raw_3cells, list) and raw_3cells, "SRC_3", "three_cells must be non-empty")
    three_cells = set()
    for i, raw in enumerate(raw_3cells):
        cell_id = _as_int(raw, "SRC_3", f"three_cells[{i}]")
        _require(cell_id > 0 and cell_id not in three_cells, "SRC_3", "three-cell labels must be unique positive integers")
        three_cells.add(cell_id)

    four_boundaries: dict[int, list[tuple[int, int]]] = {}
    _require(isinstance(raw_4cells, list) and raw_4cells, "SRC_4", "four_cells must be non-empty")
    for i, raw in enumerate(raw_4cells):
        _require(isinstance(raw, dict), "SRC_4", f"four_cells[{i}] must be an object")
        cell_id = _as_int(raw.get("id"), "SRC_4", f"four_cells[{i}].id")
        _require(cell_id > 0 and cell_id not in four_boundaries, "SRC_4", "four-cell labels must be unique positive integers")
        boundary = raw.get("boundary")
        _require(isinstance(boundary, list) and boundary, "SRC_4", f"four-cell {cell_id} needs boundary terms")
        terms = []
        for j, term in enumerate(boundary):
            _require(isinstance(term, dict), "SRC_4", f"four-cell {cell_id} boundary[{j}] must be an object")
            subcell = _as_int(term.get("three_cell"), "SRC_4", f"four-cell {cell_id} boundary[{j}].three_cell")
            sign = _as_int(term.get("sign"), "SRC_4", f"four-cell {cell_id} boundary[{j}].sign")
            _require(subcell in three_cells, "SRC_4", f"four-cell {cell_id} references undeclared three-cell {subcell}")
            _require(sign in (-1, 1), "SRC_4", "boundary sign must be +/-1")
            terms.append((subcell, sign))
        four_boundaries[cell_id] = terms

    five_boundaries: dict[int, list[tuple[int, int]]] = {}
    _require(isinstance(raw_5cells, list) and raw_5cells, "SRC_5", "five_cells must be non-empty")
    for i, raw in enumerate(raw_5cells):
        _require(isinstance(raw, dict), "SRC_5", f"five_cells[{i}] must be an object")
        cell_id = _as_int(raw.get("id"), "SRC_5", f"five_cells[{i}].id")
        _require(cell_id > 0 and cell_id not in five_boundaries, "SRC_5", "five-cell labels must be unique positive integers")
        boundary = raw.get("boundary")
        _require(isinstance(boundary, list) and boundary, "SRC_5", f"five-cell {cell_id} needs boundary terms")
        terms = []
        for j, term in enumerate(boundary):
            _require(isinstance(term, dict), "SRC_5", f"five-cell {cell_id} boundary[{j}] must be an object")
            subcell = _as_int(term.get("four_cell"), "SRC_5", f"five-cell {cell_id} boundary[{j}].four_cell")
            sign = _as_int(term.get("sign"), "SRC_5", f"five-cell {cell_id} boundary[{j}].sign")
            _require(subcell in four_boundaries, "SRC_5", f"five-cell {cell_id} references undeclared four-cell {subcell}")
            _require(sign in (-1, 1), "SRC_5", "boundary sign must be +/-1")
            terms.append((subcell, sign))
        five_boundaries[cell_id] = terms
    return three_cells, four_boundaries, five_boundaries


def _parse_starf(payload: dict[str, Any], three_cells: set[int]) -> dict[int, Fraction]:
    raw = payload.get("starF_3cochain")
    _require(isinstance(raw, dict), "SRC_6", "starF_3cochain must be present")
    values = raw.get("values")
    _require(isinstance(values, dict), "SRC_6", "starF_3cochain.values must be an object")
    parsed: dict[int, Fraction] = {}
    for key, value in values.items():
        try:
            cell_id = int(key)
        except ValueError as exc:
            raise ValidationError("SRC_6", f"starF key is not an int string: {key}") from exc
        parsed[cell_id] = _fraction(value, "SRC_6", f"starF[{key}]")
    _require(set(parsed) == three_cells, "SRC_6", "starF must assign every three-cell exactly once")
    return parsed


def _compute_delta(values: dict[int, Fraction], boundaries: dict[int, list[tuple[int, int]]]) -> dict[int, Fraction]:
    return {cell_id: sum(Fraction(sign) * values[subcell] for subcell, sign in boundary) for cell_id, boundary in boundaries.items()}


def _validate_declared_j(payload: dict[str, Any], computed: dict[int, Fraction]) -> None:
    raw = payload.get("declared_source_J")
    if raw is None:
        return
    _require(isinstance(raw, dict), "SRC_7", "declared_source_J must be an object")
    values = raw.get("values")
    _require(isinstance(values, dict), "SRC_7", "declared_source_J.values must be an object")
    declared: dict[int, Fraction] = {}
    for key, value in values.items():
        try:
            cell_id = int(key)
        except ValueError as exc:
            raise ValidationError("SRC_7", f"declared_source_J key is not an int string: {key}") from exc
        declared[cell_id] = _fraction(value, "SRC_7", f"declared_source_J[{key}]")
    _require(set(declared) == set(computed), "SRC_7", "declared_source_J must assign every four-cell exactly once")
    mismatches = {k: (declared[k], computed[k]) for k in computed if declared[k] != computed[k]}
    _require(not mismatches, "SRC_7", f"declared source J does not match delta(starF): {mismatches}")


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "SCHEMA", "wrong schema_version")
    _require(payload.get("candidate_family_id") == FAMILY_ID, "SCHEMA", "wrong candidate_family_id")
    _require(payload.get("cert_slug") == CERT_SLUG, "SCHEMA", "wrong cert_slug")
    _validate_claim_policy(payload)
    _validate_dependencies(payload)
    three_cells, four_boundaries, five_boundaries = _parse_complex(payload)
    star_f = _parse_starf(payload, three_cells)
    source_j = _compute_delta(star_f, four_boundaries)
    _validate_declared_j(payload, source_j)
    continuity = _compute_delta(source_j, five_boundaries)
    nonzero = {cell_id: value for cell_id, value in continuity.items() if value != 0}
    _require(not nonzero, "SRC_8", f"delta(J) nonzero on five-cells: {nonzero}")
    return {
        "ok": True,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "n_four_cells": len(four_boundaries),
        "n_five_cells": len(five_boundaries),
        "checks": ["SRC_1", "SRC_2", "SRC_3", "SRC_4", "SRC_5", "SRC_6", "SRC_7", "SRC_8"],
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
    parser = argparse.ArgumentParser(description="Validate QA Source Continuity cert fixtures")
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
