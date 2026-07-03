#!/usr/bin/env python3
"""QA Field 2-Form Bianchi Cert validator.

Candidate family ID: [509].

Primary mathematical anchor: the chain-complex identity d(d(A)) = 0, dual to
boundary(boundary(c)) = 0. Reference: Allen Hatcher, Algebraic Topology (2002),
Ch. 2, singular homology boundary operator; ISBN 978-0-521-79540-1. QA project
context: docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md M1, building directly on
cert [508].

Claim: for a finite declared oriented QA cell complex with integer edge
potential 1-cochain A, the field 2-cochain F = delta(A) on faces is recomputed
exactly, and delta(F) vanishes exactly on every declared volume.

Allowed claim wording: QA derives the homogeneous Maxwell/Bianchi identity for
the declared exact field carrier. This is NOT full Maxwell, NOT the
inhomogeneous equations, NOT electromagnetism, and NOT a physical field cert.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "QA_FIELD_2FORM_BIANCHI_CERT.v1"
CERT_SLUG = "qa_field_2form_bianchi_cert_v1"
FAMILY_ID = 509


class ValidationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Edge:
    edge_id: int
    tail: int
    head: int


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


def _validate_claim_policy(payload: dict[str, Any]) -> None:
    policy = payload.get("claim_policy")
    _require(isinstance(policy, dict), "BIA_1", "claim_policy must be present")

    _require(policy.get("claims_exact_field_2form_bianchi") is True, "BIA_1", "claims_exact_field_2form_bianchi must be true")
    _require(policy.get("claims_homogeneous_maxwell_bianchi") is True, "BIA_1", "claims_homogeneous_maxwell_bianchi must be true")

    forbidden_true = [
        "claims_full_maxwell_derivation",
        "claims_inhomogeneous_maxwell",
        "claims_constitutive_hodge",
        "claims_source_law",
        "claims_electromagnetism",
        "claims_physical_field",
        "claims_whittaker_operator",
        "claims_observer_field_values",
        "claims_scalar_wave_energy_physics",
    ]
    for key in forbidden_true:
        _require(policy.get(key) is False, "BIA_1", f"forbidden claim must be false: {key}")


def _parse_complex(payload: dict[str, Any]) -> tuple[set[int], dict[int, Edge], list[dict[str, Any]], list[dict[str, Any]]]:
    complex_payload = payload.get("cell_complex")
    _require(isinstance(complex_payload, dict), "BIA_2", "cell_complex must be present")

    raw_vertices = complex_payload.get("vertices")
    _require(isinstance(raw_vertices, list) and raw_vertices, "BIA_2", "vertices must be a non-empty list")
    vertices: set[int] = set()
    for i, raw_vertex in enumerate(raw_vertices):
        vertex = _as_int(raw_vertex, "BIA_2", f"vertices[{i}]")
        _require(vertex > 0, "BIA_2", "vertex labels must be positive QA labels")
        _require(vertex not in vertices, "BIA_2", f"duplicate vertex label: {vertex}")
        vertices.add(vertex)

    raw_edges = complex_payload.get("edges")
    _require(isinstance(raw_edges, list) and raw_edges, "BIA_3", "edges must be a non-empty list")
    edges: dict[int, Edge] = {}
    for i, raw_edge in enumerate(raw_edges):
        _require(isinstance(raw_edge, dict), "BIA_3", f"edges[{i}] must be an object")
        edge_id = _as_int(raw_edge.get("id"), "BIA_3", f"edges[{i}].id")
        tail = _as_int(raw_edge.get("tail"), "BIA_3", f"edges[{i}].tail")
        head = _as_int(raw_edge.get("head"), "BIA_3", f"edges[{i}].head")
        _require(edge_id > 0, "BIA_3", "edge labels must be positive")
        _require(edge_id not in edges, "BIA_3", f"duplicate edge label: {edge_id}")
        _require(tail in vertices and head in vertices, "BIA_3", f"edge {edge_id} endpoints must be declared vertices")
        _require(tail != head, "BIA_3", f"edge {edge_id} must not be a loop")
        edges[edge_id] = Edge(edge_id=edge_id, tail=tail, head=head)

    raw_faces = complex_payload.get("faces")
    _require(isinstance(raw_faces, list) and raw_faces, "BIA_4", "faces must be a non-empty list")
    seen_faces: set[int] = set()
    for i, raw_face in enumerate(raw_faces):
        _require(isinstance(raw_face, dict), "BIA_4", f"faces[{i}] must be an object")
        face_id = _as_int(raw_face.get("id"), "BIA_4", f"faces[{i}].id")
        _require(face_id > 0, "BIA_4", "face labels must be positive")
        _require(face_id not in seen_faces, "BIA_4", f"duplicate face label: {face_id}")
        seen_faces.add(face_id)
        boundary = raw_face.get("boundary")
        _require(isinstance(boundary, list) and len(boundary) >= 3, "BIA_4", f"face {face_id} needs at least 3 boundary terms")
        for j, term in enumerate(boundary):
            _require(isinstance(term, dict), "BIA_4", f"face {face_id} boundary[{j}] must be an object")
            edge_id = _as_int(term.get("edge"), "BIA_4", f"face {face_id} boundary[{j}].edge")
            sign = _as_int(term.get("sign"), "BIA_4", f"face {face_id} boundary[{j}].sign")
            _require(edge_id in edges, "BIA_4", f"face {face_id} references undeclared edge {edge_id}")
            _require(sign in (-1, 1), "BIA_4", f"face {face_id} boundary sign must be +/-1")

    raw_volumes = complex_payload.get("volumes")
    _require(isinstance(raw_volumes, list) and raw_volumes, "BIA_5", "volumes must be a non-empty list")
    seen_volumes: set[int] = set()
    for i, raw_volume in enumerate(raw_volumes):
        _require(isinstance(raw_volume, dict), "BIA_5", f"volumes[{i}] must be an object")
        volume_id = _as_int(raw_volume.get("id"), "BIA_5", f"volumes[{i}].id")
        _require(volume_id > 0, "BIA_5", "volume labels must be positive")
        _require(volume_id not in seen_volumes, "BIA_5", f"duplicate volume label: {volume_id}")
        seen_volumes.add(volume_id)
        boundary = raw_volume.get("boundary")
        _require(isinstance(boundary, list) and len(boundary) >= 4, "BIA_5", f"volume {volume_id} needs at least 4 boundary faces")
        for j, term in enumerate(boundary):
            _require(isinstance(term, dict), "BIA_5", f"volume {volume_id} boundary[{j}] must be an object")
            face_id = _as_int(term.get("face"), "BIA_5", f"volume {volume_id} boundary[{j}].face")
            sign = _as_int(term.get("sign"), "BIA_5", f"volume {volume_id} boundary[{j}].sign")
            _require(face_id in seen_faces, "BIA_5", f"volume {volume_id} references undeclared face {face_id}")
            _require(sign in (-1, 1), "BIA_5", f"volume {volume_id} boundary sign must be +/-1")

    return vertices, edges, raw_faces, raw_volumes


def _parse_edge_potential(payload: dict[str, Any], edge_ids: set[int]) -> dict[int, int]:
    raw = payload.get("edge_potential_A")
    _require(isinstance(raw, dict), "BIA_6", "edge_potential_A must be an object")
    values = raw.get("values")
    _require(isinstance(values, dict), "BIA_6", "edge_potential_A.values must be an object")
    parsed: dict[int, int] = {}
    for raw_key, raw_value in values.items():
        try:
            edge_id = int(raw_key)
        except Exception as exc:  # noqa: BLE001 - re-raised with cert code
            raise ValidationError("BIA_6", f"edge_potential_A key is not an int string: {raw_key}") from exc
        parsed[edge_id] = _as_int(raw_value, "BIA_6", f"edge_potential_A[{raw_key}]")
    _require(set(parsed) == edge_ids, "BIA_6", "edge_potential_A must assign every edge exactly once")
    return parsed


def _face_field(face: dict[str, Any], edge_potential: dict[int, int]) -> int:
    total = 0
    for term in face["boundary"]:
        total += term["sign"] * edge_potential[term["edge"]]
    return total


def _volume_bianchi(volume: dict[str, Any], face_fields: dict[int, int]) -> int:
    total = 0
    for term in volume["boundary"]:
        total += term["sign"] * face_fields[term["face"]]
    return total


def _declared_face_fields(payload: dict[str, Any]) -> dict[int, int] | None:
    raw = payload.get("declared_field_F")
    if raw is None:
        return None
    _require(isinstance(raw, dict), "BIA_7", "declared_field_F must be an object when present")
    values = raw.get("values")
    _require(isinstance(values, dict), "BIA_7", "declared_field_F.values must be an object")
    parsed: dict[int, int] = {}
    for raw_key, raw_value in values.items():
        try:
            face_id = int(raw_key)
        except Exception as exc:  # noqa: BLE001 - re-raised with cert code
            raise ValidationError("BIA_7", f"declared_field_F key is not an int string: {raw_key}") from exc
        parsed[face_id] = _as_int(raw_value, "BIA_7", f"declared_field_F[{raw_key}]")
    return parsed


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "SCHEMA", "wrong schema_version")
    _require(payload.get("candidate_family_id") == FAMILY_ID, "SCHEMA", "wrong candidate_family_id")
    _require(payload.get("cert_slug") == CERT_SLUG, "SCHEMA", "wrong cert_slug")
    source = payload.get("source_attribution")
    _require(isinstance(source, str) and ("Hatcher" in source or "Algebraic Topology" in source), "BIA_1", "source_attribution must cite the chain-complex identity")
    _validate_claim_policy(payload)

    _vertices, edges, faces, volumes = _parse_complex(payload)
    edge_potential = _parse_edge_potential(payload, set(edges))

    face_fields = {face["id"]: _face_field(face, edge_potential) for face in faces}
    declared = _declared_face_fields(payload)
    if declared is not None:
        _require(set(declared) == set(face_fields), "BIA_7", "declared_field_F must assign every face exactly once")
        mismatches = {
            face_id: {"declared": declared[face_id], "computed": computed}
            for face_id, computed in sorted(face_fields.items())
            if declared[face_id] != computed
        }
        _require(not mismatches, "BIA_7", f"declared F does not match delta(A): {mismatches}")

    nonzero: dict[int, int] = {}
    for volume in volumes:
        value = _volume_bianchi(volume, face_fields)
        if value != 0:
            nonzero[volume["id"]] = value
    _require(not nonzero, "BIA_8", f"delta(F) nonzero on volumes: {nonzero}")

    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "checks": ["BIA_1", "BIA_2", "BIA_3", "BIA_4", "BIA_5", "BIA_6", "BIA_7", "BIA_8"],
        "n_edges": len(edges),
        "n_faces": len(faces),
        "n_volumes": len(volumes),
        "face_fields": {str(k): v for k, v in sorted(face_fields.items())},
    }


def validate_file(path: str) -> dict[str, Any]:
    return validate_payload(_load_json(path))


def _fixture_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def run_self_test() -> dict[str, Any]:
    fixtures = sorted(
        os.path.join(_fixture_dir(), name)
        for name in os.listdir(_fixture_dir())
        if name.endswith(".json")
    )
    passes = 0
    fails = 0
    failures: list[dict[str, Any]] = []
    for path in fixtures:
        payload = _load_json(path)
        expected = payload.get("expected_result")
        try:
            result = validate_payload(payload)
            if expected == "PASS":
                passes += 1
            else:
                failures.append({"fixture": os.path.basename(path), "expected": expected, "actual": result})
        except ValidationError as exc:
            if expected == "FAIL" and payload.get("expected_error_code") == exc.code:
                fails += 1
            else:
                failures.append({
                    "fixture": os.path.basename(path),
                    "expected": expected,
                    "expected_error_code": payload.get("expected_error_code"),
                    "actual_error_code": exc.code,
                    "message": exc.message,
                })
    return {
        "ok": not failures,
        "schema_version": SCHEMA_VERSION,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "n_pass": passes,
        "n_fail": fails,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            result = run_self_test()
            print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            return 0 if result.get("ok") is True else 1
        if not args.file:
            raise ValidationError("SCHEMA", "missing file argument")
        result = validate_file(args.file)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except ValidationError as exc:
        print(json.dumps({"ok": False, "error_code": exc.code, "message": exc.message}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
