#!/usr/bin/env python3
"""QA Discrete Exterior Nilpotency Cert validator.

Candidate family ID: [508].

Primary source / mathematical anchor: the chain-complex identity
boundary(boundary(c)) = 0 and its dual coboundary identity delta(delta(f)) = 0
as used in standard algebraic topology. Reference: Allen Hatcher, Algebraic
Topology (2002), Ch. 2, singular homology boundary operator; ISBN
978-0-521-79540-1. QA project context:
docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md M0.

Claim: for a finite declared oriented QA cell complex with integer vertex,
edge, and face labels, the validator checks exact integer cancellation of
boundary-of-boundary on every face and exact integer cancellation of the dual
coboundary applied twice to declared 0-cochain witnesses.

This is NOT a Maxwell derivation, NOT electromagnetism, and NOT a physical
field cert. It is only the first combinatorial proof obligation in the Maxwell
derivation program.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "QA_DISCRETE_EXTERIOR_NILPOTENCY_CERT.v1"
CERT_SLUG = "qa_discrete_exterior_nilpotency_cert_v1"
FAMILY_ID = 508


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
    _require(isinstance(policy, dict), "DN_1", "claim_policy must be present")

    forbidden_true = [
        "claims_maxwell_derivation",
        "claims_full_maxwell",
        "claims_electromagnetism",
        "claims_physical_field",
        "claims_whittaker_operator",
        "claims_observer_boundary",
        "claims_continuum_limit",
        "claims_scalar_wave_energy_physics",
    ]
    for key in forbidden_true:
        _require(policy.get(key) is False, "DN_1", f"forbidden claim must be false: {key}")

    allowed = policy.get("claims_discrete_nilpotency_only")
    _require(allowed is True, "DN_1", "claims_discrete_nilpotency_only must be true")


def _parse_complex(payload: dict[str, Any]) -> tuple[set[int], dict[int, Edge], list[dict[str, Any]]]:
    complex_payload = payload.get("cell_complex")
    _require(isinstance(complex_payload, dict), "DN_2", "cell_complex must be present")

    raw_vertices = complex_payload.get("vertices")
    _require(isinstance(raw_vertices, list) and raw_vertices, "DN_2", "vertices must be a non-empty list")
    vertices: set[int] = set()
    for i, raw_vertex in enumerate(raw_vertices):
        vertex = _as_int(raw_vertex, "DN_2", f"vertices[{i}]")
        _require(vertex > 0, "DN_2", "vertex labels must be positive QA labels")
        _require(vertex not in vertices, "DN_2", f"duplicate vertex label: {vertex}")
        vertices.add(vertex)

    raw_edges = complex_payload.get("edges")
    _require(isinstance(raw_edges, list) and raw_edges, "DN_3", "edges must be a non-empty list")
    edges: dict[int, Edge] = {}
    for i, raw_edge in enumerate(raw_edges):
        _require(isinstance(raw_edge, dict), "DN_3", f"edges[{i}] must be an object")
        edge_id = _as_int(raw_edge.get("id"), "DN_3", f"edges[{i}].id")
        tail = _as_int(raw_edge.get("tail"), "DN_3", f"edges[{i}].tail")
        head = _as_int(raw_edge.get("head"), "DN_3", f"edges[{i}].head")
        _require(edge_id > 0, "DN_3", "edge labels must be positive")
        _require(edge_id not in edges, "DN_3", f"duplicate edge label: {edge_id}")
        _require(tail in vertices and head in vertices, "DN_3", f"edge {edge_id} endpoints must be declared vertices")
        _require(tail != head, "DN_3", f"edge {edge_id} must not be a loop")
        edges[edge_id] = Edge(edge_id=edge_id, tail=tail, head=head)

    raw_faces = complex_payload.get("faces")
    _require(isinstance(raw_faces, list) and raw_faces, "DN_4", "faces must be a non-empty list")
    seen_faces: set[int] = set()
    for i, raw_face in enumerate(raw_faces):
        _require(isinstance(raw_face, dict), "DN_4", f"faces[{i}] must be an object")
        face_id = _as_int(raw_face.get("id"), "DN_4", f"faces[{i}].id")
        _require(face_id > 0, "DN_4", "face labels must be positive")
        _require(face_id not in seen_faces, "DN_4", f"duplicate face label: {face_id}")
        seen_faces.add(face_id)
        boundary = raw_face.get("boundary")
        _require(isinstance(boundary, list) and len(boundary) >= 3, "DN_4", f"face {face_id} needs at least 3 boundary terms")
        for j, term in enumerate(boundary):
            _require(isinstance(term, dict), "DN_4", f"face {face_id} boundary[{j}] must be an object")
            edge_id = _as_int(term.get("edge"), "DN_4", f"face {face_id} boundary[{j}].edge")
            sign = _as_int(term.get("sign"), "DN_4", f"face {face_id} boundary[{j}].sign")
            _require(edge_id in edges, "DN_4", f"face {face_id} references undeclared edge {edge_id}")
            _require(sign in (-1, 1), "DN_4", f"face {face_id} boundary sign must be +/-1")

    return vertices, edges, raw_faces


def _boundary_of_boundary(face: dict[str, Any], edges: dict[int, Edge]) -> dict[int, int]:
    accum: dict[int, int] = {}
    for term in face["boundary"]:
        edge = edges[term["edge"]]
        sign = term["sign"]
        accum[edge.tail] = accum.get(edge.tail, 0) - sign
        accum[edge.head] = accum.get(edge.head, 0) + sign
    return {vertex: coeff for vertex, coeff in sorted(accum.items()) if coeff != 0}


def _delta0(cochain: dict[int, int], edge: Edge) -> int:
    return cochain[edge.head] - cochain[edge.tail]


def _delta1_of_delta0(face: dict[str, Any], edges: dict[int, Edge], cochain: dict[int, int]) -> int:
    total = 0
    for term in face["boundary"]:
        total += term["sign"] * _delta0(cochain, edges[term["edge"]])
    return total


def _parse_cochains(payload: dict[str, Any], vertices: set[int]) -> list[dict[int, int]]:
    raw_cochains = payload.get("zero_cochain_witnesses")
    _require(isinstance(raw_cochains, list) and raw_cochains, "DN_6", "zero_cochain_witnesses must be non-empty")

    witnesses: list[dict[int, int]] = []
    for i, raw_cochain in enumerate(raw_cochains):
        _require(isinstance(raw_cochain, dict), "DN_6", f"zero_cochain_witnesses[{i}] must be an object")
        values = raw_cochain.get("values")
        _require(isinstance(values, dict), "DN_6", f"zero_cochain_witnesses[{i}].values must be an object")
        parsed: dict[int, int] = {}
        for raw_key, raw_value in values.items():
            try:
                vertex = int(raw_key)
            except Exception as exc:  # noqa: BLE001 - schema error re-raised with cert code
                raise ValidationError("DN_6", f"cochain vertex key is not an int string: {raw_key}") from exc
            coeff = _as_int(raw_value, "DN_6", f"cochain[{raw_key}]")
            parsed[vertex] = coeff
        _require(set(parsed) == vertices, "DN_6", "each zero-cochain witness must assign every vertex exactly once")
        witnesses.append(parsed)
    return witnesses


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "SCHEMA", "wrong schema_version")
    _require(payload.get("candidate_family_id") == FAMILY_ID, "SCHEMA", "wrong candidate_family_id")
    _require(payload.get("cert_slug") == CERT_SLUG, "SCHEMA", "wrong cert_slug")
    source = payload.get("source_attribution")
    _require(isinstance(source, str) and ("Hatcher" in source or "Algebraic Topology" in source), "DN_1", "source_attribution must cite the chain-complex identity")
    _validate_claim_policy(payload)

    vertices, edges, faces = _parse_complex(payload)

    nonzero_boundaries: dict[int, dict[int, int]] = {}
    for face in faces:
        residue = _boundary_of_boundary(face, edges)
        if residue:
            nonzero_boundaries[face["id"]] = residue
    _require(not nonzero_boundaries, "DN_5", f"boundary-of-boundary not zero: {nonzero_boundaries}")

    cochains = _parse_cochains(payload, vertices)
    nonzero_coboundaries: list[dict[str, Any]] = []
    for witness_index, cochain in enumerate(cochains):
        for face in faces:
            value = _delta1_of_delta0(face, edges, cochain)
            if value != 0:
                nonzero_coboundaries.append({"witness_index": witness_index, "face": face["id"], "value": value})
    _require(not nonzero_coboundaries, "DN_7", f"delta(delta(f)) not zero: {nonzero_coboundaries}")

    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "candidate_family_id": FAMILY_ID,
        "cert_slug": CERT_SLUG,
        "checks": ["DN_1", "DN_2", "DN_3", "DN_4", "DN_5", "DN_6", "DN_7"],
        "n_vertices": len(vertices),
        "n_edges": len(edges),
        "n_faces": len(faces),
        "n_zero_cochain_witnesses": len(cochains),
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
