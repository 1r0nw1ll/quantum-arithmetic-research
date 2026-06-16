#!/usr/bin/env python3
"""
validator.py

QA_ARTEXPLORER_SCENE_ADAPTER.v1 validator (Machine Tract).

Adapter cert family that ingests ARTexplorer exported JSON scene data and
certifies typed geometric objects with RT (Rational Trigonometry) invariants.

ARTexplorer: https://github.com/arossti/ARTexplorer
- RT-pure quadrance + spread pipeline; defers sqrt until GPU render boundary
- Save/Load geometry configurations (JSON)
- Quadray (WXYZ) and Cartesian (XYZ) coordinate systems

Gates:
  1) Schema validity (jsonschema Draft-07)
  2) Determinism contract + invariant_diff presence (typed obstruction)
  3) Typed formation + illegal normalization detection
  4) Base algebra adequacy
  5) Deterministic step hash verification + RT invariant recomputation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _validate_schema(obj: Dict[str, Any]) -> None:
    schema = _load_json(_schema_path())
    try:
        import jsonschema
    except ModuleNotFoundError:
        _validate_schema_minimal(obj, schema)
        return
    jsonschema.validate(instance=obj, schema=schema)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expect_type(value: Any, expected_type: type, path: str) -> None:
    _expect(isinstance(value, expected_type), f"{path} must be {expected_type.__name__}")


def _expect_required(obj: Dict[str, Any], fields: List[str], path: str) -> None:
    missing = [field for field in fields if field not in obj]
    _expect(not missing, f"{path} missing required fields: {', '.join(missing)}")


def _expect_hash64(value: Any, path: str) -> None:
    _expect(
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value),
        f"{path} must be 64 lowercase hex chars",
    )


def _expect_number_list(value: Any, path: str, min_len: int, max_len: int) -> None:
    _expect(isinstance(value, list) and min_len <= len(value) <= max_len,
            f"{path} must be array length {min_len}..{max_len}")
    _expect(all(isinstance(item, (int, float)) for item in value), f"{path} entries must be numbers")


def _validate_schema_minimal(obj: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Dependency-free checker for this family's concrete schema.

    This preserves standalone validator behavior when jsonschema is absent.
    Semantic failures such as missing invariant_diff, degenerate triangles,
    illegal normalization, and law mismatches remain assigned to later gates.
    """

    _expect_type(obj, dict, "$")
    _expect_required(obj, schema.get("required", []), "$")
    _expect(obj.get("schema_version") == "v1", "schema_version must be v1")
    _expect(obj.get("cert_type") == "QA_ARTEXPLORER_SCENE_ADAPTER.v1", "cert_type mismatch")
    _expect_type(obj.get("cert_id"), str, "cert_id")
    _expect_type(obj.get("created_utc"), str, "created_utc")

    source = obj.get("source_semantics")
    _expect_type(source, dict, "source_semantics")
    _expect_required(source, ["upstream", "export"], "source_semantics")
    upstream = source["upstream"]
    _expect_type(upstream, dict, "source_semantics.upstream")
    _expect_required(upstream, ["name", "repo_url", "app_url"], "source_semantics.upstream")
    for field in ("name", "repo_url", "app_url"):
        _expect_type(upstream[field], str, f"source_semantics.upstream.{field}")
    export = source["export"]
    _expect_type(export, dict, "source_semantics.export")
    _expect_required(export, ["format"], "source_semantics.export")
    _expect(export["format"] == "artexplorer_scene_json", "source_semantics.export.format invalid")

    base = obj.get("base_algebra")
    _expect_type(base, dict, "base_algebra")
    _expect_required(base, ["name", "properties"], "base_algebra")
    _expect_type(base["name"], str, "base_algebra.name")
    props = base["properties"]
    _expect_type(props, dict, "base_algebra.properties")
    _expect_required(props, ["integral_domain", "field", "no_zero_divisors"], "base_algebra.properties")
    for field in ("integral_domain", "field", "no_zero_divisors"):
        _expect_type(props[field], bool, f"base_algebra.properties.{field}")

    scene = obj.get("scene")
    _expect_type(scene, dict, "scene")
    _expect_required(scene, ["scene_raw", "coordinate_system"], "scene")
    _expect(scene["coordinate_system"] in {"XYZ", "WXYZ"}, "scene.coordinate_system invalid")
    _expect_type(scene["scene_raw"], dict, "scene.scene_raw")
    if "scene_raw_sha256" in scene:
        _expect_hash64(scene["scene_raw_sha256"], "scene.scene_raw_sha256")
    if "zero_sum_normalized" in scene:
        _expect_type(scene["zero_sum_normalized"], bool, "scene.zero_sum_normalized")

    derivation = obj.get("derivation")
    _expect_type(derivation, dict, "derivation")
    _expect_required(derivation, ["parsed_objects", "steps"], "derivation")
    parsed = derivation["parsed_objects"]
    _expect(isinstance(parsed, list) and len(parsed) >= 1, "derivation.parsed_objects must be nonempty list")
    for idx, parsed_obj in enumerate(parsed):
        path = f"derivation.parsed_objects[{idx}]"
        _expect_type(parsed_obj, dict, path)
        _expect_required(parsed_obj, ["object_id", "object_type", "vertices"], path)
        _expect_type(parsed_obj["object_id"], str, f"{path}.object_id")
        _expect(parsed_obj["object_type"] in {"POINT_SET", "EDGE_SET", "TRIANGLE", "MESH", "POLYHEDRON"},
                f"{path}.object_type invalid")
        vertices = parsed_obj["vertices"]
        _expect(isinstance(vertices, list) and len(vertices) >= 1, f"{path}.vertices must be nonempty list")
        for vidx, vertex in enumerate(vertices):
            vpath = f"{path}.vertices[{vidx}]"
            _expect_type(vertex, dict, vpath)
            _expect_required(vertex, ["id", "coord"], vpath)
            _expect_type(vertex["id"], str, f"{vpath}.id")
            _expect_number_list(vertex["coord"], f"{vpath}.coord", 3, 4)
        if "faces" in parsed_obj:
            _expect_type(parsed_obj["faces"], list, f"{path}.faces")

    steps = derivation["steps"]
    _expect(isinstance(steps, list) and len(steps) >= 1, "derivation.steps must be nonempty list")
    allowed_moves = {
        "ART_PARSE_SCENE",
        "ART_NORMALIZE_WXYZ_ZERO_SUM",
        "ART_PROJECT_WXYZ_TO_XYZ",
        "RT_COMPUTE_TRIANGLE_INVARIANTS",
        "RT_VALIDATE_LAW_EQUATION",
    }
    for idx, step in enumerate(steps):
        spath = f"derivation.steps[{idx}]"
        _expect_type(step, dict, spath)
        _expect_required(step, ["step_id", "move_id", "inputs", "outputs", "step_hash_sha256"], spath)
        _expect_type(step["step_id"], str, f"{spath}.step_id")
        _expect(step["move_id"] in allowed_moves, f"{spath}.move_id invalid")
        _expect_type(step["inputs"], dict, f"{spath}.inputs")
        _expect_type(step["outputs"], dict, f"{spath}.outputs")
        _expect_hash64(step["step_hash_sha256"], f"{spath}.step_hash_sha256")

    result = obj.get("result")
    _expect_type(result, dict, "result")
    _expect_required(result, ["rt_invariants"], "result")
    rt_inv = result["rt_invariants"]
    _expect_type(rt_inv, dict, "result.rt_invariants")
    _expect_required(rt_inv, ["triangles"], "result.rt_invariants")
    _expect_type(rt_inv["triangles"], list, "result.rt_invariants.triangles")
    for idx, tri in enumerate(rt_inv["triangles"]):
        tpath = f"result.rt_invariants.triangles[{idx}]"
        _expect_type(tri, dict, tpath)
        _expect_required(tri, ["triangle_object_id", "Q", "s"], tpath)
        _expect_type(tri["triangle_object_id"], str, f"{tpath}.triangle_object_id")
        _expect_number_list(tri["Q"], f"{tpath}.Q", 3, 3)
        _expect_number_list(tri["s"], f"{tpath}.s", 3, 3)
    for optional_hash in ("object_digest_sha256", "rt_invariants_digest_sha256"):
        if optional_hash in result:
            _expect_hash64(result[optional_hash], f"result.{optional_hash}")

    dc = obj.get("determinism_contract")
    _expect_type(dc, dict, "determinism_contract")
    _expect_required(dc, ["canonical_json", "stable_sorting", "no_rng", "invariant_diff_defined"], "determinism_contract")
    for field in ("canonical_json", "stable_sorting", "no_rng", "invariant_diff_defined"):
        _expect_type(dc[field], bool, f"determinism_contract.{field}")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[i] - b[i] for i in range(min(len(a), len(b)))]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(a[i] * b[i] for i in range(min(len(a), len(b))))


def _quadrance_xyz(pa: List[float], pb: List[float]) -> float:
    d = _vec_sub(pb[:3], pa[:3])
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


def _cross_mag_sq(a: List[float], b: List[float], c: List[float]) -> float:
    """Squared magnitude of cross product (B-A) x (C-A). Zero iff collinear."""
    ab = _vec_sub(b[:3], a[:3])
    ac = _vec_sub(c[:3], a[:3])
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return cx * cx + cy * cy + cz * cz


def _spread_from_vectors(v1: List[float], v2: List[float]) -> Tuple[Optional[float], Optional[str]]:
    """s = 1 - (dot^2)/(Q1*Q2). Returns (spread, error_str)."""
    q1 = _dot(v1, v1)
    q2 = _dot(v2, v2)
    if q1 == 0.0 or q2 == 0.0:
        return None, "ZERO_DIVISOR_OBSTRUCTION"
    d = _dot(v1, v2)
    return 1.0 - (d * d) / (q1 * q2), None


def _is_zero_sum(coord: List[float]) -> bool:
    return len(coord) == 4 and abs(sum(coord)) < 1e-12


# ---------------------------------------------------------------------------
# RT law equation helpers
# ---------------------------------------------------------------------------

def _cross_law_residual(Q: List[float], s: List[float]) -> Tuple[float, float, float]:
    """Cross Law (RT_LAW_04): (Q1+Q2-Q3)^2 = 4*Q1*Q2*(1-s3).
    Returns (lhs, rhs, residual)."""
    Q1, Q2, Q3 = Q
    s3 = s[2]
    lhs = (Q1 + Q2 - Q3) ** 2
    rhs = 4.0 * Q1 * Q2 * (1.0 - s3)
    return lhs, rhs, abs(lhs - rhs)


def _triple_spread_residual(s: List[float]) -> Tuple[float, float, float]:
    """Triple Spread Formula (RT_LAW_05):
    (s1+s2+s3)^2 = 2*(s1^2+s2^2+s3^2) + 4*s1*s2*s3.
    Returns (lhs, rhs, residual)."""
    s1, s2, s3 = s
    lhs = (s1 + s2 + s3) ** 2
    rhs = 2.0 * (s1 * s1 + s2 * s2 + s3 * s3) + 4.0 * s1 * s2 * s3
    return lhs, rhs, abs(lhs - rhs)


def _pythagoras_residual(Q: List[float], s: List[float]) -> Tuple[float, float, float]:
    """Pythagoras' theorem (RT_LAW_01): Q3 = Q1 + Q2  when s3 = 1.
    Returns (lhs, rhs, residual)."""
    Q1, Q2, Q3 = Q
    lhs = Q3
    rhs = Q1 + Q2
    return float(lhs), float(rhs), abs(lhs - rhs)


_LAW_DISPATCH = {
    "RT_LAW_04": _cross_law_residual,
    "RT_LAW_05": _triple_spread_residual,
    "RT_LAW_01": _pythagoras_residual,
}


def _verify_law_step(step: Dict[str, Any]) -> Optional[GateResult]:
    """Verify an RT_VALIDATE_LAW_EQUATION step. Returns GateResult on failure, None on pass."""
    inp = step.get("inputs", {})
    out = step.get("outputs", {})
    law_id = inp.get("law_id", "")
    Q = inp.get("Q")
    s = inp.get("s")

    if law_id not in _LAW_DISPATCH:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"Unknown law_id: {law_id}",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    compute_fn = _LAW_DISPATCH[law_id]
    if law_id == "RT_LAW_05":
        lhs, rhs, residual = compute_fn(s)
    elif law_id == "RT_LAW_01":
        lhs, rhs, residual = compute_fn(Q, s)
    else:
        lhs, rhs, residual = compute_fn(Q, s)

    REL_TOL = 1e-9
    denom = max(abs(lhs), abs(rhs), 1.0)
    if residual / denom > REL_TOL:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} equation not satisfied (residual={residual:.2e})",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id"),
             "law_id": law_id, "lhs": lhs, "rhs": rhs, "residual": residual})

    if out.get("verified") is not True:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} satisfied but outputs.verified != true",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    claimed_lhs = out.get("lhs")
    claimed_rhs = out.get("rhs")
    if claimed_lhs is not None and abs(claimed_lhs - lhs) / max(abs(lhs), 1.0) > REL_TOL:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} claimed lhs mismatch",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})
    if claimed_rhs is not None and abs(claimed_rhs - rhs) / max(abs(rhs), 1.0) > REL_TOL:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} claimed rhs mismatch",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    return None


def _get_triangle_verts(obj: Dict[str, Any]) -> Tuple[Optional[List[List[float]]], Optional[str]]:
    """Extract 3 vertex coords for a TRIANGLE object. Returns ([a,b,c], None) or (None, error)."""
    verts = obj.get("vertices", [])
    faces = obj.get("faces", [])
    vid_to_coord = {v["id"]: v["coord"] for v in verts}

    if faces:
        tri_ids = faces[0]
    elif len(verts) >= 3:
        tri_ids = [verts[0]["id"], verts[1]["id"], verts[2]["id"]]
    else:
        return None, "not enough vertices"

    try:
        coords = [vid_to_coord[tid] for tid in tri_ids]
    except KeyError as e:
        return None, f"face references unknown vertex id: {e}"
    return coords, None


def _compute_triangle_rt(a: List[float], b: List[float], c: List[float]) -> Tuple[List[float], List[float], Optional[str]]:
    """Compute Q=[Q(BC),Q(CA),Q(AB)] and s=[sA,sB,sC] for triangle ABC."""
    Q1 = _quadrance_xyz(b, c)
    Q2 = _quadrance_xyz(c, a)
    Q3 = _quadrance_xyz(a, b)

    ab = _vec_sub(b[:3], a[:3])
    ac = _vec_sub(c[:3], a[:3])
    bc = _vec_sub(c[:3], b[:3])
    ba = _vec_sub(a[:3], b[:3])
    ca = _vec_sub(a[:3], c[:3])
    cb = _vec_sub(b[:3], c[:3])

    sA, eA = _spread_from_vectors(ab, ac)
    if eA:
        return [Q1, Q2, Q3], [], eA
    sB, eB = _spread_from_vectors(bc, ba)
    if eB:
        return [Q1, Q2, Q3], [], eB
    sC, eC = _spread_from_vectors(ca, cb)
    if eC:
        return [Q1, Q2, Q3], [], eC

    return [Q1, Q2, Q3], [sA, sB, sC], None


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 -- Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.PASS,
            "Valid QA_ARTEXPLORER_SCENE_ADAPTER.v1 schema"))
    except Exception as e:
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.FAIL,
            f"Schema validation failed: {e}"))
        return results

    # Gate 2 -- Determinism contract + invariant_diff presence
    dc = obj.get("determinism_contract", {})
    if (
        dc.get("canonical_json") is not True
        or dc.get("stable_sorting") is not True
        or dc.get("no_rng") is not True
        or dc.get("invariant_diff_defined") is not True
    ):
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            "Determinism contract flags not strict",
            {"fail_type": "NONDETERMINISM_CONTRACT_VIOLATION"}))
        return results

    res = obj.get("result", {})
    if "invariant_diff" not in res or not isinstance(res.get("invariant_diff"), dict):
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            "result.invariant_diff missing or malformed",
            {"fail_type": "MISSING_INVARIANT_DIFF"}))
        return results

    results.append(GateResult(
        "gate_2_determinism_contract", GateStatus.PASS,
        "Determinism contract strict + invariant_diff present"))

    # Gate 3 -- Typed formation + illegal normalization
    scene = obj.get("scene", {})
    coord_sys = scene.get("coordinate_system")
    zero_sum_flag = scene.get("zero_sum_normalized", False)

    parsed = obj.get("derivation", {}).get("parsed_objects", [])
    if not parsed:
        results.append(GateResult(
            "gate_3_typed_formation", GateStatus.FAIL,
            "derivation.parsed_objects empty",
            {"fail_type": "SCENE_PARSE_ERROR"}))
        return results

    # If WXYZ + zero_sum_normalized, require explicit normalization move
    if coord_sys == "WXYZ" and zero_sum_flag:
        moves = [s.get("move_id") for s in obj.get("derivation", {}).get("steps", [])]
        if "ART_NORMALIZE_WXYZ_ZERO_SUM" not in moves:
            results.append(GateResult(
                "gate_3_typed_formation", GateStatus.FAIL,
                "zero_sum_normalized set but ART_NORMALIZE_WXYZ_ZERO_SUM move not recorded",
                {"fail_type": "ILLEGAL_NORMALIZATION"}))
            return results

    # Check all TRIANGLE objects for non-degeneracy
    for p_obj in parsed:
        if p_obj.get("object_type") != "TRIANGLE":
            continue
        coords, err = _get_triangle_verts(p_obj)
        if err:
            results.append(GateResult(
                "gate_3_typed_formation", GateStatus.FAIL,
                f"Triangle parse error: {err}",
                {"fail_type": "SCENE_PARSE_ERROR",
                 "triangle_object_id": p_obj.get("object_id")}))
            return results

        a, b, c = coords[0], coords[1], coords[2]
        if coord_sys == "XYZ":
            if _cross_mag_sq(a, b, c) == 0.0:
                results.append(GateResult(
                    "gate_3_typed_formation", GateStatus.FAIL,
                    "Triangle points are collinear: formation rule violated",
                    {"fail_type": "DEGENERATE_TRIANGLE_COLLINEAR",
                     "triangle_object_id": p_obj.get("object_id")}))
                return results
        elif coord_sys == "WXYZ" and zero_sum_flag:
            if not all(_is_zero_sum(v) for v in [a, b, c]):
                results.append(GateResult(
                    "gate_3_typed_formation", GateStatus.FAIL,
                    "zero_sum_normalized but vertex not zero-sum",
                    {"fail_type": "ILLEGAL_NORMALIZATION",
                     "triangle_object_id": p_obj.get("object_id")}))
                return results

    results.append(GateResult(
        "gate_3_typed_formation", GateStatus.PASS,
        "Typed formation satisfied + normalization legal"))

    # Gate 4 -- Base algebra adequacy
    alg = obj.get("base_algebra", {}).get("properties", {})
    if alg.get("no_zero_divisors") is not True:
        results.append(GateResult(
            "gate_4_base_algebra_adequacy", GateStatus.FAIL,
            "Spread computation requires no_zero_divisors",
            {"fail_type": "BASE_ALGEBRA_TOO_WEAK"}))
        return results

    results.append(GateResult(
        "gate_4_base_algebra_adequacy", GateStatus.PASS,
        "Base algebra adequate (no_zero_divisors)"))

    # Gate 5 -- Step hash verification + RT invariant recomputation
    steps = obj.get("derivation", {}).get("steps", [])
    for st in steps:
        payload = {
            "inputs": st.get("inputs"),
            "move_id": st.get("move_id"),
            "outputs": st.get("outputs"),
        }
        expected = _sha256_hex(_canonical_json_compact(payload))
        got = st.get("step_hash_sha256", "")
        if got != expected:
            results.append(GateResult(
                "gate_5_step_hash_and_rt", GateStatus.FAIL,
                "Step hash mismatch",
                {"fail_type": "NONDETERMINISM_CONTRACT_VIOLATION",
                 "step_id": st.get("step_id"),
                 "expected": expected, "got": got}))
            return results

    # Verify RT_VALIDATE_LAW_EQUATION steps
    for st in steps:
        if st.get("move_id") == "RT_VALIDATE_LAW_EQUATION":
            law_fail = _verify_law_step(st)
            if law_fail is not None:
                results.append(law_fail)
                return results

    # Recompute RT invariants and compare
    # For WXYZ scenes with a projection step, use projected XYZ coordinates.
    xyz_coords_override = None
    if coord_sys == "WXYZ":
        proj_steps = [s for s in steps if s.get("move_id") == "ART_PROJECT_WXYZ_TO_XYZ"]
        if proj_steps:
            # Use projected XYZ vertices from the projection step's outputs
            projected = proj_steps[0].get("outputs", {}).get("vertices_xyz")
            if isinstance(projected, list) and len(projected) >= 3:
                xyz_coords_override = projected

    computed_triangles = []
    for p_obj in parsed:
        if p_obj.get("object_type") != "TRIANGLE":
            continue

        if coord_sys == "XYZ" or xyz_coords_override is not None:
            if xyz_coords_override is not None:
                a, b, c = xyz_coords_override[0], xyz_coords_override[1], xyz_coords_override[2]
            else:
                coords, _ = _get_triangle_verts(p_obj)
                a, b, c = coords[0], coords[1], coords[2]
            Q, s_vals, err = _compute_triangle_rt(a, b, c)
            if err:
                results.append(GateResult(
                    "gate_5_step_hash_and_rt", GateStatus.FAIL,
                    f"RT computation failed: {err}",
                    {"fail_type": err,
                     "triangle_object_id": p_obj.get("object_id")}))
                return results
            computed_triangles.append({
                "triangle_object_id": p_obj["object_id"],
                "Q": Q, "s": s_vals,
            })

    if computed_triangles:
        claimed = res.get("rt_invariants", {}).get("triangles", [])
        if claimed != computed_triangles:
            results.append(GateResult(
                "gate_5_step_hash_and_rt", GateStatus.FAIL,
                "Claimed RT invariants do not match recomputation",
                {"fail_type": "LAW_EQUATION_MISMATCH",
                 "computed": computed_triangles, "claimed": claimed}))
            return results

    results.append(GateResult(
        "gate_5_step_hash_and_rt", GateStatus.PASS,
        "Step hashes verified + RT invariants match recomputation"))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(base, "fixtures")

    fixtures = [
        ("valid_minimal.json", True, None),
        ("valid_provenance_tetrahedron.json", True, None),
        ("valid_wxyz_projected.json", True, None),
        ("valid_law_verified_tetrahedron.json", True, None),
        ("invalid_missing_invariant_diff.json", False, "gate_2_determinism_contract"),
        ("invalid_degenerate_triangle.json", False, "gate_3_typed_formation"),
        ("invalid_illegal_normalization.json", False, "gate_3_typed_formation"),
        ("invalid_law_equation_mismatch.json", False, "gate_5_step_hash_and_rt"),
    ]

    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        obj = _load_json(path)
        res = validate_cert(obj)
        passed = _report_ok(res)
        if should_pass != passed:
            ok = False
        fail_gates = [r.gate for r in res if r.status == GateStatus.FAIL]
        if (not should_pass) and expected_fail_gate and expected_fail_gate not in fail_gates:
            ok = False
        details.append({
            "fixture": name,
            "ok": passed,
            "expected_ok": should_pass,
            "failed_gates": fail_gates,
        })

    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_ARTEXPLORER_SCENE_ADAPTER.v1 SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_ARTEXPLORER_SCENE_ADAPTER.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test on fixtures")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = _load_json(args.file)
    results = validate_cert(obj)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
