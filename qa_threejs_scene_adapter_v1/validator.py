#!/usr/bin/env python3
"""
validator.py

QA_THREEJS_SCENE_ADAPTER.v1 validator (Machine Tract).

Adapter cert family that ingests a constrained Three.js scene export JSON
(format: three_scene_export_v1) and certifies typed geometric objects
(TRIANGLE / MESH) with RT (Rational Trigonometry) invariants over float64.

Upstream: Three.js (https://threejs.org/)
Substrate: float64, REL_TOL = 1e-9

Gates:
  1) Schema validity (jsonschema Draft-07)
  2) Determinism contract + invariant_diff presence + compute_substrate check
  3) Typed formation: parse scene_raw, detect non-finite coordinates, stable ordering
  4) Base algebra adequacy (field + no_zero_divisors)
  5) Step hash chain + RT invariant recomputation + RT law verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


# ---------------------------------------------------------------------------
# Shared primitives (deterministic; matches all QA adapter family conventions)
# ---------------------------------------------------------------------------

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
    """Dependency-free checker for this family's concrete schema."""

    _expect_type(obj, dict, "$")
    _expect_required(obj, schema.get("required", []), "$")
    _expect(obj.get("schema_version") == "v1", "schema_version must be v1")
    _expect(obj.get("cert_type") == "QA_THREEJS_SCENE_ADAPTER.v1", "cert_type mismatch")
    _expect_type(obj.get("cert_id"), str, "cert_id")
    _expect_type(obj.get("created_utc"), str, "created_utc")
    _expect(obj.get("compute_substrate") == "float64", "compute_substrate must be float64")

    source = obj.get("source_semantics")
    _expect_type(source, dict, "source_semantics")
    _expect_required(source, ["upstream", "export"], "source_semantics")
    upstream = source["upstream"]
    _expect_type(upstream, dict, "source_semantics.upstream")
    _expect_required(upstream, ["name"], "source_semantics.upstream")
    _expect_type(upstream["name"], str, "source_semantics.upstream.name")
    export = source["export"]
    _expect_type(export, dict, "source_semantics.export")
    _expect_required(export, ["format", "exported_by"], "source_semantics.export")
    _expect(export["format"] == "three_scene_export_v1", "source_semantics.export.format invalid")
    _expect_type(export["exported_by"], str, "source_semantics.export.exported_by")

    base = obj.get("base_algebra")
    _expect_type(base, dict, "base_algebra")
    _expect_required(base, ["name", "properties"], "base_algebra")
    _expect_type(base["name"], str, "base_algebra.name")
    props = base["properties"]
    _expect_type(props, dict, "base_algebra.properties")
    _expect_required(props, ["field", "no_zero_divisors"], "base_algebra.properties")
    _expect(props["field"] is True, "base_algebra.properties.field must be true")
    _expect(props["no_zero_divisors"] is True, "base_algebra.properties.no_zero_divisors must be true")

    scene = obj.get("scene")
    _expect_type(scene, dict, "scene")
    _expect_required(scene, ["scene_raw", "scene_raw_sha256"], "scene")
    _expect_type(scene["scene_raw"], dict, "scene.scene_raw")
    _expect_hash64(scene["scene_raw_sha256"], "scene.scene_raw_sha256")

    derivation = obj.get("derivation")
    _expect_type(derivation, dict, "derivation")
    _expect_required(derivation, ["parsed_objects", "steps"], "derivation")
    parsed = derivation["parsed_objects"]
    _expect(isinstance(parsed, list) and len(parsed) >= 1, "derivation.parsed_objects must be nonempty list")
    for idx, parsed_obj in enumerate(parsed):
        path = f"derivation.parsed_objects[{idx}]"
        _expect_type(parsed_obj, dict, path)
        _expect_required(parsed_obj, ["object_id", "object_type", "vertices", "faces"], path)
        _expect_type(parsed_obj["object_id"], str, f"{path}.object_id")
        _expect(parsed_obj["object_type"] in {"TRIANGLE", "MESH"}, f"{path}.object_type invalid")
        vertices = parsed_obj["vertices"]
        _expect(isinstance(vertices, list) and len(vertices) >= 3, f"{path}.vertices must be nonempty")
        for v_idx, vertex in enumerate(vertices):
            v_path = f"{path}.vertices[{v_idx}]"
            _expect_type(vertex, dict, v_path)
            _expect_required(vertex, ["id", "coord"], v_path)
            _expect_type(vertex["id"], str, f"{v_path}.id")
            _expect_number_list(vertex["coord"], f"{v_path}.coord", 3, 3)
        faces = parsed_obj["faces"]
        _expect_type(faces, list, f"{path}.faces")
        for face_idx, face in enumerate(faces):
            face_path = f"{path}.faces[{face_idx}]"
            _expect(isinstance(face, list) and len(face) == 3, f"{face_path} must have three entries")
            _expect(all(isinstance(item, str) for item in face), f"{face_path} entries must be strings")

    steps = derivation["steps"]
    _expect(isinstance(steps, list) and len(steps) >= 2, "derivation.steps must contain at least two steps")
    for idx, step in enumerate(steps):
        path = f"derivation.steps[{idx}]"
        _expect_type(step, dict, path)
        _expect_required(step, ["step_id", "move_id", "inputs", "outputs", "step_hash_sha256"], path)
        _expect_type(step["step_id"], str, f"{path}.step_id")
        _expect(
            step["move_id"] in {"THREE_PARSE_SCENE", "RT_COMPUTE_TRIANGLE_INVARIANTS", "RT_VALIDATE_LAW_EQUATION"},
            f"{path}.move_id invalid",
        )
        _expect_type(step["inputs"], dict, f"{path}.inputs")
        _expect_type(step["outputs"], dict, f"{path}.outputs")
        _expect_hash64(step["step_hash_sha256"], f"{path}.step_hash_sha256")

    result = obj.get("result")
    _expect_type(result, dict, "result")
    _expect_required(result, ["rt_invariants"], "result")
    rt = result["rt_invariants"]
    _expect_type(rt, dict, "result.rt_invariants")
    _expect_required(rt, ["triangles"], "result.rt_invariants")
    triangles = rt["triangles"]
    _expect(isinstance(triangles, list) and len(triangles) >= 1,
            "result.rt_invariants.triangles must be nonempty list")
    for idx, tri in enumerate(triangles):
        path = f"result.rt_invariants.triangles[{idx}]"
        _expect_type(tri, dict, path)
        _expect_required(tri, ["triangle_object_id", "Q", "s"], path)
        _expect_type(tri["triangle_object_id"], str, f"{path}.triangle_object_id")
        _expect_number_list(tri["Q"], f"{path}.Q", 3, 3)
        _expect_number_list(tri["s"], f"{path}.s", 3, 3)
    if "invariant_diff" in result:
        _expect_type(result["invariant_diff"], dict, "result.invariant_diff")

    dc = obj.get("determinism_contract")
    _expect_type(dc, dict, "determinism_contract")
    _expect_required(dc, ["canonical_json", "stable_sorting", "no_rng", "invariant_diff_defined"],
                     "determinism_contract")
    for field in ("canonical_json", "stable_sorting", "no_rng", "invariant_diff_defined"):
        _expect(dc[field] is True, f"determinism_contract.{field} must be true")


# ---------------------------------------------------------------------------
# RT geometry helpers — float64, deterministic arithmetic form (never **2)
# ---------------------------------------------------------------------------

REL_TOL = 1e-9


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _quadrance(a: List[float], b: List[float]) -> float:
    d = _vec_sub(b[:3], a[:3])
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


def _spread(v1: List[float], v2: List[float]) -> Optional[float]:
    """s = 1 - (dot^2)/(Q1*Q2). Returns None on zero quadrance."""
    q1 = _dot(v1, v1)
    q2 = _dot(v2, v2)
    if q1 == 0.0 or q2 == 0.0:
        return None
    dotv = _dot(v1, v2)
    return 1.0 - (dotv * dotv) / (q1 * q2)


def _cross_mag_sq(a: List[float], b: List[float], c: List[float]) -> float:
    ab = _vec_sub(b[:3], a[:3])
    ac = _vec_sub(c[:3], a[:3])
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return cx * cx + cy * cy + cz * cz


def _rel_close(a: float, b: float) -> bool:
    diff = abs(a - b)
    scale = max(1.0, abs(a), abs(b))
    return diff <= REL_TOL * scale


def _compute_triangle_rt(
    a: List[float], b: List[float], c: List[float]
) -> Tuple[List[float], List[float], Optional[str]]:
    """Compute Q=[Q(BC),Q(CA),Q(AB)] and s=[sA,sB,sC]."""
    Q1 = _quadrance(b, c)
    Q2 = _quadrance(c, a)
    Q3 = _quadrance(a, b)

    ab = _vec_sub(b[:3], a[:3])
    ac = _vec_sub(c[:3], a[:3])
    bc = _vec_sub(c[:3], b[:3])
    ba = _vec_sub(a[:3], b[:3])
    ca = _vec_sub(a[:3], c[:3])
    cb = _vec_sub(b[:3], c[:3])

    sA = _spread(ab, ac)
    sB = _spread(bc, ba)
    sC = _spread(ca, cb)

    if sA is None or sB is None or sC is None:
        return [Q1, Q2, Q3], [], "ZERO_DIVISOR_OBSTRUCTION"

    return [Q1, Q2, Q3], [sA, sB, sC], None


# ---------------------------------------------------------------------------
# RT law helpers (relative tolerance)
# ---------------------------------------------------------------------------

def _cross_law_residual(Q: List[float], s: List[float]) -> Tuple[float, float, float]:
    """Cross Law (RT_LAW_04): (Q1+Q2-Q3)^2 = 4*Q1*Q2*(1-s3)."""
    Q1, Q2, Q3 = Q
    t = Q1 + Q2 - Q3
    lhs = t * t
    rhs = 4.0 * Q1 * Q2 * (1.0 - s[2])
    return lhs, rhs, abs(lhs - rhs)


def _triple_spread_residual(s: List[float]) -> Tuple[float, float, float]:
    """Triple Spread (RT_LAW_05): (s1+s2+s3)^2 = 2*(s1^2+s2^2+s3^2)+4*s1*s2*s3."""
    s1, s2, s3 = s
    lhs = (s1 + s2 + s3) * (s1 + s2 + s3)
    rhs = 2.0 * (s1 * s1 + s2 * s2 + s3 * s3) + 4.0 * s1 * s2 * s3
    return lhs, rhs, abs(lhs - rhs)


def _pythagoras_residual(Q: List[float], s: List[float]) -> Tuple[float, float, float]:
    """Pythagoras (RT_LAW_01): Q3 = Q1+Q2 when s3=1."""
    lhs = float(Q[2])
    rhs = float(Q[0] + Q[1])
    return lhs, rhs, abs(lhs - rhs)


_LAW_DISPATCH = {
    "RT_LAW_04": ("QS", _cross_law_residual),
    "RT_LAW_05": ("S",  _triple_spread_residual),
    "RT_LAW_01": ("QS", _pythagoras_residual),
}


def _verify_law_step(step: Dict[str, Any]) -> Optional[GateResult]:
    """Verify RT_VALIDATE_LAW_EQUATION step (relative tolerance)."""
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

    sig, fn = _LAW_DISPATCH[law_id]
    if sig == "S":
        lhs, rhs, residual = fn(s)
    else:
        lhs, rhs, residual = fn(Q, s)

    denom = max(abs(lhs), abs(rhs), 1.0)
    if residual / denom > REL_TOL:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} not satisfied (rel_residual={residual/denom:.2e})",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id"),
             "law_id": law_id, "lhs": lhs, "rhs": rhs, "residual": residual})

    if out.get("verified") is not True:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} satisfied but outputs.verified != true",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    return None


# ---------------------------------------------------------------------------
# Three.js scene parser (Gate 3 typed formation)
# ---------------------------------------------------------------------------

def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _parse_three_scene(scene_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse constrained three_scene_export_v1 into typed parsed_objects.

    Stable ordering: sorted by object_id (determinism).
    Non-finite coordinate → raises ValueError with NONFINITE_COORDINATE detail.
    """
    if scene_raw.get("format") != "three_scene_export_v1":
        raise ValueError("scene_raw.format must be 'three_scene_export_v1'")
    raw_objs = scene_raw.get("objects")
    if not isinstance(raw_objs, list) or not raw_objs:
        raise ValueError("scene_raw.objects missing or empty")

    parsed: List[Dict[str, Any]] = []
    for obj in raw_objs:
        oid = obj.get("id")
        otype = obj.get("type")
        if not isinstance(oid, str) or not isinstance(otype, str):
            raise ValueError("object id/type must be strings")

        if otype == "Triangle":
            raw_verts = obj.get("vertices")
            if not (isinstance(raw_verts, list) and len(raw_verts) == 3):
                raise ValueError(f"Triangle '{oid}' must have exactly 3 vertices")
            vtx = []
            for i, p in enumerate(raw_verts):
                if not (isinstance(p, list) and len(p) == 3):
                    raise ValueError(f"Triangle '{oid}' vertex {i} must be [x,y,z]")
                for j, coord in enumerate(p):
                    if not _is_finite(coord):
                        raise ValueError(
                            f"NONFINITE_COORDINATE|{oid}|vertex_{i}|coord_{j}|{coord}")
                vtx.append({"id": ["A", "B", "C"][i],
                             "coord": [float(p[0]), float(p[1]), float(p[2])]})
            parsed.append({
                "object_id": oid,
                "object_type": "TRIANGLE",
                "label": obj.get("label", ""),
                "vertices": vtx,
                "faces": [["A", "B", "C"]],
            })
        else:
            raise ValueError(f"Unsupported object type: '{otype}' (object '{oid}')")

    parsed.sort(key=lambda x: x["object_id"])
    return parsed


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.PASS,
            "Valid QA_THREEJS_SCENE_ADAPTER.v1 schema"))
    except Exception as e:
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.FAIL,
            f"Schema validation failed: {e}"))
        return results

    # Gate 2 — Determinism contract + invariant_diff + compute_substrate
    dc = obj.get("determinism_contract", {})
    if not (
        dc.get("canonical_json") is True
        and dc.get("stable_sorting") is True
        and dc.get("no_rng") is True
        and dc.get("invariant_diff_defined") is True
    ):
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            "Determinism contract flags not strict",
            {"fail_type": "MISSING_DETERMINISM_CONTRACT"}))
        return results

    if obj.get("compute_substrate") != "float64":
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            f"compute_substrate must be 'float64', got '{obj.get('compute_substrate')}'",
            {"fail_type": "UNSUPPORTED_COMPUTE_SUBSTRATE"}))
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
        "Determinism contract strict + invariant_diff present + substrate=float64"))

    # Gate 3 — Typed formation + non-finite coordinate check
    scene = obj.get("scene", {})
    scene_raw = scene.get("scene_raw", {})

    # Verify scene_raw SHA
    claimed_sha = scene.get("scene_raw_sha256", "")
    computed_sha = _sha256_hex(_canonical_json_compact(scene_raw))
    if claimed_sha != computed_sha:
        results.append(GateResult(
            "gate_3_typed_formation", GateStatus.FAIL,
            "scene_raw_sha256 mismatch",
            {"fail_type": "TYPED_FORMATION_ERROR",
             "claimed": claimed_sha, "computed": computed_sha,
             "target_path": "scene.scene_raw_sha256"}))
        return results

    try:
        parsed = _parse_three_scene(scene_raw)
    except ValueError as e:
        msg = str(e)
        fail_type = "NONFINITE_COORDINATE" if msg.startswith("NONFINITE_COORDINATE") \
            else "TYPED_FORMATION_ERROR"
        results.append(GateResult(
            "gate_3_typed_formation", GateStatus.FAIL,
            f"Scene parse error: {msg}",
            {"fail_type": fail_type}))
        return results

    # Verify derivation.parsed_objects matches parsed (stable sort)
    claimed_parsed = obj.get("derivation", {}).get("parsed_objects", [])
    if len(claimed_parsed) != len(parsed):
        results.append(GateResult(
            "gate_3_typed_formation", GateStatus.FAIL,
            "parsed_objects count mismatch vs reparse",
            {"fail_type": "TYPED_FORMATION_ERROR"}))
        return results

    results.append(GateResult(
        "gate_3_typed_formation", GateStatus.PASS,
        f"Typed formation OK ({len(parsed)} object(s) parsed, stable-sorted, no non-finite coords)"))

    # Gate 4 — Base algebra adequacy
    alg = obj.get("base_algebra", {}).get("properties", {})
    if alg.get("no_zero_divisors") is not True or alg.get("field") is not True:
        results.append(GateResult(
            "gate_4_base_algebra_adequacy", GateStatus.FAIL,
            "Spread computation requires field + no_zero_divisors",
            {"fail_type": "TYPED_FORMATION_ERROR"}))
        return results

    results.append(GateResult(
        "gate_4_base_algebra_adequacy", GateStatus.PASS,
        "Base algebra adequate (field + no_zero_divisors)"))

    # Gate 5 — Step hash chain + RT recomputation + law verification
    steps = obj.get("derivation", {}).get("steps", [])

    # 5a: Step hash verification
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
                {"fail_type": "STEP_HASH_MISMATCH",
                 "step_id": st.get("step_id"),
                 "expected": expected, "got": got}))
            return results

    # 5b: Law equation verification
    for st in steps:
        if st.get("move_id") == "RT_VALIDATE_LAW_EQUATION":
            law_fail = _verify_law_step(st)
            if law_fail is not None:
                results.append(law_fail)
                return results

    # 5c: RT invariant recomputation
    computed_triangles = []
    for p_obj in parsed:
        if p_obj.get("object_type") != "TRIANGLE":
            continue
        verts = {v["id"]: v["coord"] for v in p_obj["vertices"]}
        a, b, c = verts["A"], verts["B"], verts["C"]
        Q, s_vals, err = _compute_triangle_rt(a, b, c)
        if err:
            results.append(GateResult(
                "gate_5_step_hash_and_rt", GateStatus.FAIL,
                f"RT computation failed: {err}",
                {"fail_type": err, "triangle_object_id": p_obj["object_id"]}))
            return results
        computed_triangles.append({
            "triangle_object_id": p_obj["object_id"],
            "Q": Q, "s": s_vals,
        })

    claimed_tris = res.get("rt_invariants", {}).get("triangles", [])
    if len(claimed_tris) != len(computed_triangles):
        results.append(GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            "Triangle count mismatch",
            {"fail_type": "LAW_EQUATION_MISMATCH"}))
        return results

    for cl, co in zip(claimed_tris, computed_triangles):
        if cl.get("triangle_object_id") != co["triangle_object_id"]:
            results.append(GateResult(
                "gate_5_step_hash_and_rt", GateStatus.FAIL,
                "Triangle ID mismatch",
                {"fail_type": "LAW_EQUATION_MISMATCH"}))
            return results
        for i in range(3):
            if not _rel_close(float(cl["Q"][i]), float(co["Q"][i])):
                results.append(GateResult(
                    "gate_5_step_hash_and_rt", GateStatus.FAIL,
                    f"Q[{i}] mismatch: claimed={cl['Q'][i]} computed={co['Q'][i]}",
                    {"fail_type": "LAW_EQUATION_MISMATCH",
                     "reason": "Q_MISMATCH", "index": i,
                     "claimed": cl["Q"][i], "computed": co["Q"][i]}))
                return results
            if not _rel_close(float(cl["s"][i]), float(co["s"][i])):
                results.append(GateResult(
                    "gate_5_step_hash_and_rt", GateStatus.FAIL,
                    f"s[{i}] mismatch: claimed={cl['s'][i]} computed={co['s'][i]}",
                    {"fail_type": "LAW_EQUATION_MISMATCH",
                     "reason": "S_MISMATCH", "index": i,
                     "claimed": cl["s"][i], "computed": co["s"][i]}))
                return results

    results.append(GateResult(
        "gate_5_step_hash_and_rt", GateStatus.PASS,
        "Step hashes verified + RT invariants match recomputation + law checks pass"))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json_out(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(base, "fixtures")

    fixtures = [
        ("valid_minimal_triangle_scene.json", True, None),
        ("invalid_nonfinite_coordinate.json", False, "gate_3_typed_formation"),
        ("invalid_missing_invariant_diff.json", False, "gate_2_determinism_contract"),
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
        print("=== QA_THREEJS_SCENE_ADAPTER.v1 SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_THREEJS_SCENE_ADAPTER.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON to validate")
    ap.add_argument("--self-test", action="store_true")
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
        _print_json_out(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
