#!/usr/bin/env python3
"""
validator.py

QA_ARTEXPLORER_SCENE_ADAPTER.v2 validator (Machine Tract).

Exact-arithmetic adapter cert family using unreduced rational pairs.
Ingests ARTexplorer JSON scene data with integer coordinates and certifies
typed geometric objects with RT invariants computed over exact integer
arithmetic — zero tolerance, no floats.

Compute substrate: qa_rational_pair_noreduce
- Coordinates: integers
- Quadrance: integer (sum of squared integer differences)
- Spread: unreduced rational pair {"n": int, "d": int}
  where d = Q1*Q2, n = d - dot*dot (explicit multiply, never **2)
- Law verification: cross-multiplication to integer equality (tolerance = 0)
- Fraction reduction: certified projection via RT_REDUCE_FRACTION move

ARTexplorer: https://github.com/arossti/ARTexplorer

Gates:
  1) Schema validity (jsonschema Draft-07)
  2) Determinism contract + invariant_diff presence + compute_substrate check
  3) Typed formation + integer coordinate enforcement
  4) Base algebra adequacy
  5) Step hash verification + exact RT invariant recomputation + exact law verification
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


def _expect_int_list(value: Any, path: str, min_len: int, max_len: int) -> None:
    _expect(isinstance(value, list) and min_len <= len(value) <= max_len,
            f"{path} must be array length {min_len}..{max_len}")
    _expect(all(isinstance(item, int) for item in value), f"{path} entries must be integers")


def _expect_pair(value: Any, path: str) -> None:
    _expect_type(value, dict, path)
    _expect_required(value, ["n", "d"], path)
    _expect_type(value["n"], int, f"{path}.n")
    _expect_type(value["d"], int, f"{path}.d")


def _validate_schema_minimal(obj: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Dependency-free checker for this family's concrete schema."""

    _expect_type(obj, dict, "$")
    _expect_required(obj, schema.get("required", []), "$")
    _expect(obj.get("schema_version") == "v2", "schema_version must be v2")
    _expect(obj.get("cert_type") == "QA_ARTEXPLORER_SCENE_ADAPTER.v2", "cert_type mismatch")
    _expect_type(obj.get("cert_id"), str, "cert_id")
    _expect_type(obj.get("created_utc"), str, "created_utc")
    if "compute_substrate" in obj:
        _expect(obj["compute_substrate"] == "qa_rational_pair_noreduce", "compute_substrate invalid")

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
            _expect_int_list(vertex["coord"], f"{vpath}.coord", 3, 4)
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
        "RT_REDUCE_FRACTION",
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
        _expect_int_list(tri["Q"], f"{tpath}.Q", 3, 3)
        _expect(isinstance(tri["s"], list) and len(tri["s"]) == 3, f"{tpath}.s must be length-3 array")
        for sidx, pair in enumerate(tri["s"]):
            _expect_pair(pair, f"{tpath}.s[{sidx}]")
    for optional_hash in ("object_digest_sha256", "rt_invariants_digest_sha256"):
        if optional_hash in result:
            _expect_hash64(result[optional_hash], f"result.{optional_hash}")

    dc = obj.get("determinism_contract")
    _expect_type(dc, dict, "determinism_contract")
    _expect_required(dc, ["canonical_json", "stable_sorting", "no_rng", "invariant_diff_defined"], "determinism_contract")
    for field in ("canonical_json", "stable_sorting", "no_rng", "invariant_diff_defined"):
        _expect_type(dc[field], bool, f"determinism_contract.{field}")


# ---------------------------------------------------------------------------
# Integer geometry helpers (exact arithmetic — no floats)
# ---------------------------------------------------------------------------

def _vec_sub(a: List[int], b: List[int]) -> List[int]:
    return [a[i] - b[i] for i in range(min(len(a), len(b)))]


def _dot(a: List[int], b: List[int]) -> int:
    return sum(a[i] * b[i] for i in range(min(len(a), len(b))))


def _quadrance_int(pa: List[int], pb: List[int]) -> int:
    """Integer quadrance: sum of squared integer differences (explicit multiply)."""
    d = _vec_sub(pb[:3], pa[:3])
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


def _cross_mag_sq_int(a: List[int], b: List[int], c: List[int]) -> int:
    """Squared magnitude of cross product (B-A) x (C-A). Zero iff collinear."""
    ab = _vec_sub(b[:3], a[:3])
    ac = _vec_sub(c[:3], a[:3])
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return cx * cx + cy * cy + cz * cz


def _spread_pair(v1: List[int], v2: List[int]) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """Compute unreduced spread as rational pair {n, d}.

    d = Q1 * Q2 (product of adjacent quadrances, explicit multiply)
    dot_val = dot(v1, v2)
    n = d - dot_val * dot_val

    Returns ({n, d}, None) or (None, error_str).
    """
    q1 = _dot(v1, v1)
    q2 = _dot(v2, v2)
    if q1 == 0 or q2 == 0:
        return None, "ZERO_DIVISOR_OBSTRUCTION"
    d = q1 * q2
    dot_val = _dot(v1, v2)
    n = d - dot_val * dot_val
    return {"n": n, "d": d}, None


def _compute_triangle_rt_exact(
    a: List[int], b: List[int], c: List[int]
) -> Tuple[List[int], List[Dict[str, int]], Optional[str]]:
    """Compute exact Q and s for triangle ABC.

    Q = [Q(BC), Q(CA), Q(AB)]
    s = [sA, sB, sC] as unreduced rational pairs.
    """
    Q1 = _quadrance_int(b, c)
    Q2 = _quadrance_int(c, a)
    Q3 = _quadrance_int(a, b)

    ab = _vec_sub(b[:3], a[:3])
    ac = _vec_sub(c[:3], a[:3])
    bc = _vec_sub(c[:3], b[:3])
    ba = _vec_sub(a[:3], b[:3])
    ca = _vec_sub(a[:3], c[:3])
    cb = _vec_sub(b[:3], c[:3])

    sA, eA = _spread_pair(ab, ac)
    if eA:
        return [Q1, Q2, Q3], [], eA
    sB, eB = _spread_pair(bc, ba)
    if eB:
        return [Q1, Q2, Q3], [], eB
    sC, eC = _spread_pair(ca, cb)
    if eC:
        return [Q1, Q2, Q3], [], eC

    return [Q1, Q2, Q3], [sA, sB, sC], None


# ---------------------------------------------------------------------------
# Exact law verification (cross-multiplication, zero tolerance)
# ---------------------------------------------------------------------------

def _exact_cross_law(Q: List[int], s: List[Dict[str, int]]) -> Tuple[bool, int, int]:
    """Cross Law (RT_LAW_04): (Q1+Q2-Q3)^2 = 4*Q1*Q2*(1-s3).

    Cross-multiplied with s3.d:
      (Q1+Q2-Q3)*(Q1+Q2-Q3)*s3.d == 4*Q1*Q2*(s3.d - s3.n)

    Returns (ok, lhs, rhs).
    """
    Q1, Q2, Q3 = Q
    s3n, s3d = s[2]["n"], s[2]["d"]
    t = Q1 + Q2 - Q3
    lhs = t * t * s3d
    rhs = 4 * Q1 * Q2 * (s3d - s3n)
    return lhs == rhs, lhs, rhs


def _exact_triple_spread(s: List[Dict[str, int]]) -> Tuple[bool, int, int]:
    """Triple Spread Formula (RT_LAW_05):
    (s1+s2+s3)^2 = 2*(s1^2+s2^2+s3^2) + 4*s1*s2*s3.

    All terms cross-multiplied to common denominator (d1*d2*d3)^2.

    Returns (ok, lhs, rhs).
    """
    n1, d1 = s[0]["n"], s[0]["d"]
    n2, d2 = s[1]["n"], s[1]["d"]
    n3, d3 = s[2]["n"], s[2]["d"]

    # Sum s1+s2+s3 = (n1*d2*d3 + n2*d1*d3 + n3*d1*d2) / (d1*d2*d3)
    cd = d1 * d2 * d3
    sum_num = n1 * d2 * d3 + n2 * d1 * d3 + n3 * d1 * d2

    # LHS: (sum)^2 numerator = sum_num^2, denominator = cd^2
    lhs = sum_num * sum_num

    # s1^2 = n1^2 / d1^2, cross-multiplied to cd^2:
    # s1^2 * cd^2 = n1^2 * (d2*d3)^2
    s1sq = n1 * n1 * d2 * d2 * d3 * d3
    s2sq = n2 * n2 * d1 * d1 * d3 * d3
    s3sq = n3 * n3 * d1 * d1 * d2 * d2

    # s1*s2*s3 = (n1*n2*n3) / (d1*d2*d3), cross-multiplied to cd^2:
    # s1*s2*s3 * cd^2 = n1*n2*n3 * cd = n1*n2*n3 * d1*d2*d3
    prod = n1 * n2 * n3 * d1 * d2 * d3

    rhs = 2 * (s1sq + s2sq + s3sq) + 4 * prod
    return lhs == rhs, lhs, rhs


def _exact_pythagoras(Q: List[int], s: List[Dict[str, int]]) -> Tuple[bool, int, int]:
    """Pythagoras (RT_LAW_01): Q3 = Q1 + Q2 when s3 = 1 (i.e. s3.n == s3.d).

    Returns (ok, lhs, rhs).
    """
    Q1, Q2, Q3 = Q
    return Q3 == Q1 + Q2, Q3, Q1 + Q2


_EXACT_LAW_DISPATCH = {
    "RT_LAW_04": "cross_law",
    "RT_LAW_05": "triple_spread",
    "RT_LAW_01": "pythagoras",
}


def _verify_law_step_exact(step: Dict[str, Any]) -> Optional[GateResult]:
    """Verify an RT_VALIDATE_LAW_EQUATION step with exact integer arithmetic."""
    inp = step.get("inputs", {})
    out = step.get("outputs", {})
    law_id = inp.get("law_id", "")
    Q = inp.get("Q")
    s = inp.get("s")

    if law_id not in _EXACT_LAW_DISPATCH:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"Unknown law_id: {law_id}",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    if law_id == "RT_LAW_04":
        ok, lhs, rhs = _exact_cross_law(Q, s)
    elif law_id == "RT_LAW_05":
        ok, lhs, rhs = _exact_triple_spread(s)
    elif law_id == "RT_LAW_01":
        ok, lhs, rhs = _exact_pythagoras(Q, s)
    else:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"Unimplemented law: {law_id}",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    if not ok:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} exact equation not satisfied (lhs={lhs}, rhs={rhs})",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id"),
             "law_id": law_id, "lhs": lhs, "rhs": rhs})

    if out.get("verified") is not True:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} satisfied but outputs.verified != true",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    claimed_lhs = out.get("lhs")
    claimed_rhs = out.get("rhs")
    if claimed_lhs is not None and claimed_lhs != lhs:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} claimed lhs mismatch (claimed={claimed_lhs}, computed={lhs})",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})
    if claimed_rhs is not None and claimed_rhs != rhs:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"{law_id} claimed rhs mismatch (claimed={claimed_rhs}, computed={rhs})",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step.get("step_id")})

    return None


# ---------------------------------------------------------------------------
# RT_REDUCE_FRACTION verification (certified projection)
# ---------------------------------------------------------------------------

def _verify_reduce_fraction_step(step: Dict[str, Any]) -> Optional[GateResult]:
    """Verify an RT_REDUCE_FRACTION step (certified projection with scale collapse)."""
    inp = step.get("inputs", {})
    out = step.get("outputs", {})
    step_id = step.get("step_id")

    n = inp.get("n")
    d = inp.get("d")
    if not isinstance(n, int) or not isinstance(d, int):
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            "RT_REDUCE_FRACTION inputs n,d must be integers",
            {"fail_type": "ILLEGAL_NORMALIZATION", "step_id": step_id})

    n_reduced = out.get("n_reduced")
    d_reduced = out.get("d_reduced")
    gcd_val = out.get("gcd")
    scale_before = out.get("scale_before")
    scale_after = out.get("scale_after")
    ack = out.get("non_reduction_axiom_ack")

    if not all(isinstance(v, int) for v in [n_reduced, d_reduced, gcd_val]):
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            "RT_REDUCE_FRACTION outputs must be integers",
            {"fail_type": "ILLEGAL_NORMALIZATION", "step_id": step_id})

    if n != n_reduced * gcd_val:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"n={n} != n_reduced*gcd={n_reduced}*{gcd_val}={n_reduced*gcd_val}",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step_id})

    if d != d_reduced * gcd_val:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"d={d} != d_reduced*gcd={d_reduced}*{gcd_val}={d_reduced*gcd_val}",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step_id})

    actual_gcd = math.gcd(n_reduced, d_reduced)
    if actual_gcd != 1:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"Reduced pair not coprime: gcd({n_reduced},{d_reduced})={actual_gcd}",
            {"fail_type": "LAW_EQUATION_MISMATCH", "step_id": step_id})

    if scale_before != d:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"scale_before={scale_before} != d={d}",
            {"fail_type": "ILLEGAL_NORMALIZATION", "step_id": step_id})

    if scale_after != d_reduced:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            f"scale_after={scale_after} != d_reduced={d_reduced}",
            {"fail_type": "ILLEGAL_NORMALIZATION", "step_id": step_id})

    if ack is not True:
        return GateResult(
            "gate_5_step_hash_and_rt", GateStatus.FAIL,
            "non_reduction_axiom_ack must be true",
            {"fail_type": "ILLEGAL_NORMALIZATION", "step_id": step_id})

    return None


# ---------------------------------------------------------------------------
# Spread pair comparison helpers
# ---------------------------------------------------------------------------

def _spread_pair_matches(claimed: Dict[str, Any], computed: Dict[str, int]) -> bool:
    """Check if a claimed spread pair matches the computed unreduced form.

    The claimed pair MUST exactly match the unreduced computation.
    Any reduction without a preceding RT_REDUCE_FRACTION move is ILLEGAL_NORMALIZATION.
    """
    return claimed.get("n") == computed["n"] and claimed.get("d") == computed["d"]


def _get_triangle_verts(obj: Dict[str, Any]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Extract 3 vertex coords for a TRIANGLE object."""
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
            "Valid QA_ARTEXPLORER_SCENE_ADAPTER.v2 schema"))
    except Exception as e:
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.FAIL,
            f"Schema validation failed: {e}"))
        return results

    # Gate 2 -- Determinism contract + invariant_diff + compute_substrate
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

    # compute_substrate is optional in schema but required by validator for v2
    substrate = obj.get("compute_substrate")
    if substrate != "qa_rational_pair_noreduce":
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            f"compute_substrate must be 'qa_rational_pair_noreduce', got '{substrate}'",
            {"fail_type": "UNSUPPORTED_COMPUTE_SUBSTRATE"}))
        return results

    results.append(GateResult(
        "gate_2_determinism_contract", GateStatus.PASS,
        "Determinism contract strict + invariant_diff present + substrate=qa_rational_pair_noreduce"))

    # Gate 3 -- Typed formation + integer coordinate enforcement
    scene = obj.get("scene", {})
    coord_sys = scene.get("coordinate_system")

    parsed = obj.get("derivation", {}).get("parsed_objects", [])
    if not parsed:
        results.append(GateResult(
            "gate_3_typed_formation", GateStatus.FAIL,
            "derivation.parsed_objects empty",
            {"fail_type": "SCENE_PARSE_ERROR"}))
        return results

    # Enforce integer coordinates
    for p_obj in parsed:
        for vert in p_obj.get("vertices", []):
            coord = vert.get("coord", [])
            for i, c in enumerate(coord):
                if not isinstance(c, int):
                    results.append(GateResult(
                        "gate_3_typed_formation", GateStatus.FAIL,
                        f"Non-integer coordinate: vertex {vert.get('id')} coord[{i}]={c}",
                        {"fail_type": "NON_INTEGER_COORDINATE",
                         "object_id": p_obj.get("object_id"),
                         "vertex_id": vert.get("id")}))
                    return results

    # Check TRIANGLE objects for non-degeneracy
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
            if _cross_mag_sq_int(a, b, c) == 0:
                results.append(GateResult(
                    "gate_3_typed_formation", GateStatus.FAIL,
                    "Triangle points are collinear: formation rule violated",
                    {"fail_type": "DEGENERATE_TRIANGLE_COLLINEAR",
                     "triangle_object_id": p_obj.get("object_id")}))
                return results

    results.append(GateResult(
        "gate_3_typed_formation", GateStatus.PASS,
        "Typed formation satisfied + all coordinates integer"))

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

    # Gate 5 -- Step hash verification + exact RT recomputation
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
                {"fail_type": "NONDETERMINISM_CONTRACT_VIOLATION",
                 "step_id": st.get("step_id"),
                 "expected": expected, "got": got}))
            return results

    # 5b: Verify RT_VALIDATE_LAW_EQUATION steps (exact integer arithmetic)
    for st in steps:
        if st.get("move_id") == "RT_VALIDATE_LAW_EQUATION":
            law_fail = _verify_law_step_exact(st)
            if law_fail is not None:
                results.append(law_fail)
                return results

    # 5c: Verify RT_REDUCE_FRACTION steps (certified projection)
    for st in steps:
        if st.get("move_id") == "RT_REDUCE_FRACTION":
            reduce_fail = _verify_reduce_fraction_step(st)
            if reduce_fail is not None:
                results.append(reduce_fail)
                return results

    # 5d: Recompute RT invariants and compare (exact pair matching)
    # Collect which spread fields have been reduced by RT_REDUCE_FRACTION steps
    reduced_targets = set()
    for st in steps:
        if st.get("move_id") == "RT_REDUCE_FRACTION":
            target = st.get("inputs", {}).get("target")
            if target:
                reduced_targets.add(target)

    computed_triangles = []
    for p_obj in parsed:
        if p_obj.get("object_type") != "TRIANGLE":
            continue

        if coord_sys == "XYZ":
            coords, _ = _get_triangle_verts(p_obj)
            a, b, c = coords[0], coords[1], coords[2]
            Q, s_vals, err = _compute_triangle_rt_exact(a, b, c)
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
        if len(claimed) != len(computed_triangles):
            results.append(GateResult(
                "gate_5_step_hash_and_rt", GateStatus.FAIL,
                f"Triangle count mismatch: claimed {len(claimed)} vs computed {len(computed_triangles)}",
                {"fail_type": "LAW_EQUATION_MISMATCH"}))
            return results

        for i, (cl, co) in enumerate(zip(claimed, computed_triangles)):
            if cl.get("triangle_object_id") != co["triangle_object_id"]:
                results.append(GateResult(
                    "gate_5_step_hash_and_rt", GateStatus.FAIL,
                    f"Triangle ID mismatch at index {i}",
                    {"fail_type": "LAW_EQUATION_MISMATCH"}))
                return results

            if cl.get("Q") != co["Q"]:
                results.append(GateResult(
                    "gate_5_step_hash_and_rt", GateStatus.FAIL,
                    f"Q mismatch for {co['triangle_object_id']}: claimed={cl.get('Q')} computed={co['Q']}",
                    {"fail_type": "LAW_EQUATION_MISMATCH"}))
                return results

            # Check spread pairs — must match unreduced computation exactly
            # unless an RT_REDUCE_FRACTION step explicitly covers this field
            claimed_s = cl.get("s", [])
            computed_s = co["s"]
            tri_id = co["triangle_object_id"]
            spread_labels = ["sA", "sB", "sC"]

            for j, (cs, xs) in enumerate(zip(claimed_s, computed_s)):
                field_path = f"rt_invariants.triangles[{tri_id}].s[{j}]"
                if field_path in reduced_targets:
                    # Reduced pair — just verify n/d ratio matches
                    if cs.get("n") * xs["d"] != xs["n"] * cs.get("d"):
                        results.append(GateResult(
                            "gate_5_step_hash_and_rt", GateStatus.FAIL,
                            f"Spread {spread_labels[j]} reduced but ratio mismatch",
                            {"fail_type": "LAW_EQUATION_MISMATCH",
                             "triangle_object_id": tri_id}))
                        return results
                else:
                    # Must match unreduced form exactly
                    if not _spread_pair_matches(cs, xs):
                        target_path = f"result.rt_invariants.triangles[{tri_id}].s[{j}]"
                        results.append(GateResult(
                            "gate_5_step_hash_and_rt", GateStatus.FAIL,
                            f"Spread {spread_labels[j]} pair identity mismatch: "
                            f"claimed={{n:{cs.get('n')},d:{cs.get('d')}}} "
                            f"computed={{n:{xs['n']},d:{xs['d']}}}",
                            {"fail_type": "ILLEGAL_NORMALIZATION",
                             "reason": "PAIR_IDENTITY_MISMATCH",
                             "triangle_object_id": tri_id,
                             "target_path": target_path,
                             "claimed": cs, "computed": xs}))
                        return results

    results.append(GateResult(
        "gate_5_step_hash_and_rt", GateStatus.PASS,
        "Step hashes verified + exact RT invariants match recomputation"))

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
        ("valid_exact_345_triangle.json", True, None),
        ("valid_reduction_with_projection.json", True, None),
        ("invalid_illegal_reduction.json", False, "gate_5_step_hash_and_rt"),
        ("invalid_wrong_unreduced_pair.json", False, "gate_5_step_hash_and_rt"),
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
        print("=== QA_ARTEXPLORER_SCENE_ADAPTER.v2 SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_ARTEXPLORER_SCENE_ADAPTER.v2 validator")
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
