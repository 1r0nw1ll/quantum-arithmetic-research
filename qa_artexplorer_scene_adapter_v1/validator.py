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
    import jsonschema
    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


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
