#!/usr/bin/env python3
"""
validator.py

QA_GEOGEBRA_SCENE_ADAPTER.v1 validator (Machine Tract).
Exact substrate only: qa_rational_pair_noreduce.

Coordinates are Z (integer) or unreduced Q(n,d) pairs.
Computation lifts to integer lattice via per-triangle LCM of denominators.

Deterministic arithmetic-form invariant:
  - NEVER use pow() or **2
  - Always x*x

Gates:
  1) Schema validity (jsonschema Draft-07)
  2) Determinism contract + invariant_diff + compute_substrate
  3) Typed formation: scene_raw_sha256, Z/Q coord typing, zero-denominator check
  4) Base algebra adequacy (field + no_zero_divisors)
  5) Step hash chain + exact RT recomputation + strict law verification (zero tolerance)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None


# ---------------------------------------------------------------------------
# Canonical JSON + SHA256
# ---------------------------------------------------------------------------

def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _step_hash(move_id: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
    payload = {"inputs": inputs, "move_id": move_id, "outputs": outputs}
    return _sha256_hex(_canonical_json_compact(payload))


# ---------------------------------------------------------------------------
# Gate result structures
# ---------------------------------------------------------------------------

class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate_id: str
    status: GateStatus
    message: str
    details: Dict[str, Any]


def _pass(gate_id: str, msg: str, details: Optional[Dict[str, Any]] = None) -> GateResult:
    return GateResult(gate_id, GateStatus.PASS, msg, details or {})


def _fail(gate_id: str, msg: str, details: Optional[Dict[str, Any]] = None) -> GateResult:
    return GateResult(gate_id, GateStatus.FAIL, msg, details or {})


# ---------------------------------------------------------------------------
# Rational coordinate helpers
# ---------------------------------------------------------------------------

def _is_coord_Z(x: Any) -> bool:
    return isinstance(x, dict) and x.get("k") == "Z" and isinstance(x.get("v"), int)


def _is_coord_Q(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and x.get("k") == "Q"
        and isinstance(x.get("n"), int)
        and isinstance(x.get("d"), int)
    )


def _coord_den(x: Dict[str, Any]) -> int:
    return 1 if x["k"] == "Z" else int(x["d"])


def _coord_num(x: Dict[str, Any]) -> int:
    return int(x["v"]) if x["k"] == "Z" else int(x["n"])


def _gcd(a: int, b: int) -> int:
    x, y = abs(a), abs(b)
    while y:
        x, y = y, x % y
    return x


def _lcm(a: int, b: int) -> int:
    a0, b0 = abs(a), abs(b)
    if a0 == 0 or b0 == 0:
        return 0
    return (a0 // _gcd(a0, b0)) * b0


def _lift_point_to_int_xyz(coord3: List[Dict[str, Any]], L: int) -> List[int]:
    """Lift rational/integer coords to integer lattice via LCM scale L."""
    out: List[int] = []
    for c in coord3:
        d = _coord_den(c)
        n = _coord_num(c)
        scale = L // abs(d)
        if d < 0:
            n = -n
        out.append(n * scale)
    return out


# ---------------------------------------------------------------------------
# Exact RT computations on integer-lifted coordinates
# ---------------------------------------------------------------------------

def _vec_sub_int(a: List[int], b: List[int]) -> List[int]:
    return [a[i] - b[i] for i in range(3)]


def _dot_int(a: List[int], b: List[int]) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _quadrance_int(a: List[int], b: List[int]) -> int:
    d = _vec_sub_int(b, a)
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


def _spread_pair_from_vectors_int(
    v1: List[int], v2: List[int]
) -> Optional[Dict[str, int]]:
    """Spread as unreduced pair {n,d} with d=Q1*Q2, n=d-dot*dot."""
    q1 = _dot_int(v1, v1)
    q2 = _dot_int(v2, v2)
    if q1 == 0 or q2 == 0:
        return None
    d = q1 * q2
    dotv = _dot_int(v1, v2)
    n = d - dotv * dotv
    return {"n": n, "d": d}


def _compute_rt_for_triangle_int(
    A: List[int], B: List[int], C: List[int]
) -> Tuple[List[int], List[Dict[str, int]]]:
    """Q=[Q(BC),Q(CA),Q(AB)], s=[sA,sB,sC] via exact integer arithmetic."""
    Q1 = _quadrance_int(B, C)
    Q2 = _quadrance_int(C, A)
    Q3 = _quadrance_int(A, B)

    AB = _vec_sub_int(B, A)
    AC = _vec_sub_int(C, A)
    BC = _vec_sub_int(C, B)
    BA = _vec_sub_int(A, B)
    CA = _vec_sub_int(A, C)
    CB = _vec_sub_int(B, C)

    sA = _spread_pair_from_vectors_int(AB, AC)
    sB = _spread_pair_from_vectors_int(BC, BA)
    sC = _spread_pair_from_vectors_int(CA, CB)

    if sA is None or sB is None or sC is None:
        raise ValueError("Degenerate triangle (zero quadrance in spread denominator)")

    return [Q1, Q2, Q3], [sA, sB, sC]


def _exact_cross_law(
    Q: List[int], s: List[Dict[str, int]]
) -> Tuple[int, int, bool]:
    """Cross Law: t*t*s3.d == 4*Q1*Q2*(s3.d - s3.n), t = Q1+Q2-Q3."""
    Q1, Q2, Q3 = Q
    s3n, s3d = s[2]["n"], s[2]["d"]
    t = Q1 + Q2 - Q3
    lhs = t * t * s3d
    rhs = 4 * Q1 * Q2 * (s3d - s3n)
    return lhs, rhs, lhs == rhs


def _exact_pythagoras(Q: List[int]) -> bool:
    Q1, Q2, Q3 = Q
    return (Q1 == Q2 + Q3) or (Q2 == Q1 + Q3) or (Q3 == Q1 + Q2)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_validate(schema: Dict[str, Any], cert: Dict[str, Any]) -> Optional[str]:
    if jsonschema is None:
        return "jsonschema not installed"
    try:
        jsonschema.validate(instance=cert, schema=schema)
        return None
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------

def validate_cert(cert: Dict[str, Any], schema_dir: str) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1: Schema validity
    schema_path = os.path.join(schema_dir, "schema.json")
    try:
        schema = _load_json(schema_path)
    except Exception as e:
        return [_fail("gate_1_schema_validity", "Could not load schema.json",
                      {"error": str(e)})]

    err = _schema_validate(schema, cert)
    if err is not None:
        results.append(_fail("gate_1_schema_validity",
                             "Invalid schema for QA_GEOGEBRA_SCENE_ADAPTER.v1",
                             {"fail_type": "SCHEMA_INVALID", "error": err}))
        return results
    results.append(_pass("gate_1_schema_validity",
                         "Valid QA_GEOGEBRA_SCENE_ADAPTER.v1 schema"))

    # Gate 2: Determinism contract + invariant_diff + compute_substrate
    dc = cert.get("determinism_contract")
    if not isinstance(dc, dict):
        results.append(_fail("gate_2_determinism_contract",
                             "Missing determinism_contract",
                             {"fail_type": "MISSING_DETERMINISM_CONTRACT"}))
        return results

    bad_flags = [k for k in ("canonical_json", "stable_sorting", "no_rng",
                              "invariant_diff_defined") if dc.get(k) is not True]
    if bad_flags:
        results.append(_fail("gate_2_determinism_contract",
                             "Determinism contract flags not all true",
                             {"fail_type": "MISSING_DETERMINISM_CONTRACT",
                              "bad_flags": bad_flags}))
        return results

    if cert.get("compute_substrate") != "qa_rational_pair_noreduce":
        results.append(_fail("gate_2_determinism_contract",
                             "Unsupported compute_substrate",
                             {"fail_type": "UNSUPPORTED_COMPUTE_SUBSTRATE",
                              "compute_substrate": cert.get("compute_substrate")}))
        return results

    invd = cert.get("result", {}).get("invariant_diff")
    if not isinstance(invd, dict):
        results.append(_fail("gate_2_determinism_contract",
                             "Missing result.invariant_diff (Gate 2 owns this check)",
                             {"fail_type": "MISSING_INVARIANT_DIFF"}))
        return results

    results.append(_pass("gate_2_determinism_contract",
                         "Determinism contract strict + invariant_diff present "
                         "+ substrate=qa_rational_pair_noreduce"))

    # Gate 3: Typed formation + scene_raw_sha256 + Z/Q coord typing
    scene = cert.get("scene", {})
    scene_raw = scene.get("scene_raw")
    scene_sha = scene.get("scene_raw_sha256")
    if not isinstance(scene_raw, dict) or not isinstance(scene_sha, str):
        results.append(_fail("gate_3_typed_formation", "Invalid scene fields",
                             {"fail_type": "TYPED_FORMATION_ERROR"}))
        return results

    computed_scene_sha = _sha256_hex(_canonical_json_compact(scene_raw))
    if computed_scene_sha != scene_sha:
        results.append(_fail("gate_3_typed_formation", "scene_raw_sha256 mismatch",
                             {"fail_type": "TYPED_FORMATION_ERROR",
                              "claimed": scene_sha, "computed": computed_scene_sha}))
        return results

    parsed = cert.get("derivation", {}).get("parsed_objects", [])
    if not isinstance(parsed, list):
        results.append(_fail("gate_3_typed_formation", "Missing derivation.parsed_objects",
                             {"fail_type": "TYPED_FORMATION_ERROR"}))
        return results

    # Stable sort check
    obj_ids = []
    for o in parsed:
        if not isinstance(o, dict) or "object_id" not in o:
            results.append(_fail("gate_3_typed_formation", "Malformed parsed object",
                                 {"fail_type": "TYPED_FORMATION_ERROR"}))
            return results
        obj_ids.append(o["object_id"])
    if obj_ids != sorted(obj_ids):
        results.append(_fail("gate_3_typed_formation",
                             "parsed_objects not sorted by object_id",
                             {"fail_type": "TYPED_FORMATION_ERROR",
                              "reason": "UNSTABLE_ORDERING"}))
        return results

    # Coord typing + zero-denominator check
    for o in parsed:
        if o.get("object_type") != "TRIANGLE":
            results.append(_fail("gate_3_typed_formation", "Unsupported object_type",
                                 {"fail_type": "TYPED_FORMATION_ERROR",
                                  "object_type": o.get("object_type")}))
            return results
        verts = o.get("vertices", [])
        if not isinstance(verts, list) or len(verts) != 3:
            results.append(_fail("gate_3_typed_formation", "Triangle must have 3 vertices",
                                 {"fail_type": "TYPED_FORMATION_ERROR"}))
            return results
        oid = o["object_id"]
        for vi, v in enumerate(verts):
            coord = v.get("coord", [])
            if not isinstance(coord, list) or len(coord) != 3:
                results.append(_fail("gate_3_typed_formation", "Vertex coord must be length-3",
                                     {"fail_type": "TYPED_FORMATION_ERROR",
                                      "object_id": oid,
                                      "target_path": f"derivation.parsed_objects[{oid}].vertices[{vi}].coord"}))
                return results
            for ci, c in enumerate(coord):
                tp = f"derivation.parsed_objects[{oid}].vertices[{vi}].coord[{ci}]"
                if not (_is_coord_Z(c) or _is_coord_Q(c)):
                    results.append(_fail("gate_3_typed_formation",
                                         "Coordinate is not Z or Q rational pair",
                                         {"fail_type": "NON_RATIONAL_COORDINATE",
                                          "reason": "INVALID_COORDINATE_TYPE",
                                          "object_id": oid,
                                          "target_path": tp,
                                          "bad_coord": c}))
                    return results
                if isinstance(c, dict) and c.get("k") == "Q" and int(c.get("d", 1)) == 0:
                    results.append(_fail("gate_3_typed_formation",
                                         "Zero denominator in rational coordinate",
                                         {"fail_type": "ZERO_DENOMINATOR",
                                          "reason": "RATIONAL_PAIR_DENOMINATOR_ZERO",
                                          "object_id": oid,
                                          "target_path": tp,
                                          "bad_coord": c}))
                    return results

        # LCM lift feasibility check
        denoms = [_coord_den(c) for v in verts for c in v["coord"]]
        L = 1
        for d in denoms:
            L = _lcm(L, d)
            if L == 0:
                results.append(_fail("gate_3_typed_formation",
                                     "LCM collapsed to zero (zero denominator?)",
                                     {"fail_type": "TYPED_FORMATION_ERROR"}))
                return results

    results.append(_pass("gate_3_typed_formation",
                         f"Typed formation OK ({len(parsed)} object(s), stable-sorted, "
                         "all coords Z/Q with nonzero denom)"))

    # Gate 4: Base algebra adequacy
    props = cert.get("base_algebra", {}).get("properties", {})
    if not (isinstance(props, dict)
            and props.get("field") is True
            and props.get("no_zero_divisors") is True):
        results.append(_fail("gate_4_base_algebra_adequacy", "Base algebra inadequate",
                             {"fail_type": "BASE_ALGEBRA_INADEQUATE",
                              "properties": props}))
        return results
    results.append(_pass("gate_4_base_algebra_adequacy",
                         "Base algebra adequate (field + no_zero_divisors)"))

    # Gate 5: Step hash chain + exact RT recomputation + strict law verification
    steps = cert.get("derivation", {}).get("steps", [])

    # 5a: Step hash verification
    for st in steps:
        move_id = st.get("move_id", "")
        inputs = st.get("inputs", {})
        outputs = st.get("outputs", {})
        claimed = st.get("step_hash_sha256", "")
        computed = _step_hash(move_id, inputs, outputs)
        if computed != claimed:
            results.append(_fail("gate_5_step_hash_and_rt", "step_hash_sha256 mismatch",
                                 {"fail_type": "STEP_HASH_MISMATCH",
                                  "step_id": st.get("step_id"),
                                  "claimed": claimed, "computed": computed}))
            return results

    # 5b: Law step outputs.verified check
    for st in steps:
        if st.get("move_id") != "RT_VALIDATE_LAW_EQUATION":
            continue
        out = st.get("outputs", {})
        if out.get("verified") is not True:
            results.append(_fail("gate_5_step_hash_and_rt",
                                 "Law step outputs.verified != true",
                                 {"fail_type": "LAW_EQUATION_MISMATCH",
                                  "step_id": st.get("step_id"),
                                  "law_id": st.get("inputs", {}).get("law_id")}))
            return results

    # 5c: Exact RT recomputation and pair identity check
    tri_claims = cert.get("result", {}).get("rt_invariants", {}).get("triangles", [])
    parsed_by_id: Dict[str, Any] = {o["object_id"]: o for o in parsed}

    for tc in tri_claims:
        tri_id = tc.get("triangle_object_id")
        if tri_id not in parsed_by_id:
            results.append(_fail("gate_5_step_hash_and_rt",
                                 "triangle_object_id not found in parsed_objects",
                                 {"fail_type": "TYPED_FORMATION_ERROR",
                                  "triangle_object_id": tri_id}))
            return results

        pobj = parsed_by_id[tri_id]
        verts = pobj["vertices"]

        # Compute LCM of denominators (using denom as-given, no reduction)
        denoms = [_coord_den(c) for v in verts for c in v["coord"]]
        L = 1
        for d in denoms:
            L = _lcm(L, d)

        A = _lift_point_to_int_xyz(verts[0]["coord"], L)
        B = _lift_point_to_int_xyz(verts[1]["coord"], L)
        C = _lift_point_to_int_xyz(verts[2]["coord"], L)

        try:
            Qx, sx = _compute_rt_for_triangle_int(A, B, C)
        except ValueError as e:
            results.append(_fail("gate_5_step_hash_and_rt",
                                 f"RT computation failed: {e}",
                                 {"fail_type": "ZERO_DIVISOR_OBSTRUCTION",
                                  "triangle_object_id": tri_id}))
            return results

        # Quadrance pair check (integer equality)
        claimed_Q = tc.get("Q", [])
        if claimed_Q != Qx:
            results.append(_fail("gate_5_step_hash_and_rt", "Quadrance Q mismatch",
                                 {"fail_type": "LAW_EQUATION_MISMATCH",
                                  "triangle_object_id": tri_id,
                                  "claimed": claimed_Q, "computed": Qx}))
            return results

        # Spread pair identity check (unreduced — exact pair equality)
        claimed_s = tc.get("s", [])
        if not isinstance(claimed_s, list) or len(claimed_s) != 3:
            results.append(_fail("gate_5_step_hash_and_rt", "Spread array malformed",
                                 {"fail_type": "LAW_EQUATION_MISMATCH",
                                  "triangle_object_id": tri_id}))
            return results

        spread_labels = ["sA", "sB", "sC"]
        for j in range(3):
            cs = claimed_s[j]
            xs = sx[j]
            if not isinstance(cs, dict) or "n" not in cs or "d" not in cs:
                results.append(_fail("gate_5_step_hash_and_rt", "Spread entry malformed",
                                     {"fail_type": "LAW_EQUATION_MISMATCH",
                                      "triangle_object_id": tri_id}))
                return results
            if int(cs["n"]) != int(xs["n"]) or int(cs["d"]) != int(xs["d"]):
                target_path = f"result.rt_invariants.triangles[{tri_id}].s[{j}]"
                results.append(_fail(
                    "gate_5_step_hash_and_rt",
                    f"Spread {spread_labels[j]} pair identity mismatch: "
                    f"claimed={{n:{cs['n']},d:{cs['d']}}} "
                    f"computed={{n:{xs['n']},d:{xs['d']}}}",
                    {"fail_type": "ILLEGAL_NORMALIZATION",
                     "reason": "PAIR_IDENTITY_MISMATCH",
                     "triangle_object_id": tri_id,
                     "target_path": target_path,
                     "claimed": cs, "computed": xs}))
                return results

        # 5d: Strict law verification for RT_VALIDATE_LAW_EQUATION steps
        for st in steps:
            if st.get("move_id") != "RT_VALIDATE_LAW_EQUATION":
                continue
            law_id = st.get("inputs", {}).get("law_id")
            if law_id == "RT_LAW_04":
                lhs, rhs, ok = _exact_cross_law(Qx, sx)
                if not ok:
                    results.append(_fail("gate_5_step_hash_and_rt",
                                         "Cross Law (RT_LAW_04) failed strict equality",
                                         {"fail_type": "LAW_EQUATION_MISMATCH",
                                          "triangle_object_id": tri_id,
                                          "lhs": lhs, "rhs": rhs}))
                    return results
            elif law_id == "RT_LAW_01":
                if not _exact_pythagoras(Qx):
                    results.append(_fail("gate_5_step_hash_and_rt",
                                         "Pythagoras (RT_LAW_01) failed strict equality",
                                         {"fail_type": "LAW_EQUATION_MISMATCH",
                                          "triangle_object_id": tri_id, "Q": Qx}))
                    return results

    results.append(_pass("gate_5_step_hash_and_rt",
                         "Step hashes verified + exact RT recomputation matches "
                         "+ strict law checks pass"))
    return results


# ---------------------------------------------------------------------------
# CLI + self-test
# ---------------------------------------------------------------------------

def _results_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_results(results: List[GateResult]) -> int:
    for r in results:
        prefix = "[PASS]" if r.status == GateStatus.PASS else "[FAIL]"
        print(f"{prefix} {r.gate_id}: {r.message}")
        if r.status == GateStatus.FAIL and r.details:
            print(json.dumps({"details": r.details}, indent=2, sort_keys=True))
    ok = _results_ok(results)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def self_test(schema_dir: str, fixtures_dir: str) -> int:
    print("=== QA_GEOGEBRA_SCENE_ADAPTER.v1 SELF-TEST ===")
    fixture_cases = [
        ("valid_exact_345_triangle.json",      True,  None),
        ("invalid_missing_invariant_diff.json", False, "gate_2_determinism_contract"),
        ("invalid_zero_denominator.json",       False, "gate_3_typed_formation"),
        ("invalid_law_equation_mismatch.json",  False, "gate_5_step_hash_and_rt"),
    ]
    ok = True
    for fname, expect_pass, expect_gate in fixture_cases:
        path = os.path.join(fixtures_dir, fname)
        cert = _load_json(path)
        res = validate_cert(cert, schema_dir=schema_dir)
        passed = _results_ok(res)
        if passed != expect_pass:
            ok = False
            print(f"{fname}: FAIL (expected {'PASS' if expect_pass else 'FAIL'}, "
                  f"got {'PASS' if passed else 'FAIL'})")
            continue
        if not expect_pass and expect_gate is not None:
            first_fail_gate = next(
                (r.gate_id for r in res if r.status == GateStatus.FAIL), None)
            if first_fail_gate != expect_gate:
                ok = False
                print(f"{fname}: FAIL (expected first fail at {expect_gate}, "
                      f"got {first_fail_gate})")
                continue
        print(f"{fname}: PASS (expected {'PASS' if expect_pass else 'FAIL'})")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_GEOGEBRA_SCENE_ADAPTER.v1 validator")
    ap.add_argument("path", nargs="?", help="Certificate JSON to validate")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    schema_dir = here
    fixtures_dir = os.path.join(here, "fixtures")

    if args.self_test:
        return self_test(schema_dir=schema_dir, fixtures_dir=fixtures_dir)

    if not args.path:
        ap.print_help()
        return 2

    cert = _load_json(args.path)
    results = validate_cert(cert, schema_dir=schema_dir)
    if args.json:
        out = {"ok": _results_ok(results),
               "results": [{"gate_id": r.gate_id, "status": r.status.value,
                             "message": r.message, "details": r.details}
                            for r in results]}
        print(_canonical_json_compact(out))
        return 0 if _results_ok(results) else 1
    return _print_results(results)


if __name__ == "__main__":
    from typing import List, Optional  # noqa: F401 — used in type hints above
    raise SystemExit(main())
