#!/usr/bin/env python3
"""
validator.py

QA_RATIONAL_TRIG_TYPE_SYSTEM.v1 validator (Machine Tract).

This family encodes:
1) "Types vs Sets" semantics (objects governed by formation/use rules),
2) Rational Trigonometry (quadrance + spread) laws as QA generator moves.

RT objects are typed state manifolds (Point2, Line2, Triangle) with explicit
formation rules. RT theorems are generator moves that add constraints or derive
new invariants. Failures are typed obstructions (degenerate triangle, algebra
too weak, zero divisor), not "set membership" failures.

Hash spec:
deterministic_hash per step = sha256(canonical_json({uses_law_id, inputs, outputs}))
where canonical_json uses sort_keys=True, separators=(',',':'), ensure_ascii=False.
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


FAIL_TYPES = {
    "MISSING_INVARIANT_DIFF",
    "NONDETERMINISM_CONTRACT_VIOLATION",
    "BASE_ALGEBRA_TOO_WEAK",
    "ZERO_DIVISOR_OBSTRUCTION",
    "DEGENERATE_TRIANGLE_COLLINEAR",
    "LAW_PRECONDITION_FAILED",
    "LAW_EQUATION_MISMATCH",
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


def _expect_number_pair(value: Any, path: str) -> None:
    _expect(isinstance(value, list) and len(value) == 2, f"{path} must be length-2 array")
    _expect(all(isinstance(item, (int, float)) for item in value), f"{path} entries must be numbers")


def _validate_schema_minimal(obj: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Small dependency-free checker for this family's concrete schema.

    It intentionally covers the structure used by the fixtures and leaves
    semantic requirements, such as invariant_diff presence and triangle
    non-collinearity, to the dedicated validator gates below.
    """

    _expect_type(obj, dict, "$")
    _expect_required(obj, schema.get("required", []), "$")
    _expect(obj.get("schema_version") == "v1", "schema_version must be v1")
    _expect(obj.get("cert_type") == "QA_RATIONAL_TRIG_TYPE_SYSTEM.v1", "cert_type mismatch")
    _expect_type(obj.get("cert_id"), str, "cert_id")
    _expect_type(obj.get("created_utc"), str, "created_utc")

    source = obj.get("source_semantics")
    _expect_type(source, dict, "source_semantics")
    _expect_required(source, ["video_ref", "rt_refs"], "source_semantics")
    video = source["video_ref"]
    _expect_type(video, dict, "source_semantics.video_ref")
    _expect_required(video, ["url", "timestamp_seconds", "claim_summary"], "source_semantics.video_ref")
    _expect_type(video["url"], str, "source_semantics.video_ref.url")
    _expect(isinstance(video["timestamp_seconds"], int) and video["timestamp_seconds"] >= 0,
            "source_semantics.video_ref.timestamp_seconds must be nonnegative integer")
    _expect_type(video["claim_summary"], str, "source_semantics.video_ref.claim_summary")
    _expect(isinstance(source["rt_refs"], list) and len(source["rt_refs"]) >= 1,
            "source_semantics.rt_refs must be nonempty list")

    type_system = obj.get("type_system")
    _expect_type(type_system, dict, "type_system")
    _expect_required(type_system, ["base_algebra", "types", "formation_rules"], "type_system")
    base = type_system["base_algebra"]
    _expect_type(base, dict, "type_system.base_algebra")
    _expect_required(base, ["kind", "assumptions"], "type_system.base_algebra")
    _expect(base["kind"] in {"field", "integral_domain", "commutative_ring", "semiring"},
            "type_system.base_algebra.kind invalid")
    _expect_type(base["assumptions"], list, "type_system.base_algebra.assumptions")
    types = type_system["types"]
    _expect_type(types, dict, "type_system.types")
    _expect_required(types, ["Point2", "Line2", "Triangle"], "type_system.types")
    _expect(isinstance(type_system["formation_rules"], list) and len(type_system["formation_rules"]) >= 1,
            "type_system.formation_rules must be nonempty list")

    rt_core = obj.get("rt_core")
    _expect_type(rt_core, dict, "rt_core")
    _expect_required(rt_core, ["primitives", "laws"], "rt_core")
    primitives = rt_core["primitives"]
    _expect_type(primitives, dict, "rt_core.primitives")
    _expect_required(primitives, ["quadrance", "spread"], "rt_core.primitives")
    for name in ("quadrance", "spread"):
        _expect_type(primitives[name], dict, f"rt_core.primitives.{name}")
        _expect_required(primitives[name], ["definition"], f"rt_core.primitives.{name}")
        _expect_type(primitives[name]["definition"], str, f"rt_core.primitives.{name}.definition")
    _expect_type(rt_core["laws"], list, "rt_core.laws")

    derivation = obj.get("derivation")
    _expect_type(derivation, dict, "derivation")
    _expect_required(derivation, ["typed_state", "steps"], "derivation")
    typed_state = derivation["typed_state"]
    _expect_type(typed_state, dict, "derivation.typed_state")
    _expect_required(typed_state, ["triangle", "observables"], "derivation.typed_state")
    triangle = typed_state["triangle"]
    _expect_type(triangle, dict, "derivation.typed_state.triangle")
    _expect_required(triangle, ["A", "B", "C", "non_collinear_proof"], "derivation.typed_state.triangle")
    for vertex in ("A", "B", "C"):
        _expect_number_pair(triangle[vertex], f"derivation.typed_state.triangle.{vertex}")
    _expect_type(triangle["non_collinear_proof"], str, "derivation.typed_state.triangle.non_collinear_proof")
    observables = typed_state["observables"]
    _expect_type(observables, dict, "derivation.typed_state.observables")
    _expect_required(observables, ["Q1", "Q2", "Q3", "s1", "s2", "s3"], "derivation.typed_state.observables")
    for name in ("Q1", "Q2", "Q3", "s1", "s2", "s3"):
        _expect(isinstance(observables[name], (int, float)), f"derivation.typed_state.observables.{name} must be number")

    steps = derivation["steps"]
    _expect_type(steps, list, "derivation.steps")
    for idx, step in enumerate(steps):
        spath = f"derivation.steps[{idx}]"
        _expect_type(step, dict, spath)
        _expect_required(step, ["step_id", "uses_law_id", "inputs", "outputs", "deterministic_hash"], spath)
        _expect_type(step["step_id"], str, f"{spath}.step_id")
        _expect_type(step["uses_law_id"], str, f"{spath}.uses_law_id")
        _expect_type(step["inputs"], dict, f"{spath}.inputs")
        _expect_type(step["outputs"], dict, f"{spath}.outputs")
        _expect(isinstance(step["deterministic_hash"], str) and len(step["deterministic_hash"]) == 64,
                f"{spath}.deterministic_hash must be 64-char hex string")

    dc = obj.get("determinism_contract")
    _expect_type(dc, dict, "determinism_contract")
    _expect_required(dc, ["canonical_json", "no_rng", "stable_sorting", "hash_rule"], "determinism_contract")


def _collinear(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    eps: float = 1e-12,
) -> bool:
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return abs(det) <= eps


def _step_hash(step: Dict[str, Any]) -> str:
    payload = {
        "uses_law_id": step.get("uses_law_id"),
        "inputs": step.get("inputs"),
        "outputs": step.get("outputs"),
    }
    return _sha256_hex(_canonical_json_compact(payload))


def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 -- Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.PASS,
            "Valid QA_RATIONAL_TRIG_TYPE_SYSTEM.v1 schema"))
    except Exception as e:
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.FAIL,
            f"Schema validation failed: {e}"))
        return results

    # Gate 2 -- Determinism contract + invariant_diff presence
    dc = obj.get("determinism_contract", {})
    if (
        dc.get("canonical_json") is not True
        or dc.get("no_rng") is not True
        or dc.get("stable_sorting") is not True
    ):
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            "Determinism contract flags not strict",
            {"determinism_contract": dc}))
        return results

    invd = obj.get("invariant_diff")
    if (
        not isinstance(invd, dict)
        or "invariants" not in invd
        or "diffs" not in invd
    ):
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            "invariant_diff missing or malformed",
            {"fail_type": "MISSING_INVARIANT_DIFF"}))
        return results

    if not isinstance(invd["invariants"], list) or len(invd["invariants"]) < 1:
        results.append(GateResult(
            "gate_2_determinism_contract", GateStatus.FAIL,
            "invariant_diff.invariants must be non-empty list",
            {"fail_type": "MISSING_INVARIANT_DIFF"}))
        return results

    results.append(GateResult(
        "gate_2_determinism_contract", GateStatus.PASS,
        "Determinism contract strict + invariant_diff present"))

    # Gate 3 -- Typed formation: Triangle non-collinearity
    ts = obj.get("derivation", {}).get("typed_state", {})
    tri = ts.get("triangle", {})
    try:
        A = (float(tri["A"][0]), float(tri["A"][1]))
        B = (float(tri["B"][0]), float(tri["B"][1]))
        C = (float(tri["C"][0]), float(tri["C"][1]))
    except Exception:
        results.append(GateResult(
            "gate_3_typed_formation", GateStatus.FAIL,
            "Triangle coordinates invalid or missing",
            {"fail_type": "LAW_PRECONDITION_FAILED"}))
        return results

    if _collinear(A, B, C):
        results.append(GateResult(
            "gate_3_typed_formation", GateStatus.FAIL,
            "Triangle points are collinear: formation rule violated",
            {"fail_type": "DEGENERATE_TRIANGLE_COLLINEAR",
             "A": list(A), "B": list(B), "C": list(C)}))
        return results

    results.append(GateResult(
        "gate_3_typed_formation", GateStatus.PASS,
        "Triangle non-collinear (typed formation satisfied)"))

    # Gate 4 -- Base algebra adequacy for chosen laws
    base = obj.get("type_system", {}).get("base_algebra", {})
    kind = base.get("kind")
    assumptions = base.get("assumptions", [])

    steps = obj.get("derivation", {}).get("steps", [])
    division_laws = {"RT_LAW_03"}  # Spread law requires division
    for st in steps:
        uses = st.get("uses_law_id", "")
        if uses in division_laws:
            if kind not in ("field", "integral_domain") and "no_zero_divisors" not in assumptions:
                results.append(GateResult(
                    "gate_4_base_algebra_adequacy", GateStatus.FAIL,
                    f"Law {uses} requires integral_domain/field for division",
                    {"fail_type": "BASE_ALGEBRA_TOO_WEAK",
                     "kind": kind, "assumptions": assumptions}))
                return results

    results.append(GateResult(
        "gate_4_base_algebra_adequacy", GateStatus.PASS,
        "Base algebra adequate for all used laws"))

    # Gate 5 -- Step deterministic hash verification
    bad_steps = []
    for i, st in enumerate(steps):
        expected = _step_hash(st)
        got = st.get("deterministic_hash", "")
        if got != expected:
            bad_steps.append({
                "step_index": i,
                "step_id": st.get("step_id"),
                "expected": expected,
                "got": got,
            })

    if bad_steps:
        results.append(GateResult(
            "gate_5_step_determinism", GateStatus.FAIL,
            "Step deterministic_hash mismatch",
            {"fail_type": "NONDETERMINISM_CONTRACT_VIOLATION",
             "bad_steps": bad_steps}))
        return results

    results.append(GateResult(
        "gate_5_step_determinism", GateStatus.PASS,
        "All step hashes match canonical recomputation"))

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
        ("invalid_missing_invariant_diff.json", False, "gate_2_determinism_contract"),
        ("invalid_degenerate_triangle.json", False, "gate_3_typed_formation"),
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
        print("=== QA_RATIONAL_TRIG_TYPE_SYSTEM.v1 SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_RATIONAL_TRIG_TYPE_SYSTEM.v1 validator")
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
