#!/usr/bin/env python3
"""
validator.py

QA_MAPPING_PROTOCOL.v1 validator (Machine Tract).

Gates (per formal_mapping_to_qa.odt):
  Gate 1 — Schema Validity
  Gate 2 — Invariant Diff Enforcement (generator invariant_effect present)
  Gate 3 — Failure Completeness (non-empty generators + non-empty failure_taxonomy)
  Gate 4 — Determinism Contract (invariant_diff_defined==true and proof non-empty)
  Gate 5 — State Constraint Closure (constraints array non-empty)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


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


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _validate_schema(obj: Dict[str, Any]) -> None:
    import jsonschema

    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


def validate_mapping(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema Validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Valid QA_MAPPING_PROTOCOL.v1 schema"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL, f"Schema validation failed: {e}"))
        return results

    # Gate 2 — Invariant Diff Enforcement
    generators = obj.get("generators", [])
    missing = []
    if isinstance(generators, list):
        for i, g in enumerate(generators):
            inv_eff = None
            if isinstance(g, dict):
                inv_eff = g.get("invariant_effect")
            if not isinstance(inv_eff, str) or not inv_eff.strip():
                missing.append(i)
    if missing:
        results.append(GateResult(
            "gate_2_invariant_diff_enforcement",
            GateStatus.FAIL,
            "Some generators missing invariant_effect",
            {"missing_generator_indices": missing},
        ))
    else:
        results.append(GateResult(
            "gate_2_invariant_diff_enforcement",
            GateStatus.PASS,
            "All generators declare invariant_effect",
        ))

    # Gate 3 — Failure Completeness
    failure_taxonomy = obj.get("failure_taxonomy", [])
    ok_generators = isinstance(generators, list) and len(generators) > 0
    ok_failures = isinstance(failure_taxonomy, list) and len(failure_taxonomy) > 0
    if ok_generators and ok_failures:
        results.append(GateResult(
            "gate_3_failure_completeness",
            GateStatus.PASS,
            "Non-empty generators and failure_taxonomy present",
            {"generator_count": len(generators), "failure_count": len(failure_taxonomy)},
        ))
    else:
        results.append(GateResult(
            "gate_3_failure_completeness",
            GateStatus.FAIL,
            "Missing generator or failure coverage",
            {"generator_count": len(generators) if isinstance(generators, list) else None,
             "failure_count": len(failure_taxonomy) if isinstance(failure_taxonomy, list) else None},
        ))

    # Gate 4 — Determinism Contract
    dc = obj.get("determinism_contract", {})
    inv_diff_defined = isinstance(dc, dict) and dc.get("invariant_diff_defined") is True
    proof = dc.get("nondeterminism_proof") if isinstance(dc, dict) else None
    proof_ok = isinstance(proof, str) and bool(proof.strip())
    if inv_diff_defined and proof_ok:
        results.append(GateResult(
            "gate_4_determinism_contract",
            GateStatus.PASS,
            "Determinism contract satisfied",
        ))
    else:
        results.append(GateResult(
            "gate_4_determinism_contract",
            GateStatus.FAIL,
            "Determinism contract missing/invalid",
            {
                "invariant_diff_defined": dc.get("invariant_diff_defined") if isinstance(dc, dict) else None,
                "nondeterminism_proof_present": proof_ok,
            },
        ))

    # Gate 5 — State Constraint Closure
    sm = obj.get("state_manifold", {})
    constraints = sm.get("constraints") if isinstance(sm, dict) else None
    ok_constraints = isinstance(constraints, list) and len(constraints) > 0 and all(
        isinstance(c, str) and bool(c.strip()) for c in constraints
    )
    if ok_constraints:
        results.append(GateResult(
            "gate_5_state_constraint_closure",
            GateStatus.PASS,
            "State constraints explicitly listed",
            {"constraint_count": len(constraints)},
        ))
    else:
        results.append(GateResult(
            "gate_5_state_constraint_closure",
            GateStatus.FAIL,
            "State constraints missing/empty",
            {"constraints": constraints},
        ))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json(results: List[GateResult]) -> None:
    payload = {
        "ok": _report_ok(results),
        "results": [r.to_dict() for r in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    valid = _load_json(os.path.join(base, "fixtures", "valid_min.json"))
    invalid = _load_json(os.path.join(base, "fixtures", "invalid_missing_determinism.json"))

    ok = True

    vr = validate_mapping(valid)
    ir = validate_mapping(invalid)

    ok = ok and _report_ok(vr)
    ok = ok and not _report_ok(ir)

    # Assert the intended failing gates for invalid fixture
    failed_gates = {r.gate for r in ir if r.status == GateStatus.FAIL}
    expected = {
        "gate_3_failure_completeness",
        "gate_4_determinism_contract",
        "gate_5_state_constraint_closure",
    }
    ok = ok and expected.issubset(failed_gates)

    if as_json:
        print(json.dumps({
            "ok": ok,
            "valid_ok": _report_ok(vr),
            "invalid_ok": _report_ok(ir),
            "invalid_failed_gates": sorted(failed_gates),
        }, indent=2, sort_keys=True))
    else:
        print("=== QA_MAPPING_PROTOCOL.v1 SELF-TEST ===")
        print(f"valid_min.json:  {'PASS' if _report_ok(vr) else 'FAIL'}")
        print(f"invalid_missing_determinism.json:  {'PASS' if _report_ok(ir) else 'FAIL'} (expected FAIL)")
        if not ok:
            print(f"Expected invalid failing gates include: {sorted(expected)}")
            print(f"Actual invalid failing gates:        {sorted(failed_gates)}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_MAPPING_PROTOCOL.v1 validator")
    ap.add_argument("file", nargs="?", help="Mapping JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test on fixtures")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = _load_json(args.file)
    results = validate_mapping(obj)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

