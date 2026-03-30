#!/usr/bin/env python3
"""
Validate QA_DISCOVERY_PIPELINE_RUN_SCHEMA.v1,
        QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1,
        QA_DISCOVERY_BUNDLE_SCHEMA.v1.

Run gates:
  1. Schema shape (schema_id + required fields)  -> SCHEMA_INVALID
  2. Determinism declarations present             -> DETERMINISM_MISSING
  3. Step ordering (unique indices)               -> DUPLICATE_STEP_INDEX
  4. Step results (FAIL needs typed diff)         -> RESULT_INCOMPLETE
  5. Top-level result (FAIL needs typed diff)     -> RESULT_INCOMPLETE
  6. Merkle parent present                        -> MERKLE_PARENT_MISSING

Plan gates:
  1. Schema shape                                 -> SCHEMA_INVALID
  2. Determinism declarations present             -> DETERMINISM_MISSING
  3. canonical_json must be true                  -> NONDETERMINISTIC_PLAN
  4. Merkle parent present                        -> MERKLE_PARENT_MISSING

Bundle gates:
  1. Schema shape                                 -> SCHEMA_INVALID
  2. Hash chain (this_bundle_hash non-empty)      -> HASH_CHAIN_INVALID
  3. Summary sanity (runs_success <= runs_total)   -> SUMMARY_INVALID

Usage:
  python qa_discovery_pipeline_validator.py run    <file.json> [--ci]
  python qa_discovery_pipeline_validator.py plan   <file.json> [--ci]
  python qa_discovery_pipeline_validator.py bundle <file.json> [--ci]
  python qa_discovery_pipeline_validator.py --self-test

Returns 0 on PASS, 1 on FAIL, 2 on usage error.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_SCHEMA_ID = "QA_DISCOVERY_PIPELINE_RUN_SCHEMA.v1"
PLAN_SCHEMA_ID = "QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1"
BUNDLE_SCHEMA_ID = "QA_DISCOVERY_BUNDLE_SCHEMA.v1"

VALID_OPS = frozenset([
    "LOAD_EPISODE", "VALIDATE_EPISODE", "EMIT_FRONTIER",
    "EMIT_RETURN_RECEIPT", "EMIT_NEXT_SEED",
])

VALID_STATUSES = frozenset(["SUCCESS", "FAIL"])


# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self, ok: bool, fail_type: Optional[str] = None,
                 invariant_diff: Optional[Dict[str, Any]] = None,
                 obj_id: str = ""):
        self.ok = ok
        self.fail_type = fail_type
        self.invariant_diff = invariant_diff or {}
        self.obj_id = obj_id


def _fail(ft: str, diff: Dict[str, Any], obj_id: str = "") -> ValidationResult:
    return ValidationResult(False, ft, diff, obj_id)


# ===================================================================
# Run gates
# ===================================================================

def _run_check_schema(r: Dict) -> Optional[Tuple[str, Dict]]:
    if r.get("schema_id") != RUN_SCHEMA_ID:
        return ("SCHEMA_INVALID", {"expected": RUN_SCHEMA_ID, "got": r.get("schema_id")})
    for field in ("run_id", "created_utc", "agent_id", "pipeline_id",
                  "toolchain", "inputs", "execution", "outputs", "result", "merkle_parent"):
        if field not in r:
            return ("SCHEMA_INVALID", {"missing_field": field})
    # Check toolchain subfields
    tc = r.get("toolchain", {})
    for f in ("python", "os", "toolchain_id"):
        if f not in tc:
            return ("SCHEMA_INVALID", {"missing_field": f"toolchain.{f}"})
    # Check inputs subfields
    inp = r.get("inputs", {})
    for f in ("episode_ref", "k", "determinism"):
        if f not in inp:
            return ("SCHEMA_INVALID", {"missing_field": f"inputs.{f}"})
    eref = inp.get("episode_ref", {})
    for f in ("family", "path_or_hash"):
        if f not in eref:
            return ("SCHEMA_INVALID", {"missing_field": f"inputs.episode_ref.{f}"})
    # Check execution.steps exists and non-empty
    steps = r.get("execution", {}).get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return ("SCHEMA_INVALID", {"reason": "execution.steps must be non-empty array"})
    # Check each step has required fields
    for i, s in enumerate(steps):
        for f in ("step_index", "op", "result"):
            if f not in s:
                return ("SCHEMA_INVALID", {"missing_field": f"execution.steps[{i}].{f}"})
        sr = s.get("result", {})
        if "status" not in sr:
            return ("SCHEMA_INVALID", {"missing_field": f"execution.steps[{i}].result.status"})
    # Check result.status
    res = r.get("result", {})
    if "status" not in res:
        return ("SCHEMA_INVALID", {"missing_field": "result.status"})
    return None


def _run_check_determinism(r: Dict) -> Optional[Tuple[str, Dict]]:
    det = r.get("inputs", {}).get("determinism", {})
    for k in ("canonical_json", "frontier_sort", "bfs_tie_breaker"):
        if k not in det:
            return ("DETERMINISM_MISSING", {"path": f"inputs.determinism.{k}"})
    return None


def _run_check_step_ordering(r: Dict) -> Optional[Tuple[str, Dict]]:
    seen = set()
    for s in r.get("execution", {}).get("steps", []):
        idx = s.get("step_index")
        if idx in seen:
            return ("DUPLICATE_STEP_INDEX", {"step_index": idx})
        seen.add(idx)
    return None


def _run_check_step_results(r: Dict) -> Optional[Tuple[str, Dict]]:
    for s in r.get("execution", {}).get("steps", []):
        sr = s.get("result", {})
        if sr.get("status") == "FAIL":
            if "fail_type" not in sr or "invariant_diff" not in sr:
                return ("RESULT_INCOMPLETE", {
                    "path": f"execution.steps[{s.get('step_index')}].result",
                    "reason": "FAIL step missing fail_type or invariant_diff",
                })
    return None


def _run_check_top_result(r: Dict) -> Optional[Tuple[str, Dict]]:
    res = r.get("result", {})
    if res.get("status") == "FAIL":
        if "fail_type" not in res or "invariant_diff" not in res:
            return ("RESULT_INCOMPLETE", {"path": "result", "reason": "FAIL result missing fail_type or invariant_diff"})
    return None


def _run_check_merkle(r: Dict) -> Optional[Tuple[str, Dict]]:
    if not r.get("merkle_parent"):
        return ("MERKLE_PARENT_MISSING", {"path": "merkle_parent"})
    return None


# ===================================================================
# Plan gates
# ===================================================================

def _plan_check_schema(p: Dict) -> Optional[Tuple[str, Dict]]:
    if p.get("schema_id") != PLAN_SCHEMA_ID:
        return ("SCHEMA_INVALID", {"expected": PLAN_SCHEMA_ID, "got": p.get("schema_id")})
    for field in ("plan_id", "created_utc", "agent_id", "pipeline_id",
                  "run_queue", "determinism", "budget", "merkle_parent"):
        if field not in p:
            return ("SCHEMA_INVALID", {"missing_field": field})
    rq = p.get("run_queue")
    if not isinstance(rq, list) or len(rq) == 0:
        return ("SCHEMA_INVALID", {"reason": "run_queue must be non-empty array"})
    return None


def _plan_check_determinism(p: Dict) -> Optional[Tuple[str, Dict]]:
    det = p.get("determinism", {})
    for k in ("queue_ordering", "seed_policy", "canonical_json"):
        if k not in det:
            return ("DETERMINISM_MISSING", {"path": f"determinism.{k}"})
    return None


def _plan_check_canonical(p: Dict) -> Optional[Tuple[str, Dict]]:
    det = p.get("determinism", {})
    if det.get("canonical_json") is not True:
        return ("NONDETERMINISTIC_PLAN", {"reason": "canonical_json must be true"})
    return None


def _plan_check_merkle(p: Dict) -> Optional[Tuple[str, Dict]]:
    if not p.get("merkle_parent"):
        return ("MERKLE_PARENT_MISSING", {"path": "merkle_parent"})
    return None


# ===================================================================
# Bundle gates
# ===================================================================

def _bundle_check_schema(b: Dict) -> Optional[Tuple[str, Dict]]:
    if b.get("schema_id") != BUNDLE_SCHEMA_ID:
        return ("SCHEMA_INVALID", {"expected": BUNDLE_SCHEMA_ID, "got": b.get("schema_id")})
    for field in ("bundle_id", "created_utc", "agent_id", "pipeline_id",
                  "plan_ref", "run_refs", "artifact_refs", "summary", "hash_chain"):
        if field not in b:
            return ("SCHEMA_INVALID", {"missing_field": field})
    rr = b.get("run_refs")
    if not isinstance(rr, list) or len(rr) == 0:
        return ("SCHEMA_INVALID", {"reason": "run_refs must be non-empty array"})
    ar = b.get("artifact_refs")
    if not isinstance(ar, list) or len(ar) == 0:
        return ("SCHEMA_INVALID", {"reason": "artifact_refs must be non-empty array"})
    return None


def _bundle_check_hash_chain(b: Dict) -> Optional[Tuple[str, Dict]]:
    hc = b.get("hash_chain", {})
    if not hc.get("this_bundle_hash"):
        return ("HASH_CHAIN_INVALID", {"path": "hash_chain.this_bundle_hash", "reason": "must be non-empty"})
    return None


def _bundle_check_summary(b: Dict) -> Optional[Tuple[str, Dict]]:
    summ = b.get("summary", {})
    total = summ.get("runs_total", 0)
    success = summ.get("runs_success", 0)
    if not isinstance(total, int) or not isinstance(success, int):
        return ("SUMMARY_INVALID", {"reason": "runs_total and runs_success must be integers"})
    if success > total:
        return ("SUMMARY_INVALID", {"reason": "runs_success > runs_total", "total": total, "success": success})
    return None


# ===================================================================
# Public API
# ===================================================================

def validate_run(r: Dict[str, Any]) -> ValidationResult:
    """Run all run gates in order. Returns on first failure."""
    obj_id = r.get("run_id", "<unknown>")
    for gate in [_run_check_schema, _run_check_determinism, _run_check_step_ordering,
                 _run_check_step_results, _run_check_top_result, _run_check_merkle]:
        result = gate(r)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


def validate_plan(p: Dict[str, Any]) -> ValidationResult:
    """Run all plan gates in order. Returns on first failure."""
    obj_id = p.get("plan_id", "<unknown>")
    for gate in [_plan_check_schema, _plan_check_determinism,
                 _plan_check_canonical, _plan_check_merkle]:
        result = gate(p)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


def validate_bundle(b: Dict[str, Any]) -> ValidationResult:
    """Run all bundle gates in order. Returns on first failure."""
    obj_id = b.get("bundle_id", "<unknown>")
    for gate in [_bundle_check_schema, _bundle_check_hash_chain, _bundle_check_summary]:
        result = gate(b)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    """12 built-in checks. Returns True if all pass."""
    _base_dir = Path(__file__).resolve().parent
    passed = 0
    total = 0

    def _check(label: str, ok: bool, fail_type: str, expected_ft: str):
        nonlocal passed, total
        total += 1
        if ok and expected_ft == "":
            print(f"  [PASS] {label}")
            passed += 1
        elif not ok and fail_type == expected_ft:
            print(f"  [PASS] {label} -> {fail_type}")
            passed += 1
        else:
            actual = "PASS" if ok else fail_type
            print(f"  [FAIL] {label}: expected {expected_ft or 'PASS'}, got {actual}")

    print("--- qa_discovery_pipeline_validator self-test ---")

    # --- Run tests ---
    # Load valid fixture
    with open(_base_dir / "fixtures" / "run_valid.json", "r") as f:
        run_v = json.load(f)
    r = validate_run(run_v)
    _check("valid run accepted", r.ok, r.fail_type or "", "")

    # 2: wrong schema_id
    bad = copy.deepcopy(run_v)
    bad["schema_id"] = "WRONG"
    r = validate_run(bad)
    _check("run wrong schema_id -> SCHEMA_INVALID", r.ok, r.fail_type or "", "SCHEMA_INVALID")

    # 3: missing determinism field
    bad = copy.deepcopy(run_v)
    del bad["inputs"]["determinism"]["bfs_tie_breaker"]
    r = validate_run(bad)
    _check("run missing determinism field -> DETERMINISM_MISSING", r.ok, r.fail_type or "", "DETERMINISM_MISSING")

    # 4: duplicate step_index
    bad = copy.deepcopy(run_v)
    bad["execution"]["steps"][1]["step_index"] = 0
    r = validate_run(bad)
    _check("run duplicate step_index -> DUPLICATE_STEP_INDEX", r.ok, r.fail_type or "", "DUPLICATE_STEP_INDEX")

    # 5: FAIL step without invariant_diff
    bad = copy.deepcopy(run_v)
    bad["execution"]["steps"][0]["result"] = {"status": "FAIL", "fail_type": "TEST"}
    r = validate_run(bad)
    _check("run FAIL step without invariant_diff -> RESULT_INCOMPLETE", r.ok, r.fail_type or "", "RESULT_INCOMPLETE")

    # 6: top-level FAIL without fail_type
    bad = copy.deepcopy(run_v)
    bad["result"] = {"status": "FAIL"}
    r = validate_run(bad)
    _check("run top FAIL without fail_type -> RESULT_INCOMPLETE", r.ok, r.fail_type or "", "RESULT_INCOMPLETE")

    # 7: missing merkle_parent
    bad = copy.deepcopy(run_v)
    bad["merkle_parent"] = ""
    r = validate_run(bad)
    _check("run empty merkle_parent -> MERKLE_PARENT_MISSING", r.ok, r.fail_type or "", "MERKLE_PARENT_MISSING")

    # --- Plan tests ---
    with open(_base_dir / "fixtures" / "plan_valid.json", "r") as f:
        plan_v = json.load(f)
    r = validate_plan(plan_v)
    _check("valid plan accepted", r.ok, r.fail_type or "", "")

    # 9: canonical_json = false
    bad = copy.deepcopy(plan_v)
    bad["determinism"]["canonical_json"] = False
    r = validate_plan(bad)
    _check("plan canonical_json=false -> NONDETERMINISTIC_PLAN", r.ok, r.fail_type or "", "NONDETERMINISTIC_PLAN")

    # 10: missing determinism field
    bad = copy.deepcopy(plan_v)
    del bad["determinism"]["seed_policy"]
    r = validate_plan(bad)
    _check("plan missing seed_policy -> DETERMINISM_MISSING", r.ok, r.fail_type or "", "DETERMINISM_MISSING")

    # --- Bundle tests ---
    with open(_base_dir / "fixtures" / "bundle_valid.json", "r") as f:
        bundle_v = json.load(f)
    r = validate_bundle(bundle_v)
    _check("valid bundle accepted", r.ok, r.fail_type or "", "")

    # 12: empty this_bundle_hash
    bad = copy.deepcopy(bundle_v)
    bad["hash_chain"]["this_bundle_hash"] = ""
    r = validate_bundle(bad)
    _check("bundle empty hash -> HASH_CHAIN_INVALID", r.ok, r.fail_type or "", "HASH_CHAIN_INVALID")

    print(f"\n  {passed}/{total} self-tests passed")
    return passed == total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        ok = _self_test()
        return 0 if ok else 1

    if len(sys.argv) < 3:
        print("usage: qa_discovery_pipeline_validator.py (run|plan|bundle) <file.json> [--ci]")
        print("       qa_discovery_pipeline_validator.py --self-test")
        return 2

    kind = sys.argv[1]
    path = sys.argv[2]
    ci = "--ci" in sys.argv

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if kind == "run":
        r = validate_run(obj)
    elif kind == "plan":
        r = validate_plan(obj)
    elif kind == "bundle":
        r = validate_bundle(obj)
    else:
        print(f"unknown kind: {kind}")
        return 2

    basename = Path(path).name
    if r.ok:
        h = _sha256(_canonical(obj))
        if ci:
            print(f"[PASS] {basename}: valid ({h})")
        else:
            print(f"PASS  {basename}  id={r.obj_id}  hash={h}")
        return 0

    diff_s = json.dumps(r.invariant_diff, sort_keys=True)
    if ci:
        print(f"[FAIL] {basename}: {r.fail_type} {diff_s}")
    else:
        print(f"FAIL  {basename}  fail_type={r.fail_type}")
        print(f"  invariant_diff={diff_s}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
