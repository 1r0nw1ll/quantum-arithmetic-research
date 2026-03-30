#!/usr/bin/env python3
"""
Validate QA_CONJECTURE_PROVE_EPISODE_SCHEMA.v1,
        QA_FRONTIER_SNAPSHOT_SCHEMA.v1,
        QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1.

Episode gates:
  1. invariant_diff presence              -> MISSING_INVARIANT_DIFF
  2. Schema shape                         -> SCHEMA_INVALID
  3. Step ordering (unique, sequential)   -> DUPLICATE_STEP_INDEX
  4. Step results (FAIL needs typed diff) -> RESULT_INCOMPLETE
  5. Trace ref family prefix              -> INVALID_TRACE_REF

Frontier gates:
  1. invariant_diff presence              -> MISSING_INVARIANT_DIFF
  2. Schema shape                         -> SCHEMA_INVALID
  3. Hash chain (this_snapshot_hash set)  -> HASH_CHAIN_INVALID
  4. Frontier uniqueness                  -> DUPLICATE_FRONTIER_ENTRY

Bounded-return gates:
  1. invariant_diff presence              -> MISSING_INVARIANT_DIFF
  2. Schema shape                         -> SCHEMA_INVALID
  3. Result consistency                   -> RESULT_INCOMPLETE
  4. Path length <= k                     -> PATH_EXCEEDS_K

Usage:
  python qa_conjecture_prove_validator.py episode  <file.json> [--ci]
  python qa_conjecture_prove_validator.py frontier <file.json> [--ci]
  python qa_conjecture_prove_validator.py receipt  <file.json> [--ci]
  python qa_conjecture_prove_validator.py --self-test

Returns 0 on PASS, 1 on FAIL, 2 on usage error.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODE_SCHEMA_ID = "QA_CONJECTURE_PROVE_EPISODE_SCHEMA.v1"
FRONTIER_SCHEMA_ID = "QA_FRONTIER_SNAPSHOT_SCHEMA.v1"
RECEIPT_SCHEMA_ID = "QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1"

VALID_LAYERS = frozenset(["human", "formal"])
VALID_OBJ_TYPES = frozenset(["PROVE", "REFUTE", "EXPLORE"])
VALID_STATUSES = frozenset(["SUCCESS", "FAIL"])
VALID_FINAL_STATUSES = frozenset(["SUCCESS", "FAIL", "BUDGET_EXHAUSTED"])
VALID_RETURN_STATUSES = frozenset(["RETURN_FOUND", "NO_RETURN_WITHIN_K", "BUDGET_EXHAUSTED"])
VALID_ALGORITHMS = frozenset(["BFS", "IDDFS", "ASTAR"])

# Fail types emitted by this validator
SCHEMA_INVALID = "SCHEMA_INVALID"
MISSING_INVARIANT_DIFF = "MISSING_INVARIANT_DIFF"
DUPLICATE_STEP_INDEX = "DUPLICATE_STEP_INDEX"
RESULT_INCOMPLETE = "RESULT_INCOMPLETE"
INVALID_TRACE_REF = "INVALID_TRACE_REF"
HASH_CHAIN_INVALID = "HASH_CHAIN_INVALID"
DUPLICATE_FRONTIER_ENTRY = "DUPLICATE_FRONTIER_ENTRY"
PATH_EXCEEDS_K = "PATH_EXCEEDS_K"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _is_utc_ts(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", v))


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

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"ok": self.ok, "obj_id": self.obj_id}
        if not self.ok:
            d["fail_type"] = self.fail_type
            d["invariant_diff"] = self.invariant_diff
        return d


# ===================================================================
# EPISODE VALIDATION
# ===================================================================

def _ep_check_invariant_diff(ep: Dict) -> Optional[Tuple[str, Dict]]:
    if "invariant_diff" not in ep or not isinstance(ep.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {"path": "$.invariant_diff"}
    return None


def _ep_check_schema(ep: Dict) -> Optional[Tuple[str, Dict]]:
    required = [
        "schema_id", "episode_id", "created_utc", "agent_id", "policy_id",
        "generator_set_id", "objective", "initial_state", "steps",
        "final_status", "merkle_parent", "invariant_diff",
    ]
    for f in required:
        if f not in ep:
            return SCHEMA_INVALID, {"missing_field": f, "path": f"$.{f}"}

    if ep["schema_id"] != EPISODE_SCHEMA_ID:
        return SCHEMA_INVALID, {"expected": EPISODE_SCHEMA_ID, "got": ep["schema_id"]}
    if not isinstance(ep["episode_id"], str) or len(ep["episode_id"]) < 8:
        return SCHEMA_INVALID, {"path": "$.episode_id", "reason": "must be string >=8 chars"}
    if not _is_utc_ts(ep["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc"}
    for sf in ["agent_id", "policy_id", "generator_set_id"]:
        if not isinstance(ep[sf], str) or not ep[sf]:
            return SCHEMA_INVALID, {"path": f"$.{sf}", "reason": "must be non-empty string"}

    # objective
    obj = ep["objective"]
    if not isinstance(obj, dict):
        return SCHEMA_INVALID, {"path": "$.objective"}
    if obj.get("type") not in VALID_OBJ_TYPES:
        return SCHEMA_INVALID, {"path": "$.objective.type", "expected_one_of": sorted(VALID_OBJ_TYPES)}
    if not _is_hex64(obj.get("target_hash", "")):
        return SCHEMA_INVALID, {"path": "$.objective.target_hash", "reason": "must be hex64"}

    # initial_state
    init = ep["initial_state"]
    if not isinstance(init, dict):
        return SCHEMA_INVALID, {"path": "$.initial_state"}
    if init.get("layer") not in VALID_LAYERS:
        return SCHEMA_INVALID, {"path": "$.initial_state.layer"}
    if not _is_hex64(init.get("state_hash", "")):
        return SCHEMA_INVALID, {"path": "$.initial_state.state_hash"}

    # steps
    steps = ep["steps"]
    if not isinstance(steps, list) or len(steps) < 1:
        return SCHEMA_INVALID, {"path": "$.steps", "reason": "must be non-empty array"}
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            return SCHEMA_INVALID, {"path": f"$.steps[{i}]"}
        for sf in ["step_index", "action", "input_hash", "output_hash", "trace_ref", "result"]:
            if sf not in s:
                return SCHEMA_INVALID, {"missing_field": sf, "path": f"$.steps[{i}].{sf}"}
        if not isinstance(s.get("step_index"), int):
            return SCHEMA_INVALID, {"path": f"$.steps[{i}].step_index", "reason": "must be integer"}
        if not _is_hex64(s.get("input_hash", "")):
            return SCHEMA_INVALID, {"path": f"$.steps[{i}].input_hash"}
        if not _is_hex64(s.get("output_hash", "")):
            return SCHEMA_INVALID, {"path": f"$.steps[{i}].output_hash"}
        r = s.get("result", {})
        if not isinstance(r, dict) or r.get("status") not in VALID_STATUSES:
            return SCHEMA_INVALID, {"path": f"$.steps[{i}].result.status"}

    # final_status
    fs = ep["final_status"]
    if not isinstance(fs, dict) or fs.get("status") not in VALID_FINAL_STATUSES:
        return SCHEMA_INVALID, {"path": "$.final_status.status"}
    if not _is_hex64(ep.get("merkle_parent", "")):
        return SCHEMA_INVALID, {"path": "$.merkle_parent"}

    return None


def _ep_check_step_ordering(ep: Dict) -> Optional[Tuple[str, Dict]]:
    seen = set()
    for s in ep["steps"]:
        idx = s["step_index"]
        if idx in seen:
            return DUPLICATE_STEP_INDEX, {"duplicate_index": idx}
        seen.add(idx)
    return None


def _ep_check_step_results(ep: Dict) -> Optional[Tuple[str, Dict]]:
    for s in ep["steps"]:
        r = s["result"]
        if r["status"] == "FAIL":
            if "fail_type" not in r:
                return RESULT_INCOMPLETE, {"path": f"$.steps[{s['step_index']}].result.fail_type"}
            if "invariant_diff" not in r or not isinstance(r.get("invariant_diff"), dict):
                return RESULT_INCOMPLETE, {"path": f"$.steps[{s['step_index']}].result.invariant_diff"}
    return None


def _ep_check_trace_refs(ep: Dict) -> Optional[Tuple[str, Dict]]:
    for s in ep["steps"]:
        tr = s.get("trace_ref", {})
        fam = tr.get("family", "")
        if not isinstance(fam, str) or not fam.startswith("QA_"):
            return INVALID_TRACE_REF, {
                "path": f"$.steps[{s['step_index']}].trace_ref.family",
                "got": fam, "reason": "must start with 'QA_'",
            }
    return None


def validate_episode(ep: Dict) -> ValidationResult:
    obj_id = ep.get("episode_id", "<unknown>")
    for gate in [_ep_check_invariant_diff, _ep_check_schema,
                 _ep_check_step_ordering, _ep_check_step_results,
                 _ep_check_trace_refs]:
        result = gate(ep)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# FRONTIER SNAPSHOT VALIDATION
# ===================================================================

def _fr_check_invariant_diff(snap: Dict) -> Optional[Tuple[str, Dict]]:
    if "invariant_diff" not in snap or not isinstance(snap.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {"path": "$.invariant_diff"}
    return None


def _fr_check_schema(snap: Dict) -> Optional[Tuple[str, Dict]]:
    required = [
        "schema_id", "snapshot_id", "created_utc", "agent_id",
        "generator_set_id", "frontier", "visited", "score_model",
        "hash_chain", "invariant_diff",
    ]
    for f in required:
        if f not in snap:
            return SCHEMA_INVALID, {"missing_field": f, "path": f"$.{f}"}

    if snap["schema_id"] != FRONTIER_SCHEMA_ID:
        return SCHEMA_INVALID, {"expected": FRONTIER_SCHEMA_ID, "got": snap["schema_id"]}
    if not isinstance(snap["snapshot_id"], str) or len(snap["snapshot_id"]) < 8:
        return SCHEMA_INVALID, {"path": "$.snapshot_id"}
    if not _is_utc_ts(snap["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc"}

    # score_model
    sm = snap.get("score_model", {})
    if not isinstance(sm, dict):
        return SCHEMA_INVALID, {"path": "$.score_model"}
    for wk in ["novelty_weight", "reuse_weight", "obstruction_diversity_weight"]:
        if wk not in sm or not isinstance(sm[wk], (int, float)):
            return SCHEMA_INVALID, {"missing_field": wk, "path": f"$.score_model.{wk}"}

    # hash_chain
    hc = snap.get("hash_chain", {})
    if not isinstance(hc, dict):
        return SCHEMA_INVALID, {"path": "$.hash_chain"}
    for hf in ["prev_snapshot_hash", "this_snapshot_hash"]:
        if hf not in hc:
            return SCHEMA_INVALID, {"missing_field": hf, "path": f"$.hash_chain.{hf}"}

    return None


def _fr_check_hash_chain(snap: Dict) -> Optional[Tuple[str, Dict]]:
    hc = snap["hash_chain"]
    if not hc["this_snapshot_hash"] or not isinstance(hc["this_snapshot_hash"], str):
        return HASH_CHAIN_INVALID, {"path": "$.hash_chain.this_snapshot_hash", "reason": "must be non-empty"}
    return None


def _fr_check_frontier_unique(snap: Dict) -> Optional[Tuple[str, Dict]]:
    frontier = snap.get("frontier", [])
    if not isinstance(frontier, list):
        return SCHEMA_INVALID, {"path": "$.frontier", "reason": "must be array"}
    seen = set()
    for i, item in enumerate(frontier):
        if not isinstance(item, dict):
            return SCHEMA_INVALID, {"path": f"$.frontier[{i}]"}
        key = (item.get("layer"), item.get("state_hash"))
        if key in seen:
            return DUPLICATE_FRONTIER_ENTRY, {"duplicate": {"layer": key[0], "state_hash": key[1]}}
        seen.add(key)
    return None


def validate_frontier(snap: Dict) -> ValidationResult:
    obj_id = snap.get("snapshot_id", "<unknown>")
    for gate in [_fr_check_invariant_diff, _fr_check_schema,
                 _fr_check_hash_chain, _fr_check_frontier_unique]:
        result = gate(snap)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# BOUNDED RETURN RECEIPT VALIDATION
# ===================================================================

def _br_check_invariant_diff(rec: Dict) -> Optional[Tuple[str, Dict]]:
    if "invariant_diff" not in rec or not isinstance(rec.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {"path": "$.invariant_diff"}
    return None


def _br_check_schema(rec: Dict) -> Optional[Tuple[str, Dict]]:
    required = [
        "schema_id", "receipt_id", "created_utc", "agent_id",
        "generator_set_id", "start_state", "return_target_state",
        "k", "search", "result", "merkle_parent", "invariant_diff",
    ]
    for f in required:
        if f not in rec:
            return SCHEMA_INVALID, {"missing_field": f, "path": f"$.{f}"}

    if rec["schema_id"] != RECEIPT_SCHEMA_ID:
        return SCHEMA_INVALID, {"expected": RECEIPT_SCHEMA_ID, "got": rec["schema_id"]}
    if not isinstance(rec["receipt_id"], str) or len(rec["receipt_id"]) < 8:
        return SCHEMA_INVALID, {"path": "$.receipt_id"}
    if not _is_utc_ts(rec["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc"}
    if not isinstance(rec["k"], int) or rec["k"] < 1:
        return SCHEMA_INVALID, {"path": "$.k", "reason": "must be integer >= 1"}
    if not _is_hex64(rec.get("merkle_parent", "")):
        return SCHEMA_INVALID, {"path": "$.merkle_parent"}

    # start_state / return_target_state
    for state_field in ["start_state", "return_target_state"]:
        st = rec.get(state_field, {})
        if not isinstance(st, dict):
            return SCHEMA_INVALID, {"path": f"$.{state_field}"}
        if st.get("layer") not in VALID_LAYERS:
            return SCHEMA_INVALID, {"path": f"$.{state_field}.layer"}
        if not _is_hex64(st.get("state_hash", "")):
            return SCHEMA_INVALID, {"path": f"$.{state_field}.state_hash"}

    # search
    search = rec.get("search", {})
    if not isinstance(search, dict):
        return SCHEMA_INVALID, {"path": "$.search"}
    if search.get("algorithm") not in VALID_ALGORITHMS:
        return SCHEMA_INVALID, {"path": "$.search.algorithm", "expected_one_of": sorted(VALID_ALGORITHMS)}

    # result
    res = rec.get("result", {})
    if not isinstance(res, dict) or res.get("status") not in VALID_RETURN_STATUSES:
        return SCHEMA_INVALID, {"path": "$.result.status", "expected_one_of": sorted(VALID_RETURN_STATUSES)}

    return None


def _br_check_result(rec: Dict) -> Optional[Tuple[str, Dict]]:
    res = rec["result"]
    status = res["status"]

    if status == "RETURN_FOUND":
        path = res.get("path")
        if not isinstance(path, list) or not path:
            return RESULT_INCOMPLETE, {"path": "$.result.path", "reason": "RETURN_FOUND requires non-empty path"}
    else:
        # NO_RETURN_WITHIN_K or BUDGET_EXHAUSTED require typed failure
        if "fail_type" not in res:
            return RESULT_INCOMPLETE, {"path": "$.result.fail_type", "reason": f"{status} requires fail_type"}
        if "invariant_diff" not in res or not isinstance(res.get("invariant_diff"), dict):
            return RESULT_INCOMPLETE, {"path": "$.result.invariant_diff", "reason": f"{status} requires invariant_diff"}
    return None


def _br_check_path_length(rec: Dict) -> Optional[Tuple[str, Dict]]:
    res = rec["result"]
    if res["status"] == "RETURN_FOUND":
        path = res.get("path", [])
        if len(path) > rec["k"]:
            return PATH_EXCEEDS_K, {"k": rec["k"], "path_length": len(path)}
    return None


def validate_receipt(rec: Dict) -> ValidationResult:
    obj_id = rec.get("receipt_id", "<unknown>")
    for gate in [_br_check_invariant_diff, _br_check_schema,
                 _br_check_result, _br_check_path_length]:
        result = gate(rec)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    print("--- qa_conjecture_prove_validator self-test ---")
    passed = 0
    failed = 0

    HEX64 = "a" * 64

    # ---- Episode tests ----

    valid_ep = {
        "schema_id": EPISODE_SCHEMA_ID,
        "episode_id": "EPISODE-SELFTEST01",
        "created_utc": "2026-02-10T23:40:00Z",
        "agent_id": "qa-agent-ctrl-1",
        "policy_id": "pi_conjecture_prove.v1",
        "generator_set_id": "GENSET{sigma_down,sigma_tactic,sigma_up}.v1",
        "objective": {"type": "PROVE", "target_hash": HEX64, "budget": {"max_steps": 5, "max_seconds": 30}},
        "initial_state": {"layer": "formal", "state_hash": HEX64},
        "steps": [
            {
                "step_index": 0,
                "action": {"generator": "sigma_tactic", "params": {"tactic": "simp"}},
                "input_hash": HEX64,
                "output_hash": "b" * 64,
                "trace_ref": {"family": "QA_MATH_COMPILER_STACK.v1", "trace_id": "trace_0001"},
                "result": {"status": "SUCCESS", "witness_hash": "c" * 64},
            },
        ],
        "final_status": {"status": "SUCCESS", "summary": "proved"},
        "merkle_parent": "d" * 64,
        "invariant_diff": {"note": "self-test"},
    }

    r = validate_episode(valid_ep)
    if r.ok:
        print("  [PASS] valid episode accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid episode rejected: {r.fail_type}")
        failed += 1

    # E2: missing invariant_diff
    bad = copy.deepcopy(valid_ep)
    del bad["invariant_diff"]
    r = validate_episode(bad)
    if not r.ok and r.fail_type == MISSING_INVARIANT_DIFF:
        print("  [PASS] episode missing invariant_diff -> MISSING_INVARIANT_DIFF")
        passed += 1
    else:
        print(f"  [FAIL] expected MISSING_INVARIANT_DIFF, got {r.fail_type}")
        failed += 1

    # E3: duplicate step_index
    bad = copy.deepcopy(valid_ep)
    bad["steps"].append(copy.deepcopy(bad["steps"][0]))
    r = validate_episode(bad)
    if not r.ok and r.fail_type == DUPLICATE_STEP_INDEX:
        print("  [PASS] duplicate step_index -> DUPLICATE_STEP_INDEX")
        passed += 1
    else:
        print(f"  [FAIL] expected DUPLICATE_STEP_INDEX, got {r.fail_type}")
        failed += 1

    # E4: FAIL step missing invariant_diff in result
    bad = copy.deepcopy(valid_ep)
    bad["steps"][0]["result"] = {"status": "FAIL", "fail_type": "GOAL_STUCK"}
    r = validate_episode(bad)
    if not r.ok and r.fail_type == RESULT_INCOMPLETE:
        print("  [PASS] FAIL step without result.invariant_diff -> RESULT_INCOMPLETE")
        passed += 1
    else:
        print(f"  [FAIL] expected RESULT_INCOMPLETE, got {r.fail_type}")
        failed += 1

    # E5: bad trace_ref family prefix
    bad = copy.deepcopy(valid_ep)
    bad["steps"][0]["trace_ref"]["family"] = "NOT_QA"
    r = validate_episode(bad)
    if not r.ok and r.fail_type == INVALID_TRACE_REF:
        print("  [PASS] bad trace_ref.family -> INVALID_TRACE_REF")
        passed += 1
    else:
        print(f"  [FAIL] expected INVALID_TRACE_REF, got {r.fail_type}")
        failed += 1

    # ---- Frontier tests ----

    valid_fr = {
        "schema_id": FRONTIER_SCHEMA_ID,
        "snapshot_id": "SNAP-SELFTEST0001",
        "created_utc": "2026-02-10T23:45:00Z",
        "agent_id": "qa-agent-ctrl-1",
        "generator_set_id": "GENSET.v1",
        "frontier": [
            {"state_hash": "a" * 64, "layer": "formal", "priority": 0.91},
            {"state_hash": "b" * 64, "layer": "human", "priority": 0.64},
        ],
        "visited": ["c" * 64, "d" * 64],
        "score_model": {"novelty_weight": 1.0, "reuse_weight": 0.5, "obstruction_diversity_weight": 0.75},
        "hash_chain": {"prev_snapshot_hash": "e" * 64, "this_snapshot_hash": "f" * 64},
        "invariant_diff": {"note": "self-test"},
    }

    r = validate_frontier(valid_fr)
    if r.ok:
        print("  [PASS] valid frontier accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid frontier rejected: {r.fail_type}")
        failed += 1

    # F2: empty this_snapshot_hash
    bad = copy.deepcopy(valid_fr)
    bad["hash_chain"]["this_snapshot_hash"] = ""
    r = validate_frontier(bad)
    if not r.ok and r.fail_type == HASH_CHAIN_INVALID:
        print("  [PASS] empty this_snapshot_hash -> HASH_CHAIN_INVALID")
        passed += 1
    else:
        print(f"  [FAIL] expected HASH_CHAIN_INVALID, got {r.fail_type}")
        failed += 1

    # F3: duplicate frontier entry
    bad = copy.deepcopy(valid_fr)
    bad["frontier"].append(bad["frontier"][0].copy())
    r = validate_frontier(bad)
    if not r.ok and r.fail_type == DUPLICATE_FRONTIER_ENTRY:
        print("  [PASS] duplicate frontier entry -> DUPLICATE_FRONTIER_ENTRY")
        passed += 1
    else:
        print(f"  [FAIL] expected DUPLICATE_FRONTIER_ENTRY, got {r.fail_type}")
        failed += 1

    # ---- Bounded return tests ----

    valid_br = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "receipt_id": "RCPT-SELFTEST0001",
        "created_utc": "2026-02-10T23:50:00Z",
        "agent_id": "qa-agent-ctrl-1",
        "generator_set_id": "GENSET.v1",
        "start_state": {"layer": "formal", "state_hash": "a" * 64},
        "return_target_state": {"layer": "formal", "state_hash": "a" * 64},
        "k": 2,
        "search": {
            "algorithm": "BFS",
            "budget": {"max_nodes": 100, "max_seconds": 5},
            "determinism": {"root_ordering": "lex", "tie_breaker": "stable", "toolchain_id": "lean4.12.0"},
        },
        "result": {
            "status": "NO_RETURN_WITHIN_K",
            "fail_type": "NO_RETURN_WITHIN_K",
            "invariant_diff": {"k": 2, "visited_nodes": 37, "frontier_exhausted": True},
        },
        "merkle_parent": "b" * 64,
        "invariant_diff": {"note": "self-test"},
    }

    r = validate_receipt(valid_br)
    if r.ok:
        print("  [PASS] valid bounded-return receipt accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid receipt rejected: {r.fail_type}")
        failed += 1

    # R2: NO_RETURN without fail_type
    bad = copy.deepcopy(valid_br)
    bad["result"] = {"status": "NO_RETURN_WITHIN_K"}
    r = validate_receipt(bad)
    if not r.ok and r.fail_type == RESULT_INCOMPLETE:
        print("  [PASS] NO_RETURN without fail_type -> RESULT_INCOMPLETE")
        passed += 1
    else:
        print(f"  [FAIL] expected RESULT_INCOMPLETE, got {r.fail_type}")
        failed += 1

    # R3: RETURN_FOUND with path too long
    bad = copy.deepcopy(valid_br)
    bad["result"] = {
        "status": "RETURN_FOUND",
        "path": [
            {"step_index": 0, "trace_ref": {"family": "QA_X", "trace_id": "t1"}, "input_hash": "a" * 64, "output_hash": "b" * 64},
            {"step_index": 1, "trace_ref": {"family": "QA_X", "trace_id": "t2"}, "input_hash": "b" * 64, "output_hash": "c" * 64},
            {"step_index": 2, "trace_ref": {"family": "QA_X", "trace_id": "t3"}, "input_hash": "c" * 64, "output_hash": "a" * 64},
        ],
    }
    bad["k"] = 2  # path length 3 > k=2
    r = validate_receipt(bad)
    if not r.ok and r.fail_type == PATH_EXCEEDS_K:
        print("  [PASS] path length > k -> PATH_EXCEEDS_K")
        passed += 1
    else:
        print(f"  [FAIL] expected PATH_EXCEEDS_K, got {r.fail_type}")
        failed += 1

    print(f"\n  {passed}/{passed + failed} self-tests passed")
    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--self-test" in args:
        ok = _self_test()
        sys.exit(0 if ok else 1)

    modes = {"episode": validate_episode, "frontier": validate_frontier, "receipt": validate_receipt}

    if len(args) < 2 or args[0] not in modes:
        print(f"Usage: {sys.argv[0]} episode|frontier|receipt <file.json> [--ci]", file=sys.stderr)
        print(f"       {sys.argv[0]} --self-test", file=sys.stderr)
        sys.exit(2)

    mode = args[0]
    file_path = Path(args[1])
    ci_mode = "--ci" in args

    if not file_path.exists():
        print(f"ERROR: {file_path} not found", file=sys.stderr)
        sys.exit(2)

    with file_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            if ci_mode:
                print(f"[FAIL] {file_path.name}: SCHEMA_INVALID (JSON decode: {e})")
            else:
                print(f"FAIL: {file_path.name}")
                print(json.dumps({"fail_type": SCHEMA_INVALID, "invariant_diff": {"reason": str(e)}}, indent=2))
            sys.exit(1)

    validate_fn = modes[mode]
    result = validate_fn(data)

    if ci_mode:
        if result.ok:
            print(f"[PASS] {file_path.name}: valid ({result.obj_id})")
        else:
            diff_str = json.dumps(result.invariant_diff, sort_keys=True)
            print(f"[FAIL] {file_path.name}: {result.fail_type} {diff_str}")
    else:
        if result.ok:
            print(f"PASS: {file_path.name} ({result.obj_id})")
        else:
            print(f"FAIL: {file_path.name}")
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
