#!/usr/bin/env python3
"""
Validate QA_AGENT_TRACE_COMPETENCY_CERT_SCHEMA.v1 certificates.

Checks (in order):
  1. Schema structural validation                -> SCHEMA_INVALID
  2. invariant_diff presence (hard gate)         -> MISSING_INVARIANT_DIFF
  3. Trace ref hash verification (if trace provided) -> TRACE_REF_HASH_MISMATCH
  4. Metric bounds check                         -> METRIC_OUT_OF_BOUNDS
  5. Dominance consistency (dims_won matches bools, rule satisfied) -> NONDETERMINISTIC_DERIVATION / DOMINANCE_FAILURE
  6. Verdict internal consistency                -> NONDETERMINISTIC_DERIVATION

Usage:
  python qa_agent_trace_competency_cert_validator.py <cert.json> [--ci]
  python qa_agent_trace_competency_cert_validator.py <cert.json> --trace <trace.json> [--ci]
  python qa_agent_trace_competency_cert_validator.py --self-test

Accepts cert-only JSON or wrapper format {"trace": {...}, "cert": {...}}.
In wrapper mode the bundled trace is used for hash verification unless --trace
is explicitly provided.

Returns 0 on PASS, 1 on FAIL, 2 on usage error.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_ID = "QA_AGENT_TRACE_COMPETENCY_CERT_SCHEMA.v1"
TRACE_SCHEMA_ID = "QA_AGENT_TRACE_SCHEMA.v1"
DERIVATION_ID = "QA_COMPETENCY_FROM_AGENT_TRACE.v1"

VALID_DOMINANCE_RULES = frozenset([
    "strong_dominance_4of4", "weak_dominance_3of4", "existential_1of4",
])
VALID_BASELINE_KINDS = frozenset(["null", "random", "heuristic", "historical_agent"])
VALID_TIMEBASES = frozenset(["trace_event_index", "trace_timestamp_ms"])
VALID_FAIL_TYPES = frozenset([
    "OK", "SCHEMA_INVALID", "TRACE_REF_HASH_MISMATCH",
    "MISSING_INVARIANT_DIFF", "NONDETERMINISTIC_DERIVATION",
    "DOMINANCE_FAILURE", "METRIC_OUT_OF_BOUNDS",
])

# Fail type constants
SCHEMA_INVALID = "SCHEMA_INVALID"
TRACE_REF_HASH_MISMATCH = "TRACE_REF_HASH_MISMATCH"
MISSING_INVARIANT_DIFF = "MISSING_INVARIANT_DIFF"
NONDETERMINISTIC_DERIVATION = "NONDETERMINISTIC_DERIVATION"
DOMINANCE_FAILURE = "DOMINANCE_FAILURE"
METRIC_OUT_OF_BOUNDS = "METRIC_OUT_OF_BOUNDS"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _is_utc_ts(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    import re
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", v))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class CertValidationResult:
    def __init__(self, ok: bool, fail_type: Optional[str] = None,
                 invariant_diff: Optional[Dict[str, Any]] = None,
                 cert_id: str = ""):
        self.ok = ok
        self.fail_type = fail_type
        self.invariant_diff = invariant_diff or {}
        self.cert_id = cert_id

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"ok": self.ok, "cert_id": self.cert_id}
        if not self.ok:
            d["fail_type"] = self.fail_type
            d["invariant_diff"] = self.invariant_diff
        return d


# ---------------------------------------------------------------------------
# Gate 1: Schema structural checks
# ---------------------------------------------------------------------------

def _check_schema(cert: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
    required = [
        "schema_id", "cert_id", "created_utc", "trace_ref", "task_ref",
        "derivation", "metrics", "baselines", "dominance", "verdict",
        "invariant_diff",
    ]
    for field in required:
        if field not in cert:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if cert["schema_id"] != SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": SCHEMA_ID, "got": cert["schema_id"], "path": "$.schema_id",
        }

    if not isinstance(cert["cert_id"], str) or len(cert["cert_id"]) < 8:
        return SCHEMA_INVALID, {"path": "$.cert_id", "reason": "must be string >=8 chars"}

    if not _is_utc_ts(cert["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad ISO 8601 UTC timestamp"}

    # --- trace_ref ---
    tr = cert["trace_ref"]
    if not isinstance(tr, dict):
        return SCHEMA_INVALID, {"path": "$.trace_ref", "reason": "must be object"}
    for tf in ["trace_id", "trace_sha256", "trace_schema_id"]:
        if tf not in tr:
            return SCHEMA_INVALID, {"missing_field": tf, "path": f"$.trace_ref.{tf}"}
    if not _is_hex64(tr["trace_sha256"]):
        return SCHEMA_INVALID, {"path": "$.trace_ref.trace_sha256", "reason": "must be hex64"}
    if tr["trace_schema_id"] != TRACE_SCHEMA_ID:
        return SCHEMA_INVALID, {
            "path": "$.trace_ref.trace_schema_id",
            "expected": TRACE_SCHEMA_ID, "got": tr["trace_schema_id"],
        }

    # --- task_ref ---
    task = cert["task_ref"]
    if not isinstance(task, dict):
        return SCHEMA_INVALID, {"path": "$.task_ref", "reason": "must be object"}
    for tf in ["task_id", "source_dataset", "dataset_slice_sha256", "license"]:
        if tf not in task:
            return SCHEMA_INVALID, {"missing_field": tf, "path": f"$.task_ref.{tf}"}
    if not _is_hex64(task["dataset_slice_sha256"]):
        return SCHEMA_INVALID, {"path": "$.task_ref.dataset_slice_sha256", "reason": "must be hex64"}

    # --- derivation ---
    der = cert["derivation"]
    if not isinstance(der, dict):
        return SCHEMA_INVALID, {"path": "$.derivation", "reason": "must be object"}
    for df in ["derivation_id", "version", "inputs", "canonicalization"]:
        if df not in der:
            return SCHEMA_INVALID, {"missing_field": df, "path": f"$.derivation.{df}"}
    if der["derivation_id"] != DERIVATION_ID:
        return SCHEMA_INVALID, {
            "path": "$.derivation.derivation_id",
            "expected": DERIVATION_ID, "got": der["derivation_id"],
        }
    inp = der.get("inputs", {})
    if not isinstance(inp, dict):
        return SCHEMA_INVALID, {"path": "$.derivation.inputs", "reason": "must be object"}
    for ik in ["event_types_used", "action_vocab", "timebase"]:
        if ik not in inp:
            return SCHEMA_INVALID, {"missing_field": ik, "path": f"$.derivation.inputs.{ik}"}
    if inp["timebase"] not in VALID_TIMEBASES:
        return SCHEMA_INVALID, {
            "path": "$.derivation.inputs.timebase",
            "expected_one_of": sorted(VALID_TIMEBASES), "got": inp["timebase"],
        }
    canon = der.get("canonicalization", {})
    if not isinstance(canon, dict):
        return SCHEMA_INVALID, {"path": "$.derivation.canonicalization", "reason": "must be object"}
    if canon.get("json_sort_keys") is not True:
        return SCHEMA_INVALID, {"path": "$.derivation.canonicalization.json_sort_keys", "reason": "must be true"}
    if canon.get("json_separators") != ",:":
        return SCHEMA_INVALID, {"path": "$.derivation.canonicalization.json_separators", "reason": "must be ',:'"}
    if canon.get("utf8") is not True:
        return SCHEMA_INVALID, {"path": "$.derivation.canonicalization.utf8", "reason": "must be true"}

    # --- metrics ---
    m = cert["metrics"]
    if not isinstance(m, dict):
        return SCHEMA_INVALID, {"path": "$.metrics", "reason": "must be object"}
    for mk in ["agency_index", "plasticity_index", "goal_density", "control_entropy"]:
        if mk not in m:
            return SCHEMA_INVALID, {"missing_field": mk, "path": f"$.metrics.{mk}"}
        if not isinstance(m[mk], (int, float)):
            return SCHEMA_INVALID, {"path": f"$.metrics.{mk}", "reason": "must be number"}

    # --- baselines ---
    bl = cert["baselines"]
    if not isinstance(bl, dict):
        return SCHEMA_INVALID, {"path": "$.baselines", "reason": "must be object"}
    for bk in ["baseline_set_id", "baseline_bundle_sha256", "baselines_used"]:
        if bk not in bl:
            return SCHEMA_INVALID, {"missing_field": bk, "path": f"$.baselines.{bk}"}
    if not _is_hex64(bl["baseline_bundle_sha256"]):
        return SCHEMA_INVALID, {"path": "$.baselines.baseline_bundle_sha256", "reason": "must be hex64"}
    bu = bl["baselines_used"]
    if not isinstance(bu, list) or len(bu) < 1:
        return SCHEMA_INVALID, {"path": "$.baselines.baselines_used", "reason": "must be non-empty array"}
    for i, b in enumerate(bu):
        if not isinstance(b, dict):
            return SCHEMA_INVALID, {"path": f"$.baselines.baselines_used[{i}]", "reason": "must be object"}
        for brf in ["baseline_id", "kind"]:
            if brf not in b:
                return SCHEMA_INVALID, {"missing_field": brf, "path": f"$.baselines.baselines_used[{i}].{brf}"}
        if b["kind"] not in VALID_BASELINE_KINDS:
            return SCHEMA_INVALID, {
                "path": f"$.baselines.baselines_used[{i}].kind",
                "expected_one_of": sorted(VALID_BASELINE_KINDS), "got": b["kind"],
            }

    # --- dominance ---
    dom = cert["dominance"]
    if not isinstance(dom, dict):
        return SCHEMA_INVALID, {"path": "$.dominance", "reason": "must be object"}
    for dk in ["rule", "min_margin", "wins"]:
        if dk not in dom:
            return SCHEMA_INVALID, {"missing_field": dk, "path": f"$.dominance.{dk}"}
    if dom["rule"] not in VALID_DOMINANCE_RULES:
        return SCHEMA_INVALID, {
            "path": "$.dominance.rule",
            "expected_one_of": sorted(VALID_DOMINANCE_RULES), "got": dom["rule"],
        }
    wins = dom.get("wins", {})
    if not isinstance(wins, dict):
        return SCHEMA_INVALID, {"path": "$.dominance.wins", "reason": "must be object"}
    for wk in ["dimensions_won", "by_dimension"]:
        if wk not in wins:
            return SCHEMA_INVALID, {"missing_field": wk, "path": f"$.dominance.wins.{wk}"}
    bd = wins.get("by_dimension", {})
    if not isinstance(bd, dict):
        return SCHEMA_INVALID, {"path": "$.dominance.wins.by_dimension", "reason": "must be object"}
    for dim in ["agency", "plasticity", "goal_density", "entropy"]:
        if dim not in bd or not isinstance(bd[dim], bool):
            return SCHEMA_INVALID, {"path": f"$.dominance.wins.by_dimension.{dim}", "reason": "must be boolean"}

    # --- verdict ---
    v = cert["verdict"]
    if not isinstance(v, dict):
        return SCHEMA_INVALID, {"path": "$.verdict", "reason": "must be object"}
    for vk in ["passed", "fail_type"]:
        if vk not in v:
            return SCHEMA_INVALID, {"missing_field": vk, "path": f"$.verdict.{vk}"}
    if not isinstance(v["passed"], bool):
        return SCHEMA_INVALID, {"path": "$.verdict.passed", "reason": "must be boolean"}
    if v["fail_type"] not in VALID_FAIL_TYPES:
        return SCHEMA_INVALID, {
            "path": "$.verdict.fail_type",
            "expected_one_of": sorted(VALID_FAIL_TYPES), "got": v["fail_type"],
        }

    # --- invariant_diff ---
    if not isinstance(cert["invariant_diff"], dict):
        return SCHEMA_INVALID, {"path": "$.invariant_diff", "reason": "must be object"}

    return None


# ---------------------------------------------------------------------------
# Gate 2: invariant_diff presence (hard gate, checked before schema for
#          the case where it's entirely absent vs malformed)
# ---------------------------------------------------------------------------

def _check_invariant_diff(cert: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
    if "invariant_diff" not in cert or not isinstance(cert.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {
            "path": "$.invariant_diff", "reason": "must be a JSON object",
        }
    return None


# ---------------------------------------------------------------------------
# Gate 3: Trace ref hash verification
# ---------------------------------------------------------------------------

def _check_trace_hash(cert: Dict[str, Any],
                      trace_canonical: Optional[str]) -> Optional[Tuple[str, Dict]]:
    """If trace canonical JSON is provided, verify trace_ref.trace_sha256 matches."""
    if trace_canonical is None:
        return None
    expected = cert["trace_ref"]["trace_sha256"]
    computed = _sha256(trace_canonical)
    if computed != expected:
        return TRACE_REF_HASH_MISMATCH, {
            "expected": expected, "got": computed,
        }
    return None


# ---------------------------------------------------------------------------
# Gate 4: Metric bounds
# ---------------------------------------------------------------------------

def _check_metrics(cert: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
    m = cert["metrics"]
    bounds_errors = []
    for field in ["agency_index", "plasticity_index", "goal_density"]:
        val = m[field]
        if not (0.0 <= val <= 1.0):
            bounds_errors.append({"field": field, "value": val, "bounds": [0.0, 1.0]})
    if m["control_entropy"] < 0.0:
        bounds_errors.append({"field": "control_entropy", "value": m["control_entropy"], "bounds": [0.0, "inf"]})
    if bounds_errors:
        return METRIC_OUT_OF_BOUNDS, {"errors": bounds_errors}
    return None


# ---------------------------------------------------------------------------
# Gate 5: Dominance consistency
# ---------------------------------------------------------------------------

# Dimension name in dominance.wins.by_dimension -> metric key in cert.metrics
_DIM_TO_METRIC = {
    "agency": "agency_index",
    "plasticity": "plasticity_index",
    "goal_density": "goal_density",
    "entropy": "control_entropy",
}

def _check_dominance(cert: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
    dom = cert["dominance"]
    wins = dom["wins"]
    bd = wins["by_dimension"]

    # Recount dimensions
    counted = sum(1 for d in ["agency", "plasticity", "goal_density", "entropy"] if bd[d])
    declared = wins["dimensions_won"]
    if counted != declared:
        return NONDETERMINISTIC_DERIVATION, {
            "reason": "dimensions_won_mismatch",
            "declared": declared, "computed": counted,
            "path": "$.dominance.wins.dimensions_won",
        }

    # Rule satisfaction
    rule = dom["rule"]
    if rule == "strong_dominance_4of4" and counted != 4:
        return DOMINANCE_FAILURE, {"rule": rule, "dimensions_won": counted, "required": 4}
    if rule == "weak_dominance_3of4" and counted < 3:
        return DOMINANCE_FAILURE, {"rule": rule, "dimensions_won": counted, "required": 3}
    if rule == "existential_1of4" and counted < 1:
        return DOMINANCE_FAILURE, {"rule": rule, "dimensions_won": counted, "required": 1}

    return None


# ---------------------------------------------------------------------------
# Gate 5b: Dominance recomputation (if baseline metrics present)
# ---------------------------------------------------------------------------

def _check_dominance_recompute(cert: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
    """If any baseline in baselines_used has a metrics dict, recompute wins
    and verify they match cert.dominance.wins.by_dimension.
    Tie policy: agent_metric - baseline_metric >= min_margin counts as a win.
    """
    baselines_used = cert.get("baselines", {}).get("baselines_used", [])
    agent_metrics = cert.get("metrics", {})
    min_margin = cert.get("dominance", {}).get("min_margin", 0.0)
    declared_bd = cert.get("dominance", {}).get("wins", {}).get("by_dimension", {})

    for i, bl in enumerate(baselines_used):
        bl_metrics = bl.get("metrics")
        if not isinstance(bl_metrics, dict):
            continue

        # Recompute wins for this baseline
        mismatches = []
        for dim, metric_key in _DIM_TO_METRIC.items():
            agent_val = agent_metrics.get(metric_key)
            baseline_val = bl_metrics.get(metric_key)
            if not isinstance(agent_val, (int, float)) or not isinstance(baseline_val, (int, float)):
                continue
            recomputed_win = (agent_val - baseline_val) >= min_margin
            declared_win = declared_bd.get(dim)
            if recomputed_win != declared_win:
                mismatches.append({
                    "dimension": dim,
                    "metric_key": metric_key,
                    "agent_value": agent_val,
                    "baseline_value": baseline_val,
                    "min_margin": min_margin,
                    "recomputed_win": recomputed_win,
                    "declared_win": declared_win,
                })
        if mismatches:
            return NONDETERMINISTIC_DERIVATION, {
                "reason": "dominance_recomputation_mismatch",
                "baseline_id": bl.get("baseline_id", f"baselines_used[{i}]"),
                "mismatches": mismatches,
            }

    return None


# ---------------------------------------------------------------------------
# Gate 6: Verdict internal consistency
# ---------------------------------------------------------------------------

def _check_verdict(cert: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
    v = cert["verdict"]
    passed = v["passed"]
    ft = v["fail_type"]
    if passed and ft != "OK":
        return NONDETERMINISTIC_DERIVATION, {
            "reason": "passed_but_fail_type_not_OK",
            "passed": passed, "fail_type": ft,
        }
    if not passed and ft == "OK":
        return NONDETERMINISTIC_DERIVATION, {
            "reason": "failed_but_fail_type_is_OK",
            "passed": passed, "fail_type": ft,
        }
    return None


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_cert(cert: Dict[str, Any],
                  trace_canonical: Optional[str] = None) -> CertValidationResult:
    """Run all gates in order. Returns on first failure."""
    cert_id = cert.get("cert_id", "<unknown>")

    # Gate 2 first (catch totally missing invariant_diff before schema details)
    result = _check_invariant_diff(cert)
    if result is not None:
        return CertValidationResult(False, result[0], result[1], cert_id)

    # Gate 1: Schema shape
    result = _check_schema(cert)
    if result is not None:
        return CertValidationResult(False, result[0], result[1], cert_id)

    # Gate 3: Trace hash
    result = _check_trace_hash(cert, trace_canonical)
    if result is not None:
        return CertValidationResult(False, result[0], result[1], cert_id)

    # Gate 4: Metric bounds
    result = _check_metrics(cert)
    if result is not None:
        return CertValidationResult(False, result[0], result[1], cert_id)

    # Gate 5: Dominance consistency
    result = _check_dominance(cert)
    if result is not None:
        return CertValidationResult(False, result[0], result[1], cert_id)

    # Gate 5b: Dominance recomputation (if baseline metrics present)
    result = _check_dominance_recompute(cert)
    if result is not None:
        return CertValidationResult(False, result[0], result[1], cert_id)

    # Gate 6: Verdict consistency
    result = _check_verdict(cert)
    if result is not None:
        return CertValidationResult(False, result[0], result[1], cert_id)

    return CertValidationResult(True, cert_id=cert_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    """Run built-in sanity checks. Returns True on pass."""
    import copy
    print("--- qa_agent_trace_competency_cert_validator self-test ---")
    passed = 0
    failed = 0

    valid_cert = {
        "schema_id": SCHEMA_ID,
        "cert_id": "CERT-SELFTEST0001",
        "created_utc": "2026-02-10T23:00:00Z",
        "trace_ref": {
            "trace_id": "TRACE-TEST0001",
            "trace_sha256": "a" * 64,
            "trace_schema_id": TRACE_SCHEMA_ID,
        },
        "task_ref": {
            "task_id": "django__django-12345",
            "source_dataset": "princeton-nlp/SWE-bench_Lite",
            "dataset_slice_sha256": "b" * 64,
            "license": "mit",
        },
        "derivation": {
            "derivation_id": DERIVATION_ID,
            "version": "1.0.0",
            "inputs": {
                "event_types_used": ["read", "edit", "test"],
                "action_vocab": ["edit_file", "run_tests"],
                "timebase": "trace_event_index",
            },
            "canonicalization": {
                "json_sort_keys": True,
                "json_separators": ",:",
                "utf8": True,
            },
        },
        "metrics": {
            "agency_index": 0.21,
            "plasticity_index": 0.44,
            "goal_density": 0.12,
            "control_entropy": 0.88,
        },
        "baselines": {
            "baseline_set_id": "QA_BASELINES.v1",
            "baseline_bundle_sha256": "c" * 64,
            "baselines_used": [
                {"baseline_id": "NULL_AGENT.v1", "kind": "null"},
            ],
        },
        "dominance": {
            "rule": "weak_dominance_3of4",
            "min_margin": 0.01,
            "wins": {
                "dimensions_won": 3,
                "by_dimension": {
                    "agency": True,
                    "plasticity": True,
                    "goal_density": False,
                    "entropy": True,
                },
            },
        },
        "verdict": {"passed": True, "fail_type": "OK"},
        "invariant_diff": {"note": "self-test fixture"},
    }

    # Test 1: valid cert passes
    r = validate_cert(valid_cert)
    if r.ok:
        print("  [PASS] valid cert accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid cert rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    # Test 2: missing invariant_diff
    bad = copy.deepcopy(valid_cert)
    del bad["invariant_diff"]
    r = validate_cert(bad)
    if not r.ok and r.fail_type == MISSING_INVARIANT_DIFF:
        print("  [PASS] missing invariant_diff -> MISSING_INVARIANT_DIFF")
        passed += 1
    else:
        print(f"  [FAIL] expected MISSING_INVARIANT_DIFF, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 3: trace hash mismatch
    trace_obj = {"hello": "world"}
    trace_canonical = _canonical(trace_obj)
    bad = copy.deepcopy(valid_cert)
    bad["trace_ref"]["trace_sha256"] = "f" * 64  # wrong hash
    r = validate_cert(bad, trace_canonical=trace_canonical)
    if not r.ok and r.fail_type == TRACE_REF_HASH_MISMATCH:
        print("  [PASS] wrong trace hash -> TRACE_REF_HASH_MISMATCH")
        passed += 1
    else:
        print(f"  [FAIL] expected TRACE_REF_HASH_MISMATCH, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 4: metric out of bounds
    bad = copy.deepcopy(valid_cert)
    bad["metrics"]["agency_index"] = 1.5
    r = validate_cert(bad)
    if not r.ok and r.fail_type == METRIC_OUT_OF_BOUNDS:
        print("  [PASS] agency_index=1.5 -> METRIC_OUT_OF_BOUNDS")
        passed += 1
    else:
        print(f"  [FAIL] expected METRIC_OUT_OF_BOUNDS, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 5: dimensions_won mismatch
    bad = copy.deepcopy(valid_cert)
    bad["dominance"]["wins"]["dimensions_won"] = 4  # says 4 but only 3 true
    r = validate_cert(bad)
    if not r.ok and r.fail_type == NONDETERMINISTIC_DERIVATION:
        print("  [PASS] dims_won mismatch -> NONDETERMINISTIC_DERIVATION")
        passed += 1
    else:
        print(f"  [FAIL] expected NONDETERMINISTIC_DERIVATION, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 6: dominance rule failure
    bad = copy.deepcopy(valid_cert)
    bad["dominance"]["wins"]["by_dimension"] = {
        "agency": True, "plasticity": False, "goal_density": False, "entropy": False,
    }
    bad["dominance"]["wins"]["dimensions_won"] = 1
    # rule is weak_dominance_3of4 but only 1 win
    r = validate_cert(bad)
    if not r.ok and r.fail_type == DOMINANCE_FAILURE:
        print("  [PASS] 1/4 wins with 3of4 rule -> DOMINANCE_FAILURE")
        passed += 1
    else:
        print(f"  [FAIL] expected DOMINANCE_FAILURE, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 7: verdict inconsistency (passed=true, fail_type!=OK)
    bad = copy.deepcopy(valid_cert)
    bad["verdict"]["fail_type"] = "DOMINANCE_FAILURE"
    r = validate_cert(bad)
    if not r.ok and r.fail_type == NONDETERMINISTIC_DERIVATION:
        print("  [PASS] passed+DOMINANCE_FAILURE -> NONDETERMINISTIC_DERIVATION")
        passed += 1
    else:
        print(f"  [FAIL] expected NONDETERMINISTIC_DERIVATION, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 8: dominance recomputation mismatch (baseline metrics present)
    bad = copy.deepcopy(valid_cert)
    # Add baseline metrics that make agency a clear win (0.21 - 0.05 = 0.16 >= 0.01)
    bad["baselines"]["baselines_used"][0]["metrics"] = {
        "agency_index": 0.05,
        "plasticity_index": 0.02,
        "goal_density": 0.12,  # tie: 0.12 - 0.12 = 0.00 < 0.01 -> not a win
        "control_entropy": 0.10,
    }
    # Declare agency=False (should be True per recomputation)
    bad["dominance"]["wins"]["by_dimension"]["agency"] = False
    bad["dominance"]["wins"]["dimensions_won"] = 2  # keep consistent with count
    bad["dominance"]["rule"] = "existential_1of4"  # relax rule so gate 5 passes
    r = validate_cert(bad)
    if not r.ok and r.fail_type == NONDETERMINISTIC_DERIVATION:
        print("  [PASS] dominance recomputation mismatch -> NONDETERMINISTIC_DERIVATION")
        passed += 1
    else:
        print(f"  [FAIL] expected NONDETERMINISTIC_DERIVATION, got ok={r.ok} ft={r.fail_type}")
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

    if not args or args[0].startswith("-"):
        print(f"Usage: {sys.argv[0]} <cert.json> [--trace <trace.json>] [--ci]",
              file=sys.stderr)
        print(f"       {sys.argv[0]} --self-test", file=sys.stderr)
        sys.exit(2)

    cert_path = Path(args[0])
    ci_mode = "--ci" in args
    trace_canonical = None

    if "--trace" in args:
        ti = args.index("--trace")
        if ti + 1 < len(args):
            trace_path = Path(args[ti + 1])
            with trace_path.open("r", encoding="utf-8") as f:
                trace_obj = json.load(f)
            trace_canonical = _canonical(trace_obj)

    if not cert_path.exists():
        print(f"ERROR: {cert_path} not found", file=sys.stderr)
        sys.exit(2)

    with cert_path.open("r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            if ci_mode:
                print(f"[FAIL] {cert_path.name}: SCHEMA_INVALID (JSON decode: {e})")
            else:
                print(f"FAIL: {cert_path.name}")
                print(json.dumps({
                    "fail_type": SCHEMA_INVALID,
                    "invariant_diff": {"reason": f"JSON decode error: {e}"},
                }, indent=2))
            sys.exit(1)

    # Auto-detect wrapper format: {"trace": {...}, "cert": {...}}
    if isinstance(raw, dict) and "trace" in raw and "cert" in raw:
        cert = raw["cert"]
        if trace_canonical is None:  # only if --trace not already provided
            trace_canonical = _canonical(raw["trace"])
    else:
        cert = raw

    result = validate_cert(cert, trace_canonical=trace_canonical)

    if ci_mode:
        if result.ok:
            print(f"[PASS] {cert_path.name}: valid ({result.cert_id})")
        else:
            diff_str = json.dumps(result.invariant_diff, sort_keys=True)
            print(f"[FAIL] {cert_path.name}: {result.fail_type} {diff_str}")
        sys.exit(0 if result.ok else 1)
    else:
        if result.ok:
            print(f"PASS: {cert_path.name} ({result.cert_id})")
        else:
            print(f"FAIL: {cert_path.name}")
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
