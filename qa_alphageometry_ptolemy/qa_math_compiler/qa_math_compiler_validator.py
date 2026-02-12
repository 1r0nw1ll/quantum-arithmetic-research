#!/usr/bin/env python3
"""
Validate the QA Math Compiler stack artifacts.

Supported schemas:
  - QA_MATH_COMPILER_TRACE_SCHEMA.v1
  - QA_COMPILER_PAIR_CERT_SCHEMA.v1
  - QA_FORMAL_TASK_SCHEMA.v1
  - QA_MATH_COMPILER_REPLAY_BUNDLE_SCHEMA.v1
  - QA_HUMAN_FORMAL_PAIR_CERT.v1
  - QA_LEMMA_MINING_SCHEMA.v1

Usage:
  python qa_math_compiler_validator.py trace   <trace.json> [--ci]
  python qa_math_compiler_validator.py pair    <pair.json> [--ci]
  python qa_math_compiler_validator.py task    <task.json> [--ci]
  python qa_math_compiler_validator.py replay  <replay_bundle.json> [--ci]
  python qa_math_compiler_validator.py pair_v1 <pair_v1.json> [--ci]
  python qa_math_compiler_validator.py lemma   <lemma_pack.json> [--ci]
  python qa_math_compiler_validator.py --self-test

Returns 0 on PASS, 1 on FAIL, 2 on usage error.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACE_SCHEMA_ID = "QA_MATH_COMPILER_TRACE_SCHEMA.v1"
PAIR_SCHEMA_ID = "QA_COMPILER_PAIR_CERT_SCHEMA.v1"
TASK_SCHEMA_ID = "QA_FORMAL_TASK_SCHEMA.v1"
REPLAY_SCHEMA_ID = "QA_MATH_COMPILER_REPLAY_BUNDLE_SCHEMA.v1"
PAIR_V1_SCHEMA_ID = "QA_HUMAN_FORMAL_PAIR_CERT.v1"
LEMMA_SCHEMA_ID = "QA_LEMMA_MINING_SCHEMA.v1"

VALID_LAYERS = frozenset(["human", "formal"])
VALID_STATUSES = frozenset(["SUCCESS", "FAIL"])
VALID_REPLAY_STATUSES = frozenset(["SUCCESS", "FAIL", "INFRA_FLAKE"])
VALID_PAIR_STATUSES = frozenset(["PROVED", "UNPROVED", "AMBIGUOUS"])
VALID_PROOF_STATUSES = frozenset(["FOUND", "NEEDS_PROOF"])

VALID_TRACE_FAIL_TYPES = frozenset([
    "PARSE_AMBIGUITY", "MISSING_DEPENDENCY", "TYPE_MISMATCH",
    "NON_CANONICAL_NORMALIZATION", "UNSUPPORTED_IDIOM", "SPEC_DRIFT",
    "GOAL_STUCK", "SEARCH_BUDGET_EXCEEDED", "LIBRARY_GAP",
    "NONDETERMINISTIC_TACTIC", "COUNTEREXAMPLE_FOUND", "INVARIANT_BREAK",
    "EXPLANATION_COVERAGE_LOW", "LOSSY_MAP_UNDECLARED",
])

# Fail types emitted by this validator
SCHEMA_INVALID = "SCHEMA_INVALID"
MISSING_INVARIANT_DIFF = "MISSING_INVARIANT_DIFF"
LAYER_VIOLATION = "LAYER_VIOLATION"
RESULT_INCOMPLETE = "RESULT_INCOMPLETE"
HASH_SELF_BINDING = "HASH_SELF_BINDING"
BINDING_SIGNATURE_MISMATCH = "BINDING_SIGNATURE_MISMATCH"
COVERAGE_BELOW_THRESHOLD = "COVERAGE_BELOW_THRESHOLD"
DETERMINISM_MISMATCH = "DETERMINISM_MISMATCH"
REPLAY_COUNTS_MISMATCH = "REPLAY_COUNTS_MISMATCH"
REPLAY_BELOW_THRESHOLD = "REPLAY_BELOW_THRESHOLD"
PROVED_PAIR_REPLAY_MISMATCH = "PROVED_PAIR_REPLAY_MISMATCH"
COMPRESSION_METRIC_MISMATCH = "COMPRESSION_METRIC_MISMATCH"
COMPRESSION_BELOW_TARGET = "COMPRESSION_BELOW_TARGET"
NEEDS_PROOF_UNACCOUNTED = "NEEDS_PROOF_UNACCOUNTED"

MIN_COVERAGE_RATIO = 0.5

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



def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())



def _is_nonempty_str_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(_is_nonempty_str(x) for x in v)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ValidationResult:
    def __init__(
        self,
        ok: bool,
        fail_type: Optional[str] = None,
        invariant_diff: Optional[Dict[str, Any]] = None,
        obj_id: str = "",
    ):
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
# TRACE VALIDATION (existing Family [31] contract)
# ===================================================================


def _trace_check_invariant_diff(t: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if "invariant_diff" not in t or not isinstance(t.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {
            "path": "$.invariant_diff",
            "reason": "must be a JSON object",
        }
    return None



def _trace_check_schema(t: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    required = [
        "schema_id",
        "trace_id",
        "created_utc",
        "agent_id",
        "source_layer",
        "target_layer",
        "generator",
        "input_hash",
        "output_hash",
        "toolchain_id",
        "result",
        "merkle_parent",
        "invariant_diff",
    ]
    for field in required:
        if field not in t:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if t["schema_id"] != TRACE_SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": TRACE_SCHEMA_ID,
            "got": t["schema_id"],
            "path": "$.schema_id",
        }
    if not _is_hex64(t["trace_id"]):
        return SCHEMA_INVALID, {"path": "$.trace_id", "reason": "must be hex64"}
    if not _is_utc_ts(t["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad UTC timestamp"}
    if not _is_nonempty_str(t["agent_id"]):
        return SCHEMA_INVALID, {"path": "$.agent_id", "reason": "must be non-empty string"}
    if t["source_layer"] not in VALID_LAYERS:
        return SCHEMA_INVALID, {
            "path": "$.source_layer",
            "reason": f"must be one of {sorted(VALID_LAYERS)}",
        }
    if t["target_layer"] not in VALID_LAYERS:
        return SCHEMA_INVALID, {
            "path": "$.target_layer",
            "reason": f"must be one of {sorted(VALID_LAYERS)}",
        }
    if not _is_nonempty_str(t["generator"]):
        return SCHEMA_INVALID, {"path": "$.generator", "reason": "must be non-empty string"}
    if not _is_hex64(t["input_hash"]):
        return SCHEMA_INVALID, {"path": "$.input_hash", "reason": "must be hex64"}
    if not _is_hex64(t["output_hash"]):
        return SCHEMA_INVALID, {"path": "$.output_hash", "reason": "must be hex64"}
    if not _is_nonempty_str(t["toolchain_id"]):
        return SCHEMA_INVALID, {"path": "$.toolchain_id", "reason": "must be non-empty string"}
    if not _is_hex64(t["merkle_parent"]):
        return SCHEMA_INVALID, {"path": "$.merkle_parent", "reason": "must be hex64"}

    result = t["result"]
    if not isinstance(result, dict):
        return SCHEMA_INVALID, {"path": "$.result", "reason": "must be object"}
    if result.get("status") not in VALID_STATUSES:
        return SCHEMA_INVALID, {
            "path": "$.result.status",
            "reason": f"must be one of {sorted(VALID_STATUSES)}",
        }

    return None



def _trace_check_layers(t: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if t["source_layer"] == t["target_layer"]:
        return LAYER_VIOLATION, {
            "source_layer": t["source_layer"],
            "target_layer": t["target_layer"],
            "reason": "cross-layer compilation requires different source and target layers",
        }
    return None



def _trace_check_result(t: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    result = t["result"]
    if result["status"] == "FAIL":
        if "fail_type" not in result:
            return RESULT_INCOMPLETE, {
                "path": "$.result.fail_type",
                "reason": "FAIL requires fail_type",
            }
        if "invariant_diff" not in result or not isinstance(result.get("invariant_diff"), dict):
            return RESULT_INCOMPLETE, {
                "path": "$.result.invariant_diff",
                "reason": "FAIL requires invariant_diff object",
            }
        if result["fail_type"] not in VALID_TRACE_FAIL_TYPES:
            return SCHEMA_INVALID, {
                "path": "$.result.fail_type",
                "expected_one_of": sorted(VALID_TRACE_FAIL_TYPES),
                "got": result["fail_type"],
            }
    return None



def validate_trace(t: Dict[str, Any]) -> ValidationResult:
    obj_id = t.get("trace_id", "<unknown>")
    for gate in [_trace_check_invariant_diff, _trace_check_schema, _trace_check_layers, _trace_check_result]:
        result = gate(t)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# PAIR VALIDATION (existing Family [31] contract)
# ===================================================================


def _pair_check_invariant_diff(p: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if "invariant_diff" not in p or not isinstance(p.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {
            "path": "$.invariant_diff",
            "reason": "must be a JSON object",
        }
    return None



def _pair_check_schema(p: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    required = [
        "schema_id",
        "pair_id",
        "created_utc",
        "human_hash",
        "formal_hash",
        "alignment_metrics",
        "binding_signature",
        "invariant_diff",
    ]
    for field in required:
        if field not in p:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if p["schema_id"] != PAIR_SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": PAIR_SCHEMA_ID,
            "got": p["schema_id"],
            "path": "$.schema_id",
        }
    if not isinstance(p["pair_id"], str) or len(p["pair_id"]) < 8:
        return SCHEMA_INVALID, {"path": "$.pair_id", "reason": "must be string >=8 chars"}
    if not _is_utc_ts(p["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad UTC timestamp"}
    if not _is_hex64(p["human_hash"]):
        return SCHEMA_INVALID, {"path": "$.human_hash", "reason": "must be hex64"}
    if not _is_hex64(p["formal_hash"]):
        return SCHEMA_INVALID, {"path": "$.formal_hash", "reason": "must be hex64"}
    if not _is_hex64(p["binding_signature"]):
        return SCHEMA_INVALID, {"path": "$.binding_signature", "reason": "must be hex64"}

    am = p["alignment_metrics"]
    if not isinstance(am, dict):
        return SCHEMA_INVALID, {"path": "$.alignment_metrics", "reason": "must be object"}
    for mk in ["coverage_ratio", "definition_overlap", "lemma_overlap"]:
        if mk not in am:
            return SCHEMA_INVALID, {"missing_field": mk, "path": f"$.alignment_metrics.{mk}"}
    if not isinstance(am["coverage_ratio"], (int, float)):
        return SCHEMA_INVALID, {
            "path": "$.alignment_metrics.coverage_ratio",
            "reason": "must be number",
        }
    if not isinstance(am["definition_overlap"], int) or am["definition_overlap"] < 0:
        return SCHEMA_INVALID, {
            "path": "$.alignment_metrics.definition_overlap",
            "reason": "must be non-negative integer",
        }
    if not isinstance(am["lemma_overlap"], int) or am["lemma_overlap"] < 0:
        return SCHEMA_INVALID, {
            "path": "$.alignment_metrics.lemma_overlap",
            "reason": "must be non-negative integer",
        }

    return None



def _pair_check_hash_self_binding(p: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if p["human_hash"] == p["formal_hash"]:
        return HASH_SELF_BINDING, {
            "human_hash": p["human_hash"],
            "formal_hash": p["formal_hash"],
            "reason": "human and formal hashes must differ (would mean same object in both layers)",
        }
    return None



def _pair_check_binding_signature(p: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    expected = _sha256(_canonical({"formal_hash": p["formal_hash"], "human_hash": p["human_hash"]}))
    if p["binding_signature"] != expected:
        return BINDING_SIGNATURE_MISMATCH, {
            "expected": expected,
            "got": p["binding_signature"],
            "reason": "binding_signature must be sha256(canonical({formal_hash, human_hash}))",
        }
    return None



def _pair_check_coverage(p: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    coverage_ratio = p["alignment_metrics"]["coverage_ratio"]
    if not (0.0 <= coverage_ratio <= 1.0):
        return SCHEMA_INVALID, {
            "path": "$.alignment_metrics.coverage_ratio",
            "reason": "must be in [0, 1]",
        }
    if coverage_ratio < MIN_COVERAGE_RATIO:
        return COVERAGE_BELOW_THRESHOLD, {
            "coverage_ratio": coverage_ratio,
            "threshold": MIN_COVERAGE_RATIO,
            "reason": f"coverage_ratio {coverage_ratio} below minimum {MIN_COVERAGE_RATIO}",
        }
    return None



def validate_pair(p: Dict[str, Any]) -> ValidationResult:
    obj_id = p.get("pair_id", "<unknown>")
    for gate in [
        _pair_check_invariant_diff,
        _pair_check_schema,
        _pair_check_hash_self_binding,
        _pair_check_binding_signature,
        _pair_check_coverage,
    ]:
        result = gate(p)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# FORMAL TASK VALIDATION
# ===================================================================


def _task_check_invariant_diff(task: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if "invariant_diff" not in task or not isinstance(task.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {
            "path": "$.invariant_diff",
            "reason": "must be a JSON object",
        }
    return None



def _task_check_schema(task: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    required = [
        "schema_id",
        "task_id",
        "created_utc",
        "formal_goal",
        "imports",
        "context",
        "constraints",
        "invariant_diff",
    ]
    for field in required:
        if field not in task:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if task["schema_id"] != TASK_SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": TASK_SCHEMA_ID,
            "got": task["schema_id"],
            "path": "$.schema_id",
        }
    if not _is_nonempty_str(task["task_id"]):
        return SCHEMA_INVALID, {"path": "$.task_id", "reason": "must be non-empty string"}
    if not _is_utc_ts(task["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad UTC timestamp"}

    if "nl_statement" in task and task["nl_statement"] is not None and not _is_nonempty_str(task["nl_statement"]):
        return SCHEMA_INVALID, {
            "path": "$.nl_statement",
            "reason": "must be non-empty string or null",
        }

    if not _is_nonempty_str(task["formal_goal"]):
        return SCHEMA_INVALID, {"path": "$.formal_goal", "reason": "must be non-empty string"}

    if not isinstance(task["imports"], list) or not all(_is_nonempty_str(x) for x in task["imports"]):
        return SCHEMA_INVALID, {
            "path": "$.imports",
            "reason": "must be array of non-empty strings",
        }
    if not isinstance(task["context"], list) or not all(_is_nonempty_str(x) for x in task["context"]):
        return SCHEMA_INVALID, {
            "path": "$.context",
            "reason": "must be array of non-empty strings",
        }

    constraints = task["constraints"]
    if not isinstance(constraints, dict):
        return SCHEMA_INVALID, {"path": "$.constraints", "reason": "must be object"}
    for field in ["max_seconds", "max_memory_mb", "allowed_tactics"]:
        if field not in constraints:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.constraints.{field}"}
    if not isinstance(constraints["max_seconds"], int) or constraints["max_seconds"] <= 0:
        return SCHEMA_INVALID, {
            "path": "$.constraints.max_seconds",
            "reason": "must be positive integer",
        }
    if not isinstance(constraints["max_memory_mb"], int) or constraints["max_memory_mb"] <= 0:
        return SCHEMA_INVALID, {
            "path": "$.constraints.max_memory_mb",
            "reason": "must be positive integer",
        }
    if not _is_nonempty_str_list(constraints["allowed_tactics"]):
        return SCHEMA_INVALID, {
            "path": "$.constraints.allowed_tactics",
            "reason": "must be non-empty array of non-empty strings",
        }

    return None



def validate_task(task: Dict[str, Any]) -> ValidationResult:
    obj_id = task.get("task_id", "<unknown>")
    for gate in [_task_check_invariant_diff, _task_check_schema]:
        result = gate(task)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# REPLAY BUNDLE VALIDATION
# ===================================================================


def _replay_check_invariant_diff(bundle: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if "invariant_diff" not in bundle or not isinstance(bundle.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {
            "path": "$.invariant_diff",
            "reason": "must be a JSON object",
        }
    return None



def _replay_check_schema(bundle: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    required = [
        "schema_id",
        "bundle_id",
        "created_utc",
        "toolchain",
        "benchmark",
        "traces",
        "metrics",
        "invariant_diff",
    ]
    for field in required:
        if field not in bundle:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if bundle["schema_id"] != REPLAY_SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": REPLAY_SCHEMA_ID,
            "got": bundle["schema_id"],
            "path": "$.schema_id",
        }
    if not _is_nonempty_str(bundle["bundle_id"]):
        return SCHEMA_INVALID, {"path": "$.bundle_id", "reason": "must be non-empty string"}
    if not _is_utc_ts(bundle["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad UTC timestamp"}

    toolchain = bundle["toolchain"]
    if not isinstance(toolchain, dict):
        return SCHEMA_INVALID, {"path": "$.toolchain", "reason": "must be object"}
    for field in ["lean_version", "lake_lock_hash", "toolchain_hash"]:
        if field not in toolchain:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.toolchain.{field}"}
    if not _is_nonempty_str(toolchain["lean_version"]):
        return SCHEMA_INVALID, {"path": "$.toolchain.lean_version", "reason": "must be non-empty string"}
    if not _is_hex64(toolchain["lake_lock_hash"]):
        return SCHEMA_INVALID, {"path": "$.toolchain.lake_lock_hash", "reason": "must be hex64"}
    if not _is_hex64(toolchain["toolchain_hash"]):
        return SCHEMA_INVALID, {"path": "$.toolchain.toolchain_hash", "reason": "must be hex64"}

    benchmark = bundle["benchmark"]
    if not isinstance(benchmark, dict):
        return SCHEMA_INVALID, {"path": "$.benchmark", "reason": "must be object"}
    for field in ["trace_count", "min_replay_rate"]:
        if field not in benchmark:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.benchmark.{field}"}
    if not isinstance(benchmark["trace_count"], int) or benchmark["trace_count"] <= 0:
        return SCHEMA_INVALID, {
            "path": "$.benchmark.trace_count",
            "reason": "must be positive integer",
        }
    if not isinstance(benchmark["min_replay_rate"], (int, float)) or not (0.0 <= benchmark["min_replay_rate"] <= 1.0):
        return SCHEMA_INVALID, {
            "path": "$.benchmark.min_replay_rate",
            "reason": "must be number in [0,1]",
        }

    traces = bundle["traces"]
    if not isinstance(traces, list) or len(traces) == 0:
        return SCHEMA_INVALID, {"path": "$.traces", "reason": "must be non-empty array"}
    for idx, tr in enumerate(traces):
        if not isinstance(tr, dict):
            return SCHEMA_INVALID, {"path": f"$.traces[{idx}]", "reason": "must be object"}
        for field in ["trace_id", "seed", "trace_hash", "replay_hash", "result_status", "replay_status"]:
            if field not in tr:
                return SCHEMA_INVALID, {"missing_field": field, "path": f"$.traces[{idx}].{field}"}
        if not _is_hex64(tr["trace_id"]):
            return SCHEMA_INVALID, {"path": f"$.traces[{idx}].trace_id", "reason": "must be hex64"}
        if not isinstance(tr["seed"], int) or tr["seed"] < 0:
            return SCHEMA_INVALID, {"path": f"$.traces[{idx}].seed", "reason": "must be non-negative integer"}
        if not _is_hex64(tr["trace_hash"]):
            return SCHEMA_INVALID, {"path": f"$.traces[{idx}].trace_hash", "reason": "must be hex64"}
        if not _is_hex64(tr["replay_hash"]):
            return SCHEMA_INVALID, {"path": f"$.traces[{idx}].replay_hash", "reason": "must be hex64"}
        if tr["result_status"] not in VALID_STATUSES:
            return SCHEMA_INVALID, {
                "path": f"$.traces[{idx}].result_status",
                "reason": f"must be one of {sorted(VALID_STATUSES)}",
            }
        if tr["replay_status"] not in VALID_REPLAY_STATUSES:
            return SCHEMA_INVALID, {
                "path": f"$.traces[{idx}].replay_status",
                "reason": f"must be one of {sorted(VALID_REPLAY_STATUSES)}",
            }

    metrics = bundle["metrics"]
    if not isinstance(metrics, dict):
        return SCHEMA_INVALID, {"path": "$.metrics", "reason": "must be object"}
    for field in [
        "deterministic_replays",
        "total_replays",
        "replay_successes",
        "replay_rate",
        "infra_flake_count",
    ]:
        if field not in metrics:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.metrics.{field}"}
    int_fields = ["deterministic_replays", "total_replays", "replay_successes", "infra_flake_count"]
    for field in int_fields:
        if not isinstance(metrics[field], int) or metrics[field] < 0:
            return SCHEMA_INVALID, {"path": f"$.metrics.{field}", "reason": "must be non-negative integer"}
    if not isinstance(metrics["replay_rate"], (int, float)) or not (0.0 <= metrics["replay_rate"] <= 1.0):
        return SCHEMA_INVALID, {"path": "$.metrics.replay_rate", "reason": "must be number in [0,1]"}

    return None



def _replay_check_determinism(bundle: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    for tr in bundle["traces"]:
        if tr["replay_status"] == "INFRA_FLAKE":
            continue
        if tr["trace_hash"] != tr["replay_hash"]:
            return DETERMINISM_MISMATCH, {
                "trace_id": tr["trace_id"],
                "trace_hash": tr["trace_hash"],
                "replay_hash": tr["replay_hash"],
                "reason": "same seed + toolchain must replay to identical hash",
            }
    return None



def _replay_check_metrics(bundle: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    traces = bundle["traces"]
    metrics = bundle["metrics"]
    benchmark = bundle["benchmark"]

    total_replays = len(traces)
    deterministic_replays = sum(
        1
        for tr in traces
        if tr["replay_status"] != "INFRA_FLAKE" and tr["trace_hash"] == tr["replay_hash"]
    )
    replay_successes = sum(1 for tr in traces if tr["replay_status"] == "SUCCESS")
    infra_flake_count = sum(1 for tr in traces if tr["replay_status"] == "INFRA_FLAKE")
    replay_rate = replay_successes / total_replays if total_replays > 0 else 0.0

    checks = {
        "benchmark.trace_count": (benchmark["trace_count"], total_replays),
        "metrics.total_replays": (metrics["total_replays"], total_replays),
        "metrics.deterministic_replays": (metrics["deterministic_replays"], deterministic_replays),
        "metrics.replay_successes": (metrics["replay_successes"], replay_successes),
        "metrics.infra_flake_count": (metrics["infra_flake_count"], infra_flake_count),
    }
    for path, (declared, computed) in checks.items():
        if declared != computed:
            return REPLAY_COUNTS_MISMATCH, {
                "path": path,
                "declared": declared,
                "computed": computed,
            }

    if abs(float(metrics["replay_rate"]) - replay_rate) > 1e-12:
        return REPLAY_COUNTS_MISMATCH, {
            "path": "$.metrics.replay_rate",
            "declared": float(metrics["replay_rate"]),
            "computed": replay_rate,
        }

    if replay_rate < float(benchmark["min_replay_rate"]):
        return REPLAY_BELOW_THRESHOLD, {
            "replay_rate": replay_rate,
            "min_replay_rate": benchmark["min_replay_rate"],
            "reason": "bundle replay rate below configured threshold",
        }

    return None



def validate_replay_bundle(bundle: Dict[str, Any]) -> ValidationResult:
    obj_id = bundle.get("bundle_id", "<unknown>")
    for gate in [_replay_check_invariant_diff, _replay_check_schema, _replay_check_determinism, _replay_check_metrics]:
        result = gate(bundle)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# HUMAN <-> FORMAL PAIR CERT v1 VALIDATION
# ===================================================================


def _pair_v1_check_invariant_diff(pair: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if "invariant_diff" not in pair or not isinstance(pair.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {
            "path": "$.invariant_diff",
            "reason": "must be a JSON object",
        }
    return None



def _pair_v1_check_schema(pair: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    required = [
        "schema_id",
        "pair_id",
        "created_utc",
        "natural_language_claim",
        "formal_statement",
        "alignment_evidence",
        "trace_ref",
        "status",
        "objections",
        "invariant_diff",
    ]
    for field in required:
        if field not in pair:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if pair["schema_id"] != PAIR_V1_SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": PAIR_V1_SCHEMA_ID,
            "got": pair["schema_id"],
            "path": "$.schema_id",
        }
    if not _is_nonempty_str(pair["pair_id"]):
        return SCHEMA_INVALID, {"path": "$.pair_id", "reason": "must be non-empty string"}
    if not _is_utc_ts(pair["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad UTC timestamp"}
    if not _is_nonempty_str(pair["natural_language_claim"]):
        return SCHEMA_INVALID, {"path": "$.natural_language_claim", "reason": "must be non-empty string"}
    if not _is_nonempty_str(pair["formal_statement"]):
        return SCHEMA_INVALID, {"path": "$.formal_statement", "reason": "must be non-empty string"}

    evidence = pair["alignment_evidence"]
    if not isinstance(evidence, dict):
        return SCHEMA_INVALID, {"path": "$.alignment_evidence", "reason": "must be object"}
    for field in ["key_lemmas", "span_mappings"]:
        if field not in evidence:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.alignment_evidence.{field}"}
    if not isinstance(evidence["key_lemmas"], list) or not all(_is_nonempty_str(x) for x in evidence["key_lemmas"]):
        return SCHEMA_INVALID, {
            "path": "$.alignment_evidence.key_lemmas",
            "reason": "must be array of non-empty strings",
        }
    if not isinstance(evidence["span_mappings"], list) or len(evidence["span_mappings"]) == 0:
        return SCHEMA_INVALID, {
            "path": "$.alignment_evidence.span_mappings",
            "reason": "must be non-empty array",
        }
    for idx, mapping in enumerate(evidence["span_mappings"]):
        if not isinstance(mapping, dict):
            return SCHEMA_INVALID, {
                "path": f"$.alignment_evidence.span_mappings[{idx}]",
                "reason": "must be object",
            }
        for field in ["nl_span", "formal_identifiers"]:
            if field not in mapping:
                return SCHEMA_INVALID, {
                    "missing_field": field,
                    "path": f"$.alignment_evidence.span_mappings[{idx}].{field}",
                }
        if not _is_nonempty_str(mapping["nl_span"]):
            return SCHEMA_INVALID, {
                "path": f"$.alignment_evidence.span_mappings[{idx}].nl_span",
                "reason": "must be non-empty string",
            }
        if not _is_nonempty_str_list(mapping["formal_identifiers"]):
            return SCHEMA_INVALID, {
                "path": f"$.alignment_evidence.span_mappings[{idx}].formal_identifiers",
                "reason": "must be non-empty array of non-empty strings",
            }

    trace_ref = pair["trace_ref"]
    if not isinstance(trace_ref, dict):
        return SCHEMA_INVALID, {"path": "$.trace_ref", "reason": "must be object"}
    for field in ["trace_id", "result_status", "replay_status"]:
        if field not in trace_ref:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.trace_ref.{field}"}
    if not _is_hex64(trace_ref["trace_id"]):
        return SCHEMA_INVALID, {"path": "$.trace_ref.trace_id", "reason": "must be hex64"}
    if trace_ref["result_status"] not in VALID_STATUSES:
        return SCHEMA_INVALID, {
            "path": "$.trace_ref.result_status",
            "reason": f"must be one of {sorted(VALID_STATUSES)}",
        }
    if trace_ref["replay_status"] not in VALID_REPLAY_STATUSES:
        return SCHEMA_INVALID, {
            "path": "$.trace_ref.replay_status",
            "reason": f"must be one of {sorted(VALID_REPLAY_STATUSES)}",
        }

    if pair["status"] not in VALID_PAIR_STATUSES:
        return SCHEMA_INVALID, {
            "path": "$.status",
            "reason": f"must be one of {sorted(VALID_PAIR_STATUSES)}",
        }

    if not isinstance(pair["objections"], list):
        return SCHEMA_INVALID, {"path": "$.objections", "reason": "must be array"}
    for idx, objection in enumerate(pair["objections"]):
        if not isinstance(objection, dict):
            return SCHEMA_INVALID, {"path": f"$.objections[{idx}]", "reason": "must be object"}
        for field in ["type", "message", "pointer"]:
            if field not in objection:
                return SCHEMA_INVALID, {
                    "missing_field": field,
                    "path": f"$.objections[{idx}].{field}",
                }
            if not _is_nonempty_str(objection[field]):
                return SCHEMA_INVALID, {
                    "path": f"$.objections[{idx}].{field}",
                    "reason": "must be non-empty string",
                }

    return None



def _pair_v1_check_proved(pair: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if pair["status"] != "PROVED":
        return None
    trace_ref = pair["trace_ref"]
    if trace_ref["result_status"] != "SUCCESS" or trace_ref["replay_status"] != "SUCCESS":
        return PROVED_PAIR_REPLAY_MISMATCH, {
            "status": pair["status"],
            "result_status": trace_ref["result_status"],
            "replay_status": trace_ref["replay_status"],
            "reason": "PROVED pairs must reference SUCCESS traces that replay to SUCCESS",
        }
    return None



def validate_pair_v1(pair: Dict[str, Any]) -> ValidationResult:
    obj_id = pair.get("pair_id", "<unknown>")
    for gate in [_pair_v1_check_invariant_diff, _pair_v1_check_schema, _pair_v1_check_proved]:
        result = gate(pair)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# LEMMA MINING VALIDATION
# ===================================================================


def _lemma_check_invariant_diff(pack: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if "invariant_diff" not in pack or not isinstance(pack.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {
            "path": "$.invariant_diff",
            "reason": "must be a JSON object",
        }
    return None



def _lemma_check_schema(pack: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    required = [
        "schema_id",
        "mining_id",
        "created_utc",
        "source_traces",
        "baseline_trace_lengths",
        "compressed_trace_lengths",
        "target_median_reduction",
        "lemma_candidates",
        "failure_records",
        "metrics",
        "invariant_diff",
    ]
    for field in required:
        if field not in pack:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if pack["schema_id"] != LEMMA_SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": LEMMA_SCHEMA_ID,
            "got": pack["schema_id"],
            "path": "$.schema_id",
        }
    if not _is_nonempty_str(pack["mining_id"]):
        return SCHEMA_INVALID, {"path": "$.mining_id", "reason": "must be non-empty string"}
    if not _is_utc_ts(pack["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad UTC timestamp"}

    if not isinstance(pack["source_traces"], list) or len(pack["source_traces"]) == 0:
        return SCHEMA_INVALID, {"path": "$.source_traces", "reason": "must be non-empty array"}
    for idx, value in enumerate(pack["source_traces"]):
        if not _is_hex64(value):
            return SCHEMA_INVALID, {
                "path": f"$.source_traces[{idx}]",
                "reason": "must be hex64",
            }

    baseline = pack["baseline_trace_lengths"]
    compressed = pack["compressed_trace_lengths"]
    if not isinstance(baseline, list) or not isinstance(compressed, list) or len(baseline) == 0:
        return SCHEMA_INVALID, {
            "path": "$.baseline_trace_lengths",
            "reason": "must be non-empty array",
        }
    if len(baseline) != len(compressed):
        return SCHEMA_INVALID, {
            "path": "$.compressed_trace_lengths",
            "reason": "must have same length as baseline_trace_lengths",
        }
    for idx, (base_len, comp_len) in enumerate(zip(baseline, compressed)):
        if not isinstance(base_len, int) or base_len <= 0:
            return SCHEMA_INVALID, {
                "path": f"$.baseline_trace_lengths[{idx}]",
                "reason": "must be positive integer",
            }
        if not isinstance(comp_len, int) or comp_len <= 0:
            return SCHEMA_INVALID, {
                "path": f"$.compressed_trace_lengths[{idx}]",
                "reason": "must be positive integer",
            }
        if comp_len > base_len:
            return SCHEMA_INVALID, {
                "path": f"$.compressed_trace_lengths[{idx}]",
                "reason": "compressed length cannot exceed baseline length",
            }

    target = pack["target_median_reduction"]
    if not isinstance(target, (int, float)) or not (0.0 <= target <= 1.0):
        return SCHEMA_INVALID, {
            "path": "$.target_median_reduction",
            "reason": "must be number in [0,1]",
        }

    candidates = pack["lemma_candidates"]
    if not isinstance(candidates, list) or len(candidates) == 0:
        return SCHEMA_INVALID, {
            "path": "$.lemma_candidates",
            "reason": "must be non-empty array",
        }
    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            return SCHEMA_INVALID, {"path": f"$.lemma_candidates[{idx}]", "reason": "must be object"}
        for field in ["lemma_id", "statement", "usage_count", "compression_gain_steps", "proof_status", "dependency_imports"]:
            if field not in cand:
                return SCHEMA_INVALID, {
                    "missing_field": field,
                    "path": f"$.lemma_candidates[{idx}].{field}",
                }
        if not _is_nonempty_str(cand["lemma_id"]):
            return SCHEMA_INVALID, {"path": f"$.lemma_candidates[{idx}].lemma_id", "reason": "must be non-empty string"}
        if not _is_nonempty_str(cand["statement"]):
            return SCHEMA_INVALID, {"path": f"$.lemma_candidates[{idx}].statement", "reason": "must be non-empty string"}
        if not isinstance(cand["usage_count"], int) or cand["usage_count"] < 0:
            return SCHEMA_INVALID, {"path": f"$.lemma_candidates[{idx}].usage_count", "reason": "must be non-negative integer"}
        if not isinstance(cand["compression_gain_steps"], int) or cand["compression_gain_steps"] < 0:
            return SCHEMA_INVALID, {
                "path": f"$.lemma_candidates[{idx}].compression_gain_steps",
                "reason": "must be non-negative integer",
            }
        if cand["proof_status"] not in VALID_PROOF_STATUSES:
            return SCHEMA_INVALID, {
                "path": f"$.lemma_candidates[{idx}].proof_status",
                "reason": f"must be one of {sorted(VALID_PROOF_STATUSES)}",
            }
        if not _is_nonempty_str_list(cand["dependency_imports"]):
            return SCHEMA_INVALID, {
                "path": f"$.lemma_candidates[{idx}].dependency_imports",
                "reason": "must be non-empty array of non-empty strings",
            }

    records = pack["failure_records"]
    if not isinstance(records, list):
        return SCHEMA_INVALID, {"path": "$.failure_records", "reason": "must be array"}
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            return SCHEMA_INVALID, {"path": f"$.failure_records[{idx}]", "reason": "must be object"}
        for field in ["lemma_id", "fail_type", "invariant_diff"]:
            if field not in record:
                return SCHEMA_INVALID, {
                    "missing_field": field,
                    "path": f"$.failure_records[{idx}].{field}",
                }
        if not _is_nonempty_str(record["lemma_id"]):
            return SCHEMA_INVALID, {
                "path": f"$.failure_records[{idx}].lemma_id",
                "reason": "must be non-empty string",
            }
        if not _is_nonempty_str(record["fail_type"]):
            return SCHEMA_INVALID, {
                "path": f"$.failure_records[{idx}].fail_type",
                "reason": "must be non-empty string",
            }
        if not isinstance(record["invariant_diff"], dict):
            return SCHEMA_INVALID, {
                "path": f"$.failure_records[{idx}].invariant_diff",
                "reason": "must be object",
            }

    metrics = pack["metrics"]
    if not isinstance(metrics, dict):
        return SCHEMA_INVALID, {"path": "$.metrics", "reason": "must be object"}
    for field in ["median_reduction", "total_candidates"]:
        if field not in metrics:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.metrics.{field}"}
    if not isinstance(metrics["median_reduction"], (int, float)) or not (0.0 <= metrics["median_reduction"] <= 1.0):
        return SCHEMA_INVALID, {
            "path": "$.metrics.median_reduction",
            "reason": "must be number in [0,1]",
        }
    if not isinstance(metrics["total_candidates"], int) or metrics["total_candidates"] < 0:
        return SCHEMA_INVALID, {
            "path": "$.metrics.total_candidates",
            "reason": "must be non-negative integer",
        }

    return None



def _lemma_check_compression(pack: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    baseline = pack["baseline_trace_lengths"]
    compressed = pack["compressed_trace_lengths"]

    reductions = [(float(base_len - comp_len) / float(base_len)) for base_len, comp_len in zip(baseline, compressed)]
    observed_median = float(statistics.median(reductions))
    declared_median = float(pack["metrics"]["median_reduction"])

    if abs(observed_median - declared_median) > 1e-12:
        return COMPRESSION_METRIC_MISMATCH, {
            "declared_median_reduction": declared_median,
            "observed_median_reduction": observed_median,
        }

    if pack["metrics"]["total_candidates"] != len(pack["lemma_candidates"]):
        return SCHEMA_INVALID, {
            "path": "$.metrics.total_candidates",
            "declared": pack["metrics"]["total_candidates"],
            "observed": len(pack["lemma_candidates"]),
        }

    target = float(pack["target_median_reduction"])
    if observed_median < target:
        return COMPRESSION_BELOW_TARGET, {
            "observed_median_reduction": observed_median,
            "target_median_reduction": target,
            "reason": "median compression below target",
        }

    return None



def _lemma_check_needs_proof(pack: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    record_by_lemma = {record["lemma_id"]: record for record in pack["failure_records"]}
    for candidate in pack["lemma_candidates"]:
        if candidate["proof_status"] == "NEEDS_PROOF":
            record = record_by_lemma.get(candidate["lemma_id"])
            if record is None:
                return NEEDS_PROOF_UNACCOUNTED, {
                    "lemma_id": candidate["lemma_id"],
                    "reason": "NEEDS_PROOF candidate must have failure record",
                }
            if "fail_type" not in record or "invariant_diff" not in record:
                return NEEDS_PROOF_UNACCOUNTED, {
                    "lemma_id": candidate["lemma_id"],
                    "reason": "failure record must include fail_type and invariant_diff",
                }
    return None



def validate_lemma_mining(pack: Dict[str, Any]) -> ValidationResult:
    obj_id = pack.get("mining_id", "<unknown>")
    for gate in [_lemma_check_invariant_diff, _lemma_check_schema, _lemma_check_compression, _lemma_check_needs_proof]:
        result = gate(pack)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    print("--- qa_math_compiler_validator self-test ---")
    passed = 0
    failed = 0

    # -- Trace tests --
    valid_trace = {
        "schema_id": TRACE_SCHEMA_ID,
        "trace_id": "a" * 64,
        "created_utc": "2026-02-10T23:00:00Z",
        "agent_id": "qa-agent-1",
        "source_layer": "human",
        "target_layer": "formal",
        "generator": "sigma_autoformalize",
        "input_hash": "b" * 64,
        "output_hash": "c" * 64,
        "toolchain_id": "lean4.12.0",
        "result": {"status": "SUCCESS", "witness_hash": "d" * 64},
        "merkle_parent": "e" * 64,
        "invariant_diff": {"note": "self-test"},
    }

    r = validate_trace(valid_trace)
    if r.ok:
        print("  [PASS] valid trace accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid trace rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    bad = copy.deepcopy(valid_trace)
    del bad["invariant_diff"]
    r = validate_trace(bad)
    if not r.ok and r.fail_type == MISSING_INVARIANT_DIFF:
        print("  [PASS] trace missing invariant_diff -> MISSING_INVARIANT_DIFF")
        passed += 1
    else:
        print(f"  [FAIL] expected MISSING_INVARIANT_DIFF, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    bad = copy.deepcopy(valid_trace)
    bad["source_layer"] = "formal"
    bad["target_layer"] = "formal"
    r = validate_trace(bad)
    if not r.ok and r.fail_type == LAYER_VIOLATION:
        print("  [PASS] same-layer trace -> LAYER_VIOLATION")
        passed += 1
    else:
        print(f"  [FAIL] expected LAYER_VIOLATION, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    bad = copy.deepcopy(valid_trace)
    bad["result"] = {"status": "FAIL", "fail_type": "TYPE_MISMATCH"}
    r = validate_trace(bad)
    if not r.ok and r.fail_type == RESULT_INCOMPLETE:
        print("  [PASS] FAIL without result.invariant_diff -> RESULT_INCOMPLETE")
        passed += 1
    else:
        print(f"  [FAIL] expected RESULT_INCOMPLETE, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    good_fail = copy.deepcopy(valid_trace)
    good_fail["result"] = {
        "status": "FAIL",
        "fail_type": "TYPE_MISMATCH",
        "invariant_diff": {"expected_type": "Nat", "found_type": "Int"},
    }
    r = validate_trace(good_fail)
    if r.ok:
        print("  [PASS] valid FAIL trace accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid FAIL trace rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    # -- Pair tests --
    human_hash = "a1" * 32
    formal_hash = "b2" * 32
    binding = _sha256(_canonical({"formal_hash": formal_hash, "human_hash": human_hash}))

    valid_pair = {
        "schema_id": PAIR_SCHEMA_ID,
        "pair_id": "PAIR-SELFTEST0001",
        "created_utc": "2026-02-10T23:00:00Z",
        "human_hash": human_hash,
        "formal_hash": formal_hash,
        "alignment_metrics": {
            "coverage_ratio": 0.87,
            "definition_overlap": 12,
            "lemma_overlap": 9,
        },
        "binding_signature": binding,
        "invariant_diff": {"note": "self-test pair"},
    }

    r = validate_pair(valid_pair)
    if r.ok:
        print("  [PASS] valid pair accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid pair rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    bad = copy.deepcopy(valid_pair)
    bad["formal_hash"] = bad["human_hash"]
    bad["binding_signature"] = _sha256(
        _canonical({"formal_hash": bad["formal_hash"], "human_hash": bad["human_hash"]})
    )
    r = validate_pair(bad)
    if not r.ok and r.fail_type == HASH_SELF_BINDING:
        print("  [PASS] human==formal hash -> HASH_SELF_BINDING")
        passed += 1
    else:
        print(f"  [FAIL] expected HASH_SELF_BINDING, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    bad = copy.deepcopy(valid_pair)
    bad["binding_signature"] = "f" * 64
    r = validate_pair(bad)
    if not r.ok and r.fail_type == BINDING_SIGNATURE_MISMATCH:
        print("  [PASS] bad binding_signature -> BINDING_SIGNATURE_MISMATCH")
        passed += 1
    else:
        print(f"  [FAIL] expected BINDING_SIGNATURE_MISMATCH, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    bad = copy.deepcopy(valid_pair)
    bad["alignment_metrics"]["coverage_ratio"] = 0.3
    r = validate_pair(bad)
    if not r.ok and r.fail_type == COVERAGE_BELOW_THRESHOLD:
        print("  [PASS] coverage 0.3 -> COVERAGE_BELOW_THRESHOLD")
        passed += 1
    else:
        print(f"  [FAIL] expected COVERAGE_BELOW_THRESHOLD, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # -- Task tests --
    valid_task = {
        "schema_id": TASK_SCHEMA_ID,
        "task_id": "TASK-SELFTEST-0001",
        "created_utc": "2026-02-11T00:00:00Z",
        "nl_statement": "If n is even then n + n is even.",
        "formal_goal": "theorem even_add_even (n : Nat) (h : Even n) : Even (n + n)",
        "imports": ["Mathlib.Data.Nat.Parity"],
        "context": ["open Nat"],
        "constraints": {
            "max_seconds": 60,
            "max_memory_mb": 2048,
            "allowed_tactics": ["simp", "exact", "constructor"],
        },
        "invariant_diff": {"note": "task self-test"},
    }

    r = validate_task(valid_task)
    if r.ok:
        print("  [PASS] valid task accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid task rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    bad = copy.deepcopy(valid_task)
    bad["formal_goal"] = ""
    r = validate_task(bad)
    if not r.ok and r.fail_type == SCHEMA_INVALID:
        print("  [PASS] empty formal_goal -> SCHEMA_INVALID")
        passed += 1
    else:
        print(f"  [FAIL] expected SCHEMA_INVALID, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # -- Replay tests --
    valid_replay = {
        "schema_id": REPLAY_SCHEMA_ID,
        "bundle_id": "REPLAY-SELFTEST-0001",
        "created_utc": "2026-02-11T00:10:00Z",
        "toolchain": {
            "lean_version": "4.12.0",
            "lake_lock_hash": "1" * 64,
            "toolchain_hash": "2" * 64,
        },
        "benchmark": {"trace_count": 3, "min_replay_rate": 0.33},
        "traces": [
            {
                "trace_id": "a" * 64,
                "seed": 7,
                "trace_hash": "b" * 64,
                "replay_hash": "b" * 64,
                "result_status": "SUCCESS",
                "replay_status": "SUCCESS",
            },
            {
                "trace_id": "c" * 64,
                "seed": 8,
                "trace_hash": "d" * 64,
                "replay_hash": "d" * 64,
                "result_status": "FAIL",
                "replay_status": "FAIL",
            },
            {
                "trace_id": "e" * 64,
                "seed": 9,
                "trace_hash": "f" * 64,
                "replay_hash": "f" * 64,
                "result_status": "SUCCESS",
                "replay_status": "INFRA_FLAKE",
            },
        ],
        "metrics": {
            "deterministic_replays": 2,
            "total_replays": 3,
            "replay_successes": 1,
            "replay_rate": 1.0 / 3.0,
            "infra_flake_count": 1,
        },
        "invariant_diff": {"note": "replay self-test"},
    }

    r = validate_replay_bundle(valid_replay)
    if r.ok:
        print("  [PASS] valid replay bundle accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid replay bundle rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    bad = copy.deepcopy(valid_replay)
    bad["traces"][1]["replay_hash"] = "0" * 64
    r = validate_replay_bundle(bad)
    if not r.ok and r.fail_type == DETERMINISM_MISMATCH:
        print("  [PASS] hash mismatch replay -> DETERMINISM_MISMATCH")
        passed += 1
    else:
        print(f"  [FAIL] expected DETERMINISM_MISMATCH, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # -- Pair v1 tests --
    valid_pair_v1 = {
        "schema_id": PAIR_V1_SCHEMA_ID,
        "pair_id": "PAIR-V1-SELFTEST-0001",
        "created_utc": "2026-02-11T00:20:00Z",
        "natural_language_claim": "Addition of even naturals is even.",
        "formal_statement": "theorem even_add_even : Even n -> Even m -> Even (n + m)",
        "alignment_evidence": {
            "key_lemmas": ["Nat.even_add"],
            "span_mappings": [
                {
                    "nl_span": "even naturals",
                    "formal_identifiers": ["Even", "Nat"],
                }
            ],
        },
        "trace_ref": {
            "trace_id": "9" * 64,
            "result_status": "SUCCESS",
            "replay_status": "SUCCESS",
        },
        "status": "PROVED",
        "objections": [],
        "invariant_diff": {"note": "pair v1 self-test"},
    }

    r = validate_pair_v1(valid_pair_v1)
    if r.ok:
        print("  [PASS] valid pair_v1 accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid pair_v1 rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    bad = copy.deepcopy(valid_pair_v1)
    bad["trace_ref"]["replay_status"] = "FAIL"
    r = validate_pair_v1(bad)
    if not r.ok and r.fail_type == PROVED_PAIR_REPLAY_MISMATCH:
        print("  [PASS] PROVED with replay FAIL -> PROVED_PAIR_REPLAY_MISMATCH")
        passed += 1
    else:
        print(f"  [FAIL] expected PROVED_PAIR_REPLAY_MISMATCH, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # -- Lemma mining tests --
    valid_lemma = {
        "schema_id": LEMMA_SCHEMA_ID,
        "mining_id": "LEMMA-SELFTEST-0001",
        "created_utc": "2026-02-11T00:30:00Z",
        "source_traces": ["a" * 64, "b" * 64, "c" * 64],
        "baseline_trace_lengths": [100, 120, 80],
        "compressed_trace_lengths": [75, 90, 60],
        "target_median_reduction": 0.20,
        "lemma_candidates": [
            {
                "lemma_id": "LEMMA_EVEN_ADD",
                "statement": "Even n -> Even m -> Even (n + m)",
                "usage_count": 12,
                "compression_gain_steps": 35,
                "proof_status": "FOUND",
                "dependency_imports": ["Mathlib.Data.Nat.Parity"],
            },
            {
                "lemma_id": "LEMMA_RING_REWRITE",
                "statement": "n + m = m + n",
                "usage_count": 3,
                "compression_gain_steps": 6,
                "proof_status": "NEEDS_PROOF",
                "dependency_imports": ["Mathlib.Algebra.Group.Basic"],
            },
        ],
        "failure_records": [
            {
                "lemma_id": "LEMMA_RING_REWRITE",
                "fail_type": "NEEDS_PROOF",
                "invariant_diff": {"reason": "proof search exhausted"},
            }
        ],
        "metrics": {
            "median_reduction": 0.25,
            "total_candidates": 2,
        },
        "invariant_diff": {"note": "lemma self-test"},
    }

    r = validate_lemma_mining(valid_lemma)
    if r.ok:
        print("  [PASS] valid lemma mining pack accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid lemma mining pack rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    bad = copy.deepcopy(valid_lemma)
    bad["target_median_reduction"] = 0.30
    r = validate_lemma_mining(bad)
    if not r.ok and r.fail_type == COMPRESSION_BELOW_TARGET:
        print("  [PASS] low median compression -> COMPRESSION_BELOW_TARGET")
        passed += 1
    else:
        print(f"  [FAIL] expected COMPRESSION_BELOW_TARGET, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    total = passed + failed
    print(f"\n  {passed}/{total} self-tests passed")
    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_mode(mode: str, data: Dict[str, Any]) -> ValidationResult:
    if mode == "trace":
        return validate_trace(data)
    if mode == "pair":
        return validate_pair(data)
    if mode == "task":
        return validate_task(data)
    if mode == "replay":
        return validate_replay_bundle(data)
    if mode == "pair_v1":
        return validate_pair_v1(data)
    if mode == "lemma":
        return validate_lemma_mining(data)
    raise ValueError(f"unsupported mode: {mode}")



def main() -> None:
    args = sys.argv[1:]

    if "--self-test" in args:
        ok = _self_test()
        sys.exit(0 if ok else 1)

    valid_modes = ("trace", "pair", "task", "replay", "pair_v1", "lemma")
    if len(args) < 2 or args[0] not in valid_modes:
        print(
            f"Usage: {sys.argv[0]} trace   <file.json> [--ci]", file=sys.stderr
        )
        print(
            f"       {sys.argv[0]} pair    <file.json> [--ci]", file=sys.stderr
        )
        print(
            f"       {sys.argv[0]} task    <file.json> [--ci]", file=sys.stderr
        )
        print(
            f"       {sys.argv[0]} replay  <file.json> [--ci]", file=sys.stderr
        )
        print(
            f"       {sys.argv[0]} pair_v1 <file.json> [--ci]", file=sys.stderr
        )
        print(
            f"       {sys.argv[0]} lemma   <file.json> [--ci]", file=sys.stderr
        )
        print(f"       {sys.argv[0]} --self-test", file=sys.stderr)
        sys.exit(2)

    mode = args[0]
    file_path = Path(args[1])
    ci_mode = "--ci" in args

    if not file_path.exists():
        print(f"ERROR: {file_path} not found", file=sys.stderr)
        sys.exit(2)

    with file_path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            if ci_mode:
                print(f"[FAIL] {file_path.name}: SCHEMA_INVALID (JSON decode: {exc})")
            else:
                print(f"FAIL: {file_path.name}")
                print(json.dumps({"fail_type": SCHEMA_INVALID, "invariant_diff": {"reason": str(exc)}}, indent=2))
            sys.exit(1)

    result = _validate_mode(mode, data)

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
