#!/usr/bin/env python3
"""
Validate QA_AGENT_TRACE_SCHEMA.v1 trace containers.

Checks (in order):
  1. JSON schema structural validation          -> SCHEMA_INVALID
  2. Event ordering: strictly increasing, no gaps -> NONDETERMINISTIC_EVENT_ORDER
  3. invariant_diff present on every event       -> MISSING_INVARIANT_DIFF
  4. Recompute event hashes and chain integrity  -> HASH_MISMATCH / HASH_CHAIN_BROKEN
  5. Summary counter consistency                 -> SUMMARY_MISMATCH
  6. Provenance policy checks                    -> PROVENANCE_INVALID / REDACTION_POLICY_VIOLATION

Usage:
  python qa_agent_trace_validator.py <trace.json>
  python qa_agent_trace_validator.py <trace.json> --ci
  python qa_agent_trace_validator.py --self-test

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

SCHEMA_ID = "QA_AGENT_TRACE_SCHEMA.v1"
HEX64_ZERO = "0" * 64

VALID_EVENT_TYPES = frozenset([
    "read", "search", "plan", "edit", "run", "test", "error", "final",
])

VALID_COLLECTION_METHODS = frozenset(["live_run", "replay", "synthetic"])
VALID_ARTIFACT_POLICIES = frozenset(["hash_only", "redacted", "full"])
VALID_OUTCOMES = frozenset(["resolved", "unresolved", "error", "timeout", "unknown"])
VALID_TASK_TYPES = frozenset(["swe_bench", "swe_bench_lite", "custom"])
VALID_RUNNERS = frozenset(["local", "docker", "ci", "cloud"])

# Fail types
SCHEMA_INVALID = "SCHEMA_INVALID"
NONDETERMINISTIC_EVENT_ORDER = "NONDETERMINISTIC_EVENT_ORDER"
HASH_MISMATCH = "HASH_MISMATCH"
HASH_CHAIN_BROKEN = "HASH_CHAIN_BROKEN"
SUMMARY_MISMATCH = "SUMMARY_MISMATCH"
PROVENANCE_INVALID = "PROVENANCE_INVALID"
REDACTION_POLICY_VIOLATION = "REDACTION_POLICY_VIOLATION"
MISSING_INVARIANT_DIFF = "MISSING_INVARIANT_DIFF"

# Canonical JSON for hashing
def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class TraceValidationResult:
    """Structured validation result."""

    def __init__(self, ok: bool, fail_type: Optional[str] = None,
                 invariant_diff: Optional[Dict[str, Any]] = None,
                 trace_id: str = ""):
        self.ok = ok
        self.fail_type = fail_type
        self.invariant_diff = invariant_diff or {}
        self.trace_id = trace_id

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"ok": self.ok, "trace_id": self.trace_id}
        if not self.ok:
            d["fail_type"] = self.fail_type
            d["invariant_diff"] = self.invariant_diff
        return d


# ---------------------------------------------------------------------------
# Structural (schema) validation — no external deps
# ---------------------------------------------------------------------------

def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _is_utc_ts(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    import re
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", v))


def _check_schema(trace: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
    """Check structural shape. Returns (fail_type, diff) or None on success."""
    # Top-level required fields
    required = [
        "schema_id", "trace_id", "created_utc", "provenance",
        "task", "agent", "environment", "events", "trace_hashes", "summary",
    ]
    for field in required:
        if field not in trace:
            return SCHEMA_INVALID, {"missing_field": field, "path": f"$.{field}"}

    if trace["schema_id"] != SCHEMA_ID:
        return SCHEMA_INVALID, {
            "expected": SCHEMA_ID, "got": trace["schema_id"], "path": "$.schema_id",
        }

    if not isinstance(trace["trace_id"], str) or not trace["trace_id"]:
        return SCHEMA_INVALID, {"path": "$.trace_id", "reason": "must be non-empty string"}

    if not _is_utc_ts(trace["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc", "reason": "bad ISO 8601 UTC timestamp"}

    # --- provenance ---
    prov = trace["provenance"]
    if not isinstance(prov, dict):
        return SCHEMA_INVALID, {"path": "$.provenance", "reason": "must be object"}
    for pf in ["collector", "collection_method", "source_dataset",
                "source_url", "license", "raw_artifacts_policy", "privacy_redactions"]:
        if pf not in prov:
            return SCHEMA_INVALID, {"missing_field": pf, "path": f"$.provenance.{pf}"}
    if prov["collection_method"] not in VALID_COLLECTION_METHODS:
        return SCHEMA_INVALID, {
            "path": "$.provenance.collection_method",
            "expected_one_of": sorted(VALID_COLLECTION_METHODS),
            "got": prov["collection_method"],
        }
    if prov["raw_artifacts_policy"] not in VALID_ARTIFACT_POLICIES:
        return SCHEMA_INVALID, {
            "path": "$.provenance.raw_artifacts_policy",
            "expected_one_of": sorted(VALID_ARTIFACT_POLICIES),
            "got": prov["raw_artifacts_policy"],
        }
    if not isinstance(prov["privacy_redactions"], list):
        return SCHEMA_INVALID, {"path": "$.provenance.privacy_redactions", "reason": "must be array"}

    # --- task ---
    task = trace["task"]
    if not isinstance(task, dict):
        return SCHEMA_INVALID, {"path": "$.task", "reason": "must be object"}
    for tf in ["task_type", "instance_id", "repo", "base_commit", "problem_ref"]:
        if tf not in task:
            return SCHEMA_INVALID, {"missing_field": tf, "path": f"$.task.{tf}"}
    if task["task_type"] not in VALID_TASK_TYPES:
        return SCHEMA_INVALID, {
            "path": "$.task.task_type",
            "expected_one_of": sorted(VALID_TASK_TYPES),
            "got": task["task_type"],
        }
    pref = task["problem_ref"]
    if not isinstance(pref, dict):
        return SCHEMA_INVALID, {"path": "$.task.problem_ref", "reason": "must be object"}
    if not _is_hex64(pref.get("sha256", "")):
        return SCHEMA_INVALID, {"path": "$.task.problem_ref.sha256", "reason": "must be hex64"}
    if not isinstance(pref.get("length_bytes"), int) or pref["length_bytes"] < 0:
        return SCHEMA_INVALID, {"path": "$.task.problem_ref.length_bytes", "reason": "must be non-negative int"}

    # --- agent ---
    agent = trace["agent"]
    if not isinstance(agent, dict):
        return SCHEMA_INVALID, {"path": "$.agent", "reason": "must be object"}
    for af in ["name", "version", "model", "policy"]:
        if af not in agent:
            return SCHEMA_INVALID, {"missing_field": af, "path": f"$.agent.{af}"}
    pol = agent.get("policy", {})
    if not isinstance(pol, dict):
        return SCHEMA_INVALID, {"path": "$.agent.policy", "reason": "must be object"}
    for pk in ["tooling", "guardrails"]:
        if pk not in pol or not isinstance(pol[pk], list):
            return SCHEMA_INVALID, {"path": f"$.agent.policy.{pk}", "reason": "must be array"}

    # --- environment ---
    env = trace["environment"]
    if not isinstance(env, dict):
        return SCHEMA_INVALID, {"path": "$.environment", "reason": "must be object"}
    for ef in ["os", "python", "runner"]:
        if ef not in env:
            return SCHEMA_INVALID, {"missing_field": ef, "path": f"$.environment.{ef}"}
    if env["runner"] not in VALID_RUNNERS:
        return SCHEMA_INVALID, {
            "path": "$.environment.runner",
            "expected_one_of": sorted(VALID_RUNNERS),
            "got": env["runner"],
        }

    # --- events array ---
    events = trace["events"]
    if not isinstance(events, list) or len(events) < 1:
        return SCHEMA_INVALID, {"path": "$.events", "reason": "must be non-empty array"}
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            return SCHEMA_INVALID, {"path": f"$.events[{i}]", "reason": "must be object"}
        for ef in ["event_index", "ts_utc", "event_type", "payload", "invariant_diff"]:
            if ef not in ev:
                return SCHEMA_INVALID, {"missing_field": ef, "path": f"$.events[{i}].{ef}"}
        if not isinstance(ev["event_index"], int):
            return SCHEMA_INVALID, {"path": f"$.events[{i}].event_index", "reason": "must be int"}
        if ev["event_type"] not in VALID_EVENT_TYPES:
            return SCHEMA_INVALID, {
                "path": f"$.events[{i}].event_type",
                "expected_one_of": sorted(VALID_EVENT_TYPES),
                "got": ev["event_type"],
            }
        if not isinstance(ev["payload"], dict):
            return SCHEMA_INVALID, {"path": f"$.events[{i}].payload", "reason": "must be object"}

    # --- trace_hashes ---
    th = trace["trace_hashes"]
    if not isinstance(th, dict):
        return SCHEMA_INVALID, {"path": "$.trace_hashes", "reason": "must be object"}
    for hf in ["events_sha256", "trace_sha256", "hash_chain"]:
        if hf not in th:
            return SCHEMA_INVALID, {"missing_field": hf, "path": f"$.trace_hashes.{hf}"}
    if not _is_hex64(th["events_sha256"]):
        return SCHEMA_INVALID, {"path": "$.trace_hashes.events_sha256", "reason": "must be hex64"}
    if not _is_hex64(th["trace_sha256"]):
        return SCHEMA_INVALID, {"path": "$.trace_hashes.trace_sha256", "reason": "must be hex64"}
    chain = th["hash_chain"]
    if not isinstance(chain, list) or len(chain) < 1:
        return SCHEMA_INVALID, {"path": "$.trace_hashes.hash_chain", "reason": "must be non-empty array"}
    for i, link in enumerate(chain):
        if not isinstance(link, dict):
            return SCHEMA_INVALID, {"path": f"$.trace_hashes.hash_chain[{i}]", "reason": "must be object"}
        for lf in ["event_index", "event_sha256", "prev_sha256"]:
            if lf not in link:
                return SCHEMA_INVALID, {"missing_field": lf, "path": f"$.trace_hashes.hash_chain[{i}].{lf}"}
        if not _is_hex64(link["event_sha256"]) or not _is_hex64(link["prev_sha256"]):
            return SCHEMA_INVALID, {
                "path": f"$.trace_hashes.hash_chain[{i}]",
                "reason": "event_sha256 and prev_sha256 must be hex64",
            }

    # --- summary ---
    summary = trace["summary"]
    if not isinstance(summary, dict):
        return SCHEMA_INVALID, {"path": "$.summary", "reason": "must be object"}
    for sf in ["event_count", "tool_call_count", "edit_count", "test_run_count", "outcome"]:
        if sf not in summary:
            return SCHEMA_INVALID, {"missing_field": sf, "path": f"$.summary.{sf}"}
    if summary["outcome"] not in VALID_OUTCOMES:
        return SCHEMA_INVALID, {
            "path": "$.summary.outcome",
            "expected_one_of": sorted(VALID_OUTCOMES),
            "got": summary["outcome"],
        }

    return None


# ---------------------------------------------------------------------------
# Gate 2: Deterministic event ordering
# ---------------------------------------------------------------------------

def _check_event_order(events: List[Dict]) -> Optional[Tuple[str, Dict]]:
    """Events must be strictly increasing by event_index, gap-free, starting at 0."""
    for i, ev in enumerate(events):
        idx = ev["event_index"]
        if idx != i:
            return NONDETERMINISTIC_EVENT_ORDER, {
                "expected_index": i,
                "got_index": idx,
                "path": f"$.events[{i}].event_index",
            }
    return None


# ---------------------------------------------------------------------------
# Gate 3: invariant_diff presence
# ---------------------------------------------------------------------------

def _check_invariant_diff(events: List[Dict]) -> Optional[Tuple[str, Dict]]:
    """Every event must have a non-None invariant_diff dict."""
    for i, ev in enumerate(events):
        inv = ev.get("invariant_diff")
        if not isinstance(inv, dict):
            return MISSING_INVARIANT_DIFF, {
                "path": f"$.events[{i}].invariant_diff",
                "event_index": ev.get("event_index", i),
                "reason": "must be a JSON object",
            }
    return None


# ---------------------------------------------------------------------------
# Gate 4: Hash integrity
# ---------------------------------------------------------------------------

def _compute_event_hash(event: Dict) -> str:
    """SHA-256 of canonical JSON of a single event."""
    return _sha256(_canonical(event))


def _check_hashes(trace: Dict) -> Optional[Tuple[str, Dict]]:
    """Recompute events_sha256 and the hash chain."""
    events = trace["events"]
    th = trace["trace_hashes"]

    # 4a: events_sha256 = SHA-256 of canonical JSON of events array
    expected_events_hash = _sha256(_canonical(events))
    if th["events_sha256"] != expected_events_hash:
        return HASH_MISMATCH, {
            "field": "events_sha256",
            "expected": expected_events_hash,
            "got": th["events_sha256"],
        }

    # 4b: hash_chain length must match events
    chain = th["hash_chain"]
    if len(chain) != len(events):
        return HASH_CHAIN_BROKEN, {
            "reason": "hash_chain length mismatch",
            "expected": len(events),
            "got": len(chain),
        }

    # 4c: Each chain link must have correct event_index and hashes
    prev = HEX64_ZERO
    for i, (ev, link) in enumerate(zip(events, chain)):
        # event_index alignment
        if link["event_index"] != ev["event_index"]:
            return HASH_CHAIN_BROKEN, {
                "path": f"$.trace_hashes.hash_chain[{i}].event_index",
                "expected": ev["event_index"],
                "got": link["event_index"],
            }

        # prev_sha256 must match prior event_sha256
        if link["prev_sha256"] != prev:
            return HASH_CHAIN_BROKEN, {
                "path": f"$.trace_hashes.hash_chain[{i}].prev_sha256",
                "expected": prev,
                "got": link["prev_sha256"],
            }

        # event_sha256 must match recomputed
        computed = _compute_event_hash(ev)
        if link["event_sha256"] != computed:
            return HASH_MISMATCH, {
                "path": f"$.trace_hashes.hash_chain[{i}].event_sha256",
                "expected": computed,
                "got": link["event_sha256"],
            }

        prev = link["event_sha256"]

    return None


# ---------------------------------------------------------------------------
# Gate 5: Summary consistency
# ---------------------------------------------------------------------------

# Event types that count as tool calls (agent actions)
_TOOL_CALL_TYPES = frozenset(["read", "search", "edit", "run", "test"])

def _check_summary(trace: Dict) -> Optional[Tuple[str, Dict]]:
    """Recompute summary counters from events."""
    events = trace["events"]
    summary = trace["summary"]

    expected_count = len(events)
    if summary["event_count"] != expected_count:
        return SUMMARY_MISMATCH, {
            "field": "event_count",
            "expected": expected_count,
            "got": summary["event_count"],
        }

    tool_calls = sum(1 for ev in events if ev["event_type"] in _TOOL_CALL_TYPES)
    if summary["tool_call_count"] != tool_calls:
        return SUMMARY_MISMATCH, {
            "field": "tool_call_count",
            "expected": tool_calls,
            "got": summary["tool_call_count"],
        }

    edits = sum(1 for ev in events if ev["event_type"] == "edit")
    if summary["edit_count"] != edits:
        return SUMMARY_MISMATCH, {
            "field": "edit_count",
            "expected": edits,
            "got": summary["edit_count"],
        }

    tests = sum(1 for ev in events if ev["event_type"] == "test")
    if summary["test_run_count"] != tests:
        return SUMMARY_MISMATCH, {
            "field": "test_run_count",
            "expected": tests,
            "got": summary["test_run_count"],
        }

    return None


# ---------------------------------------------------------------------------
# Gate 6: Provenance policy checks
# ---------------------------------------------------------------------------

def _check_provenance(trace: Dict) -> Optional[Tuple[str, Dict]]:
    """Check provenance integrity and redaction policy consistency."""
    prov = trace["provenance"]

    # License must be non-empty
    if not prov.get("license", "").strip():
        return PROVENANCE_INVALID, {
            "path": "$.provenance.license",
            "reason": "license must be non-empty",
        }

    # source_dataset must be non-empty
    if not prov.get("source_dataset", "").strip():
        return PROVENANCE_INVALID, {
            "path": "$.provenance.source_dataset",
            "reason": "source_dataset must be non-empty",
        }

    # Redaction policy: if raw_artifacts_policy is hash_only, event payloads
    # must not contain raw text fields (heuristic: no value longer than 256 chars)
    if prov["raw_artifacts_policy"] == "hash_only":
        events = trace["events"]
        for i, ev in enumerate(events):
            payload = ev.get("payload", {})
            for key, val in payload.items():
                if isinstance(val, str) and len(val) > 256 and not _is_hex64(val):
                    return REDACTION_POLICY_VIOLATION, {
                        "path": f"$.events[{i}].payload.{key}",
                        "reason": "raw_artifacts_policy=hash_only but payload contains "
                                  f"non-hash string of length {len(val)}",
                        "policy": "hash_only",
                    }

    return None


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_trace(trace: Dict[str, Any]) -> TraceValidationResult:
    """
    Run all gates in order. Returns on first failure.
    """
    trace_id = trace.get("trace_id", "<unknown>")

    # Gate 1: Schema shape
    result = _check_schema(trace)
    if result is not None:
        return TraceValidationResult(False, result[0], result[1], trace_id)

    events = trace["events"]

    # Gate 2: Event ordering
    result = _check_event_order(events)
    if result is not None:
        return TraceValidationResult(False, result[0], result[1], trace_id)

    # Gate 3: invariant_diff presence
    result = _check_invariant_diff(events)
    if result is not None:
        return TraceValidationResult(False, result[0], result[1], trace_id)

    # Gate 4: Hash integrity
    result = _check_hashes(trace)
    if result is not None:
        return TraceValidationResult(False, result[0], result[1], trace_id)

    # Gate 5: Summary consistency
    result = _check_summary(trace)
    if result is not None:
        return TraceValidationResult(False, result[0], result[1], trace_id)

    # Gate 6: Provenance
    result = _check_provenance(trace)
    if result is not None:
        return TraceValidationResult(False, result[0], result[1], trace_id)

    return TraceValidationResult(True, trace_id=trace_id)


# ---------------------------------------------------------------------------
# Fixture helpers (for building valid traces programmatically)
# ---------------------------------------------------------------------------

def build_hash_chain(events: List[Dict]) -> List[Dict[str, Any]]:
    """Compute the hash chain for a list of events."""
    chain = []
    prev = HEX64_ZERO
    for ev in events:
        h = _compute_event_hash(ev)
        chain.append({
            "event_index": ev["event_index"],
            "event_sha256": h,
            "prev_sha256": prev,
        })
        prev = h
    return chain


def compute_events_sha256(events: List[Dict]) -> str:
    """SHA-256 of the canonical JSON of the events array."""
    return _sha256(_canonical(events))


def build_summary(events: List[Dict], outcome: str = "unresolved") -> Dict[str, Any]:
    """Derive summary from events."""
    return {
        "event_count": len(events),
        "tool_call_count": sum(1 for ev in events if ev["event_type"] in _TOOL_CALL_TYPES),
        "edit_count": sum(1 for ev in events if ev["event_type"] == "edit"),
        "test_run_count": sum(1 for ev in events if ev["event_type"] == "test"),
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    """Run built-in sanity checks. Returns True on pass."""
    print("--- qa_agent_trace_validator self-test ---")
    passed = 0
    failed = 0

    # Build a minimal valid trace
    events = [
        {
            "event_index": 0,
            "ts_utc": "2026-02-10T22:10:01Z",
            "event_type": "read",
            "payload": {"path_hash": "a" * 64},
            "invariant_diff": {"note": "read file"},
        },
        {
            "event_index": 1,
            "ts_utc": "2026-02-10T22:10:10Z",
            "event_type": "test",
            "payload": {"cmd_hash": "b" * 64},
            "invariant_diff": {"exit_code": 1},
        },
    ]
    chain = build_hash_chain(events)
    events_hash = compute_events_sha256(events)
    summary = build_summary(events)

    valid_trace = {
        "schema_id": SCHEMA_ID,
        "trace_id": "TRACE-TEST0001",
        "created_utc": "2026-02-10T22:10:00Z",
        "provenance": {
            "collector": "self_test",
            "collection_method": "synthetic",
            "source_dataset": "test",
            "source_url": "https://example.com",
            "license": "mit",
            "raw_artifacts_policy": "hash_only",
            "privacy_redactions": ["tokens"],
        },
        "task": {
            "task_type": "custom",
            "instance_id": "test-001",
            "repo": "test/repo",
            "base_commit": "abcdef1",
            "problem_ref": {"sha256": "c" * 64, "length_bytes": 100},
        },
        "agent": {
            "name": "test_agent",
            "version": "0.1.0",
            "model": "test-model",
            "policy": {"tooling": ["read_file"], "guardrails": ["qa_guardrail.v1"]},
        },
        "environment": {"os": "linux", "python": "3.11", "runner": "local"},
        "events": events,
        "trace_hashes": {
            "events_sha256": events_hash,
            "trace_sha256": HEX64_ZERO,  # placeholder
            "hash_chain": chain,
        },
        "summary": summary,
    }

    # Test 1: valid trace passes
    r = validate_trace(valid_trace)
    if r.ok:
        print("  [PASS] valid trace accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid trace rejected: {r.fail_type} {r.invariant_diff}")
        failed += 1

    # Test 2: missing schema_id
    import copy
    bad = copy.deepcopy(valid_trace)
    del bad["schema_id"]
    r = validate_trace(bad)
    if not r.ok and r.fail_type == SCHEMA_INVALID:
        print("  [PASS] missing schema_id -> SCHEMA_INVALID")
        passed += 1
    else:
        print(f"  [FAIL] expected SCHEMA_INVALID, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 3: nondeterministic order
    bad = copy.deepcopy(valid_trace)
    bad["events"][0]["event_index"] = 1
    bad["events"][1]["event_index"] = 0
    r = validate_trace(bad)
    if not r.ok and r.fail_type == NONDETERMINISTIC_EVENT_ORDER:
        print("  [PASS] swapped indices -> NONDETERMINISTIC_EVENT_ORDER")
        passed += 1
    else:
        print(f"  [FAIL] expected NONDETERMINISTIC_EVENT_ORDER, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 4: missing invariant_diff
    bad = copy.deepcopy(valid_trace)
    del bad["events"][1]["invariant_diff"]
    # Must also pass schema check first — but the schema check will catch it!
    r = validate_trace(bad)
    if not r.ok and r.fail_type in (SCHEMA_INVALID, MISSING_INVARIANT_DIFF):
        print(f"  [PASS] missing invariant_diff -> {r.fail_type}")
        passed += 1
    else:
        print(f"  [FAIL] expected MISSING_INVARIANT_DIFF, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 5: hash mismatch
    bad = copy.deepcopy(valid_trace)
    bad["trace_hashes"]["events_sha256"] = "f" * 64
    r = validate_trace(bad)
    if not r.ok and r.fail_type == HASH_MISMATCH:
        print("  [PASS] wrong events_sha256 -> HASH_MISMATCH")
        passed += 1
    else:
        print(f"  [FAIL] expected HASH_MISMATCH, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 6: broken chain
    bad = copy.deepcopy(valid_trace)
    bad["trace_hashes"]["hash_chain"][1]["prev_sha256"] = "e" * 64
    r = validate_trace(bad)
    if not r.ok and r.fail_type == HASH_CHAIN_BROKEN:
        print("  [PASS] broken chain link -> HASH_CHAIN_BROKEN")
        passed += 1
    else:
        print(f"  [FAIL] expected HASH_CHAIN_BROKEN, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 7: summary mismatch
    bad = copy.deepcopy(valid_trace)
    bad["summary"]["event_count"] = 99
    r = validate_trace(bad)
    if not r.ok and r.fail_type == SUMMARY_MISMATCH:
        print("  [PASS] bad event_count -> SUMMARY_MISMATCH")
        passed += 1
    else:
        print(f"  [FAIL] expected SUMMARY_MISMATCH, got ok={r.ok} ft={r.fail_type}")
        failed += 1

    # Test 8: redaction policy violation
    bad = copy.deepcopy(valid_trace)
    bad["events"][0]["payload"]["raw_code"] = "x" * 300  # too long for hash_only
    # Must recompute hashes after modifying events
    bad["trace_hashes"]["events_sha256"] = compute_events_sha256(bad["events"])
    bad["trace_hashes"]["hash_chain"] = build_hash_chain(bad["events"])
    bad["summary"] = build_summary(bad["events"])
    r = validate_trace(bad)
    if not r.ok and r.fail_type == REDACTION_POLICY_VIOLATION:
        print("  [PASS] long raw string in hash_only -> REDACTION_POLICY_VIOLATION")
        passed += 1
    else:
        print(f"  [FAIL] expected REDACTION_POLICY_VIOLATION, got ok={r.ok} ft={r.fail_type}")
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
        print(f"Usage: {sys.argv[0]} <trace.json> [--ci]", file=sys.stderr)
        print(f"       {sys.argv[0]} --self-test", file=sys.stderr)
        sys.exit(2)

    trace_path = Path(args[0])
    ci_mode = "--ci" in args

    if not trace_path.exists():
        print(f"ERROR: {trace_path} not found", file=sys.stderr)
        sys.exit(2)

    with trace_path.open("r", encoding="utf-8") as f:
        try:
            trace = json.load(f)
        except json.JSONDecodeError as e:
            if ci_mode:
                print(f"[FAIL] {trace_path.name}: SCHEMA_INVALID (JSON decode: {e})")
            else:
                print(f"FAIL: {trace_path.name}")
                print(json.dumps({
                    "fail_type": SCHEMA_INVALID,
                    "invariant_diff": {"reason": f"JSON decode error: {e}"},
                }, indent=2))
            sys.exit(1)

    result = validate_trace(trace)

    if ci_mode:
        if result.ok:
            print(f"[PASS] {trace_path.name}: valid ({result.trace_id})")
        else:
            diff_str = json.dumps(result.invariant_diff, sort_keys=True)
            print(f"[FAIL] {trace_path.name}: {result.fail_type} {diff_str}")
        sys.exit(0 if result.ok else 1)
    else:
        if result.ok:
            print(f"PASS: {trace_path.name} ({result.trace_id})")
        else:
            print(f"FAIL: {trace_path.name}")
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
