#!/usr/bin/env python3
"""
QA Episode Regime Transitions Cert validator — Family [81]
Schema: QA_EPISODE_REGIME_CERT.v1.0

5 gates:
  Gate 1: JSON schema validity + schema_version const check
  Gate 2: label validity (every primary_label in ALLOWED_PRIMARY_LABELS)
  Gate 3: regime_sequence, counts, transition_matrix recompute
  Gate 4: drift_declaration recompute
  Gate 5: max_run_length + max_run_label recompute

Output-only (non-cert): transition_probabilities, first_recovery_index,
last_escalation_index, cert_sha256.

Usage:
  python3 validator.py path/to/cert.json
  python3 validator.py --self-test [--json]
"""

import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Label classification constants
# ---------------------------------------------------------------------------

ALLOWED_PRIMARY_LABELS = {
    "RETURN_TO_REF",
    "MONOTONE_ESCALATION",
    "MONOTONE_RECOVERY",
    "OSCILLATORY",
    "STASIS",
}

ESCALATION_CLASS = {"MONOTONE_ESCALATION"}
RECOVERY_CLASS   = {"MONOTONE_RECOVERY", "RETURN_TO_REF"}
NEUTRAL_CLASS    = {"OSCILLATORY", "STASIS"}


def _regime_for_label(label: str) -> str:
    if label in ESCALATION_CLASS:
        return "ESCALATION"
    if label in RECOVERY_CLASS:
        return "RECOVERY"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Recompute helpers
# ---------------------------------------------------------------------------

def _recompute_regime_and_counts(labels: List[str]) -> Tuple[List[str], int, int, int]:
    """Return (regime_sequence, escalation_count, recovery_count, neutral_count)."""
    regime_seq: List[str] = []
    esc = rec = neu = 0
    for lbl in labels:
        r = _regime_for_label(lbl)
        regime_seq.append(r)
        if r == "ESCALATION":
            esc += 1
        elif r == "RECOVERY":
            rec += 1
        else:
            neu += 1
    return regime_seq, esc, rec, neu


def _recompute_transition_matrix(labels: List[str]) -> List[Dict[str, Any]]:
    """Sparse transition matrix from consecutive label pairs, sorted lex."""
    counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(labels) - 1):
        key = (labels[i], labels[i + 1])
        counts[key] = counts.get(key, 0) + 1
    result = []
    for (f, t) in sorted(counts.keys()):
        result.append({"from_label": f, "to_label": t, "count": counts[(f, t)]})
    return result


def _recompute_drift(esc: int, rec: int, neu: int) -> str:
    """
    Drift declaration rules (integer-only, exhaustive):
      ESCALATING : esc > rec and esc > neu
      RECOVERING : rec > esc and rec > neu
      STABLE     : neu > esc and neu > rec
      MIXED      : otherwise (ties or no single dominant class)
    """
    if esc > rec and esc > neu:
        return "ESCALATING"
    if rec > esc and rec > neu:
        return "RECOVERING"
    if neu > esc and neu > rec:
        return "STABLE"
    return "MIXED"


def _recompute_max_run(labels: List[str]) -> Tuple[int, str]:
    """Max consecutive run of the same label. Tie-break: first occurrence wins (strict >)."""
    if not labels:
        return 0, ""
    best_len = 1
    best_lbl = labels[0]
    cur_len = 1
    cur_lbl = labels[0]
    for lbl in labels[1:]:
        if lbl == cur_lbl:
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_lbl = cur_lbl
        else:
            cur_lbl = lbl
            cur_len = 1
    return best_len, best_lbl


# ---------------------------------------------------------------------------
# Output-only helpers
# ---------------------------------------------------------------------------

def _cert_sha256(cert: Dict[str, Any]) -> str:
    canonical = json.dumps(cert, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _transition_probabilities(labels: List[str]) -> List[Dict[str, Any]]:
    """
    Exact rational transition probabilities: p(from→to) = count / (episode_count - 1).
    Returns list sorted lex by (from_label, to_label).
    """
    n = len(labels)
    if n <= 1:
        return []
    denom = n - 1
    counts: Dict[Tuple[str, str], int] = {}
    for i in range(n - 1):
        key = (labels[i], labels[i + 1])
        counts[key] = counts.get(key, 0) + 1
    result = []
    for (f, t) in sorted(counts.keys()):
        result.append({
            "from_label": f,
            "to_label": t,
            "count": counts[(f, t)],
            "prob_num": counts[(f, t)],
            "prob_den": denom,
        })
    return result


def _first_recovery_index(labels: List[str]) -> Optional[int]:
    for i, lbl in enumerate(labels):
        if lbl in RECOVERY_CLASS:
            return i
    return None


def _last_escalation_index(labels: List[str]) -> Optional[int]:
    result = None
    for i, lbl in enumerate(labels):
        if lbl in ESCALATION_CLASS:
            result = i
    return result


# ---------------------------------------------------------------------------
# JSON schema validation (subset, no external deps)
# ---------------------------------------------------------------------------

def _load_schema() -> Dict[str, Any]:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    with open(schema_path) as f:
        return json.load(f)


def _validate_schema(cert: Any, schema: Dict[str, Any]) -> Optional[str]:
    """Minimal JSON-schema Draft-07 validation sufficient for this schema."""
    if not isinstance(cert, dict):
        return "cert is not an object"

    # schema_version const
    sv = cert.get("schema_version")
    if sv != "QA_EPISODE_REGIME_CERT.v1.0":
        return f"schema_version must be 'QA_EPISODE_REGIME_CERT.v1.0', got {sv!r}"

    # required top-level keys
    for key in ("cert_id", "episodes", "expected_summary"):
        if key not in cert:
            return f"missing required field: {key!r}"

    if not isinstance(cert["cert_id"], str) or not cert["cert_id"]:
        return "cert_id must be a non-empty string"

    episodes = cert["episodes"]
    if not isinstance(episodes, list) or len(episodes) == 0:
        return "episodes must be a non-empty array"

    for i, ep in enumerate(episodes):
        if not isinstance(ep, dict):
            return f"episodes[{i}] is not an object"
        for field in ("episode_id", "primary_label"):
            if field not in ep:
                return f"episodes[{i}] missing required field {field!r}"
            if not isinstance(ep[field], str) or not ep[field]:
                return f"episodes[{i}].{field} must be a non-empty string"
        extra = set(ep.keys()) - {"episode_id", "primary_label"}
        if extra:
            return f"episodes[{i}] has unexpected fields: {sorted(extra)}"

    es = cert["expected_summary"]
    if not isinstance(es, dict):
        return "expected_summary must be an object"

    for ifield in ("episode_count", "escalation_count", "recovery_count", "neutral_count",
                   "max_run_length"):
        v = es.get(ifield)
        if not isinstance(v, int):
            return f"expected_summary.{ifield} must be an integer"
        if ifield != "max_run_length" and v < 0:
            return f"expected_summary.{ifield} must be >= 0"
        if ifield == "max_run_length" and v < 1:
            return f"expected_summary.max_run_length must be >= 1"

    if es.get("drift_declaration") not in ("ESCALATING", "RECOVERING", "STABLE", "MIXED"):
        return f"expected_summary.drift_declaration invalid: {es.get('drift_declaration')!r}"

    rs = es.get("regime_sequence")
    if not isinstance(rs, list):
        return "expected_summary.regime_sequence must be an array"
    for i, r in enumerate(rs):
        if r not in ("ESCALATION", "RECOVERY", "NEUTRAL"):
            return f"expected_summary.regime_sequence[{i}] invalid: {r!r}"

    tm = es.get("transition_matrix")
    if not isinstance(tm, list):
        return "expected_summary.transition_matrix must be an array"
    for i, entry in enumerate(tm):
        if not isinstance(entry, dict):
            return f"transition_matrix[{i}] is not an object"
        for field in ("from_label", "to_label", "count"):
            if field not in entry:
                return f"transition_matrix[{i}] missing {field!r}"
        if not isinstance(entry["count"], int) or entry["count"] < 1:
            return f"transition_matrix[{i}].count must be integer >= 1"

    if "max_run_label" not in es:
        return "expected_summary.max_run_label is required"
    if not isinstance(es["max_run_label"], str) or not es["max_run_label"]:
        return "expected_summary.max_run_label must be a non-empty string"

    extra = set(es.keys()) - {
        "episode_count", "escalation_count", "recovery_count", "neutral_count",
        "drift_declaration", "regime_sequence", "transition_matrix",
        "max_run_length", "max_run_label",
    }
    if extra:
        return f"expected_summary has unexpected fields: {sorted(extra)}"

    return None


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate(cert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all 5 gates. Returns result dict:
      {"ok": bool, "gates_passed": int, "error": str|None, ...output-only fields...}
    """
    schema = _load_schema()

    # ----- Gate 1: schema validity -----
    schema_err = _validate_schema(cert, schema)
    if schema_err:
        return {"ok": False, "gates_passed": 0, "error": f"SCHEMA_INVALID: {schema_err}"}

    episodes = cert["episodes"]
    labels: List[str] = [ep["primary_label"] for ep in episodes]
    es = cert["expected_summary"]

    # ----- Gate 2: label validity -----
    for ep in episodes:
        lbl = ep["primary_label"]
        if lbl not in ALLOWED_PRIMARY_LABELS:
            return {
                "ok": False, "gates_passed": 1,
                "error": f"LABEL_INVALID: episode {ep['episode_id']!r} has unknown primary_label {lbl!r}",
            }

    # ----- Gate 3: regime_sequence, counts, transition_matrix -----
    exp_count = es["episode_count"]
    if exp_count != len(labels):
        return {
            "ok": False, "gates_passed": 2,
            "error": f"COUNT_MISMATCH: expected_summary.episode_count={exp_count} but episodes array has {len(labels)}",
        }

    regime_seq, esc, rec, neu = _recompute_regime_and_counts(labels)

    if regime_seq != es["regime_sequence"]:
        return {
            "ok": False, "gates_passed": 2,
            "error": f"REGIME_SEQUENCE_MISMATCH: computed {regime_seq} != declared {es['regime_sequence']}",
        }
    if esc != es["escalation_count"]:
        return {
            "ok": False, "gates_passed": 2,
            "error": f"COUNT_MISMATCH: escalation_count computed={esc} declared={es['escalation_count']}",
        }
    if rec != es["recovery_count"]:
        return {
            "ok": False, "gates_passed": 2,
            "error": f"COUNT_MISMATCH: recovery_count computed={rec} declared={es['recovery_count']}",
        }
    if neu != es["neutral_count"]:
        return {
            "ok": False, "gates_passed": 2,
            "error": f"COUNT_MISMATCH: neutral_count computed={neu} declared={es['neutral_count']}",
        }

    tm_computed = _recompute_transition_matrix(labels)
    tm_declared = sorted(
        [{"from_label": e["from_label"], "to_label": e["to_label"], "count": e["count"]}
         for e in es["transition_matrix"]],
        key=lambda x: (x["from_label"], x["to_label"]),
    )
    if tm_computed != tm_declared:
        return {
            "ok": False, "gates_passed": 2,
            "error": f"TRANSITION_MISMATCH: computed={tm_computed} declared={tm_declared}",
        }

    # ----- Gate 4: drift_declaration -----
    drift_computed = _recompute_drift(esc, rec, neu)
    if drift_computed != es["drift_declaration"]:
        return {
            "ok": False, "gates_passed": 3,
            "error": f"DRIFT_MISMATCH: computed={drift_computed!r} declared={es['drift_declaration']!r}",
        }

    # ----- Gate 5: max_run_length + max_run_label -----
    run_len, run_lbl = _recompute_max_run(labels)
    if run_len != es["max_run_length"]:
        return {
            "ok": False, "gates_passed": 4,
            "error": f"MAX_RUN_MISMATCH: computed length={run_len} declared={es['max_run_length']}",
        }
    if run_lbl != es["max_run_label"]:
        return {
            "ok": False, "gates_passed": 4,
            "error": f"MAX_RUN_MISMATCH: computed label={run_lbl!r} declared={es['max_run_label']!r}",
        }

    # ----- All gates passed — build output-only envelope -----
    return {
        "ok": True,
        "gates_passed": 5,
        "error": None,
        "transition_probabilities": _transition_probabilities(labels),
        "first_recovery_index": _first_recovery_index(labels),
        "last_escalation_index": _last_escalation_index(labels),
        "cert_sha256": _cert_sha256(cert),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_cert(path: str) -> int:
    with open(path) as f:
        cert = json.load(f)
    result = validate(cert)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# Self-test fixtures
# ---------------------------------------------------------------------------

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

_FIXTURES = [
    ("PASS_RECOVERING.json",   True),
    ("PASS_ESCALATING.json",   True),
    ("PASS_MIXED.json",        True),
    ("FAIL_LABEL.json",        False),
    ("FAIL_TRANSITION.json",   False),
    ("FAIL_DRIFT.json",        False),
]


def _run_self_test(json_output: bool = False) -> int:
    results = []
    all_ok = True

    for fname, expected_ok in _FIXTURES:
        path = os.path.join(_FIXTURES_DIR, fname)
        try:
            with open(path) as f:
                cert = json.load(f)
            result = validate(cert)
            passed = result["ok"] == expected_ok
            results.append({
                "fixture": fname,
                "expected_ok": expected_ok,
                "ok": result["ok"],
                "gates_passed": result["gates_passed"],
                "error": result.get("error"),
                "match": passed,
            })
            if not passed:
                all_ok = False
        except Exception as exc:
            results.append({
                "fixture": fname,
                "expected_ok": expected_ok,
                "ok": None,
                "error": str(exc),
                "match": False,
            })
            all_ok = False

    if json_output:
        print(json.dumps({"ok": all_ok, "fixtures": results}, indent=2))
    else:
        for r in results:
            status = "PASS" if r["match"] else "FAIL"
            print(f"  [{status}] {r['fixture']} (expected_ok={r['expected_ok']}, got ok={r['ok']}, gates={r.get('gates_passed','?')})")
            if not r["match"]:
                print(f"        error: {r.get('error')}")
        total = len(results)
        passed = sum(1 for r in results if r["match"])
        print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'} ({passed}/{total} fixtures)")

    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--self-test" in args:
        json_flag = "--json" in args
        sys.exit(_run_self_test(json_output=json_flag))
    elif len(args) == 1:
        sys.exit(_run_cert(args[0]))
    else:
        print("Usage: validator.py <cert.json> | --self-test [--json]", file=sys.stderr)
        sys.exit(2)
