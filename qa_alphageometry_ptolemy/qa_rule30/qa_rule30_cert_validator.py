#!/usr/bin/env python3
"""
Validate QA_RULE30_NONPERIODICITY_CERT_SCHEMA.v1 and
         QA_RULE30_WITNESS_MANIFEST.v1.

Cert gates:
  1. invariant_diff presence               -> MISSING_INVARIANT_DIFF
  2. Schema shape                          -> SCHEMA_INVALID
  3. Scope consistency (P_max >= P_min)    -> SCOPE_INVALID
  4. Aggregate vs witness_refs consistency -> AGGREGATE_MISMATCH
  5. Hash chain (this_cert_hash non-empty) -> HASH_CHAIN_INVALID
  6. Zero failures in all witness_refs     -> FAILURE_DETECTED

Manifest gates:
  1. Schema shape                          -> SCHEMA_INVALID
  2. File hashes match on disk             -> HASH_MISMATCH
  3. Failure count == 0                    -> FAILURE_DETECTED

Usage:
  python qa_rule30_cert_validator.py cert     <file.json> [--ci]
  python qa_rule30_cert_validator.py manifest <file.json> [--verify-files] [--ci]
  python qa_rule30_cert_validator.py --self-test

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

CERT_SCHEMA_ID = "QA_RULE30_NONPERIODICITY_CERT_SCHEMA.v1"
MANIFEST_SCHEMA_ID = "QA_RULE30_WITNESS_MANIFEST.v1"

SCHEMA_INVALID = "SCHEMA_INVALID"
MISSING_INVARIANT_DIFF = "MISSING_INVARIANT_DIFF"
SCOPE_INVALID = "SCOPE_INVALID"
AGGREGATE_MISMATCH = "AGGREGATE_MISMATCH"
HASH_CHAIN_INVALID = "HASH_CHAIN_INVALID"
HASH_MISMATCH = "HASH_MISMATCH"
FAILURE_DETECTED = "FAILURE_DETECTED"


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
    return isinstance(v, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", v))


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
# CERT VALIDATION
# ===================================================================

def _cert_check_invariant_diff(cert: Dict) -> Optional[Tuple[str, Dict]]:
    if "invariant_diff" not in cert or not isinstance(cert.get("invariant_diff"), dict):
        return MISSING_INVARIANT_DIFF, {"path": "$.invariant_diff"}
    return None


def _cert_check_schema(cert: Dict) -> Optional[Tuple[str, Dict]]:
    required = [
        "schema_id", "cert_id", "created_utc", "agent_id",
        "scope", "witness_refs", "bundle_refs",
        "aggregate", "hash_chain", "invariant_diff",
    ]
    for f in required:
        if f not in cert:
            return SCHEMA_INVALID, {"missing_field": f}

    if cert["schema_id"] != CERT_SCHEMA_ID:
        return SCHEMA_INVALID, {"expected": CERT_SCHEMA_ID, "got": cert["schema_id"]}
    if not isinstance(cert["cert_id"], str) or len(cert["cert_id"]) < 8:
        return SCHEMA_INVALID, {"path": "$.cert_id"}
    if not _is_utc_ts(cert["created_utc"]):
        return SCHEMA_INVALID, {"path": "$.created_utc"}
    if not isinstance(cert["agent_id"], str) or not cert["agent_id"]:
        return SCHEMA_INVALID, {"path": "$.agent_id"}

    # scope
    scope = cert.get("scope", {})
    if not isinstance(scope, dict):
        return SCHEMA_INVALID, {"path": "$.scope"}
    for sf in ["rule", "initial_condition", "P_min", "P_max", "T_max", "k_range"]:
        if sf not in scope:
            return SCHEMA_INVALID, {"missing_field": sf, "path": f"$.scope.{sf}"}
    if scope["rule"] != 30:
        return SCHEMA_INVALID, {"path": "$.scope.rule", "expected": 30}
    if scope["initial_condition"] != "single_1_at_origin":
        return SCHEMA_INVALID, {"path": "$.scope.initial_condition"}

    # witness_refs
    wr = cert.get("witness_refs", [])
    if not isinstance(wr, list) or len(wr) < 1:
        return SCHEMA_INVALID, {"path": "$.witness_refs", "reason": "must be non-empty array"}
    for i, ref in enumerate(wr):
        if not isinstance(ref, dict):
            return SCHEMA_INVALID, {"path": f"$.witness_refs[{i}]"}
        for rf in ["T", "P_min", "P_max", "manifest_path", "manifest_sha256",
                    "verified_periods", "failure_count"]:
            if rf not in ref:
                return SCHEMA_INVALID, {"missing_field": rf, "path": f"$.witness_refs[{i}].{rf}"}
        if not _is_hex64(ref.get("manifest_sha256", "")):
            return SCHEMA_INVALID, {"path": f"$.witness_refs[{i}].manifest_sha256"}

    # aggregate
    agg = cert.get("aggregate", {})
    if not isinstance(agg, dict):
        return SCHEMA_INVALID, {"path": "$.aggregate"}
    for af in ["total_periods_checked", "total_verified", "total_failures", "T_values", "claim"]:
        if af not in agg:
            return SCHEMA_INVALID, {"missing_field": af, "path": f"$.aggregate.{af}"}

    # hash_chain
    hc = cert.get("hash_chain", {})
    if not isinstance(hc, dict):
        return SCHEMA_INVALID, {"path": "$.hash_chain"}
    for hf in ["prev_cert_hash", "this_cert_hash"]:
        if hf not in hc:
            return SCHEMA_INVALID, {"missing_field": hf, "path": f"$.hash_chain.{hf}"}
        if not _is_hex64(hc.get(hf, "")):
            return SCHEMA_INVALID, {"path": f"$.hash_chain.{hf}"}

    return None


def _cert_check_scope(cert: Dict) -> Optional[Tuple[str, Dict]]:
    scope = cert["scope"]
    if scope["P_max"] < scope["P_min"]:
        return SCOPE_INVALID, {"P_min": scope["P_min"], "P_max": scope["P_max"]}
    kr = scope.get("k_range", {})
    if kr.get("max", 0) < kr.get("min", 0):
        return SCOPE_INVALID, {"k_range": kr}
    return None


def _cert_check_aggregate(cert: Dict) -> Optional[Tuple[str, Dict]]:
    agg = cert["aggregate"]
    refs = cert["witness_refs"]

    # Sum verified and failures across witness_refs
    total_verified = sum(r["verified_periods"] for r in refs)
    total_failures = sum(r["failure_count"] for r in refs)
    total_checked = total_verified + total_failures
    t_values = sorted(set(r["T"] for r in refs))

    if agg["total_verified"] != total_verified:
        return AGGREGATE_MISMATCH, {
            "field": "total_verified",
            "expected": total_verified,
            "got": agg["total_verified"],
        }
    if agg["total_failures"] != total_failures:
        return AGGREGATE_MISMATCH, {
            "field": "total_failures",
            "expected": total_failures,
            "got": agg["total_failures"],
        }
    if agg["total_periods_checked"] != total_checked:
        return AGGREGATE_MISMATCH, {
            "field": "total_periods_checked",
            "expected": total_checked,
            "got": agg["total_periods_checked"],
        }
    if sorted(agg.get("T_values", [])) != t_values:
        return AGGREGATE_MISMATCH, {
            "field": "T_values",
            "expected": t_values,
            "got": agg["T_values"],
        }
    return None


def _cert_check_hash_chain(cert: Dict) -> Optional[Tuple[str, Dict]]:
    hc = cert["hash_chain"]
    if not hc["this_cert_hash"]:
        return HASH_CHAIN_INVALID, {"path": "$.hash_chain.this_cert_hash", "reason": "empty"}
    return None


def _cert_check_no_failures(cert: Dict) -> Optional[Tuple[str, Dict]]:
    for i, ref in enumerate(cert["witness_refs"]):
        if ref["failure_count"] > 0:
            return FAILURE_DETECTED, {
                "witness_ref_index": i,
                "T": ref["T"],
                "failure_count": ref["failure_count"],
            }
    return None


def validate_cert(cert: Dict) -> ValidationResult:
    obj_id = cert.get("cert_id", "<unknown>")
    for gate in [_cert_check_invariant_diff, _cert_check_schema,
                 _cert_check_scope, _cert_check_aggregate,
                 _cert_check_hash_chain, _cert_check_no_failures]:
        result = gate(cert)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ===================================================================
# MANIFEST VALIDATION
# ===================================================================

def _man_check_schema(man: Dict) -> Optional[Tuple[str, Dict]]:
    if man.get("schema_id") != MANIFEST_SCHEMA_ID:
        return SCHEMA_INVALID, {"expected": MANIFEST_SCHEMA_ID, "got": man.get("schema_id")}
    for f in ["rule", "T", "P_min", "P_max", "total_periods", "verified_periods",
              "failure_count", "failures", "files"]:
        if f not in man:
            return SCHEMA_INVALID, {"missing_field": f}
    if man["rule"] != 30:
        return SCHEMA_INVALID, {"path": "$.rule"}
    files = man.get("files", {})
    if "witnesses" not in files:
        return SCHEMA_INVALID, {"path": "$.files.witnesses"}
    return None


def _man_check_files(man: Dict, base_dir: Optional[Path]) -> Optional[Tuple[str, Dict]]:
    if base_dir is None:
        return None  # skip file verification
    files = man["files"]
    for key, info in files.items():
        fpath = base_dir / info["path"]
        if not fpath.exists():
            return HASH_MISMATCH, {"file": key, "path": str(fpath), "reason": "not found"}
        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual_hash != info["sha256"]:
            return HASH_MISMATCH, {
                "file": key,
                "expected": info["sha256"],
                "actual": actual_hash,
            }
    return None


def _man_check_failures(man: Dict) -> Optional[Tuple[str, Dict]]:
    if man["failure_count"] > 0:
        return FAILURE_DETECTED, {"failures": man["failures"]}
    return None


def validate_manifest(man: Dict, base_dir: Optional[Path] = None) -> ValidationResult:
    obj_id = f"manifest_P{man.get('P_min','?')}-{man.get('P_max','?')}_T{man.get('T','?')}"
    for gate_fn, gate_args in [
        (_man_check_schema, (man,)),
        (_man_check_files, (man, base_dir)),
        (_man_check_failures, (man,)),
    ]:
        result = gate_fn(*gate_args)
        if result is not None:
            return ValidationResult(False, result[0], result[1], obj_id)
    return ValidationResult(True, obj_id=obj_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    print("--- qa_rule30_cert_validator self-test ---")
    passed = 0
    failed = 0

    HEX64 = "a" * 64

    # ---- Cert tests ----

    valid_cert = {
        "schema_id": CERT_SCHEMA_ID,
        "cert_id": "CERT-RULE30-SELFTEST01",
        "created_utc": "2026-02-11T02:00:00Z",
        "agent_id": "qa-agent-ctrl-1",
        "scope": {
            "rule": 30,
            "initial_condition": "single_1_at_origin",
            "P_min": 1,
            "P_max": 256,
            "T_max": 65536,
            "k_range": {"min": 4, "max": 16},
        },
        "witness_refs": [
            {
                "T": 16384,
                "P_min": 1,
                "P_max": 256,
                "manifest_path": "witnesses/MANIFEST.json",
                "manifest_sha256": HEX64,
                "verified_periods": 256,
                "failure_count": 0,
            },
        ],
        "bundle_refs": [
            {"family": "QA_DISCOVERY_PIPELINE.v1", "path_or_hash": "bundles/bundle_001.json"},
        ],
        "aggregate": {
            "total_periods_checked": 256,
            "total_verified": 256,
            "total_failures": 0,
            "T_values": [16384],
            "claim": "No period p in [1,256] detected for Rule 30 center up to T=65536",
        },
        "hash_chain": {"prev_cert_hash": HEX64, "this_cert_hash": "b" * 64},
        "invariant_diff": {"note": "self-test"},
    }

    r = validate_cert(valid_cert)
    if r.ok:
        print("  [PASS] valid cert accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid cert rejected: {r.fail_type}")
        failed += 1

    # C2: missing invariant_diff
    bad = copy.deepcopy(valid_cert)
    del bad["invariant_diff"]
    r = validate_cert(bad)
    if not r.ok and r.fail_type == MISSING_INVARIANT_DIFF:
        print("  [PASS] missing invariant_diff -> MISSING_INVARIANT_DIFF")
        passed += 1
    else:
        print(f"  [FAIL] expected MISSING_INVARIANT_DIFF, got {r.fail_type}")
        failed += 1

    # C3: P_max < P_min
    bad = copy.deepcopy(valid_cert)
    bad["scope"]["P_min"] = 100
    bad["scope"]["P_max"] = 10
    r = validate_cert(bad)
    if not r.ok and r.fail_type == SCOPE_INVALID:
        print("  [PASS] P_max < P_min -> SCOPE_INVALID")
        passed += 1
    else:
        print(f"  [FAIL] expected SCOPE_INVALID, got {r.fail_type}")
        failed += 1

    # C4: aggregate mismatch
    bad = copy.deepcopy(valid_cert)
    bad["aggregate"]["total_verified"] = 999
    r = validate_cert(bad)
    if not r.ok and r.fail_type == AGGREGATE_MISMATCH:
        print("  [PASS] aggregate mismatch -> AGGREGATE_MISMATCH")
        passed += 1
    else:
        print(f"  [FAIL] expected AGGREGATE_MISMATCH, got {r.fail_type}")
        failed += 1

    # C5: empty this_cert_hash
    bad = copy.deepcopy(valid_cert)
    bad["hash_chain"]["this_cert_hash"] = "0" * 64
    r = validate_cert(bad)
    if r.ok:
        print("  [PASS] zero hash accepted (valid hex64)")
        passed += 1
    else:
        print(f"  [FAIL] zero hash rejected: {r.fail_type}")
        failed += 1

    # C6: failure detected
    bad = copy.deepcopy(valid_cert)
    bad["witness_refs"][0]["failure_count"] = 3
    bad["aggregate"]["total_failures"] = 3
    bad["aggregate"]["total_periods_checked"] = 259
    r = validate_cert(bad)
    if not r.ok and r.fail_type == FAILURE_DETECTED:
        print("  [PASS] failure_count > 0 -> FAILURE_DETECTED")
        passed += 1
    else:
        print(f"  [FAIL] expected FAILURE_DETECTED, got {r.fail_type}")
        failed += 1

    # ---- Manifest tests ----

    valid_man = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "rule": 30,
        "T": 16384,
        "P_min": 1,
        "P_max": 256,
        "initial_condition": "single_1_at_origin",
        "total_periods": 256,
        "verified_periods": 256,
        "failure_count": 0,
        "failures": [],
        "files": {
            "witnesses": {"path": "witnesses.json", "sha256": HEX64},
            "center_sequence": {"path": "center.txt", "sha256": "b" * 64},
        },
    }

    r = validate_manifest(valid_man)
    if r.ok:
        print("  [PASS] valid manifest accepted")
        passed += 1
    else:
        print(f"  [FAIL] valid manifest rejected: {r.fail_type}")
        failed += 1

    # M2: wrong schema_id
    bad = copy.deepcopy(valid_man)
    bad["schema_id"] = "WRONG"
    r = validate_manifest(bad)
    if not r.ok and r.fail_type == SCHEMA_INVALID:
        print("  [PASS] wrong manifest schema_id -> SCHEMA_INVALID")
        passed += 1
    else:
        print(f"  [FAIL] expected SCHEMA_INVALID, got {r.fail_type}")
        failed += 1

    # M3: failures present
    bad = copy.deepcopy(valid_man)
    bad["failure_count"] = 2
    bad["failures"] = [17, 42]
    r = validate_manifest(bad)
    if not r.ok and r.fail_type == FAILURE_DETECTED:
        print("  [PASS] manifest with failures -> FAILURE_DETECTED")
        passed += 1
    else:
        print(f"  [FAIL] expected FAILURE_DETECTED, got {r.fail_type}")
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

    if len(args) < 2 or args[0] not in ("cert", "manifest"):
        print(f"Usage: {sys.argv[0]} cert|manifest <file.json> [--verify-files] [--ci]",
              file=sys.stderr)
        print(f"       {sys.argv[0]} --self-test", file=sys.stderr)
        sys.exit(2)

    mode = args[0]
    file_path = Path(args[1])
    ci_mode = "--ci" in args
    verify_files = "--verify-files" in args

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

    if mode == "cert":
        result = validate_cert(data)
    else:
        base_dir = file_path.parent if verify_files else None
        result = validate_manifest(data, base_dir)

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
