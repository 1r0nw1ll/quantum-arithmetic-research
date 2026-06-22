#!/usr/bin/env python3
"""Validate typed dependency-installation and replay receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
SCHEMA_ID = "QA_MATH_COMPILER_DEPENDENCY_REPLAY_RECEIPT.v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def validate_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if receipt.get("schema_id") != SCHEMA_ID:
        errors.append("SCHEMA_ID_MISMATCH")
    if receipt.get("checkout_status") != "SUCCESS":
        errors.append("DEPENDENCY_CHECKOUT_FAILED")
    expected_lock = receipt.get("expected_lock_sha256")
    observed_lock = receipt.get("observed_lock_sha256")
    if (
        not isinstance(expected_lock, str)
        or len(expected_lock) != 64
        or not isinstance(observed_lock, str)
        or len(observed_lock) != 64
    ):
        errors.append("DEPENDENCY_LOCK_INVALID")
    elif expected_lock != observed_lock:
        errors.append("DEPENDENCY_LOCK_DRIFT")
    if receipt.get("cache_required") is True and receipt.get("cache_status") != "AVAILABLE":
        errors.append("DEPENDENCY_CACHE_MISSING")
    trace_hash = receipt.get("trace_execution_sha256")
    replay_hash = receipt.get("replay_execution_sha256")
    if (
        not isinstance(trace_hash, str)
        or len(trace_hash) != 64
        or not isinstance(replay_hash, str)
        or len(replay_hash) != 64
    ):
        errors.append("DEPENDENCY_REPLAY_HASH_INVALID")
    elif trace_hash != replay_hash:
        errors.append("DEPENDENCY_REPLAY_MISMATCH")
    return {"ok": not errors, "errors": errors}


LIVE_RECEIPTS = [
    ROOT / "coq_dependency_receipt.json",
    ROOT / "isabelle_dependency_receipt.json",
]


def self_test() -> dict[str, Any]:
    cases = [
        # Lean/Mathlib cases
        ("dependency_replay_valid.json", None),
        ("dependency_fail_missing_cache.json", "DEPENDENCY_CACHE_MISSING"),
        ("dependency_fail_lock_drift.json", "DEPENDENCY_LOCK_DRIFT"),
        ("dependency_fail_checkout.json", "DEPENDENCY_CHECKOUT_FAILED"),
        ("dependency_fail_replay_mismatch.json", "DEPENDENCY_REPLAY_MISMATCH"),
        # Coq/Rocq cases
        ("coq_dependency_replay_valid.json", None),
        ("coq_dependency_fail_lock_drift.json", "DEPENDENCY_LOCK_DRIFT"),
        # Isabelle cases
        ("isabelle_dependency_replay_valid.json", None),
        ("isabelle_dependency_fail_lock_drift.json", "DEPENDENCY_LOCK_DRIFT"),
    ]
    checks = []
    for filename, expected in cases:
        result = validate_receipt(load_json(FIXTURES / filename))
        ok = result["ok"] if expected is None else expected in result["errors"]
        checks.append({"name": filename, "expected": expected or "PASS", "ok": ok})
    # Also validate the live receipts against the schema
    for path in LIVE_RECEIPTS:
        result = validate_receipt(load_json(path))
        checks.append({
            "name": path.name,
            "expected": "PASS",
            "ok": result["ok"],
            "errors": result.get("errors", []),
        })
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.receipt is not None:
        result = validate_receipt(load_json(args.receipt))
    else:
        parser.error("receipt or --self-test required")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
