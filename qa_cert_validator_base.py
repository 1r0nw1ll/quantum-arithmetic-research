#!/usr/bin/env python3
"""
qa_cert_validator_base.py

Reusable infrastructure for QA certificate validators.

Extracts the common 5-gate adapter pattern proven by family [45]
(ARTexplorer Scene Adapter). Validators can opt-in to import from here
instead of duplicating GateStatus, GateResult, hashing, CLI, and self-test
boilerplate.

Usage:
    from qa_cert_validator_base import (
        GateStatus, GateResult, canonical_json_compact, sha256_hex,
        load_json, validate_schema, report_ok, print_human, print_json,
        run_self_test, build_cli,
    )
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


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


def canonical_json_compact(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, no ASCII escape."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(obj: Dict[str, Any], schema_path: str) -> None:
    """Validate obj against a JSON Schema (Draft-07). Raises on failure."""
    import jsonschema
    schema = load_json(schema_path)
    jsonschema.validate(instance=obj, schema=schema)


def report_ok(results: List[GateResult]) -> bool:
    """True iff every gate passed."""
    return all(r.status == GateStatus.PASS for r in results)


def print_human(results: List[GateResult]) -> None:
    """Print gate results in human-readable format."""
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def print_json(results: List[GateResult]) -> None:
    """Print gate results as JSON."""
    payload = {"ok": report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_self_test(
    fixtures_dir: str,
    fixture_specs: List[Tuple[str, bool, Optional[str]]],
    validate_fn: Callable[[Dict[str, Any]], List[GateResult]],
    label: str,
    as_json: bool = False,
) -> int:
    """Run self-test over fixtures.

    Args:
        fixtures_dir: directory containing fixture JSON files
        fixture_specs: list of (filename, should_pass, expected_fail_gate)
        validate_fn: the validator's main function (obj -> List[GateResult])
        label: display label for the self-test header
        as_json: emit JSON output instead of human-readable
    Returns:
        0 on success, 1 on failure
    """
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixture_specs:
        path = os.path.join(fixtures_dir, name)
        obj = load_json(path)
        res = validate_fn(obj)
        passed = report_ok(res)
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
        print(f"=== {label} SELF-TEST ===")
        for d in details:
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"{d['fixture']}: {status} (expected {'PASS' if d['expected_ok'] else 'FAIL'})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_cli(
    label: str,
    validate_fn: Callable[[Dict[str, Any]], List[GateResult]],
    self_test_fn: Callable[[bool], int],
    argv: Optional[List[str]] = None,
) -> int:
    """Build a standard CLI for a cert validator.

    Args:
        label: validator display name
        validate_fn: obj -> List[GateResult]
        self_test_fn: (as_json) -> exit code
        argv: optional argv override
    Returns:
        exit code
    """
    ap = argparse.ArgumentParser(description=f"{label} validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test on fixtures")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test_fn(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = load_json(args.file)
    results = validate_fn(obj)
    if args.json:
        print_json(results)
    else:
        print_human(results)
        print(f"\nRESULT: {'PASS' if report_ok(results) else 'FAIL'}")
    return 0 if report_ok(results) else 1
