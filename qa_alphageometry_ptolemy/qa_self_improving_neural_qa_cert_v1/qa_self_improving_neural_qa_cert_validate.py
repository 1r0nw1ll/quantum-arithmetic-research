#!/usr/bin/env python3
"""Validator for QA_SELF_IMPROVING_NEURAL_QA_CERT.v1.

This is an unregistered cert-family stub. It validates compact ledger fixtures
for the Self-Improving Neural QA v0 lane without adding the family to the
project meta-validator.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator — validates self-improving neural QA v0 packet ledgers; "
    "accepted updates require zero-harm deterministic replay; rejected updates "
    "preserve failed gate evidence; capacity/configuration proposals require "
    "bounds, rollback, and manual activation"
)

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
PACKET_HASH_DOMAIN = "qa_self_improving_neural_qa_packet_v0"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def load_tool_function(tool_name: str, function_name: str) -> Any:
    path = TOOLS / tool_name
    namespace: dict[str, Any] = {}
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[function_name]


def validate_ledger_file(path: Path) -> dict[str, Any]:
    validate_packet = load_tool_function("qa_self_improving_neural_qa_v0.py", "validate_packet")
    errors: list[str] = []
    rows = 0
    accepted = 0
    rejected = 0
    update_ids: set[str] = set()
    packet_hashes: set[str] = set()
    source_replay_hashes: set[str] = set()

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            errors.append(f"line {line_no}: empty line")
            continue
        rows += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if canonical_json(row) != raw:
            errors.append(f"line {line_no}: non-canonical JSON row")
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: row must be object")
            continue
        packet = row.get("packet")
        packet_hash = row.get("packet_hash")
        if not isinstance(packet, dict):
            errors.append(f"line {line_no}: packet must be object")
            continue
        if not isinstance(packet_hash, str):
            errors.append(f"line {line_no}: packet_hash must be string")
            continue
        expected_hash = domain_sha256(PACKET_HASH_DOMAIN, packet)
        if packet_hash != expected_hash:
            errors.append(f"line {line_no}: packet_hash mismatch")
        if packet_hash in packet_hashes:
            errors.append(f"line {line_no}: duplicate packet_hash")
        packet_hashes.add(packet_hash)
        update_id = packet.get("update_id")
        if not isinstance(update_id, str) or not update_id:
            errors.append(f"line {line_no}: missing update_id")
        elif update_id in update_ids:
            errors.append(f"line {line_no}: duplicate update_id")
        else:
            update_ids.add(update_id)
        try:
            summary = validate_packet(packet)
            evidence = packet.get("evidence")
            if isinstance(evidence, dict):
                source_replay_hash = evidence.get("source_replay_hash")
                if isinstance(source_replay_hash, str):
                    if source_replay_hash in source_replay_hashes:
                        errors.append(f"line {line_no}: duplicate source_replay_hash")
                    source_replay_hashes.add(source_replay_hash)
            decision = summary["decision"]
            if decision == "accepted":
                accepted += 1
            elif decision == "rejected":
                rejected += 1
        except Exception as exc:  # noqa: BLE001 - fixtures report validation errors.
            errors.append(f"line {line_no}: packet validation failed: {exc}")

    return {
        "ok": not errors,
        "rows": rows,
        "accepted": accepted,
        "rejected": rejected,
        "unique_update_ids": len(update_ids),
        "unique_packet_hashes": len(packet_hashes),
        "unique_source_replay_hashes": len(source_replay_hashes),
        "errors": errors,
    }


def self_test() -> dict[str, Any]:
    fixtures = Path(__file__).resolve().parent / "fixtures"
    cases = [
        ("pass_ledger.jsonl", True),
        ("pass_capacity_patch_ledger.jsonl", True),
        ("pass_live_ledger.jsonl", True),
        ("fail_duplicate_update_id.jsonl", False),
        ("fail_accepted_harm.jsonl", False),
        ("fail_duplicate_source_replay_hash.jsonl", False),
        ("fail_unbounded_capacity_patch.jsonl", False),
    ]
    results = []
    ok = True
    for name, should_pass in cases:
        result = validate_ledger_file(fixtures / name)
        passed = bool(result["ok"])
        case_ok = passed is should_pass
        ok = ok and case_ok
        results.append({"fixture": name, "ok": case_ok, "passed": passed, "result": result})
    pass_result = validate_ledger_file(fixtures / "pass_ledger.jsonl")
    if pass_result["accepted"] < 1 or pass_result["rejected"] < 1:
        ok = False
        results.append({
            "fixture": "pass_ledger.jsonl",
            "ok": False,
            "error": "PASS fixture must include at least one accepted and one rejected packet",
        })
    return {"ok": ok, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    paths = args.paths or sorted((Path(__file__).resolve().parent / "fixtures").glob("*.jsonl"))
    all_ok = True
    for path in paths:
        result = validate_ledger_file(path)
        all_ok = all_ok and bool(result["ok"])
        print(json.dumps({"path": str(path), **result}, sort_keys=True))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
