#!/usr/bin/env python3
"""Validate Self-Improving Neural QA v0 append-only ledgers."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_ledger_validator — verifies canonical ledger "
    "rows, packet hashes, unique update IDs, per-packet replay gates, and "
    "bounded capacity/configuration proposals"
)

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PACKET_HASH_DOMAIN = "qa_self_improving_neural_qa_packet_v0"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def load_packet_validator() -> Any:
    tool_path = Path(__file__).resolve().with_name("qa_self_improving_neural_qa_v0.py")
    namespace: dict[str, Any] = {}
    code = tool_path.read_text(encoding="utf-8")
    exec(compile(code, str(tool_path), "exec"), namespace)
    return namespace["validate_packet"]


def validate_ledger(path: Path) -> dict[str, Any]:
    validate_packet = load_packet_validator()
    errors: list[str] = []
    update_ids: set[str] = set()
    packet_hashes: set[str] = set()
    source_replay_hashes: set[str] = set()
    accepted = 0
    rejected = 0
    rows = 0

    if not path.exists():
        return {"ok": False, "errors": [f"{path}: file not found"], "rows": 0}

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            errors.append(f"line {line_no}: empty lines are not allowed")
            continue
        rows += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: row must be a JSON object")
            continue
        if canonical_json(row) != raw:
            errors.append(f"line {line_no}: row is not canonical JSON")

        packet = row.get("packet")
        packet_hash = row.get("packet_hash")
        if not isinstance(packet, dict):
            errors.append(f"line {line_no}: packet must be an object")
            continue
        if not isinstance(packet_hash, str):
            errors.append(f"line {line_no}: packet_hash must be a string")
            continue

        computed_hash = domain_sha256(PACKET_HASH_DOMAIN, packet)
        if packet_hash != computed_hash:
            errors.append(
                f"line {line_no}: packet_hash mismatch: expected {computed_hash}, got {packet_hash}"
            )
        if packet_hash in packet_hashes:
            errors.append(f"line {line_no}: duplicate packet_hash {packet_hash}")
        packet_hashes.add(packet_hash)

        update_id = packet.get("update_id")
        if not isinstance(update_id, str) or not update_id:
            errors.append(f"line {line_no}: packet.update_id must be a non-empty string")
        elif update_id in update_ids:
            errors.append(f"line {line_no}: duplicate update_id {update_id}")
        else:
            update_ids.add(update_id)

        try:
            validate_packet(packet)
            evidence = packet.get("evidence")
            if isinstance(evidence, dict):
                source_replay_hash = evidence.get("source_replay_hash")
                if isinstance(source_replay_hash, str):
                    if source_replay_hash in source_replay_hashes:
                        errors.append(
                            f"line {line_no}: duplicate evidence.source_replay_hash {source_replay_hash}"
                        )
                    source_replay_hashes.add(source_replay_hash)
            decision = (packet.get("promotion") or {}).get("decision")
            if decision == "accepted":
                accepted += 1
            elif decision == "rejected":
                rejected += 1
        except Exception as exc:  # noqa: BLE001 - report packet validation error.
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


def _self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        packet = {
            "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0",
            "update_id": "sinqa.v0.ledger.selftest",
            "base_model": {"name": "selftest", "kind": "classifier", "neural": True},
            "candidate": {
                "kind": "correction_rule",
                "description": "ledger self-test",
                "artifact_ref": "results/selftest.json",
            },
            "evidence": {
                "source_replay_ref": "results/selftest_replay.json",
                "source_replay_hash": hashlib.sha256(b"source-replay").hexdigest(),
            },
            "replay_gate": {
                "new_failures_fixed": 1,
                "protected_cases_replayed": 2,
                "protected_cases_harmed": 0,
                "deterministic_replay": True,
                "trace_hash_before": hashlib.sha256(b"before").hexdigest(),
                "trace_hash_after": hashlib.sha256(b"after").hexdigest(),
            },
            "invariant_checks": [{"name": "zero_harm", "passed": True}],
            "promotion": {
                "decision": "accepted",
                "ledger_hash": hashlib.sha256(b"ledger").hexdigest(),
            },
        }
        rejected_packet = json.loads(canonical_json(packet))
        rejected_packet["update_id"] = "sinqa.v0.ledger.selftest.rejected"
        rejected_packet["evidence"]["source_replay_ref"] = "results/selftest_replay_rejected.json"
        rejected_packet["evidence"]["source_replay_hash"] = hashlib.sha256(
            b"source-replay-rejected"
        ).hexdigest()
        rejected_packet["replay_gate"]["new_failures_fixed"] = 0
        rejected_packet["invariant_checks"][0]["passed"] = False
        rejected_packet["promotion"] = {
            "decision": "rejected",
            "rejection_reason": "no_positive_incremental_fix",
            "ledger_hash": hashlib.sha256(b"rejected-ledger").hexdigest(),
        }
        row = {"packet_hash": domain_sha256(PACKET_HASH_DOMAIN, packet), "packet": packet}
        rejected_row = {
            "packet_hash": domain_sha256(PACKET_HASH_DOMAIN, rejected_packet),
            "packet": rejected_packet,
        }
        good = root / "good.jsonl"
        good.write_text(canonical_json(row) + "\n" + canonical_json(rejected_row) + "\n", encoding="utf-8")
        bad = root / "bad.jsonl"
        bad_row = dict(row)
        bad_row["packet_hash"] = "0" * 64
        bad.write_text(canonical_json(bad_row) + "\n", encoding="utf-8")
        duplicate_source = root / "duplicate_source.jsonl"
        duplicate_packet = json.loads(canonical_json(packet))
        duplicate_packet["update_id"] = "sinqa.v0.ledger.selftest.duplicate_source"
        duplicate_packet["promotion"]["ledger_hash"] = hashlib.sha256(b"duplicate-source").hexdigest()
        duplicate_row = {
            "packet_hash": domain_sha256(PACKET_HASH_DOMAIN, duplicate_packet),
            "packet": duplicate_packet,
        }
        duplicate_source.write_text(
            canonical_json(row) + "\n" + canonical_json(duplicate_row) + "\n",
            encoding="utf-8",
        )

        good_result = validate_ledger(good)
        bad_result = validate_ledger(bad)
        duplicate_source_result = validate_ledger(duplicate_source)
        ok = (
            good_result["ok"] is True
            and bad_result["ok"] is False
            and duplicate_source_result["ok"] is False
        )
        return {
            "ok": ok,
            "good": good_result,
            "bad": bad_result,
            "duplicate_source": duplicate_source_result,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    if args.ledger is None:
        print("error: provide a ledger path or --self-test", file=sys.stderr)
        return 2
    result = validate_ledger(args.ledger)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
