#!/usr/bin/env python3
"""Validate Self-Improving Neural QA multi-round loop transcripts."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_transcript_validator — verifies continual "
    "learning round hash chains, ledger deltas, and rejected-state immutability"
)

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_LOOP_ROUND.v0"
ZERO_HASH = "0" * 64
HEX64 = set("0123456789abcdef")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in HEX64 for ch in value)


def require_summary(record: dict[str, Any], key: str) -> dict[str, Any]:
    summary = record.get(key)
    if not isinstance(summary, dict):
        raise ValueError(f"{key}: expected object")
    for field in ("ok", "rows", "accepted", "rejected"):
        if field not in summary:
            raise ValueError(f"{key}.{field}: missing")
    if summary["ok"] is not True:
        raise ValueError(f"{key}.ok: expected true")
    for field in ("rows", "accepted", "rejected"):
        val = summary[field]
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ValueError(f"{key}.{field}: expected non-negative integer")
    return summary


def validate_record(record: dict[str, Any], line_no: int, expected_prev: str) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"line {line_no}: schema_version mismatch")
    if record.get("previous_record_hash") != expected_prev:
        errors.append(f"line {line_no}: previous_record_hash mismatch")
    for key in ("ledger_hash_before", "ledger_hash_after", "packet_hash", "source_replay_hash"):
        if not is_hex64(record.get(key)):
            errors.append(f"line {line_no}: {key} must be 64-char lowercase hex")
    decision = record.get("decision")
    if decision not in {"accepted", "rejected"}:
        errors.append(f"line {line_no}: decision must be accepted or rejected")
    for key in ("round", "fixed", "harmed", "protected"):
        val = record.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            errors.append(f"line {line_no}: {key} must be non-negative integer")
    if record.get("round") != line_no:
        errors.append(f"line {line_no}: round must equal transcript line number")
    if not isinstance(record.get("update_id"), str) or not record["update_id"]:
        errors.append(f"line {line_no}: update_id must be non-empty string")
    if not isinstance(record.get("replay"), str) or not record["replay"]:
        errors.append(f"line {line_no}: replay must be non-empty string")

    try:
        before = require_summary(record, "ledger_summary_before")
        after = require_summary(record, "ledger_summary_after")
    except Exception as exc:
        errors.append(f"line {line_no}: {exc}")
        return errors

    if after["rows"] != before["rows"] + 1:
        errors.append(f"line {line_no}: ledger rows must increase by exactly 1")
    if decision == "accepted":
        if record.get("promoted_state_mutated") is not True:
            errors.append(f"line {line_no}: accepted round must mutate promoted state")
        if record.get("fixed", 0) <= 0:
            errors.append(f"line {line_no}: accepted round requires fixed > 0")
        if record.get("harmed") != 0:
            errors.append(f"line {line_no}: accepted round requires harmed == 0")
        if after["accepted"] != before["accepted"] + 1:
            errors.append(f"line {line_no}: accepted count must increase by 1")
        if after["rejected"] != before["rejected"]:
            errors.append(f"line {line_no}: rejected count must not change on accepted round")
    elif decision == "rejected":
        if record.get("promoted_state_mutated") is not False:
            errors.append(f"line {line_no}: rejected round must not mutate promoted state")
        if after["rejected"] != before["rejected"] + 1:
            errors.append(f"line {line_no}: rejected count must increase by 1")
        if after["accepted"] != before["accepted"]:
            errors.append(f"line {line_no}: accepted count must not change on rejected round")
    return errors


def validate_transcript(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows = 0
    accepted = 0
    rejected = 0
    packet_hashes: set[str] = set()
    source_hashes: set[str] = set()
    expected_prev = ZERO_HASH

    if not path.exists():
        return {"ok": False, "errors": [f"{path}: file not found"], "rows": 0}

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            errors.append(f"line {line_no}: empty line")
            continue
        rows += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if canonical_json(row) != raw:
            errors.append(f"line {line_no}: row is not canonical JSON")
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: row must be object")
            continue
        record = row.get("record")
        record_hash = row.get("record_hash")
        if not isinstance(record, dict):
            errors.append(f"line {line_no}: record must be object")
            continue
        expected_hash = domain_sha256("qa_self_improving_neural_qa_loop_record_v0", record)
        if record_hash != expected_hash:
            errors.append(f"line {line_no}: record_hash mismatch")
        errors.extend(validate_record(record, line_no, expected_prev))
        expected_prev = record_hash if isinstance(record_hash, str) else expected_hash
        decision = record.get("decision")
        if decision == "accepted":
            accepted += 1
        elif decision == "rejected":
            rejected += 1
        packet_hash = record.get("packet_hash")
        if isinstance(packet_hash, str):
            if packet_hash in packet_hashes:
                errors.append(f"line {line_no}: duplicate packet_hash")
            packet_hashes.add(packet_hash)
        source_hash = record.get("source_replay_hash")
        if isinstance(source_hash, str):
            if source_hash in source_hashes:
                errors.append(f"line {line_no}: duplicate source_replay_hash")
            source_hashes.add(source_hash)

    return {
        "ok": not errors,
        "rows": rows,
        "accepted": accepted,
        "rejected": rejected,
        "unique_packet_hashes": len(packet_hashes),
        "unique_source_replay_hashes": len(source_hashes),
        "errors": errors,
    }


def self_test() -> dict[str, Any]:
    loop_path = Path(__file__).resolve().with_name("qa_self_improving_neural_qa_loop.py")
    namespace: dict[str, Any] = {"__file__": str(loop_path), "__name__": "_loop_selftest"}
    exec(compile(loop_path.read_text(encoding="utf-8"), str(loop_path), "exec"), namespace)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        args = argparse.Namespace(
            rounds=2,
            glob="results/qa_hsi_salinas*correction_replay*.json",
            config_glob=str(root / "*sinqa_config_proposal*.json"),
            ledger=root / "ledger.jsonl",
            out_dir=root / "packets",
            transcript=root / "transcript.jsonl",
            base_model_name="transcript_selftest_hsi_ensemble",
            base_model_kind="classifier",
            max_accepted=1,
            max_rejected=1,
            dry_run=False,
        )
        run_result = namespace["run_loop"](args)
        good = validate_transcript(args.transcript)

        lines = args.transcript.read_text(encoding="utf-8").splitlines()
        tampered = root / "tampered.jsonl"
        row = json.loads(lines[-1])
        row["record"]["promoted_state_mutated"] = True
        lines[-1] = canonical_json(row)
        tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
        bad = validate_transcript(tampered)

        ok = run_result["ok"] is True and good["ok"] is True and bad["ok"] is False
        return {"ok": ok, "run": run_result, "good": good, "bad": bad}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.transcript is None:
        print("error: provide a transcript path or --self-test", file=sys.stderr)
        return 2
    result = validate_transcript(args.transcript)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
