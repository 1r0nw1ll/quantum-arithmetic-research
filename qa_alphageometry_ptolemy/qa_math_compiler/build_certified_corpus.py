#!/usr/bin/env python3
"""
Build or verify a deterministic certified-corpus manifest from a demo pack.

Usage:
  python build_certified_corpus.py build <demo_pack_dir>
  python build_certified_corpus.py check <demo_pack_dir>

The build command prints canonical JSON to stdout. The check command compares
the deterministic rebuild with <demo_pack_dir>/corpus.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from qa_math_compiler_validator import validate_corpus, validate_demo_pack_v1


CORPUS_SCHEMA_ID = "QA_CERTIFIED_PROOF_CORPUS_SCHEMA.v1"
CORPUS_HASH_DOMAIN = "QA_CERTIFIED_PROOF_CORPUS_SCHEMA.v1:entries"
SPLIT_BY_INDEX = ("train", "train", "train", "validation", "test")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def domain_hash(domain: str, obj: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + canonical_json(obj).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def replay_row_for_trace(replay: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    rows = [
        row
        for row in replay.get("traces", [])
        if isinstance(row, dict) and row.get("trace_id") == trace_id
    ]
    if len(rows) != 1:
        raise ValueError(
            f"replay bundle must contain exactly one row for trace_id={trace_id}; got {len(rows)}"
        )
    return rows[0]


def failure_record_for(
    status: str,
    trace: Dict[str, Any],
    replay_row: Dict[str, Any],
) -> Dict[str, Any] | None:
    if status == "PROVED":
        return None
    result = trace.get("result")
    if isinstance(result, dict) and result.get("status") == "FAIL":
        return {
            "fail_type": result.get("fail_type", "UNCLASSIFIED_TRACE_FAILURE"),
            "invariant_diff": result.get("invariant_diff", {}),
        }
    return {
        "fail_type": "REPLAY_NOT_CERTIFIED",
        "invariant_diff": {
            "result_status": replay_row.get("result_status"),
            "replay_status": replay_row.get("replay_status"),
        },
    }


def created_utc_from_tasks(tasks: List[Dict[str, Any]]) -> str:
    timestamps = [task.get("created_utc") for task in tasks]
    if not all(isinstance(value, str) for value in timestamps):
        raise ValueError("every task must have created_utc")
    return max(timestamps)


def build_manifest(pack_dir: Path) -> Dict[str, Any]:
    demo_result = validate_demo_pack_v1(str(pack_dir))
    if not demo_result.ok:
        raise ValueError(
            f"demo pack invalid: {demo_result.fail_type} "
            f"{canonical_json(demo_result.invariant_diff)}"
        )

    index = load_json(pack_dir / "index.json")
    examples = index.get("examples")
    if not isinstance(examples, list):
        raise ValueError("index.examples must be a list")
    if len(examples) != len(SPLIT_BY_INDEX):
        raise ValueError(
            f"demo_pack_v1 split policy requires {len(SPLIT_BY_INDEX)} examples; "
            f"got {len(examples)}"
        )

    entries: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    for idx, index_entry in enumerate(examples):
        if not isinstance(index_entry, dict) or not isinstance(index_entry.get("id"), str):
            raise ValueError(f"index.examples[{idx}] must contain string id")
        example_id = index_entry["id"]
        example_dir = pack_dir / "examples" / example_id
        task = load_json(example_dir / "task.json")
        trace = load_json(example_dir / "trace.json")
        replay = load_json(example_dir / "replay.json")
        pair = load_json(example_dir / "pair.json")
        status = load_json(example_dir / "status.json")
        tasks.append(task)

        trace_id = trace.get("trace_id")
        if not isinstance(trace_id, str):
            raise ValueError(f"{example_id}: trace_id missing")
        replay_row = replay_row_for_trace(replay, trace_id)
        pair_status = pair.get("status")
        status_value = status.get("status")
        if status_value != pair_status:
            raise ValueError(
                f"{example_id}: status.json status {status_value!r} "
                f"does not match pair.json status {pair_status!r}"
            )

        corpus_status = "CERTIFIED" if status_value == "PROVED" else "FAILED"
        entries.append(
            {
                "example_id": example_id,
                "split": SPLIT_BY_INDEX[idx],
                "proof_assistant": "lean4",
                "task_hash": canonical_hash(task),
                "trace_hash": canonical_hash(trace),
                "replay_hash": canonical_hash(replay),
                "pair_hash": canonical_hash(pair),
                "status": corpus_status,
                "replay_status": replay_row.get("replay_status"),
                "failure_record": failure_record_for(status_value, trace, replay_row),
            }
        )

    entry_count = len(entries)
    certified_count = sum(entry["status"] == "CERTIFIED" for entry in entries)
    failed_count = entry_count - certified_count
    replay_success_count = sum(entry["replay_status"] == "SUCCESS" for entry in entries)
    manifest = {
        "schema_id": CORPUS_SCHEMA_ID,
        "corpus_id": f"{pack_dir.name}-certified-corpus-v1",
        "created_utc": created_utc_from_tasks(tasks),
        "entries": entries,
        "metrics": {
            "entry_count": entry_count,
            "certified_count": certified_count,
            "failed_count": failed_count,
            "replay_success_rate": replay_success_count / entry_count,
        },
        "corpus_sha256": domain_hash(CORPUS_HASH_DOMAIN, entries),
        "invariant_diff": {
            "source_pack": pack_dir.name,
            "split_policy": list(SPLIT_BY_INDEX),
            "hash_policy": "sha256(canonical_json(artifact))",
            "corpus_hash_policy": "sha256(domain + NUL + canonical_json(entries))",
        },
    }
    result = validate_corpus(manifest)
    if not result.ok:
        raise ValueError(
            f"generated corpus invalid: {result.fail_type} "
            f"{canonical_json(result.invariant_diff)}"
        )
    return manifest


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"build", "check"}:
        print(f"Usage: {sys.argv[0]} build|check <demo_pack_dir>", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    pack_dir = Path(sys.argv[2]).resolve()
    try:
        manifest = build_manifest(pack_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(canonical_json({"ok": False, "error": str(exc)}))
        return 1

    if mode == "build":
        print(canonical_json(manifest))
        return 0

    corpus_path = pack_dir / "corpus.json"
    try:
        stored = load_json(corpus_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(canonical_json({"ok": False, "error": str(exc)}))
        return 1
    if stored != manifest:
        print(
            canonical_json(
                {
                    "ok": False,
                    "fail_type": "CORPUS_REBUILD_MISMATCH",
                    "expected_sha256": canonical_hash(manifest),
                    "stored_sha256": canonical_hash(stored),
                }
            )
        )
        return 1
    print(
        canonical_json(
            {
                "ok": True,
                "corpus_id": manifest["corpus_id"],
                "entry_count": manifest["metrics"]["entry_count"],
                "corpus_sha256": manifest["corpus_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
