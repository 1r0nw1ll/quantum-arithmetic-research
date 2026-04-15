#!/usr/bin/env python3
"""Append reproducibility records for QA experiment and benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception as exc:
        print(f"qa_reproducibility: warning: git commit lookup failed: {exc}", file=sys.stderr)
        return "unknown"


def _load_protocol(protocol_path: Path) -> dict[str, Any]:
    with protocol_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError("protocol JSON must be an object")
    return obj


def _ledger_path(protocol_path: Path, protocol: dict[str, Any]) -> Path:
    repro = protocol.get("reproducibility")
    if not isinstance(repro, dict):
        raise ValueError("protocol missing reproducibility object")
    raw = repro.get("results_ledger")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("protocol reproducibility.results_ledger missing")
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        path = protocol_path.parent / path
    return path


def log_run(
    protocol_path: str | os.PathLike[str],
    *,
    status: str = "started",
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one canonical JSONL record to the protocol's results ledger.

    The protocol controls the ledger location through
    `reproducibility.results_ledger`. Relative ledger paths are resolved
    relative to the protocol JSON.
    """
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = _load_protocol(protocol_file)
    repro = protocol.get("reproducibility")
    if not isinstance(repro, dict):
        raise ValueError("protocol missing reproducibility object")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    protocol_id = protocol.get("experiment_id") or protocol.get("benchmark_id") or protocol_file.stem

    record = {
        "schema_version": "QA_REPRODUCIBILITY_RUN.v1",
        "ts_utc": now.isoformat().replace("+00:00", "Z"),
        "protocol_path": str(protocol_file),
        "protocol_sha256": _sha256_file(protocol_file),
        "protocol_version": protocol.get("protocol_version"),
        "run_id": f"{protocol_id}:{stamp}",
        "status": status,
        "seed": repro.get("seed"),
        "data_sha256": repro.get("data_sha256"),
        "package_versions": repro.get("package_versions"),
        "commit": _git_commit(protocol_file.parent),
        "results": results or {},
    }

    ledger = _ledger_path(protocol_file, protocol)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(record) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a QA reproducibility run record")
    parser.add_argument("protocol", help="experiment_protocol.json or benchmark_protocol.json")
    parser.add_argument("--status", default="started")
    parser.add_argument("--json", action="store_true", help="Print appended record as JSON")
    args = parser.parse_args(argv)

    record = log_run(args.protocol, status=args.status)
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"qa_reproducibility: logged {record['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
