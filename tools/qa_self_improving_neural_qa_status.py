#!/usr/bin/env python3
"""Summarize live Self-Improving Neural QA status."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_status — read-only operational status summary "
    "for scheduled SINQA learning, ledger health, curriculum lifecycle, and "
    "latest neural worker output"
)

import argparse
import glob as globlib
import json
import plistlib
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULER_LOG = Path("results/self_improving_neural_qa/scheduler_runs.jsonl")
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
DEFAULT_TRANSCRIPT = Path("results/self_improving_neural_qa/loop_transcript.jsonl")
DEFAULT_ACTIVE_PROPOSAL_GLOB = "results/self_improving_neural_qa/curriculum_proposals/*.json"
DEFAULT_ARCHIVE_GLOB = "results/self_improving_neural_qa/curriculum_archive/**/qa_curriculum_lifecycle_*.json"
DEFAULT_NEURAL_GLOB = "experiments/qa_ml/results_sinqa_neural_general_adapter*.json"
DEFAULT_LAUNCHD_PLIST = Path("/Users/player3/Library/LaunchAgents/com.qa.self-improving-neural-qa.plist")
CURRENT_FOCUSED_CHECK_MIN_RUN = 13
CURRENT_FOCUSED_CHECK_COUNT = 6


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def iter_glob_paths(glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
    return sorted(Path(match) for match in globlib.glob(pattern, recursive=True))


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return [], [f"{path}: file not found"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_no, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            errors.append(f"{path}: line {line_no}: empty line")
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_no}: invalid JSON: {exc}")
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            errors.append(f"{path}: line {line_no}: row must be object")
    return rows, errors


def load_tool_function(tool_name: str, function_name: str) -> Any:
    path = ROOT / "tools" / tool_name
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": f"_status_{tool_name}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace[function_name]


def lifecycle_check_state(record: dict[str, Any]) -> dict[str, Any]:
    checks = record.get("focused_checks")
    run_no = record.get("run")
    required = isinstance(run_no, int) and run_no >= CURRENT_FOCUSED_CHECK_MIN_RUN
    result = {
        "required": required,
        "present": False,
        "active_ok": False,
        "archive_ok": False,
        "focused_check_count": len(checks) if isinstance(checks, list) else 0,
    }
    if not isinstance(checks, list):
        return result
    result["present"] = len(checks) >= CURRENT_FOCUSED_CHECK_COUNT
    if len(checks) >= 4:
        active = checks[2]
        archive = checks[3]
        if isinstance(active, dict):
            argv = active.get("argv")
            result["active_ok"] = (
                active.get("ok") is True
                and isinstance(argv, list)
                and "tools/qa_curriculum_lifecycle.py" in argv
                and "validate-active" in argv
            )
        if isinstance(archive, dict):
            argv = archive.get("argv")
            result["archive_ok"] = (
                archive.get("ok") is True
                and isinstance(argv, list)
                and "tools/qa_curriculum_lifecycle.py" in argv
                and "validate-archive" in argv
            )
    return result


def latest_neural_from_scheduler(record: dict[str, Any]) -> dict[str, Any]:
    worker = record.get("neural_worker")
    if not isinstance(worker, dict):
        return {"ok": False, "source": "scheduler", "reason": "latest row has no neural_worker object"}
    stdout_json = worker.get("stdout_json")
    if not isinstance(stdout_json, dict):
        return {"ok": False, "source": "scheduler", "reason": "neural_worker stdout_json missing"}
    keys = ["task_id", "run_id", "steps", "parameter_count", "out", "out_hash", "candidate_control", "baseline_control"]
    return {"ok": worker.get("ok") is True, "source": "scheduler", **{key: stdout_json.get(key) for key in keys}}


def latest_neural_from_files(glob_pattern: str) -> dict[str, Any]:
    paths = iter_glob_paths(glob_pattern)
    best: tuple[int, Path] | None = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    if best is None:
        return {"ok": False, "source": "files", "reason": "no neural result files found"}
    path = best[1]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - status should report parse failures.
        return {"ok": False, "source": "files", "path": str(path), "reason": f"invalid JSON: {exc}"}
    return {
        "ok": True,
        "source": "files",
        "path": str(path),
        "schema": obj.get("schema"),
        "task_id": obj.get("task_id"),
        "run_id": obj.get("run_id"),
        "steps": obj.get("steps"),
        "parameter_count": obj.get("parameter_count"),
    }


def launchd_status(plist_path: Path) -> dict[str, Any]:
    resolved = resolve_path(plist_path)
    if not resolved.exists():
        return {"configured": False, "path": str(plist_path)}
    try:
        with resolved.open("rb") as fh:
            plist = plistlib.load(fh)
    except Exception as exc:  # noqa: BLE001 - status should report parse failures.
        return {"configured": False, "path": str(plist_path), "error": str(exc)}
    argv = plist.get("ProgramArguments")
    return {
        "configured": True,
        "path": str(resolved),
        "label": plist.get("Label"),
        "start_interval_sec": plist.get("StartInterval"),
        "run_at_load": plist.get("RunAtLoad"),
        "working_directory": plist.get("WorkingDirectory"),
        "program": argv[0] if isinstance(argv, list) and argv else None,
        "uses_python314": isinstance(argv, list) and bool(argv) and "python3.14" in str(argv[0]),
    }


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    scheduler_rows, scheduler_errors = read_jsonl(args.scheduler_log)
    latest_record = scheduler_rows[-1] if scheduler_rows else {}

    validate_ledger = load_tool_function("qa_self_improving_neural_qa_ledger_validate.py", "validate_ledger")
    validate_transcript = load_tool_function(
        "qa_self_improving_neural_qa_transcript_validate.py",
        "validate_transcript",
    )
    validate_lifecycle_paths = load_tool_function("qa_curriculum_lifecycle.py", "validate_lifecycle_paths")

    ledger = validate_ledger(resolve_path(args.ledger))
    transcript = validate_transcript(resolve_path(args.transcript))
    active_paths = iter_glob_paths(args.active_proposal_glob)
    archive_paths = iter_glob_paths(args.archive_glob)
    archive = validate_lifecycle_paths(archive_paths)

    latest_run_ok = latest_record.get("ok") is True
    lifecycle = lifecycle_check_state(latest_record)
    neural = latest_neural_from_scheduler(latest_record)
    if not neural.get("ok"):
        neural = latest_neural_from_files(args.neural_glob)

    ok = (
        not scheduler_errors
        and latest_run_ok
        and ledger.get("ok") is True
        and transcript.get("ok") is True
        and len(active_paths) == 0
        and archive.get("ok") is True
        and (not lifecycle["required"] or (lifecycle["active_ok"] and lifecycle["archive_ok"]))
    )
    return {
        "ok": ok,
        "scheduler": {
            "path": str(args.scheduler_log),
            "rows": len(scheduler_rows),
            "errors": scheduler_errors,
            "latest_run": latest_record.get("run"),
            "latest_ok": latest_run_ok,
            "latest_stop_reason": latest_record.get("stop_reason"),
            "checkpoint_due": latest_record.get("checkpoint_due"),
            "unix_time": latest_record.get("unix_time"),
        },
        "neural": neural,
        "ledger": {
            "path": str(args.ledger),
            "ok": ledger.get("ok"),
            "rows": ledger.get("rows"),
            "accepted": ledger.get("accepted"),
            "rejected": ledger.get("rejected"),
            "errors": ledger.get("errors", [])[:5],
        },
        "transcript": {
            "path": str(args.transcript),
            "ok": transcript.get("ok"),
            "rows": transcript.get("rows"),
            "accepted": transcript.get("accepted"),
            "rejected": transcript.get("rejected"),
            "errors": transcript.get("errors", [])[:5],
        },
        "curriculum": {
            "active_proposals": len(active_paths),
            "archive_records": archive.get("count"),
            "archive_ok": archive.get("ok"),
            "lifecycle_checks": lifecycle,
        },
        "launchd": launchd_status(args.launchd_plist),
    }


def print_text(status: dict[str, Any]) -> None:
    scheduler = status["scheduler"]
    neural = status["neural"]
    ledger = status["ledger"]
    transcript = status["transcript"]
    curriculum = status["curriculum"]
    launchd = status["launchd"]
    print(f"SINQA status: {'OK' if status['ok'] else 'ATTENTION'}")
    print(
        f"latest run: {scheduler['latest_run']} ok={scheduler['latest_ok']} "
        f"stop={scheduler['latest_stop_reason']} rows={scheduler['rows']}"
    )
    print(
        f"neural: task={neural.get('task_id')} run={neural.get('run_id')} "
        f"steps={neural.get('steps')} params={neural.get('parameter_count')}"
    )
    print(
        f"ledger: rows={ledger['rows']} accepted={ledger['accepted']} "
        f"rejected={ledger['rejected']} ok={ledger['ok']}"
    )
    print(
        f"transcript: rows={transcript['rows']} accepted={transcript['accepted']} "
        f"rejected={transcript['rejected']} ok={transcript['ok']}"
    )
    lifecycle = curriculum["lifecycle_checks"]
    print(
        f"curriculum: active={curriculum['active_proposals']} archive={curriculum['archive_records']} "
        f"archive_ok={curriculum['archive_ok']} lifecycle_checks="
        f"{lifecycle['active_ok'] and lifecycle['archive_ok']}"
    )
    print(
        f"launchd: configured={launchd['configured']} interval={launchd.get('start_interval_sec')} "
        f"python314={launchd.get('uses_python314')}"
    )


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = root / "ledger.jsonl"
        transcript = root / "transcript.jsonl"
        scheduler = root / "scheduler.jsonl"
        active = root / "active" / "*.json"
        archive = root / "archive" / "**" / "qa_curriculum_lifecycle_*.json"
        ledger.write_text("", encoding="utf-8")
        transcript.write_text("", encoding="utf-8")
        row = {
            "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_RUN.v0",
            "run": 13,
            "ok": True,
            "unix_time": 1,
            "stop_reason": "commit_caps_reached",
            "checkpoint_due": False,
            "focused_checks": [
                {"ok": True, "returncode": 0, "stdout_json": {"ok": True}, "argv": ["python3", "ledger"]},
                {"ok": True, "returncode": 0, "stdout_json": {"ok": True}, "argv": ["python3", "transcript"]},
                {
                    "ok": True,
                    "returncode": 0,
                    "stdout_json": {"ok": True},
                    "argv": ["python3", "tools/qa_curriculum_lifecycle.py", "validate-active"],
                },
                {
                    "ok": True,
                    "returncode": 0,
                    "stdout_json": {"ok": True},
                    "argv": ["python3", "tools/qa_curriculum_lifecycle.py", "validate-archive"],
                },
                {"ok": True, "returncode": 0, "stdout_json": {"ok": True}, "argv": ["python3", "524"]},
                {"ok": True, "returncode": 0, "stdout_json": {"ok": True}, "argv": ["python3", "525"]},
            ],
            "neural_worker": {
                "ok": True,
                "stdout_json": {"ok": True, "task_id": "qa_residue_mod3", "run_id": 1, "steps": 60, "parameter_count": 10},
            },
        }
        scheduler.write_text(canonical_json(row) + "\n", encoding="utf-8")
        args = argparse.Namespace(
            scheduler_log=scheduler,
            ledger=ledger,
            transcript=transcript,
            active_proposal_glob=str(active),
            archive_glob=str(archive),
            neural_glob=str(root / "missing*.json"),
            launchd_plist=root / "missing.plist",
        )
        status = build_status(args)
        ok = (
            status["scheduler"]["latest_run"] == 13
            and status["curriculum"]["lifecycle_checks"]["active_ok"] is True
            and status["curriculum"]["lifecycle_checks"]["archive_ok"] is True
            and status["neural"]["task_id"] == "qa_residue_mod3"
        )
        return {"ok": ok, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-log", type=Path, default=DEFAULT_SCHEDULER_LOG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--active-proposal-glob", default=DEFAULT_ACTIVE_PROPOSAL_GLOB)
    parser.add_argument("--archive-glob", default=DEFAULT_ARCHIVE_GLOB)
    parser.add_argument("--neural-glob", default=DEFAULT_NEURAL_GLOB)
    parser.add_argument("--launchd-plist", type=Path, default=DEFAULT_LAUNCHD_PLIST)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    status = build_status(args)
    if args.text:
        print_text(status)
    else:
        print(json.dumps(status, sort_keys=True))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
