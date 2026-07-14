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
DEFAULT_PRUNE_PLAN_GLOB = "results/self_improving_neural_qa/prune_plans/qa_sinqa_artifact_prune_plan_*.json"
DEFAULT_ACTIVATION_CANARY_GLOB = "results/self_improving_neural_qa/activation_canary/results/*_activation_canary.json"
DEFAULT_PRIORITIZED_REPLAY_GLOB = "results/self_improving_neural_qa/prioritized_replay/qa_sinqa_prioritized_replay_*.json"
DEFAULT_RUNTIME_CONFIG = Path("results/self_improving_neural_qa/runtime/qa_general_ml_adapter_config.json")
DEFAULT_LAUNCHD_PLIST = Path("/Users/player3/Library/LaunchAgents/com.qa.self-improving-neural-qa.plist")
DEFAULT_TREND_WINDOW = 20
DEFAULT_ANTIFORGETTING_MAX_ITEMS = 32
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


def latest_prune_plan(glob_pattern: str, validate_prune_plan: Any | None = None) -> dict[str, Any]:
    paths = iter_glob_paths(glob_pattern)
    if not paths:
        return {"exists": False, "group_count": 0, "prune_candidate_count": 0}
    best: tuple[int, Path] | None = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    if best is None:
        return {"exists": False, "group_count": 0, "prune_candidate_count": 0}
    path = best[1]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - status should report parse failures.
        return {"exists": True, "ok": False, "path": str(path), "error": str(exc)}
    result = {
        "exists": True,
        "ok": obj.get("schema_version") == "QA_SINQA_ARTIFACT_PRUNE_PLAN.v0",
        "path": str(path),
        "plan_hash": obj.get("plan_hash"),
        "group_count": obj.get("group_count"),
        "prune_candidate_count": obj.get("prune_candidate_count"),
        "mode": obj.get("mode"),
    }
    if validate_prune_plan is not None:
        validation = validate_prune_plan(path)
        result["archive_safe"] = validation.get("ok") is True
        result["referenced_candidate_count"] = validation.get("referenced_candidate_count")
        result["validation_errors"] = validation.get("errors", [])[:3]
    return result


def latest_json_artifact(glob_pattern: str) -> dict[str, Any]:
    paths = iter_glob_paths(glob_pattern)
    if not paths:
        return {"exists": False}
    best: tuple[int, Path] | None = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    if best is None:
        return {"exists": False}
    path = best[1]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - status should report parse failures.
        return {"exists": True, "ok": False, "path": str(path), "error": str(exc)}
    if not isinstance(obj, dict):
        return {"exists": True, "ok": False, "path": str(path), "error": "JSON root must be object"}
    return {"exists": True, "path": str(path), **obj}


def runtime_config_status(path: Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {"exists": False, "path": str(path)}
    try:
        obj = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - status should report parse failures.
        return {"exists": True, "ok": False, "path": str(path), "error": str(exc)}
    if not isinstance(obj, dict):
        return {"exists": True, "ok": False, "path": str(path), "error": "JSON root must be object"}
    return {
        "exists": True,
        "ok": True,
        "path": str(path),
        "activation_update_id": obj.get("activation_update_id"),
        "activation_plan_hash": obj.get("activation_plan_hash"),
        "adapter_rank": obj.get("adapter.rank"),
        "training_max_steps": obj.get("training.max_steps"),
        "runtime_max_parameters": obj.get("runtime.max_parameters"),
        "runtime_max_memory_mb": obj.get("runtime.max_memory_mb"),
        "runtime_max_runtime_sec": obj.get("runtime.max_runtime_sec"),
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
    validate_prune_plan_fn = load_tool_function("qa_sinqa_artifact_prune_plan_validate.py", "validate_plan")
    build_trends_fn = load_tool_function("qa_self_improving_neural_qa_trends.py", "build_trends")
    validate_antiforgetting_fn = load_tool_function(
        "qa_self_improving_neural_qa_antiforgetting.py",
        "validate_antiforgetting",
    )

    ledger = validate_ledger(resolve_path(args.ledger))
    transcript = validate_transcript(resolve_path(args.transcript))
    active_paths = iter_glob_paths(args.active_proposal_glob)
    archive_paths = iter_glob_paths(args.archive_glob)
    archive = validate_lifecycle_paths(archive_paths)
    prune = latest_prune_plan(args.prune_plan_glob, validate_prune_plan_fn)
    activation_canary = latest_json_artifact(args.activation_canary_glob)
    prioritized_replay = latest_json_artifact(args.prioritized_replay_glob)
    runtime_config = runtime_config_status(args.runtime_config)
    trends = build_trends_fn(
        argparse.Namespace(
            ledger=args.ledger,
            scheduler_log=args.scheduler_log,
            neural_glob=args.neural_glob,
            window=args.trend_window,
        )
    )
    antiforgetting = validate_antiforgetting_fn(
        resolve_path(args.ledger),
        args.antiforgetting_max_items,
        args.antiforgetting_min_accepted,
    )

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
        and antiforgetting.get("ok") is True
        and (runtime_config.get("exists") is not True or runtime_config.get("ok") is True)
        and (activation_canary.get("exists") is not True or activation_canary.get("ok") is True)
        and (prioritized_replay.get("exists") is not True or prioritized_replay.get("ok") is True)
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
        "prune": prune,
        "activation": {
            "runtime_config": runtime_config,
            "latest_canary": {
                "exists": activation_canary.get("exists"),
                "ok": activation_canary.get("ok"),
                "path": activation_canary.get("path"),
                "update_id": activation_canary.get("update_id"),
                "activation_status": activation_canary.get("activation_status"),
                "runtime_mutated": activation_canary.get("runtime_mutated"),
                "canary_runtime_mutated": activation_canary.get("canary_runtime_mutated"),
                "config_after": activation_canary.get("config_after"),
                "result_hash": activation_canary.get("result_hash"),
            },
        },
        "prioritized_replay": {
            "exists": prioritized_replay.get("exists"),
            "ok": prioritized_replay.get("ok"),
            "path": prioritized_replay.get("path"),
            "sample_hash": prioritized_replay.get("sample_hash"),
            "selected_count": prioritized_replay.get("selected_count"),
            "candidate_count": prioritized_replay.get("candidate_count"),
            "eligible_count": prioritized_replay.get("eligible_count"),
            "unavailable_count": prioritized_replay.get("unavailable_count"),
            "selected_domains": prioritized_replay.get("selected_domains"),
            "selected_decisions": prioritized_replay.get("selected_decisions"),
            "source_missing": prioritized_replay.get("source_missing"),
            "source_hash_mismatch": prioritized_replay.get("source_hash_mismatch"),
        },
        "trends": trends,
        "antiforgetting": {
            "ok": antiforgetting.get("ok"),
            "checked": antiforgetting.get("checked"),
            "accepted_total": antiforgetting.get("accepted_total"),
            "missing": antiforgetting.get("missing"),
            "hash_mismatches": antiforgetting.get("hash_mismatches"),
            "harm_regressions": antiforgetting.get("harm_regressions"),
            "domains": antiforgetting.get("domains"),
            "source_kinds": antiforgetting.get("source_kinds"),
            "errors": antiforgetting.get("errors", [])[:5],
        },
        "launchd": launchd_status(args.launchd_plist),
    }


def print_text(status: dict[str, Any]) -> None:
    scheduler = status["scheduler"]
    neural = status["neural"]
    ledger = status["ledger"]
    transcript = status["transcript"]
    curriculum = status["curriculum"]
    prune = status["prune"]
    activation = status["activation"]
    prioritized_replay = status["prioritized_replay"]
    trends = status["trends"]
    antiforgetting = status["antiforgetting"]
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
    if prune["exists"]:
        print(
            f"prune: groups={prune.get('group_count')} candidates={prune.get('prune_candidate_count')} "
            f"mode={prune.get('mode')} archive_safe={prune.get('archive_safe')} "
            f"referenced={prune.get('referenced_candidate_count')}"
        )
    runtime_config = activation["runtime_config"]
    latest_canary = activation["latest_canary"]
    print(
        f"activation: runtime_rank={runtime_config.get('adapter_rank')} "
        f"runtime_steps={runtime_config.get('training_max_steps')} "
        f"runtime_update={runtime_config.get('activation_update_id')} "
        f"canary_ok={latest_canary.get('ok')} canary_update={latest_canary.get('update_id')}"
    )
    print(
        f"prioritized replay: selected={prioritized_replay.get('selected_count')} "
        f"eligible={prioritized_replay.get('eligible_count')} unavailable={prioritized_replay.get('unavailable_count')} "
        f"domains={prioritized_replay.get('selected_domains')} decisions={prioritized_replay.get('selected_decisions')}"
    )
    recent = trends["ledger"]["recent"]
    print(
        f"trends: recent{trends['ledger']['window']} accepted={recent['accepted']} "
        f"rejected={recent['rejected']} harm_attempts={recent['harm_attempts']} "
        f"domains={recent['domains']}"
    )
    print(
        f"anti-forgetting: checked={antiforgetting['checked']} "
        f"accepted_total={antiforgetting['accepted_total']} ok={antiforgetting['ok']} "
        f"missing={antiforgetting['missing']} hash_mismatches={antiforgetting['hash_mismatches']} "
        f"harm_regressions={antiforgetting['harm_regressions']}"
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
            prune_plan_glob=str(root / "missing_prune*.json"),
            activation_canary_glob=str(root / "missing_canary*.json"),
            prioritized_replay_glob=str(root / "missing_prioritized*.json"),
            runtime_config=root / "missing_runtime.json",
            launchd_plist=root / "missing.plist",
            trend_window=20,
            antiforgetting_max_items=32,
            antiforgetting_min_accepted=0,
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
    parser.add_argument("--prune-plan-glob", default=DEFAULT_PRUNE_PLAN_GLOB)
    parser.add_argument("--activation-canary-glob", default=DEFAULT_ACTIVATION_CANARY_GLOB)
    parser.add_argument("--prioritized-replay-glob", default=DEFAULT_PRIORITIZED_REPLAY_GLOB)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--launchd-plist", type=Path, default=DEFAULT_LAUNCHD_PLIST)
    parser.add_argument("--trend-window", type=int, default=DEFAULT_TREND_WINDOW)
    parser.add_argument("--antiforgetting-max-items", type=int, default=DEFAULT_ANTIFORGETTING_MAX_ITEMS)
    parser.add_argument("--antiforgetting-min-accepted", type=int, default=1)
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
