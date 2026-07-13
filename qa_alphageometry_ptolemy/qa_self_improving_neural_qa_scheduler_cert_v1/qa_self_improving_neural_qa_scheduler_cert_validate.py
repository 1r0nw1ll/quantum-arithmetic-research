#!/usr/bin/env python3
"""Validator for QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_CERT.v1."""

from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator — validates scheduled self-improving neural QA run records; "
    "scheduler may run capped general-ML neural training, replay producer, "
    "capacity proposal, and curriculum discovery phases, must forward "
    "replay/config discovery and isolated supervisor paths, and must pass "
    "focused checks"
)

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_RUN.v0"
ROOT = Path(__file__).resolve().parents[2]
LIVE_LOG = ROOT / "results/self_improving_neural_qa/scheduler_runs.jsonl"
DEFAULT_MIGRATIONS = ROOT / "results/self_improving_neural_qa/scheduler_log_migrations.json"
REQUIRED_SUPERVISOR_ARGS = {
    "--glob",
    "--config-glob",
    "--runtime-config",
    "--ledger",
    "--out-dir",
    "--transcript",
    "--state",
    "--heartbeat",
    "--lock",
}
LEGACY_FOCUSED_CHECK_COUNT = 4
CURRENT_FOCUSED_CHECK_COUNT = 6
STOP_REASONS = {
    "commit_caps_reached",
    "no_candidate",
    "limits_reached",
    "loop_failed",
    "validation_failed",
    "cycle_limit_reached",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_command_result(obj: Any, label: str, errors: list[str], require_json_ok: bool) -> None:
    if not isinstance(obj, dict):
        errors.append(f"{label}: expected object")
        return
    if obj.get("ok") is not True:
        errors.append(f"{label}: ok must be true")
    if obj.get("returncode") != 0:
        errors.append(f"{label}: returncode must be 0")
    if require_json_ok:
        stdout_json = obj.get("stdout_json")
        if not isinstance(stdout_json, dict) or stdout_json.get("ok") is not True:
            errors.append(f"{label}: stdout_json.ok must be true")


def migration_relaxations(migrations: dict[int, set[str]], line_no: int) -> set[str]:
    return migrations.get(line_no, set())


def validate_record(record: dict[str, Any], line_no: int, migrations: dict[int, set[str]] | None = None) -> list[str]:
    relax = migration_relaxations(migrations or {}, line_no)
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"line {line_no}: schema_version mismatch")
    if record.get("ok") is not True and "scheduled_run_ok_false" not in relax:
        errors.append(f"line {line_no}: scheduled run ok must be true")
    if not isinstance(record.get("run"), int) or isinstance(record.get("run"), bool) or record["run"] < 1:
        errors.append(f"line {line_no}: run must be positive integer")
    if not isinstance(record.get("unix_time"), int) or isinstance(record.get("unix_time"), bool):
        errors.append(f"line {line_no}: unix_time must be integer")
    stop_reason = record.get("stop_reason")
    if stop_reason not in STOP_REASONS:
        errors.append(f"line {line_no}: stop_reason must be known bounded-run reason")

    producer = record.get("producer")
    if producer is not None:
        validate_command_result(producer, f"line {line_no}: producer", errors, True)

    neural_worker = record.get("neural_worker")
    if neural_worker is not None and "neural_worker_failed_missing_numpy" not in relax:
        validate_command_result(neural_worker, f"line {line_no}: neural_worker", errors, True)

    capacity_producer = record.get("capacity_producer")
    if capacity_producer is not None:
        validate_command_result(capacity_producer, f"line {line_no}: capacity_producer", errors, True)

    curriculum_discovery = record.get("curriculum_discovery")
    if curriculum_discovery is not None:
        validate_command_result(curriculum_discovery, f"line {line_no}: curriculum_discovery", errors, True)

    supervisor = record.get("supervisor")
    validate_command_result(supervisor, f"line {line_no}: supervisor", errors, True)
    if isinstance(supervisor, dict):
        argv = supervisor.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            errors.append(f"line {line_no}: supervisor.argv must be list[str]")
        else:
            missing = sorted(arg for arg in REQUIRED_SUPERVISOR_ARGS if arg not in argv)
            if "supervisor_runtime_config_arg_missing" in relax and missing == ["--runtime-config"]:
                missing = []
            if missing:
                errors.append(f"line {line_no}: supervisor.argv missing path args {missing}")

    checks = record.get("focused_checks")
    if not isinstance(checks, list) or len(checks) not in {LEGACY_FOCUSED_CHECK_COUNT, CURRENT_FOCUSED_CHECK_COUNT}:
        errors.append(
            f"line {line_no}: focused_checks must contain "
            f"{LEGACY_FOCUSED_CHECK_COUNT} legacy or {CURRENT_FOCUSED_CHECK_COUNT} current results"
        )
    elif not all(isinstance(check, dict) for check in checks):
        errors.append(f"line {line_no}: focused_checks entries must be objects")
    else:
        for idx, check in enumerate(checks, start=1):
            validate_command_result(check, f"line {line_no}: focused_checks[{idx}]", errors, True)
        if len(checks) == CURRENT_FOCUSED_CHECK_COUNT:
            active_argv = checks[2].get("argv")
            archive_argv = checks[3].get("argv")
            if not isinstance(active_argv, list) or "tools/qa_curriculum_lifecycle.py" not in active_argv:
                errors.append(f"line {line_no}: focused_checks[3] must run curriculum lifecycle validator")
            elif "validate-active" not in active_argv:
                errors.append(f"line {line_no}: focused_checks[3] must validate active curriculum proposals")
            if not isinstance(archive_argv, list) or "tools/qa_curriculum_lifecycle.py" not in archive_argv:
                errors.append(f"line {line_no}: focused_checks[4] must run curriculum lifecycle validator")
            elif "validate-archive" not in archive_argv:
                errors.append(f"line {line_no}: focused_checks[4] must validate curriculum archive")

    checkpoint_due = record.get("checkpoint_due")
    if not isinstance(checkpoint_due, bool):
        errors.append(f"line {line_no}: checkpoint_due must be boolean")
    meta = record.get("meta")
    if checkpoint_due is True:
        validate_command_result(meta, f"line {line_no}: meta", errors, False)
    elif meta is not None:
        errors.append(f"line {line_no}: meta must be null when checkpoint_due is false")
    return errors


def load_migrations(log_path: Path, migration_path: Path | None) -> tuple[dict[int, set[str]], list[str]]:
    if migration_path is None:
        return {}, []
    if not migration_path.exists():
        return {}, [f"{migration_path}: migration file not found"]
    try:
        obj = json.loads(migration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"{migration_path}: invalid JSON: {exc}"]
    if not isinstance(obj, dict):
        return {}, [f"{migration_path}: migration file must be object"]
    errors: list[str] = []
    migrations: dict[int, set[str]] = {}
    if obj.get("schema_version") != "QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_LOG_MIGRATIONS.v0":
        errors.append(f"{migration_path}: schema_version mismatch")
    log_ref = obj.get("log_ref")
    if isinstance(log_ref, str):
        expected = ROOT / log_ref
        if expected.resolve() != log_path.resolve():
            errors.append(f"{migration_path}: log_ref does not match validated log")
    else:
        errors.append(f"{migration_path}: log_ref must be string")
    entries = obj.get("migrations")
    if not isinstance(entries, list):
        errors.append(f"{migration_path}: migrations must be list")
        return migrations, errors
    allowed = {
        "supervisor_runtime_config_arg_missing",
        "scheduled_run_ok_false",
        "neural_worker_failed_missing_numpy",
    }
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"{migration_path}: migrations[{idx}] must be object")
            continue
        rows = entry.get("rows")
        relaxations = entry.get("allowed_relaxations")
        if not isinstance(rows, list) or not rows:
            errors.append(f"{migration_path}: migrations[{idx}].rows must be non-empty list")
            continue
        if not isinstance(relaxations, list) or not relaxations:
            errors.append(f"{migration_path}: migrations[{idx}].allowed_relaxations must be non-empty list")
            continue
        relaxation_set = set()
        for relaxation in relaxations:
            if not isinstance(relaxation, str) or relaxation not in allowed:
                errors.append(f"{migration_path}: migrations[{idx}] unknown relaxation {relaxation!r}")
                continue
            relaxation_set.add(relaxation)
        for row in rows:
            if not isinstance(row, int) or isinstance(row, bool) or row < 1:
                errors.append(f"{migration_path}: migrations[{idx}].rows contains invalid row")
                continue
            migrations.setdefault(row, set()).update(relaxation_set)
    return migrations, errors


def default_migration_for(path: Path) -> Path | None:
    if path.resolve() == LIVE_LOG.resolve() and DEFAULT_MIGRATIONS.exists():
        return DEFAULT_MIGRATIONS
    return None


def validate_scheduler_log(path: Path, migration_path: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    rows = 0
    checkpoints = 0
    migrations, migration_errors = load_migrations(path, migration_path)
    errors.extend(migration_errors)
    if not path.exists():
        return {"ok": False, "rows": 0, "errors": [f"{path}: file not found"]}

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            errors.append(f"line {line_no}: empty line")
            continue
        rows += 1
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if canonical_json(record) != raw:
            errors.append(f"line {line_no}: row is not canonical JSON")
        if not isinstance(record, dict):
            errors.append(f"line {line_no}: row must be object")
            continue
        if record.get("checkpoint_due") is True:
            checkpoints += 1
        errors.extend(validate_record(record, line_no, migrations))

    return {
        "ok": not errors,
        "rows": rows,
        "checkpoints": checkpoints,
        "migrated_rows": sorted(row for row in migrations if row <= rows),
        "errors": errors,
    }


def self_test() -> dict[str, Any]:
    fixtures = Path(__file__).resolve().parent / "fixtures"
    cases = [
        ("pass_scheduler_run.jsonl", True),
        ("fail_missing_supervisor_path_forwarding.jsonl", False),
        ("fail_failed_focused_check.jsonl", False),
    ]
    results = []
    ok = True
    for name, should_pass in cases:
        result = validate_scheduler_log(fixtures / name)
        passed = bool(result["ok"])
        case_ok = passed is should_pass
        ok = ok and case_ok
        results.append({"fixture": name, "ok": case_ok, "passed": passed, "result": result})
    return {"ok": ok, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--migrations", type=Path, default=None)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    paths = args.paths or sorted((Path(__file__).resolve().parent / "fixtures").glob("*.jsonl"))
    all_ok = True
    for path in paths:
        migration_path = args.migrations if args.migrations is not None else default_migration_for(path)
        result = validate_scheduler_log(path, migration_path)
        all_ok = all_ok and bool(result["ok"])
        print(json.dumps({"path": str(path), **result}, sort_keys=True))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
