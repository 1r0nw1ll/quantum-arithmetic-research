#!/usr/bin/env python3
"""Report SINQA learning trends over ledger, scheduler, and neural artifacts."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_trends — read-only trend report for SINQA "
    "learning progress, safety events, task coverage, and repetition pressure"
)

import argparse
import glob as globlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
DEFAULT_SCHEDULER_LOG = Path("results/self_improving_neural_qa/scheduler_runs.jsonl")
DEFAULT_NEURAL_GLOB = "experiments/qa_ml/results_sinqa_neural_general_adapter*.json"
DEFAULT_WINDOW = 20


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return [], [f"{path}: file not found"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_no, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
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


def iter_glob_paths(glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
    return sorted(Path(match) for match in globlib.glob(pattern, recursive=True))


def packet_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    packet = row.get("packet")
    return packet if isinstance(packet, dict) else None


def decision(packet: dict[str, Any]) -> str:
    promotion = packet.get("promotion")
    value = promotion.get("decision") if isinstance(promotion, dict) else None
    return value if value in {"accepted", "rejected"} else "unknown"


def replay_gate(packet: dict[str, Any]) -> dict[str, Any]:
    gate = packet.get("replay_gate")
    return gate if isinstance(gate, dict) else {}


def candidate(packet: dict[str, Any]) -> dict[str, Any]:
    cand = packet.get("candidate")
    return cand if isinstance(cand, dict) else {}


def evidence(packet: dict[str, Any]) -> dict[str, Any]:
    ev = packet.get("evidence")
    return ev if isinstance(ev, dict) else {}


def classify_domain(packet: dict[str, Any]) -> str:
    cand = candidate(packet)
    ev = evidence(packet)
    text = " ".join(
        str(part).lower()
        for part in [
            cand.get("artifact_ref"),
            cand.get("description"),
            ev.get("source_replay_ref"),
        ]
        if part is not None
    )
    if "self_improving_neural_qa/general_ml" in text or "general_ml" in text:
        return "general_ml"
    if "hsi" in text or "salinas" in text or "paviau" in text or "pavia" in text or "indian_pines" in text:
        return "hsi"
    if cand.get("kind") in {"capacity_patch", "configuration_patch"}:
        return "config"
    return "other"


def candidate_kind(packet: dict[str, Any]) -> str:
    kind = candidate(packet).get("kind")
    return kind if isinstance(kind, str) else "unknown"


def int_field(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def ledger_trend(rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    packets = [packet for row in rows if (packet := packet_from_row(row)) is not None]
    recent = packets[-window:] if window > 0 else packets

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        decisions = [decision(packet) for packet in items]
        gates = [replay_gate(packet) for packet in items]
        source_hashes = [evidence(packet).get("source_replay_hash") for packet in items]
        typed_hashes = [value for value in source_hashes if isinstance(value, str)]
        duplicate_sources = len(typed_hashes) - len(set(typed_hashes))
        return {
            "rows": len(items),
            "accepted": decisions.count("accepted"),
            "rejected": decisions.count("rejected"),
            "fixed": sum(int_field(gate, "new_failures_fixed") for gate in gates),
            "protected_replayed": sum(int_field(gate, "protected_cases_replayed") for gate in gates),
            "protected_harmed": sum(int_field(gate, "protected_cases_harmed") for gate in gates),
            "harm_attempts": sum(1 for gate in gates if int_field(gate, "protected_cases_harmed") > 0),
            "duplicate_source_hashes": duplicate_sources,
            "domains": count_values([classify_domain(packet) for packet in items]),
            "candidate_kinds": count_values([candidate_kind(packet) for packet in items]),
        }

    return {"all": summarize(packets), "recent": summarize(recent), "window": window}


def scheduler_trend(rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    recent = rows[-window:] if window > 0 else rows
    stop_reasons = [str(row.get("stop_reason")) for row in recent if row.get("stop_reason") is not None]
    neural_tasks = []
    for row in recent:
        worker = row.get("neural_worker")
        stdout_json = worker.get("stdout_json") if isinstance(worker, dict) else None
        if isinstance(stdout_json, dict) and isinstance(stdout_json.get("task_id"), str):
            neural_tasks.append(stdout_json["task_id"])
    return {
        "rows": len(rows),
        "recent_rows": len(recent),
        "latest_run": rows[-1].get("run") if rows else None,
        "recent_ok": sum(1 for row in recent if row.get("ok") is True),
        "recent_stop_reasons": count_values(stop_reasons),
        "recent_neural_tasks": count_values(neural_tasks),
    }


def neural_artifact_trend(glob_pattern: str) -> dict[str, Any]:
    paths = iter_glob_paths(glob_pattern)
    tasks: list[str] = []
    schemas: list[str] = []
    parameter_counts: list[int] = []
    for path in paths:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        task_id = obj.get("task_id") or (obj.get("task") if isinstance(obj.get("task"), dict) else {}).get("task_id")
        if isinstance(task_id, str):
            tasks.append(task_id)
        schema = obj.get("schema")
        if isinstance(schema, str):
            schemas.append(schema)
        settings = obj.get("settings") if isinstance(obj.get("settings"), dict) else obj
        param_count = settings.get("parameter_count") if isinstance(settings, dict) else None
        if isinstance(param_count, int) and not isinstance(param_count, bool):
            parameter_counts.append(param_count)
    return {
        "artifacts": len(paths),
        "tasks": count_values(tasks),
        "schemas": count_values(schemas),
        "parameter_count_min": min(parameter_counts) if parameter_counts else None,
        "parameter_count_max": max(parameter_counts) if parameter_counts else None,
    }


def build_trends(args: argparse.Namespace) -> dict[str, Any]:
    ledger_rows, ledger_errors = read_jsonl(args.ledger)
    scheduler_rows, scheduler_errors = read_jsonl(args.scheduler_log)
    return {
        "ok": not ledger_errors and not scheduler_errors,
        "errors": ledger_errors + scheduler_errors,
        "ledger": ledger_trend(ledger_rows, args.window),
        "scheduler": scheduler_trend(scheduler_rows, args.window),
        "neural_artifacts": neural_artifact_trend(args.neural_glob),
    }


def print_text(report: dict[str, Any]) -> None:
    ledger = report["ledger"]
    scheduler = report["scheduler"]
    neural = report["neural_artifacts"]
    recent = ledger["recent"]
    all_rows = ledger["all"]
    print(f"SINQA trends: {'OK' if report['ok'] else 'ATTENTION'}")
    print(
        f"ledger all: rows={all_rows['rows']} accepted={all_rows['accepted']} "
        f"rejected={all_rows['rejected']} harm_attempts={all_rows['harm_attempts']}"
    )
    print(
        f"ledger recent({ledger['window']}): accepted={recent['accepted']} rejected={recent['rejected']} "
        f"fixed={recent['fixed']} harmed={recent['protected_harmed']} domains={recent['domains']}"
    )
    print(
        f"scheduler: rows={scheduler['rows']} latest_run={scheduler['latest_run']} "
        f"recent_ok={scheduler['recent_ok']} stops={scheduler['recent_stop_reasons']}"
    )
    print(f"neural artifacts: count={neural['artifacts']} tasks={neural['tasks']}")


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = root / "ledger.jsonl"
        scheduler = root / "scheduler.jsonl"
        neural = root / "results_sinqa_neural_general_adapter_demo.json"
        packets = [
            {
                "packet": {
                    "candidate": {"kind": "correction_rule", "artifact_ref": "results/self_improving_neural_qa/general_ml/rules.json"},
                    "evidence": {"source_replay_hash": "a" * 64, "source_replay_ref": "results/self_improving_neural_qa/general_ml/replay.json"},
                    "promotion": {"decision": "accepted"},
                    "replay_gate": {"new_failures_fixed": 2, "protected_cases_replayed": 4, "protected_cases_harmed": 0},
                }
            },
            {
                "packet": {
                    "candidate": {"kind": "correction_rule", "artifact_ref": "results/qa_hsi_demo.json"},
                    "evidence": {"source_replay_hash": "b" * 64, "source_replay_ref": "results/qa_hsi_demo_replay.json"},
                    "promotion": {"decision": "rejected"},
                    "replay_gate": {"new_failures_fixed": 1, "protected_cases_replayed": 10, "protected_cases_harmed": 1},
                }
            },
        ]
        ledger.write_text("\n".join(canonical_json(row) for row in packets) + "\n", encoding="utf-8")
        scheduler.write_text(
            canonical_json(
                {
                    "run": 1,
                    "ok": True,
                    "stop_reason": "commit_caps_reached",
                    "neural_worker": {"stdout_json": {"task_id": "demo_task"}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        neural.write_text(
            canonical_json({"schema": "QA_GENERAL_ML_NEURAL_DEMO.v0", "task_id": "demo_task", "settings": {"parameter_count": 12}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(ledger=ledger, scheduler_log=scheduler, neural_glob=str(root / "results_sinqa*.json"), window=1)
        report = build_trends(args)
        ok = (
            report["ok"] is True
            and report["ledger"]["all"]["accepted"] == 1
            and report["ledger"]["all"]["harm_attempts"] == 1
            and report["ledger"]["recent"]["rejected"] == 1
            and report["scheduler"]["recent_neural_tasks"] == {"demo_task": 1}
            and report["neural_artifacts"]["parameter_count_max"] == 12
        )
        return {"ok": ok, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--scheduler-log", type=Path, default=DEFAULT_SCHEDULER_LOG)
    parser.add_argument("--neural-glob", default=DEFAULT_NEURAL_GLOB)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.window < 1:
        print(json.dumps({"ok": False, "errors": ["--window must be >= 1"]}, sort_keys=True))
        return 2
    report = build_trends(args)
    if args.text:
        print_text(report)
    else:
        print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
