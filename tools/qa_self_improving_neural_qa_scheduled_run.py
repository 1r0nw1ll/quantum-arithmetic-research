#!/usr/bin/env python3
"""Scheduled checkpoint runner for Self-Improving Neural QA.

This entrypoint is designed for launchd/cron. Each invocation runs a small
supervisor batch, validates focused certs, updates scheduler state, and only
runs the heavier meta gate at configured checkpoints.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_scheduled_run — scheduled capped learner runner "
    "with focused validation and periodic meta checkpoints"
)

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path("results/self_improving_neural_qa/scheduler_state.json")
DEFAULT_LOG = Path("results/self_improving_neural_qa/scheduler_runs.jsonl")
DEFAULT_REPLAY_GLOB = "results/**/*correction_replay*.json"
DEFAULT_CONFIG_GLOB = "results/**/*sinqa_config_proposal*.json"
DEFAULT_RUNTIME_CONFIG = "results/self_improving_neural_qa/runtime/qa_general_ml_adapter_config.json"
DEFAULT_CURRICULUM_REGISTRY = "results/self_improving_neural_qa/curriculum_registry.json"
DEFAULT_PRODUCER_SOURCE_GLOB = "experiments/qa_ml/results_*.json"
GENERAL_ML_NEURAL_WORKER = Path("tools/qa_general_ml_neural_worker.py")
GENERAL_ML_REPLAY_WORKER = Path("tools/qa_general_ml_replay_worker.py")
GENERAL_ML_CAPACITY_WORKER = Path("tools/qa_general_ml_capacity_proposal_worker.py")
CURRICULUM_DISCOVERY_WORKER = Path("tools/qa_curriculum_discovery_worker.py")
CURRICULUM_LIFECYCLE = Path("tools/qa_curriculum_lifecycle.py")
ARTIFACT_PRUNE_PLANNER = Path("tools/qa_sinqa_artifact_prune_plan.py")
ANTIFORGETTING_VALIDATOR = Path("tools/qa_self_improving_neural_qa_antiforgetting.py")
SUPERVISOR = Path("tools/qa_self_improving_neural_qa_supervisor.py")
LEDGER_VALIDATOR = Path("tools/qa_self_improving_neural_qa_ledger_validate.py")
TRANSCRIPT_VALIDATOR = Path("tools/qa_self_improving_neural_qa_transcript_validate.py")
CERT_524 = Path(
    "qa_alphageometry_ptolemy/qa_self_improving_neural_qa_cert_v1/"
    "qa_self_improving_neural_qa_cert_validate.py"
)
CERT_525 = Path(
    "qa_alphageometry_ptolemy/qa_self_improving_neural_qa_loop_cert_v1/"
    "qa_self_improving_neural_qa_loop_cert_validate.py"
)
META_VALIDATOR = Path("qa_alphageometry_ptolemy/qa_meta_validator.py")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_STATE.v0",
            "runs": 0,
            "last_ok": None,
            "last_stop_reason": None,
            "last_unix": None,
        }
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(obj) + "\n")


def run_command(argv: list[str], timeout: int) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    parsed = None
    if stdout.startswith("{") and stdout.endswith("}"):
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "duration_sec": round(time.time() - started, 3),
        "stdout_json": parsed,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def supervisor_argv(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(SUPERVISOR),
        "--cycles",
        "1",
        "--rounds-per-cycle",
        str(args.rounds_per_run),
        "--glob",
        args.glob,
        "--config-glob",
        args.config_glob,
        "--runtime-config",
        args.runtime_config,
        "--max-total-accepted",
        str(args.max_accepted_per_run),
        "--max-total-rejected",
        str(args.max_rejected_per_run),
        "--max-accepted-per-cycle",
        str(args.max_accepted_per_run),
        "--max-rejected-per-cycle",
        str(args.max_rejected_per_run),
        "--ledger",
        args.ledger,
        "--out-dir",
        args.out_dir,
        "--transcript",
        args.transcript,
        "--state",
        args.supervisor_state,
        "--heartbeat",
        args.supervisor_heartbeat,
        "--lock",
        args.supervisor_lock,
    ]


def producer_argv(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(GENERAL_ML_REPLAY_WORKER),
        "--source-glob",
        args.producer_source_glob,
        "--out-dir",
        args.producer_out_dir,
        "--ledger",
        args.ledger,
        "--runtime-config",
        args.runtime_config,
        "--max-emits",
        str(args.producer_max_emits),
    ]


def neural_worker_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        str(GENERAL_ML_NEURAL_WORKER),
        "--out-dir",
        args.neural_worker_out_dir,
        "--state",
        args.neural_worker_state,
        "--runtime-config",
        args.runtime_config,
        "--curriculum-registry",
        args.curriculum_registry,
    ]
    if args.neural_worker_max_steps is not None:
        argv.extend(["--max-steps", str(args.neural_worker_max_steps)])
    return argv


def capacity_worker_argv(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(GENERAL_ML_CAPACITY_WORKER),
        "--ledger",
        args.ledger,
        "--out-dir",
        args.producer_out_dir,
        "--runtime-config",
        args.runtime_config,
        "--max-emits",
        str(args.capacity_producer_max_emits),
    ]


def curriculum_discovery_argv(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(CURRICULUM_DISCOVERY_WORKER),
        "--source-glob",
        args.producer_source_glob,
        "--registry",
        args.curriculum_registry,
        "--out-dir",
        args.curriculum_proposal_out_dir,
        "--max-emits",
        str(args.curriculum_discovery_max_emits),
    ]


def focused_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        run_command([sys.executable, str(LEDGER_VALIDATOR), args.ledger], args.validator_timeout),
        run_command([sys.executable, str(TRANSCRIPT_VALIDATOR), args.transcript], args.validator_timeout),
        run_command(
            [
                sys.executable,
                str(CURRICULUM_LIFECYCLE),
                "validate-active",
                "--registry",
                args.curriculum_registry,
                "--proposal-glob",
                str(Path(args.curriculum_proposal_out_dir) / "*.json"),
            ],
            args.validator_timeout,
        ),
        run_command([sys.executable, str(CURRICULUM_LIFECYCLE), "validate-archive"], args.validator_timeout),
        run_command([sys.executable, str(CERT_524), "--self-test"], args.validator_timeout),
        run_command([sys.executable, str(CERT_525), "--self-test"], args.validator_timeout),
        run_command(
            [
                sys.executable,
                str(ARTIFACT_PRUNE_PLANNER),
                "--max-groups",
                "20",
                "--exclude-referenced-candidates",
                "--validate-plan",
            ],
            args.validator_timeout,
        ),
        run_command(
            [
                sys.executable,
                str(ANTIFORGETTING_VALIDATOR),
                "--ledger",
                args.ledger,
                "--max-items",
                "32",
                "--min-accepted",
                "0",
            ],
            args.validator_timeout,
        ),
    ]


def run_scheduled(args: argparse.Namespace) -> dict[str, Any]:
    state = load_json(args.state)
    run_no = int(state.get("runs", 0)) + 1
    neural_worker = None
    if args.train_general_ml_neural:
        neural_worker = run_command(neural_worker_argv(args), args.neural_worker_timeout)
    producer = None
    if args.produce_general_ml_replays:
        producer = run_command(producer_argv(args), args.producer_timeout)
    capacity_producer = None
    if args.produce_capacity_proposals:
        capacity_producer = run_command(capacity_worker_argv(args), args.capacity_producer_timeout)
    curriculum_discovery = None
    if args.discover_curriculum:
        curriculum_discovery = run_command(curriculum_discovery_argv(args), args.curriculum_discovery_timeout)
    supervisor = run_command(supervisor_argv(args), args.supervisor_timeout)
    checks = focused_checks(args)
    checkpoint_due = args.meta_every > 0 and run_no % args.meta_every == 0
    meta = None
    if checkpoint_due:
        meta_args = [sys.executable, str(META_VALIDATOR)]
        if args.meta_fast:
            meta_args.append("--fast")
        meta = run_command(meta_args, args.meta_timeout)

    ok = (
        (neural_worker is None or neural_worker["ok"])
        and
        (producer is None or producer["ok"])
        and
        (capacity_producer is None or capacity_producer["ok"])
        and
        (curriculum_discovery is None or curriculum_discovery["ok"])
        and supervisor["ok"]
        and all(check["ok"] for check in checks)
        and (meta is None or meta["ok"])
    )
    stop_reason = None
    supervisor_json = supervisor.get("stdout_json")
    if isinstance(supervisor_json, dict):
        stop_reason = supervisor_json.get("stop_reason")
    record = {
        "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_RUN.v0",
        "run": run_no,
        "ok": ok,
        "unix_time": int(time.time()),
        "neural_worker": neural_worker,
        "producer": producer,
        "capacity_producer": capacity_producer,
        "curriculum_discovery": curriculum_discovery,
        "supervisor": supervisor,
        "focused_checks": checks,
        "checkpoint_due": checkpoint_due,
        "meta": meta,
        "stop_reason": stop_reason,
    }
    append_jsonl(args.log, record)
    write_json(
        args.state,
        {
            "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_STATE.v0",
            "runs": run_no,
            "last_ok": ok,
            "last_stop_reason": stop_reason,
            "last_unix": record["unix_time"],
            "last_log": str(args.log),
        },
    )
    return record


def self_test() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        args = argparse.Namespace(
            state=root / "scheduler_state.json",
            log=root / "scheduler_runs.jsonl",
            ledger=str(root / "ledger.jsonl"),
            out_dir=str(root / "packets"),
            transcript=str(root / "transcript.jsonl"),
            supervisor_state=str(root / "supervisor_state.json"),
            supervisor_heartbeat=str(root / "supervisor_heartbeat.json"),
            supervisor_lock=str(root / "supervisor.lock"),
            glob="results/qa_hsi_salinas*correction_replay*.json",
            config_glob=str(root / "*sinqa_config_proposal*.json"),
            runtime_config=str(root / "missing_runtime_config.json"),
            curriculum_registry=str(root / "missing_curriculum_registry.json"),
            rounds_per_run=2,
            max_accepted_per_run=1,
            max_rejected_per_run=1,
            meta_every=1,
            meta_fast=True,
            train_general_ml_neural=True,
            neural_worker_out_dir=str(root / "qa_ml"),
            neural_worker_state=str(root / "neural_worker_state.json"),
            neural_worker_max_steps=60,
            neural_worker_timeout=120,
            produce_general_ml_replays=True,
            producer_source_glob=str(root / "qa_ml" / "results_*.json"),
            producer_out_dir=str(root / "general_ml"),
            producer_max_emits=1,
            producer_timeout=120,
            produce_capacity_proposals=True,
            capacity_producer_max_emits=1,
            capacity_producer_timeout=120,
            discover_curriculum=True,
            curriculum_proposal_out_dir=str(root / "curriculum_proposals"),
            curriculum_discovery_max_emits=1,
            curriculum_discovery_timeout=120,
            supervisor_timeout=120,
            validator_timeout=120,
            meta_timeout=120,
        )
        result = run_scheduled(args)
        ok = (
            result["ok"] is True
            and result["checkpoint_due"] is True
            and result["meta"] is not None
            and args.state.exists()
            and args.log.exists()
        )
        return {"ok": ok, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--ledger", default="results/self_improving_neural_qa/ledger.jsonl")
    parser.add_argument("--out-dir", default="results/self_improving_neural_qa")
    parser.add_argument("--transcript", default="results/self_improving_neural_qa/loop_transcript.jsonl")
    parser.add_argument("--glob", default=DEFAULT_REPLAY_GLOB)
    parser.add_argument("--config-glob", default=DEFAULT_CONFIG_GLOB)
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--curriculum-registry", default=DEFAULT_CURRICULUM_REGISTRY)
    parser.add_argument("--produce-general-ml-replays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--producer-source-glob", default=DEFAULT_PRODUCER_SOURCE_GLOB)
    parser.add_argument("--producer-out-dir", default="results/self_improving_neural_qa/general_ml")
    parser.add_argument("--producer-max-emits", type=int, default=1)
    parser.add_argument("--produce-capacity-proposals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capacity-producer-max-emits", type=int, default=1)
    parser.add_argument("--discover-curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curriculum-proposal-out-dir", default="results/self_improving_neural_qa/curriculum_proposals")
    parser.add_argument("--curriculum-discovery-max-emits", type=int, default=1)
    parser.add_argument("--train-general-ml-neural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neural-worker-out-dir", default="experiments/qa_ml")
    parser.add_argument("--neural-worker-state", default="results/self_improving_neural_qa/general_ml/neural_worker_state.json")
    parser.add_argument("--neural-worker-max-steps", type=int, default=None)
    parser.add_argument("--supervisor-state", default="results/self_improving_neural_qa/supervisor_state.json")
    parser.add_argument("--supervisor-heartbeat", default="results/self_improving_neural_qa/supervisor_heartbeat.json")
    parser.add_argument("--supervisor-lock", default="results/self_improving_neural_qa/supervisor.lock")
    parser.add_argument("--rounds-per-run", type=int, default=2)
    parser.add_argument("--max-accepted-per-run", type=int, default=1)
    parser.add_argument("--max-rejected-per-run", type=int, default=1)
    parser.add_argument("--meta-every", type=int, default=12)
    parser.add_argument("--meta-fast", action="store_true")
    parser.add_argument("--supervisor-timeout", type=int, default=600)
    parser.add_argument("--producer-timeout", type=int, default=180)
    parser.add_argument("--capacity-producer-timeout", type=int, default=180)
    parser.add_argument("--curriculum-discovery-timeout", type=int, default=180)
    parser.add_argument("--neural-worker-timeout", type=int, default=180)
    parser.add_argument("--validator-timeout", type=int, default=180)
    parser.add_argument("--meta-timeout", type=int, default=1800)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.rounds_per_run < 1:
        print("error: --rounds-per-run must be >= 1", file=sys.stderr)
        return 2
    if args.max_accepted_per_run < 0 or args.max_rejected_per_run < 0:
        print("error: per-run caps must be non-negative", file=sys.stderr)
        return 2
    if args.producer_max_emits < 0:
        print("error: --producer-max-emits must be non-negative", file=sys.stderr)
        return 2
    if args.capacity_producer_max_emits < 0:
        print("error: --capacity-producer-max-emits must be non-negative", file=sys.stderr)
        return 2
    if args.curriculum_discovery_max_emits < 0:
        print("error: --curriculum-discovery-max-emits must be non-negative", file=sys.stderr)
        return 2
    if args.neural_worker_max_steps is not None and args.neural_worker_max_steps < 1:
        print("error: --neural-worker-max-steps must be >= 1", file=sys.stderr)
        return 2
    result = run_scheduled(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
