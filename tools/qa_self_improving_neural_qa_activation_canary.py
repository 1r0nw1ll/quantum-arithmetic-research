#!/usr/bin/env python3
"""Run a SINQA activation canary against an isolated runtime config."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_activation_canary — exercises accepted config "
    "activation on an isolated canary runtime config with [527] validation"
)

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa/activation_canary")
DEFAULT_CONFIG = Path("results/self_improving_neural_qa/activation_canary/runtime/qa_general_ml_adapter_config_canary.json")
SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_ACTIVATION_CANARY.v0"
RESULT_HASH_DOMAIN = "qa_self_improving_neural_qa_activation_canary_v0"
FILE_HASH_DOMAIN = b"qa_general_ml_config_proposal_file_v0\x00"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(FILE_HASH_DOMAIN)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def load_tool_function(tool_name: str, function_name: str) -> Any:
    path = ROOT / "tools" / tool_name
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": f"_canary_{tool_name}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace[function_name]


def config_before_from_diff(diff: list[dict[str, Any]]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for item in diff:
        key = item.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("config diff key must be non-empty string")
        config[key] = item.get("before")
    return config


def write_canary_result(out_dir: Path, result: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_core = dict(result)
    result_core.pop("result_hash", None)
    result_with_hash = dict(result_core)
    result_with_hash["result_hash"] = domain_sha256(RESULT_HASH_DOMAIN, result_core)
    update_id = str(result["update_id"]).replace(".", "_")
    out_path = out_dir / f"{update_id}_activation_canary.json"
    out_path.write_text(canonical_json(result_with_hash), encoding="utf-8")
    return out_path


def run_canary(
    ledger: Path,
    out_dir: Path,
    config: Path,
    update_id: str | None,
    reset_canary_config: bool,
) -> dict[str, Any]:
    write_plan = load_tool_function("qa_self_improving_neural_qa_activation_gate.py", "write_plan")
    apply_plan = load_tool_function("qa_self_improving_neural_qa_apply_activation_plan.py", "apply_plan")
    validate_plan = load_tool_function(
        "../qa_alphageometry_ptolemy/qa_self_improving_neural_qa_activation_cert_v1/qa_self_improving_neural_qa_activation_cert_validate.py",
        "validate_plan",
    )

    plan_dir = out_dir / "plans"
    applied_dir = out_dir / "applied_plans"
    result_dir = out_dir / "results"
    plan_result = write_plan(ledger, plan_dir, update_id)
    plan_path = Path(plan_result["plan"])
    plan = load_json(plan_path)
    diff = plan.get("config_diff")
    if not isinstance(diff, list) or not diff:
        raise ValueError("activation plan config_diff must be non-empty list")

    config.parent.mkdir(parents=True, exist_ok=True)
    if reset_canary_config or not config.exists():
        config.write_text(canonical_json(config_before_from_diff(diff)), encoding="utf-8")

    applied = apply_plan(plan_path, config, applied_dir)
    applied_path = Path(applied["applied_plan"])
    validation = validate_plan(applied_path)
    final_config = load_json(config)
    result_core = {
        "schema_version": SCHEMA_VERSION,
        "ok": validation.get("ok") is True and applied.get("ok") is True,
        "update_id": plan["update_id"],
        "plan": repo_relative(plan_path),
        "plan_hash": plan["plan_hash"],
        "applied_plan": repo_relative(applied_path),
        "applied_plan_hash": file_hash(applied_path),
        "canary_config": repo_relative(config),
        "canary_config_hash": file_hash(config),
        "rollback_snapshot": repo_relative(Path(applied["rollback_snapshot"])),
        "rollback_snapshot_hash": file_hash(Path(applied["rollback_snapshot"])),
        "reset_canary_config": reset_canary_config,
        "runtime_mutated": False,
        "canary_runtime_mutated": True,
        "activation_status": "canary_applied_with_post_replay",
        "config_before": config_before_from_diff(diff),
        "config_after": final_config,
        "validation": validation,
    }
    result_path = write_canary_result(result_dir, result_core)
    result = load_json(result_path)
    return {"ok": result["ok"], "result": str(result_path), **result}


def self_test() -> dict[str, Any]:
    write_plan = load_tool_function("qa_self_improving_neural_qa_activation_gate.py", "write_plan")
    file_hash_fn = load_tool_function("qa_self_improving_neural_qa_activation_gate.py", "file_hash")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rollback = root / "rollback.json"
        rollback.write_text(canonical_json({"restore_config": {"adapter.rank": 8}}), encoding="utf-8")
        packet = {
            "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0",
            "update_id": "sinqa.v0.activation.canary.selftest",
            "base_model": {"name": "selftest", "kind": "classifier", "neural": True},
            "candidate": {
                "kind": "capacity_patch",
                "description": "self-test activation canary",
                "artifact_ref": "selftest_proposal.json",
                "config_patch": {
                    "objective": "self-test",
                    "activation_policy": "manual_after_cert",
                    "diff": [{"op": "increase", "key": "adapter.rank", "before": 8, "after": 12}],
                    "resource_bounds": {"max_parameters": 1000, "max_memory_mb": 128, "max_runtime_sec": 60},
                    "rollback": {"artifact_ref": str(rollback), "artifact_hash": file_hash_fn(rollback)},
                },
            },
            "evidence": {"source_replay_ref": "selftest_replay.json", "source_replay_hash": hashlib.sha256(b"source").hexdigest()},
            "replay_gate": {
                "new_failures_fixed": 1,
                "protected_cases_replayed": 2,
                "protected_cases_harmed": 0,
                "deterministic_replay": True,
                "trace_hash_before": hashlib.sha256(b"before").hexdigest(),
                "trace_hash_after": hashlib.sha256(b"after").hexdigest(),
            },
            "invariant_checks": [
                {"name": "zero_harm", "passed": True},
                {"name": "bounded_resource_delta", "passed": True},
                {"name": "rollback_available", "passed": True},
                {"name": "manual_activation_required", "passed": True},
            ],
            "promotion": {"decision": "accepted", "ledger_hash": hashlib.sha256(b"ledger").hexdigest()},
        }
        row = {
            "packet_hash": domain_sha256("qa_self_improving_neural_qa_packet_v0", packet),
            "packet": packet,
        }
        ledger = root / "ledger.jsonl"
        ledger.write_text(canonical_json(row) + "\n", encoding="utf-8")
        result = run_canary(
            ledger=ledger,
            out_dir=root / "canary",
            config=root / "canary_config.json",
            update_id="sinqa.v0.activation.canary.selftest",
            reset_canary_config=True,
        )
        ok = result["ok"] is True and result["config_after"]["adapter.rank"] == 12
        # Keep write_plan referenced in self-test so missing activation-gate imports fail loudly.
        assert write_plan is not None
        return {"ok": ok, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--update-id", default=None)
    parser.add_argument("--no-reset-canary-config", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    result = run_canary(
        ledger=repo_path(args.ledger),
        out_dir=repo_path(args.out_dir),
        config=repo_path(args.config),
        update_id=args.update_id,
        reset_canary_config=not args.no_reset_canary_config,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
