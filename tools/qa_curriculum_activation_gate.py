#!/usr/bin/env python3
"""Plan and optionally apply validated SINQA curriculum task proposals.

Default behavior is non-mutating: write an activation plan with registry diff
and rollback hash. Use --apply to append the disabled proposal task to the
active registry.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "curriculum_activation_gate — creates rollback-bound activation plans for "
    "validated SINQA curriculum proposals and applies only with explicit flag"
)

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = Path("results/self_improving_neural_qa/curriculum_registry.json")
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa/curriculum_activation_plans")
SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_ACTIVATION_PLAN.v0"
REGISTRY_SCHEMA = "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_REGISTRY.v0"
PLAN_HASH_DOMAIN = "qa_curriculum_activation_plan_v0"
REGISTRY_HASH_DOMAIN = b"qa_curriculum_registry_file_v0\x00"
PROPOSAL_HASH_DOMAIN = b"qa_curriculum_discovery_file_v0\x00"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def file_hash(path: Path, domain: bytes) -> str:
    h = hashlib.sha256()
    h.update(domain)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path: Path) -> Path:
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


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def load_proposal_validator() -> Any:
    path = ROOT / "tools" / "qa_curriculum_proposal_validate.py"
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": "_curriculum_proposal_validator"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["validate_proposal"]


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        errors.append("registry schema_version mismatch")
    tasks = registry.get("tasks")
    if not isinstance(tasks, list):
        errors.append("registry tasks must be list")
        return errors
    seen: set[str] = set()
    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            errors.append(f"task {idx}: must be object")
            continue
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"task {idx}: task_id must be non-empty string")
            continue
        if task_id in seen:
            errors.append(f"task {idx}: duplicate task_id {task_id}")
        seen.add(task_id)
        if task.get("enabled") is not True:
            errors.append(f"task {idx}: activated registry task must have enabled=true")
    return errors


def existing_task_ids(registry: dict[str, Any]) -> set[str]:
    return {
        task["task_id"]
        for task in registry.get("tasks", [])
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }


def existing_task_sources(registry: dict[str, Any]) -> tuple[set[str], set[str]]:
    source_globs: set[str] = set()
    source_hashes: set[str] = set()
    for task in registry.get("tasks", []):
        if not isinstance(task, dict):
            continue
        source_glob = task.get("source_glob")
        source_hash = task.get("source_hash")
        if isinstance(source_glob, str) and source_glob:
            source_globs.add(source_glob)
        if isinstance(source_hash, str) and source_hash:
            source_hashes.add(source_hash)
    return source_globs, source_hashes


def activated_task_from_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    task = dict(proposal["task"])
    task["enabled"] = True
    task["activation_source_proposal_hash"] = proposal["evidence"]["source_hash"]
    return task


def build_updated_registry(registry: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    task = activated_task_from_proposal(proposal)
    if task["task_id"] in existing_task_ids(registry):
        raise ValueError(f"task_id already active: {task['task_id']}")
    source_globs, source_hashes = existing_task_sources(registry)
    source_glob = task.get("source_glob")
    source_hash = task.get("source_hash")
    if isinstance(source_glob, str) and source_glob and source_glob in source_globs:
        raise ValueError(f"source_glob already active: {source_glob}")
    if isinstance(source_hash, str) and source_hash and source_hash in source_hashes:
        raise ValueError(f"source_hash already active: {source_hash}")
    updated = dict(registry)
    updated["tasks"] = list(registry.get("tasks", [])) + [task]
    return updated


def plan_path_for(out_dir: Path, proposal: dict[str, Any]) -> Path:
    task_id = proposal["task"]["task_id"]
    return out_dir / f"qa_curriculum_activation_plan_{task_id}.json"


def build_plan(proposal_path: Path, registry_path: Path, out_dir: Path) -> dict[str, Any]:
    validate_proposal = load_proposal_validator()
    validation = validate_proposal(proposal_path, registry_path)
    if validation.get("ok") is not True:
        raise ValueError(f"curriculum proposal validation failed: {validation.get('errors')}")
    proposal = load_json(proposal_path)
    registry = load_json(registry_path)
    updated = build_updated_registry(registry, proposal)
    registry_errors = validate_registry(updated)
    if registry_errors:
        raise ValueError(f"updated registry invalid: {registry_errors}")
    rollback_path = out_dir / f"qa_curriculum_registry_rollback_before_{proposal['task']['task_id']}.json"
    plan_core = {
        "schema_version": SCHEMA_VERSION,
        "activation_status": "planned_not_applied",
        "activation_policy": "manual_after_validation",
        "manual_approval_required": True,
        "registry_mutated": False,
        "proposal_ref": repo_relative(proposal_path),
        "proposal_hash": file_hash(proposal_path, PROPOSAL_HASH_DOMAIN),
        "registry_ref": repo_relative(registry_path),
        "registry_hash_before": file_hash(registry_path, REGISTRY_HASH_DOMAIN),
        "task_id": proposal["task"]["task_id"],
        "registry_diff": {
            "op": "append_task",
            "task": activated_task_from_proposal(proposal),
        },
        "rollback": {
            "artifact_ref": repo_relative(rollback_path),
            "artifact_hash": file_hash(registry_path, REGISTRY_HASH_DOMAIN),
        },
        "checks": [
            {"name": "proposal_validated", "passed": True},
            {"name": "manual_activation_required", "passed": True},
            {"name": "rollback_available", "passed": True},
            {"name": "registry_not_mutated_by_plan", "passed": True},
            {"name": "no_duplicate_task_id", "passed": True},
            {"name": "no_duplicate_source", "passed": True},
        ],
    }
    plan = dict(plan_core)
    plan["plan_hash"] = domain_sha256(PLAN_HASH_DOMAIN, plan_core)
    return plan


def write_plan(proposal_path: Path, registry_path: Path, out_dir: Path) -> dict[str, Any]:
    proposal = load_json(proposal_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollback_path = out_dir / f"qa_curriculum_registry_rollback_before_{proposal['task']['task_id']}.json"
    rollback_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    plan = build_plan(proposal_path, registry_path, out_dir)
    plan_path = plan_path_for(out_dir, proposal)
    write_json(plan_path, plan)
    return {
        "ok": True,
        "applied": False,
        "plan": str(plan_path),
        "rollback": str(rollback_path),
        "task_id": proposal["task"]["task_id"],
        "plan_hash": plan["plan_hash"],
    }


def apply_plan(plan_path: Path, registry_path: Path, out_dir: Path) -> dict[str, Any]:
    plan = load_json(plan_path)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("activation plan schema_version mismatch")
    if plan.get("activation_status") != "planned_not_applied":
        raise ValueError("input plan must be planned_not_applied")
    if plan.get("registry_mutated") is not False:
        raise ValueError("input plan must be non-mutating")
    current_hash = file_hash(registry_path, REGISTRY_HASH_DOMAIN)
    if current_hash != plan.get("registry_hash_before"):
        raise ValueError("registry hash changed since activation plan was created")
    registry = load_json(registry_path)
    task = plan["registry_diff"]["task"]
    if task["task_id"] in existing_task_ids(registry):
        raise ValueError(f"task_id already active: {task['task_id']}")
    proposal_like = {"task": dict(task), "evidence": {"source_hash": task.get("activation_source_proposal_hash")}}
    updated = build_updated_registry(registry, proposal_like)
    errors = validate_registry(updated)
    if errors:
        raise ValueError(f"updated registry invalid: {errors}")
    write_json(registry_path, updated)
    applied_core = {key: val for key, val in plan.items() if key != "plan_hash"}
    applied_core["activation_status"] = "applied"
    applied_core["registry_mutated"] = True
    applied_core["registry_hash_after"] = file_hash(registry_path, REGISTRY_HASH_DOMAIN)
    applied_core["checks"] = list(applied_core.get("checks", [])) + [
        {"name": "registry_mutated_only_with_apply", "passed": True},
        {"name": "updated_registry_valid", "passed": True},
    ]
    applied = dict(applied_core)
    applied["plan_hash"] = domain_sha256(PLAN_HASH_DOMAIN, applied_core)
    out_dir.mkdir(parents=True, exist_ok=True)
    applied_path = out_dir / f"qa_curriculum_activation_plan_{plan['task_id']}_applied.json"
    write_json(applied_path, applied)
    return {
        "ok": True,
        "applied": True,
        "applied_plan": str(applied_path),
        "registry": str(registry_path),
        "task_id": plan["task_id"],
        "registry_hash_after": applied["registry_hash_after"],
    }


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.json"
        write_json(source, {"ok": True})
        registry = root / "registry.json"
        write_json(registry, {"schema_version": REGISTRY_SCHEMA, "selection_policy": "round_robin_enabled", "tasks": []})
        source_hash = hashlib.sha256(b"qa_curriculum_discovery_file_v0\x00" + source.read_bytes()).hexdigest()
        proposal = root / "proposal.json"
        write_json(
            proposal,
            {
                "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_PROPOSAL.v0",
                "proposal_kind": "curriculum_task",
                "activation_policy": "manual_after_validation",
                "task": {
                    "task_id": "discovered_selftest",
                    "task_type": "qa_ml_benchmark_selector",
                    "enabled": False,
                    "metric": "valid_rate_mean",
                    "source_glob": str(source),
                    "source_hash": source_hash,
                },
                "evidence": {
                    "source_ref": str(source),
                    "source_hash": source_hash,
                    "case_count": 2,
                    "min_controls_per_case": 2,
                    "non_hsi": True,
                    "duplicate_existing_task": False,
                },
            },
        )
        plan_result = write_plan(proposal, registry, root / "plans")
        before = load_json(registry)
        apply_result = apply_plan(Path(plan_result["plan"]), registry, root / "plans")
        after = load_json(registry)
        duplicate_failed = False
        try:
            apply_plan(Path(plan_result["plan"]), registry, root / "plans")
        except ValueError:
            duplicate_failed = True
        ok = (
            plan_result["ok"] is True
            and plan_result["applied"] is False
            and len(before["tasks"]) == 0
            and apply_result["ok"] is True
            and len(after["tasks"]) == 1
            and after["tasks"][0]["enabled"] is True
            and duplicate_failed
        )
        return {"ok": ok, "plan": plan_result, "apply": apply_result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=False)
    parser.add_argument("--plan", type=Path, required=False)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    registry = resolve_path(args.registry)
    out_dir = resolve_path(args.out_dir)
    if args.apply:
        if args.plan is None:
            parser.error("--apply requires --plan")
        result = apply_plan(resolve_path(args.plan), registry, out_dir)
    else:
        if args.proposal is None:
            parser.error("--proposal is required unless --apply or --self-test is used")
        result = write_plan(resolve_path(args.proposal), registry, out_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
