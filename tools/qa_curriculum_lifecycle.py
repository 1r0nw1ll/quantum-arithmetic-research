#!/usr/bin/env python3
"""Manage SINQA curriculum proposal lifecycle artifacts."""

from __future__ import annotations

QA_COMPLIANCE = (
    "curriculum_lifecycle — archives rejected SINQA curriculum proposals with "
    "hash-bound provenance while keeping the active proposal queue actionable"
)

import argparse
import glob as globlib
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_LIFECYCLE.v0"
DEFAULT_REGISTRY = Path("results/self_improving_neural_qa/curriculum_registry.json")
DEFAULT_ACTIVE_PROPOSAL_GLOB = "results/self_improving_neural_qa/curriculum_proposals/*.json"
DEFAULT_ARCHIVE_GLOB = "results/self_improving_neural_qa/curriculum_archive/**/qa_curriculum_lifecycle_*.json"
DEFAULT_ARCHIVE_DIR = Path("results/self_improving_neural_qa/curriculum_archive")
PROPOSAL_HASH_DOMAIN = b"qa_curriculum_discovery_file_v0\x00"
PLAN_HASH_DOMAIN = b"qa_curriculum_activation_plan_file_v0\x00"
ROLLBACK_HASH_DOMAIN = b"qa_curriculum_registry_file_v0\x00"
LIFECYCLE_HASH_DOMAIN = "qa_curriculum_lifecycle_record_v0"
REJECTED_STATUSES = {"rejected_duplicate_source", "rejected_duplicate_task", "superseded"}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + canonical_json(obj).encode("utf-8")).hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def file_hash(path: Path, domain: bytes) -> str:
    h = hashlib.sha256()
    h.update(domain)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def iter_glob_paths(glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
    return sorted(Path(match) for match in globlib.glob(pattern, recursive=True))


def load_proposal_validator() -> Any:
    path = ROOT / "tools" / "qa_curriculum_proposal_validate.py"
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": "_curriculum_proposal_validator"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["validate_proposal"]


def copy_or_move(src: Path, dst: Path, dry_run: bool) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = {"source": repo_relative(src), "archive": repo_relative(dst), "moved": not dry_run}
    if dry_run:
        return result
    if dst.exists():
        raise FileExistsError(f"archive target exists: {dst}")
    shutil.move(str(src), str(dst))
    return result


def archive_rejected_duplicate_source(args: argparse.Namespace) -> dict[str, Any]:
    proposal_path = resolve_path(args.proposal)
    registry_path = resolve_path(args.registry)
    proposal = load_json(proposal_path)
    validate_proposal = load_proposal_validator()
    validation = validate_proposal(proposal_path, registry_path)
    if validation.get("ok") is True:
        raise ValueError("proposal is still valid; refusing to archive as rejected")
    errors = list(validation.get("errors") or [])
    if not any("duplicates active registry source" in str(error) for error in errors):
        raise ValueError(f"proposal rejection is not duplicate-source: {errors}")
    task_id = proposal.get("task", {}).get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("proposal task.task_id must be non-empty string")

    archive_dir = resolve_path(args.archive_dir) / "rejected_duplicate_source" / task_id
    archived_proposal = archive_dir / proposal_path.name
    proposal_hash_before = file_hash(proposal_path, PROPOSAL_HASH_DOMAIN)
    moves = [copy_or_move(proposal_path, archived_proposal, args.dry_run)]

    archived_plan = None
    archived_rollback = None
    plan_hash_before = None
    rollback_hash_before = None
    if args.plan is not None:
        plan_path = resolve_path(args.plan)
        plan = load_json(plan_path)
        if plan.get("task_id") != task_id:
            raise ValueError("plan task_id does not match proposal task_id")
        plan_hash_before = file_hash(plan_path, PLAN_HASH_DOMAIN)
        archived_plan = archive_dir / plan_path.name
        moves.append(copy_or_move(plan_path, archived_plan, args.dry_run))
        rollback_ref = plan.get("rollback", {}).get("artifact_ref")
        if isinstance(rollback_ref, str) and rollback_ref:
            rollback_path = resolve_path(Path(rollback_ref))
            if rollback_path.exists():
                rollback_hash_before = file_hash(rollback_path, ROLLBACK_HASH_DOMAIN)
                archived_rollback = archive_dir / rollback_path.name
                moves.append(copy_or_move(rollback_path, archived_rollback, args.dry_run))

    record_core = {
        "schema_version": SCHEMA_VERSION,
        "status": "rejected_duplicate_source",
        "task_id": task_id,
        "reason": "proposal source_glob/source_hash duplicates an active curriculum registry task",
        "registry_ref": repo_relative(registry_path),
        "validation_errors": errors,
        "proposal": {
            "original_ref": repo_relative(proposal_path),
            "archive_ref": repo_relative(archived_proposal),
            "hash_before": proposal_hash_before,
            "hash_after": None if args.dry_run else file_hash(archived_proposal, PROPOSAL_HASH_DOMAIN),
        },
        "activation_plan": None,
        "rollback": None,
    }
    if archived_plan is not None:
        record_core["activation_plan"] = {
            "original_ref": repo_relative(resolve_path(args.plan)),
            "archive_ref": repo_relative(archived_plan),
            "hash_before": plan_hash_before,
            "hash_after": None if args.dry_run else file_hash(archived_plan, PLAN_HASH_DOMAIN),
            "status": "stale_not_applyable",
        }
    if archived_rollback is not None:
        record_core["rollback"] = {
            "archive_ref": repo_relative(archived_rollback),
            "hash_before": rollback_hash_before,
            "hash_after": None if args.dry_run else file_hash(archived_rollback, ROLLBACK_HASH_DOMAIN),
        }
    record = dict(record_core)
    record["record_hash"] = domain_sha256(LIFECYCLE_HASH_DOMAIN, record_core)
    record_path = archive_dir / f"qa_curriculum_lifecycle_{task_id}.json"
    if not args.dry_run:
        write_json(record_path, record)
    return {
        "ok": True,
        "dry_run": args.dry_run,
        "status": "rejected_duplicate_source",
        "task_id": task_id,
        "record": repo_relative(record_path),
        "moves": moves,
    }


def validate_lifecycle_record(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
        record = json.loads(raw)
    except Exception as exc:
        return {"ok": False, "errors": [f"{path}: invalid JSON: {exc}"]}
    if canonical_json(record) + "\n" != raw and canonical_json(record) != raw:
        errors.append("lifecycle record is not canonical JSON")
    if not isinstance(record, dict):
        return {"ok": False, "errors": ["lifecycle record must be object"]}
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if record.get("status") not in REJECTED_STATUSES:
        errors.append("status is not an allowed lifecycle terminal status")
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        errors.append("task_id must be non-empty string")
    proposal = record.get("proposal")
    if not isinstance(proposal, dict):
        errors.append("proposal must be object")
    else:
        archive_ref = proposal.get("archive_ref")
        hash_after = proposal.get("hash_after")
        if not isinstance(archive_ref, str) or not archive_ref:
            errors.append("proposal.archive_ref must be non-empty string")
        else:
            archive_path = resolve_path(Path(archive_ref))
            if not archive_path.exists():
                errors.append("proposal.archive_ref does not exist")
            elif file_hash(archive_path, PROPOSAL_HASH_DOMAIN) != hash_after:
                errors.append("proposal.hash_after mismatch")
    plan = record.get("activation_plan")
    if plan is not None:
        if not isinstance(plan, dict):
            errors.append("activation_plan must be object or null")
        else:
            plan_ref = plan.get("archive_ref")
            plan_hash = plan.get("hash_after")
            if isinstance(plan_ref, str) and plan_ref:
                plan_path = resolve_path(Path(plan_ref))
                if not plan_path.exists():
                    errors.append("activation_plan.archive_ref does not exist")
                elif file_hash(plan_path, PLAN_HASH_DOMAIN) != plan_hash:
                    errors.append("activation_plan.hash_after mismatch")
            if plan.get("status") != "stale_not_applyable":
                errors.append("activation_plan.status must be stale_not_applyable")
    rollback = record.get("rollback")
    if rollback is not None:
        if not isinstance(rollback, dict):
            errors.append("rollback must be object or null")
        else:
            rollback_ref = rollback.get("archive_ref")
            rollback_hash = rollback.get("hash_after")
            if isinstance(rollback_ref, str) and rollback_ref:
                rollback_path = resolve_path(Path(rollback_ref))
                if not rollback_path.exists():
                    errors.append("rollback.archive_ref does not exist")
                elif file_hash(rollback_path, ROLLBACK_HASH_DOMAIN) != rollback_hash:
                    errors.append("rollback.hash_after mismatch")
    record_hash = record.get("record_hash")
    if not isinstance(record_hash, str) or len(record_hash) != 64:
        errors.append("record_hash must be 64-char hex")
    else:
        core = {key: val for key, val in record.items() if key != "record_hash"}
        if domain_sha256(LIFECYCLE_HASH_DOMAIN, core) != record_hash:
            errors.append("record_hash mismatch")
    return {"ok": not errors, "errors": errors}


def validate_lifecycle_paths(paths: list[Path]) -> dict[str, Any]:
    records = []
    ok = True
    for path in paths:
        result = validate_lifecycle_record(path)
        ok = ok and bool(result["ok"])
        records.append({"path": str(path), **result})
    return {"ok": ok, "count": len(paths), "results": records}


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.json"
        write_json(source, {"ok": True})
        source_hash = file_hash(source, PROPOSAL_HASH_DOMAIN)
        registry = root / "registry.json"
        write_json(
            registry,
            {
                "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_REGISTRY.v0",
                "tasks": [{"task_id": "active", "enabled": True, "source_glob": str(source), "source_hash": source_hash}],
            },
        )
        proposal = root / "proposal.json"
        write_json(
            proposal,
            {
                "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_PROPOSAL.v0",
                "proposal_kind": "curriculum_task",
                "activation_policy": "manual_after_validation",
                "task": {
                    "task_id": "duplicate_source",
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
                    "duplicate_existing_source": False,
                },
            },
        )
        args = argparse.Namespace(
            proposal=proposal,
            plan=None,
            registry=registry,
            archive_dir=root / "archive",
            dry_run=False,
        )
        archived = archive_rejected_duplicate_source(args)
        active_clean = not proposal.exists()
        validation = validate_lifecycle_paths(
            iter_glob_paths(str(root / "archive" / "**" / "qa_curriculum_lifecycle_*.json"))
        )
        ok = archived["ok"] is True and active_clean and validation["ok"] is True and validation["count"] == 1
        return {"ok": ok, "archive": archived, "validation": validation}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    archive = sub.add_parser("reject-duplicate-source")
    archive.add_argument("--proposal", type=Path, required=True)
    archive.add_argument("--plan", type=Path)
    archive.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    archive.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    archive.add_argument("--dry-run", action="store_true")

    validate_active = sub.add_parser("validate-active")
    validate_active.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    validate_active.add_argument("--proposal-glob", default=DEFAULT_ACTIVE_PROPOSAL_GLOB)

    validate_archive = sub.add_parser("validate-archive")
    validate_archive.add_argument("--archive-glob", default=DEFAULT_ARCHIVE_GLOB)

    sub.add_parser("self-test")
    args = parser.parse_args()

    if args.cmd == "self-test":
        result = self_test()
    elif args.cmd == "reject-duplicate-source":
        result = archive_rejected_duplicate_source(args)
    elif args.cmd == "validate-active":
        validate_proposal = load_proposal_validator()
        paths = iter_glob_paths(args.proposal_glob)
        results = []
        ok = True
        for path in paths:
            item = validate_proposal(path, args.registry)
            ok = ok and bool(item["ok"])
            results.append({"path": str(path), **item})
        result = {"ok": ok, "count": len(paths), "results": results}
    elif args.cmd == "validate-archive":
        result = validate_lifecycle_paths(iter_glob_paths(args.archive_glob))
    else:
        raise AssertionError(args.cmd)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
