#!/usr/bin/env python3
"""Validate SINQA curriculum task proposal artifacts."""

from __future__ import annotations

QA_COMPLIANCE = (
    "curriculum_proposal_validator — gates proposed self-improving neural QA "
    "curriculum tasks before registry activation"
)

import argparse
import glob as globlib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_PROPOSAL.v0"
REGISTRY_SCHEMA = "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_REGISTRY.v0"
DEFAULT_REGISTRY = Path("results/self_improving_neural_qa/curriculum_registry.json")
DEFAULT_PROPOSAL_GLOB = "results/self_improving_neural_qa/curriculum_proposals/*.json"
ALLOWED_TASK_TYPES = {"qa_ml_benchmark_selector"}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(b"qa_curriculum_discovery_file_v0\x00")
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_glob_paths(glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
    return sorted(Path(match) for match in globlib.glob(pattern, recursive=True))


def existing_task_ids(registry_path: Path) -> set[str]:
    try:
        registry = load_json(resolve_path(registry_path))
    except Exception:
        return set()
    if not isinstance(registry, dict) or registry.get("schema_version") != REGISTRY_SCHEMA:
        return set()
    ids = set()
    for task in registry.get("tasks", []):
        if isinstance(task, dict) and isinstance(task.get("task_id"), str):
            ids.add(task["task_id"])
    return ids


def existing_task_sources(registry_path: Path) -> tuple[set[str], set[str]]:
    try:
        registry = load_json(resolve_path(registry_path))
    except Exception:
        return set(), set()
    if not isinstance(registry, dict) or registry.get("schema_version") != REGISTRY_SCHEMA:
        return set(), set()
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


def validate_proposal(path: Path, registry_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
        proposal = json.loads(raw)
    except Exception as exc:
        return {"ok": False, "errors": [f"{path}: invalid JSON: {exc}"]}
    if canonical_json(proposal) + "\n" != raw and canonical_json(proposal) != raw:
        errors.append("proposal is not canonical JSON")
    if not isinstance(proposal, dict):
        return {"ok": False, "errors": ["proposal must be object"]}
    if proposal.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if proposal.get("proposal_kind") != "curriculum_task":
        errors.append("proposal_kind must be curriculum_task")
    if proposal.get("activation_policy") != "manual_after_validation":
        errors.append("activation_policy must be manual_after_validation")
    task = proposal.get("task")
    evidence = proposal.get("evidence")
    if not isinstance(task, dict):
        errors.append("task must be object")
        task = {}
    if not isinstance(evidence, dict):
        errors.append("evidence must be object")
        evidence = {}

    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        errors.append("task.task_id must be non-empty string")
    elif task_id in existing_task_ids(registry_path):
        errors.append("task.task_id duplicates active registry")
    if task.get("task_type") not in ALLOWED_TASK_TYPES:
        errors.append("task.task_type is not allowed")
    if task.get("enabled") is not False:
        errors.append("task.enabled must be false in proposal")
    if not isinstance(task.get("metric"), str) or not task["metric"]:
        errors.append("task.metric must be non-empty string")
    task_source_glob = task.get("source_glob")
    if not isinstance(task_source_glob, str) or not task_source_glob:
        errors.append("task.source_glob must be non-empty string")
    else:
        active_source_globs, active_source_hashes = existing_task_sources(registry_path)
        if task_source_glob in active_source_globs:
            errors.append("task.source_glob duplicates active registry source")

    if evidence.get("non_hsi") is not True:
        errors.append("evidence.non_hsi must be true")
    if not isinstance(evidence.get("case_count"), int) or evidence["case_count"] < 2:
        errors.append("evidence.case_count must be >= 2")
    if not isinstance(evidence.get("min_controls_per_case"), int) or evidence["min_controls_per_case"] < 2:
        errors.append("evidence.min_controls_per_case must be >= 2")
    if evidence.get("duplicate_existing_task") is not False:
        errors.append("evidence.duplicate_existing_task must be false")
    if evidence.get("duplicate_existing_source") is True:
        errors.append("evidence.duplicate_existing_source must not be true")
    source_ref = evidence.get("source_ref")
    source_hash = evidence.get("source_hash")
    if not isinstance(source_ref, str) or not source_ref:
        errors.append("evidence.source_ref must be non-empty string")
    else:
        source_path = resolve_path(Path(source_ref))
        if not source_path.exists():
            errors.append("evidence.source_ref does not exist")
        elif file_hash(source_path) != source_hash:
            errors.append("evidence.source_hash mismatch")
    if source_hash != task.get("source_hash"):
        errors.append("task.source_hash must match evidence.source_hash")
    elif isinstance(source_hash, str) and source_hash:
        _active_source_globs, active_source_hashes = existing_task_sources(registry_path)
        if source_hash in active_source_hashes:
            errors.append("task.source_hash duplicates active registry source hash")
    return {"ok": not errors, "errors": errors}


def validate_paths(paths: list[Path], registry_path: Path) -> dict[str, Any]:
    results = []
    ok = True
    for path in paths:
        result = validate_proposal(path, registry_path)
        ok = ok and bool(result["ok"])
        results.append({"path": str(path), **result})
    return {"ok": ok, "count": len(paths), "results": results}


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.json"
        source.write_text(canonical_json({"ok": True}), encoding="utf-8")
        registry = root / "registry.json"
        registry.write_text(canonical_json({"schema_version": REGISTRY_SCHEMA, "tasks": []}), encoding="utf-8")
        proposal = root / "proposal.json"
        obj = {
            "schema_version": SCHEMA_VERSION,
            "proposal_kind": "curriculum_task",
            "activation_policy": "manual_after_validation",
            "task": {
                "task_id": "discovered_demo",
                "task_type": "qa_ml_benchmark_selector",
                "enabled": False,
                "metric": "valid_rate_mean",
                "source_glob": str(source),
                "source_hash": file_hash(source),
            },
            "evidence": {
                "source_ref": str(source),
                "source_hash": file_hash(source),
                "case_count": 2,
                "min_controls_per_case": 2,
                "non_hsi": True,
                "duplicate_existing_task": False,
                "duplicate_existing_source": False,
            },
        }
        proposal.write_text(canonical_json(obj), encoding="utf-8")
        valid = validate_proposal(proposal, registry)
        dup_registry = root / "dup_registry.json"
        dup_registry.write_text(
            canonical_json({"schema_version": REGISTRY_SCHEMA, "tasks": [{"task_id": "discovered_demo"}]}),
            encoding="utf-8",
        )
        duplicate = validate_proposal(proposal, dup_registry)
        dup_source_registry = root / "dup_source_registry.json"
        dup_source_registry.write_text(
            canonical_json(
                {
                    "schema_version": REGISTRY_SCHEMA,
                    "tasks": [
                        {
                            "task_id": "already_active_other_id",
                            "source_glob": str(source),
                            "source_hash": file_hash(source),
                            "enabled": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        duplicate_source = validate_proposal(proposal, dup_source_registry)
        flagged = dict(obj)
        flagged["evidence"] = dict(obj["evidence"])
        flagged["evidence"]["duplicate_existing_source"] = True
        flagged_proposal = root / "flagged_proposal.json"
        flagged_proposal.write_text(canonical_json(flagged), encoding="utf-8")
        flagged_duplicate_source = validate_proposal(flagged_proposal, registry)
        ok = (
            valid["ok"] is True
            and duplicate["ok"] is False
            and duplicate_source["ok"] is False
            and flagged_duplicate_source["ok"] is False
        )
        return {
            "ok": ok,
            "valid": valid,
            "duplicate": duplicate,
            "duplicate_source": duplicate_source,
            "flagged_duplicate_source": flagged_duplicate_source,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    paths = [Path(p) for p in args.paths] if args.paths else iter_glob_paths(DEFAULT_PROPOSAL_GLOB)
    result = validate_paths(paths, args.registry)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
