#!/usr/bin/env python3
"""Validate SINQA artifact prune plans before any archive operation."""

from __future__ import annotations

QA_COMPLIANCE = (
    "sinqa_artifact_prune_plan_validate — verifies hash-bound SINQA prune "
    "plans and fails closed when candidates are still provenance references"
)

import argparse
import glob as globlib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "QA_SINQA_ARTIFACT_PRUNE_PLAN.v0"
PLAN_HASH_DOMAIN = "qa_sinqa_artifact_prune_plan_v0"
FILE_HASH_DOMAIN = b"qa_sinqa_artifact_file_v0\x00"
DEFAULT_REFERENCE_GLOBS = [
    "results/self_improving_neural_qa/ledger.jsonl",
    "results/self_improving_neural_qa/loop_transcript.jsonl",
    "results/self_improving_neural_qa/supervisor_state.json",
    "results/self_improving_neural_qa/sinqa_v0_*.json",
]


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


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(FILE_HASH_DOMAIN)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_glob_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for glob_pattern in patterns:
        pattern_path = Path(glob_pattern)
        pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
        paths.update(Path(match) for match in globlib.glob(pattern, recursive=True))
    return sorted(paths)


def load_json_or_jsonl(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = []
        for line in raw.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    return [json.loads(raw)]


def path_forms(value: str) -> set[str]:
    forms = {value}
    path = Path(value)
    if path.is_absolute():
        forms.add(repo_relative(path))
    else:
        forms.add(repo_relative(resolve_path(path)))
    return forms


def collect_referenced_paths(reference_globs: list[str]) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}

    def visit(value: Any, source: Path) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item, source)
            return
        if isinstance(value, list):
            for item in value:
                visit(item, source)
            return
        if not isinstance(value, str):
            return
        if "/" not in value or not (value.endswith(".json") or value.endswith(".jsonl")):
            return
        for form in path_forms(value):
            references.setdefault(form, []).append(repo_relative(source))

    for path in iter_glob_paths(reference_globs):
        try:
            objects = load_json_or_jsonl(path)
        except Exception:
            continue
        for obj in objects:
            visit(obj, path)
    return references


def validate_plan(path: Path, reference_globs: list[str] | None = None, allow_referenced: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    obj = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {"ok": False, "errors": ["plan root must be object"], "warnings": []}
    if obj.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    plan_hash = obj.get("plan_hash")
    if not isinstance(plan_hash, str) or len(plan_hash) != 64:
        errors.append("plan_hash must be 64 hex characters")
    else:
        core = dict(obj)
        core.pop("plan_hash", None)
        computed = domain_sha256(PLAN_HASH_DOMAIN, core)
        if computed != plan_hash:
            errors.append(f"plan_hash mismatch: expected {plan_hash}, computed {computed}")

    groups = obj.get("groups")
    if not isinstance(groups, list):
        errors.append("groups must be a list")
        groups = []
    if obj.get("group_count") != len(groups):
        errors.append("group_count does not match groups length")

    references = collect_referenced_paths(reference_globs or DEFAULT_REFERENCE_GLOBS)
    referenced_candidates: list[dict[str, Any]] = []
    candidate_count = 0
    keep_paths: set[str] = set()
    candidate_paths: set[str] = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"group {group_index}: must be object")
            continue
        keep = group.get("keep")
        candidates = group.get("prune_candidates")
        if not isinstance(keep, dict):
            errors.append(f"group {group_index}: keep must be object")
            continue
        if not isinstance(candidates, list):
            errors.append(f"group {group_index}: prune_candidates must be list")
            continue
        candidate_count += len(candidates)
        keep_path = keep.get("path")
        if isinstance(keep_path, str):
            keep_paths.add(repo_relative(resolve_path(Path(keep_path))))
        for label, artifact in [("keep", keep), *[(f"candidate {idx}", item) for idx, item in enumerate(candidates)]]:
            if not isinstance(artifact, dict):
                errors.append(f"group {group_index}: {label} must be object")
                continue
            artifact_path = artifact.get("path")
            artifact_sha = artifact.get("sha256")
            if not isinstance(artifact_path, str):
                errors.append(f"group {group_index}: {label} path missing")
                continue
            rel = repo_relative(resolve_path(Path(artifact_path)))
            resolved = resolve_path(Path(artifact_path))
            if not resolved.exists():
                errors.append(f"group {group_index}: {label} source missing: {artifact_path}")
                continue
            if not isinstance(artifact_sha, str) or file_hash(resolved) != artifact_sha:
                errors.append(f"group {group_index}: {label} sha256 mismatch: {artifact_path}")
            if label.startswith("candidate"):
                if rel in candidate_paths:
                    errors.append(f"duplicate prune candidate: {artifact_path}")
                candidate_paths.add(rel)
                ref_sources = references.get(rel, [])
                if ref_sources:
                    referenced_candidates.append(
                        {"path": rel, "reference_count": len(ref_sources), "reference_sources": sorted(set(ref_sources))[:8]}
                    )
        group_count = group.get("count")
        if isinstance(group_count, int) and group_count != len(candidates) + 1:
            errors.append(f"group {group_index}: count does not equal keep plus candidates")

    if obj.get("prune_candidate_count") != candidate_count:
        errors.append("prune_candidate_count does not match candidate list length")
    overlap = keep_paths.intersection(candidate_paths)
    if overlap:
        errors.append(f"keep path also listed as prune candidate: {sorted(overlap)[0]}")
    if referenced_candidates and not allow_referenced:
        errors.append(f"{len(referenced_candidates)} prune candidates are still referenced by SINQA provenance")
    elif referenced_candidates:
        warnings.append(f"{len(referenced_candidates)} referenced candidates allowed by override")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "plan": repo_relative(resolve_path(path)),
        "group_count": len(groups),
        "prune_candidate_count": candidate_count,
        "referenced_candidate_count": len(referenced_candidates),
        "referenced_candidates": referenced_candidates[:20],
    }


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact = root / "artifact.json"
        keep = root / "keep.json"
        artifact.write_text(canonical_json({"x": 1}) + "\n", encoding="utf-8")
        keep.write_text(canonical_json({"x": 2}) + "\n", encoding="utf-8")
        plan_core = {
            "schema_version": SCHEMA_VERSION,
            "mode": "plan_only",
            "archive_dir": "archive",
            "source_globs": [str(root / "*.json")],
            "criteria": {"min_group_size": 2, "max_groups": -1, "keep_policy": "latest_mtime_per_signature", "apply_requires_explicit_flag": True},
            "scan": {"scanned": 2, "invalid_json": 0, "unsupported": 0, "supported": 2},
            "group_count": 1,
            "prune_candidate_count": 1,
            "groups": [
                {
                    "signature_hash": "a" * 64,
                    "signature": {"artifact_kind": "fixture"},
                    "count": 2,
                    "keep": {"path": str(keep), "sha256": file_hash(keep), "size_bytes": keep.stat().st_size, "mtime_ns": keep.stat().st_mtime_ns, "signature_hash": "a" * 64},
                    "prune_candidates": [
                        {"path": str(artifact), "sha256": file_hash(artifact), "size_bytes": artifact.stat().st_size, "mtime_ns": artifact.stat().st_mtime_ns, "signature_hash": "a" * 64}
                    ],
                }
            ],
        }
        plan = dict(plan_core)
        plan["plan_hash"] = domain_sha256(PLAN_HASH_DOMAIN, plan_core)
        plan_path = root / "plan.json"
        plan_path.write_text(canonical_json(plan) + "\n", encoding="utf-8")
        ref_path = root / "ledger.jsonl"
        ref_path.write_text(canonical_json({"candidate": {"artifact_ref": str(artifact)}}) + "\n", encoding="utf-8")
        blocked = validate_plan(plan_path, [str(ref_path)])
        allowed = validate_plan(plan_path, [str(ref_path)], allow_referenced=True)
        return {"ok": blocked["ok"] is False and allowed["ok"] is True, "blocked": blocked, "allowed": allowed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, nargs="?")
    parser.add_argument("--reference-glob", action="append", default=None)
    parser.add_argument("--allow-referenced", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.plan is None:
        print(json.dumps({"ok": False, "errors": ["plan path required"]}, sort_keys=True))
        return 2
    result = validate_plan(args.plan, args.reference_glob or DEFAULT_REFERENCE_GLOBS, args.allow_referenced)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
