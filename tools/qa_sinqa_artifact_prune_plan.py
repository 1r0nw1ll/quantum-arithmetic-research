#!/usr/bin/env python3
"""Plan pruning/archival of repetitive SINQA generated artifacts.

Default mode is plan-only. It writes a canonical JSON plan and never deletes
or moves files unless --apply is explicitly supplied.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "sinqa_artifact_prune_plan — creates hash-bound, plan-first archive "
    "proposals for repetitive self-improving neural QA generated artifacts"
)

import argparse
import glob as globlib
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from qa_sinqa_artifact_prune_plan_validate import DEFAULT_REFERENCE_GLOBS, collect_referenced_paths, validate_plan


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "QA_SINQA_ARTIFACT_PRUNE_PLAN.v0"
PLAN_HASH_DOMAIN = "qa_sinqa_artifact_prune_plan_v0"
FILE_HASH_DOMAIN = b"qa_sinqa_artifact_file_v0\x00"
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa/prune_plans")
DEFAULT_ARCHIVE_DIR = Path("results/self_improving_neural_qa/artifact_archive")
DEFAULT_GLOBS = [
    "experiments/qa_ml/results_sinqa_neural_general_adapter*.json",
    "results/self_improving_neural_qa/general_ml/qa_general_ml_correction_replay_*.json",
    "results/self_improving_neural_qa/general_ml/qa_general_ml_rules_*.json",
    "results/self_improving_neural_qa/general_ml/qa_general_ml_sinqa_config_proposal_*.json",
    "results/self_improving_neural_qa/general_ml/qa_general_ml_config_rollback_*.json",
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


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def iter_glob_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for glob_pattern in patterns:
        pattern_path = Path(glob_pattern)
        pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
        paths.update(Path(match) for match in globlib.glob(pattern, recursive=True))
    return sorted(paths)


def rounded_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    return value


def stable_profile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): stable_profile(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [stable_profile(v) for v in obj]
    return rounded_float(obj)


def neural_signature(obj: dict[str, Any]) -> dict[str, Any] | None:
    schema = obj.get("schema")
    if not isinstance(schema, str) or not schema.startswith("QA_GENERAL_ML_NEURAL_"):
        return None
    settings = obj.get("settings") if isinstance(obj.get("settings"), dict) else {}
    task = obj.get("task") if isinstance(obj.get("task"), dict) else {}
    return {
        "artifact_kind": "neural_result",
        "schema": schema,
        "task_id": obj.get("task_id") or task.get("task_id"),
        "task_type": task.get("task_type"),
        "adapter_rank": settings.get("adapter_rank"),
        "steps": settings.get("steps"),
        "parameter_count": settings.get("parameter_count"),
        "test_moduli": settings.get("test_moduli"),
        "benchmark_sources": settings.get("benchmark_sources"),
    }


def replay_signature(obj: dict[str, Any]) -> dict[str, Any] | None:
    if obj.get("schema_version") != "QA_GENERAL_ML_CORRECTION_REPLAY.v0":
        return None
    paired_cases = obj.get("paired_cases")
    case_profile = []
    if isinstance(paired_cases, list):
        for case in paired_cases:
            if isinstance(case, dict):
                case_profile.append(
                    {
                        "case_id": case.get("case_id"),
                        "fixed": case.get("fixed"),
                        "harmed": case.get("harmed"),
                        "delta": rounded_float(case.get("delta")),
                    }
                )
    return {
        "artifact_kind": "general_ml_replay",
        "schema_version": obj.get("schema_version"),
        "dataset_slug": obj.get("dataset_slug"),
        "boundary": obj.get("boundary"),
        "baseline_control": obj.get("baseline_control"),
        "candidate_control": obj.get("candidate_control"),
        "fixed": obj.get("fixed_ensemble_errors"),
        "harmed": obj.get("harmed_correct_ensemble_rows"),
        "protected": obj.get("test_rows"),
        "accuracy_corrected": rounded_float(obj.get("accuracy_corrected")),
        "accuracy_ensemble": rounded_float(obj.get("accuracy_ensemble")),
        "case_profile": case_profile,
    }


def config_signature(obj: dict[str, Any]) -> dict[str, Any] | None:
    if obj.get("schema_version") != "QA_GENERAL_ML_SINQA_CONFIG_PROPOSAL.v0":
        return None
    patch = obj.get("config_patch") if isinstance(obj.get("config_patch"), dict) else {}
    return {
        "artifact_kind": "config_proposal",
        "schema_version": obj.get("schema_version"),
        "kind": obj.get("kind"),
        "diff": stable_profile(patch.get("diff")),
        "resource_bounds": stable_profile(patch.get("resource_bounds")),
        "fixed": obj.get("new_failures_fixed"),
        "harmed": obj.get("protected_cases_harmed"),
        "protected": obj.get("protected_cases_replayed"),
    }


def rollback_signature(path: Path, obj: dict[str, Any]) -> dict[str, Any] | None:
    if "qa_general_ml_config_rollback_" not in path.name:
        return None
    return {
        "artifact_kind": "config_rollback",
        "config": stable_profile(obj),
    }


def rules_signature(path: Path, obj: dict[str, Any]) -> dict[str, Any] | None:
    if "qa_general_ml_rules_" not in path.name:
        return None
    return {
        "artifact_kind": "general_ml_rules",
        "content": stable_profile(obj),
    }


def artifact_signature(path: Path, obj: dict[str, Any]) -> dict[str, Any] | None:
    return (
        neural_signature(obj)
        or replay_signature(obj)
        or config_signature(obj)
        or rollback_signature(path, obj)
        or rules_signature(path, obj)
    )


def artifact_record(path: Path, signature: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": repo_relative(path),
        "sha256": file_hash(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "signature_hash": domain_sha256("qa_sinqa_artifact_signature_v0", signature),
    }


def group_artifacts(paths: list[Path], min_group_size: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, dict[str, Any]] = {}
    counts = {"scanned": 0, "invalid_json": 0, "unsupported": 0, "supported": 0}
    for path in paths:
        counts["scanned"] += 1
        obj = load_json(path)
        if obj is None:
            counts["invalid_json"] += 1
            continue
        signature = artifact_signature(path, obj)
        if signature is None:
            counts["unsupported"] += 1
            continue
        counts["supported"] += 1
        sig_hash = domain_sha256("qa_sinqa_artifact_signature_v0", signature)
        groups.setdefault(sig_hash, {"signature_hash": sig_hash, "signature": signature, "artifacts": []})
        groups[sig_hash]["artifacts"].append(artifact_record(path, signature))
    prune_groups: list[dict[str, Any]] = []
    for group in groups.values():
        artifacts = sorted(group["artifacts"], key=lambda item: (item["mtime_ns"], item["path"]))
        if len(artifacts) < min_group_size:
            continue
        keep = artifacts[-1]
        prune = artifacts[:-1]
        prune_groups.append(
            {
                "signature_hash": group["signature_hash"],
                "signature": group["signature"],
                "count": len(artifacts),
                "keep": keep,
                "prune_candidates": prune,
            }
        )
    prune_groups.sort(key=lambda item: (-len(item["prune_candidates"]), item["signature_hash"]))
    return prune_groups, counts


def exclude_referenced_candidates(groups: list[dict[str, Any]], reference_globs: list[str]) -> tuple[list[dict[str, Any]], int]:
    references = collect_referenced_paths(reference_globs)
    filtered_groups = []
    excluded = 0
    for group in groups:
        candidates = []
        for artifact in group.get("prune_candidates", []):
            artifact_path = artifact.get("path")
            if not isinstance(artifact_path, str):
                continue
            rel = repo_relative(resolve_path(Path(artifact_path)))
            if rel in references:
                excluded += 1
                continue
            candidates.append(artifact)
        if not candidates:
            continue
        next_group = dict(group)
        next_group["prune_candidates"] = candidates
        next_group["count"] = len(candidates) + 1
        filtered_groups.append(next_group)
    filtered_groups.sort(key=lambda item: (-len(item["prune_candidates"]), item["signature_hash"]))
    return filtered_groups, excluded


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    paths = iter_glob_paths(args.glob)
    groups, counts = group_artifacts(paths, args.min_group_size)
    referenced_excluded = 0
    if getattr(args, "exclude_referenced_candidates", False):
        groups, referenced_excluded = exclude_referenced_candidates(groups, args.reference_glob or DEFAULT_REFERENCE_GLOBS)
    if args.max_groups >= 0:
        groups = groups[: args.max_groups]
    total_prune = sum(len(group["prune_candidates"]) for group in groups)
    plan_core = {
        "schema_version": SCHEMA_VERSION,
        "mode": "plan_only" if not args.apply else "apply_archive",
        "archive_dir": repo_relative(resolve_path(args.archive_dir)),
        "source_globs": args.glob,
        "criteria": {
            "min_group_size": args.min_group_size,
            "max_groups": args.max_groups,
            "keep_policy": "latest_mtime_per_signature",
            "apply_requires_explicit_flag": True,
            "exclude_referenced_candidates": bool(getattr(args, "exclude_referenced_candidates", False)),
        },
        "scan": {**counts, "referenced_excluded": referenced_excluded},
        "group_count": len(groups),
        "prune_candidate_count": total_prune,
        "groups": groups,
    }
    plan = dict(plan_core)
    plan["plan_hash"] = domain_sha256(PLAN_HASH_DOMAIN, plan_core)
    return plan


def plan_output_path(out_dir: Path, plan: dict[str, Any]) -> Path:
    return out_dir / f"qa_sinqa_artifact_prune_plan_{plan['plan_hash'][:16]}.json"


def apply_plan(plan: dict[str, Any]) -> dict[str, Any]:
    archive_dir = resolve_path(Path(plan["archive_dir"]))
    moved = []
    for group in plan.get("groups", []):
        signature_hash = group["signature_hash"]
        for artifact in group.get("prune_candidates", []):
            src = resolve_path(Path(artifact["path"]))
            if not src.exists():
                moved.append({"source": artifact["path"], "ok": False, "reason": "source_missing"})
                continue
            if file_hash(src) != artifact["sha256"]:
                moved.append({"source": artifact["path"], "ok": False, "reason": "source_hash_changed"})
                continue
            dst = archive_dir / signature_hash / Path(artifact["path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                moved.append({"source": artifact["path"], "archive": repo_relative(dst), "ok": False, "reason": "archive_exists"})
                continue
            shutil.move(str(src), str(dst))
            moved.append({"source": artifact["path"], "archive": repo_relative(dst), "ok": True})
    return {"ok": all(item["ok"] for item in moved), "moved": moved}


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for run_id in [1, 2, 3]:
            write_json(
                root / f"results_sinqa_neural_general_adapter_qa_residue_mod3_run{run_id:04d}.json",
                {
                    "schema": "QA_GENERAL_ML_NEURAL_ADAPTER_QA_RESIDUE.v0",
                    "task_id": "qa_residue_mod3",
                    "settings": {
                        "adapter_rank": 12,
                        "steps": 60,
                        "parameter_count": 1155,
                        "test_moduli": [33, 45],
                    },
                    "task": {"task_id": "qa_residue_mod3", "task_type": "qa_residue_mod3"},
                },
            )
        args = argparse.Namespace(
            glob=[str(root / "results_sinqa_neural_general_adapter*.json")],
            out_dir=root / "plans",
            archive_dir=root / "archive",
            min_group_size=2,
            max_groups=-1,
            apply=False,
            reference_glob=[],
            exclude_referenced_candidates=False,
        )
        plan = build_plan(args)
        ok = plan["group_count"] == 1 and plan["prune_candidate_count"] == 2 and plan["groups"][0]["keep"]["path"].endswith("run0003.json")
        return {"ok": ok, "plan": plan}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=-1)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reference-glob", action="append", default=None)
    parser.add_argument("--exclude-referenced-candidates", action="store_true")
    parser.add_argument("--allow-referenced-archive", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.glob is None:
        args.glob = list(DEFAULT_GLOBS)

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.min_group_size < 2:
        print("error: --min-group-size must be >= 2")
        return 2
    plan = build_plan(args)
    out_path = plan_output_path(resolve_path(args.out_dir), plan)
    write_json(out_path, plan)
    result: dict[str, Any] = {
        "ok": True,
        "plan": repo_relative(out_path),
        "plan_hash": plan["plan_hash"],
        "group_count": plan["group_count"],
        "prune_candidate_count": plan["prune_candidate_count"],
        "applied": False,
    }
    if args.apply:
        validation = validate_plan(out_path, args.reference_glob or DEFAULT_REFERENCE_GLOBS, args.allow_referenced_archive)
        result["pre_apply_validation"] = validation
        if validation["ok"]:
            apply_result = apply_plan(plan)
            result["applied"] = True
            result["apply"] = apply_result
            result["ok"] = apply_result["ok"]
        else:
            result["applied"] = False
            result["ok"] = False
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
