#!/usr/bin/env python3
"""Discover bounded SINQA curriculum task proposals from QA-ML result files.

The worker emits proposal artifacts only. It does not edit the active
curriculum registry.
"""

from __future__ import annotations

QA_COMPLIANCE = (
    "curriculum_discovery_worker — proposes bounded non-HSI curriculum tasks "
    "from paired-control QA-ML benchmark summaries without mutating registry"
)

import argparse
import glob as globlib
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_GLOB = "experiments/qa_ml/results_*.json"
DEFAULT_REGISTRY = Path("results/self_improving_neural_qa/curriculum_registry.json")
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa/curriculum_proposals")
SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_PROPOSAL.v0"
REGISTRY_SCHEMA = "QA_SELF_IMPROVING_NEURAL_QA_CURRICULUM_REGISTRY.v0"
NEURAL_SCHEMA_PREFIX = "QA_GENERAL_ML_NEURAL_"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(b"qa_curriculum_discovery_file_v0\x00")
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


def iter_glob_paths(glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    pattern = str(pattern_path if pattern_path.is_absolute() else ROOT / glob_pattern)
    return sorted(Path(match) for match in globlib.glob(pattern, recursive=True))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "task"


def existing_task_ids(registry_path: Path) -> set[str]:
    registry = load_json(resolve_path(registry_path)) or {}
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        return set()
    ids: set[str] = set()
    for task in registry.get("tasks", []):
        if isinstance(task, dict) and isinstance(task.get("task_id"), str):
            ids.add(task["task_id"])
    return ids


def existing_task_sources(registry_path: Path) -> tuple[set[str], set[str]]:
    registry = load_json(resolve_path(registry_path)) or {}
    if registry.get("schema_version") != REGISTRY_SCHEMA:
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


def paired_cases(result: dict[str, Any]) -> tuple[int, int, set[str]]:
    moduli_results = result.get("moduli_results")
    if not isinstance(moduli_results, dict):
        return 0, 0, set()
    cases = 0
    min_controls = 0
    controls_seen: set[str] = set()
    for case in moduli_results.values():
        if not isinstance(case, dict) or not isinstance(case.get("controls"), dict):
            continue
        controls = [str(name) for name, val in case["controls"].items() if isinstance(val, dict)]
        if len(controls) < 2:
            continue
        cases += 1
        min_controls = len(controls) if min_controls == 0 else min(min_controls, len(controls))
        controls_seen.update(controls)
    return cases, min_controls, controls_seen


def proposal_for_source(
    source_path: Path,
    result: dict[str, Any],
    registry_ids: set[str],
    registry_source_globs: set[str],
    registry_source_hashes: set[str],
) -> dict[str, Any] | None:
    schema = str(result.get("schema") or "")
    if schema.startswith(NEURAL_SCHEMA_PREFIX):
        return None
    if "hsi" in source_path.name.lower() or "hsi" in schema.lower():
        return None
    case_count, min_controls, controls_seen = paired_cases(result)
    if case_count < 2 or min_controls < 2:
        return None
    stem = source_path.stem.replace("results_", "")
    task_id = f"discovered_{slugify(stem)}"
    if task_id in registry_ids:
        return None
    source_glob = repo_relative(source_path)
    source_hash = file_hash(source_path)
    if source_glob in registry_source_globs or source_hash in registry_source_hashes:
        return None
    controls = sorted(controls_seen)
    static_control = "full_unconstrained_unseeded" if "full_unconstrained_unseeded" in controls_seen else controls[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_kind": "curriculum_task",
        "task": {
            "task_id": task_id,
            "task_type": "qa_ml_benchmark_selector",
            "enabled": False,
            "result_schema": "QA_GENERAL_ML_NEURAL_BENCHMARK_SELECTOR.v0",
            "metric": "valid_rate_mean",
            "source_glob": source_glob,
            "source_hash": source_hash,
            "baseline_control": "baseline_static_selector",
            "candidate_control_template": "neural_benchmark_selector_rank{rank}",
            "static_source_control": static_control,
            "description": "Discovered neural selector task over existing project QA-ML paired-control benchmark summaries.",
        },
        "evidence": {
            "source_ref": repo_relative(source_path),
            "source_hash": source_hash,
            "source_schema": schema,
            "case_count": case_count,
            "min_controls_per_case": min_controls,
            "control_count": len(controls_seen),
            "controls": controls,
            "non_hsi": True,
            "duplicate_existing_task": False,
            "duplicate_existing_source": False,
        },
        "activation_policy": "manual_after_validation",
    }


def output_path(out_dir: Path, proposal: dict[str, Any]) -> Path:
    task = proposal["task"]
    return out_dir / f"qa_curriculum_task_proposal_{task['task_id']}.json"


def discover(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_path(args.out_dir)
    registry_ids = existing_task_ids(args.registry)
    registry_source_globs, registry_source_hashes = existing_task_sources(args.registry)
    proposals: list[dict[str, Any]] = []
    counts = {
        "sources_scanned": 0,
        "sources_skipped_invalid": 0,
        "sources_skipped_no_pairing": 0,
        "sources_skipped_neural": 0,
        "sources_skipped_hsi": 0,
        "sources_skipped_duplicate_task": 0,
        "sources_skipped_duplicate_source": 0,
        "outputs_skipped_existing": 0,
    }
    for source_path in iter_glob_paths(args.source_glob):
        counts["sources_scanned"] += 1
        result = load_json(source_path)
        if result is None:
            counts["sources_skipped_invalid"] += 1
            continue
        schema = str(result.get("schema") or "")
        if schema.startswith(NEURAL_SCHEMA_PREFIX):
            counts["sources_skipped_neural"] += 1
            continue
        if "hsi" in source_path.name.lower() or "hsi" in schema.lower():
            counts["sources_skipped_hsi"] += 1
            continue
        case_count, min_controls, _controls = paired_cases(result)
        if case_count < 2 or min_controls < 2:
            counts["sources_skipped_no_pairing"] += 1
            continue
        stem = source_path.stem.replace("results_", "")
        task_id = f"discovered_{slugify(stem)}"
        if task_id in registry_ids:
            counts["sources_skipped_duplicate_task"] += 1
            continue
        source_glob = repo_relative(source_path)
        source_hash = file_hash(source_path)
        if source_glob in registry_source_globs or source_hash in registry_source_hashes:
            counts["sources_skipped_duplicate_source"] += 1
            continue
        proposal = proposal_for_source(
            source_path,
            result,
            registry_ids,
            registry_source_globs,
            registry_source_hashes,
        )
        if proposal is None:
            counts["sources_skipped_no_pairing"] += 1
            continue
        path = output_path(out_dir, proposal)
        if path.exists():
            counts["outputs_skipped_existing"] += 1
            continue
        proposals.append(proposal)
    proposals.sort(key=lambda item: item["task"]["task_id"])

    emitted = []
    for proposal in proposals[: max(0, args.max_emits)]:
        path = output_path(out_dir, proposal)
        if args.dry_run:
            emitted.append({"dry_run": True, "proposal": repo_relative(path), "task_id": proposal["task"]["task_id"]})
            continue
        write_json(path, proposal)
        emitted.append({"ok": True, "proposal": str(path), "task_id": proposal["task"]["task_id"]})
    return {"ok": True, "dry_run": args.dry_run, "discovery": counts, "candidate_count": len(proposals), "emitted": emitted}


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "results_demo_general.json"
        write_json(
            source,
            {
                "schema": "QA_GENERAL_DISCOVERY_SELFTEST.v1",
                "moduli_results": {
                    "a": {"controls": {"control_a": {"valid_rate_mean": 0.2}, "control_b": {"valid_rate_mean": 0.4}}},
                    "b": {"controls": {"control_a": {"valid_rate_mean": 0.1}, "control_b": {"valid_rate_mean": 0.7}}},
                },
            },
        )
        registry = root / "registry.json"
        write_json(registry, {"schema_version": REGISTRY_SCHEMA, "tasks": []})
        args = argparse.Namespace(
            source_glob=str(root / "results_*.json"),
            registry=registry,
            out_dir=root / "out",
            max_emits=1,
            dry_run=False,
        )
        first = discover(args)
        second = discover(args)
        proposal_paths = sorted((root / "out").glob("*.json"))
        proposal = load_json(proposal_paths[0]) if proposal_paths else None
        ok = (
            first["ok"] is True
            and len(first["emitted"]) == 1
            and second["ok"] is True
            and len(second["emitted"]) == 0
            and proposal is not None
            and proposal["task"]["enabled"] is False
            and proposal["evidence"]["case_count"] == 2
            and proposal["evidence"]["duplicate_existing_source"] is False
        )
        duplicate_registry = root / "duplicate_registry.json"
        write_json(
            duplicate_registry,
            {
                "schema_version": REGISTRY_SCHEMA,
                "tasks": [
                    {
                        "task_id": "already_active_other_id",
                        "enabled": True,
                        "source_glob": repo_relative(source),
                        "source_hash": file_hash(source),
                    }
                ],
            },
        )
        duplicate_args = argparse.Namespace(
            source_glob=str(root / "results_*.json"),
            registry=duplicate_registry,
            out_dir=root / "duplicate_out",
            max_emits=1,
            dry_run=False,
        )
        duplicate_source = discover(duplicate_args)
        ok = (
            ok
            and duplicate_source["candidate_count"] == 0
            and duplicate_source["discovery"]["sources_skipped_duplicate_source"] == 1
        )
        return {"ok": ok, "first": first, "second": second, "duplicate_source": duplicate_source}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glob", default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-emits", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.max_emits < 0:
        print("error: --max-emits must be non-negative")
        return 2
    result = discover(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
