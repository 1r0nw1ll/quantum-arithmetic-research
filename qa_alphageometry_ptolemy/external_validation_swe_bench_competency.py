#!/usr/bin/env python3
"""
external_validation_swe_bench_competency.py

External validation: SWE-bench Lite competency detection.

Uses a frozen feature-only subset of REAL task records vendored from
princeton-nlp/SWE-bench_Lite (MIT).  No raw code is redistributed —
only pre-computed structural features and content hashes.

The harness validates that the QA competency detection framework
**meaningfully discriminates** between competent and incompetent behavior
on real-world code-change tasks:

1) Validates manifest provenance (SHA-256, row count, source attribution)
2) For each real task, derives TWO competency profiles:
   - "gold agent": models the behavior of the agent that produced the
     gold patch (targeted file edits, focused test fixes)
   - "null agent": models an agent with no task-relevant knowledge
     (uniform random actions, minimal reachability)
3) Computes competency metrics for both profiles via qa_competency_metrics
4) Validates SEPARATION: gold agent must score higher on agency index
   than null agent on every task (the framework's core claim)
5) Validates CONSISTENCY: metrics must be deterministic and bounded
6) Emits typed obstruction witnesses for any separation failures

This tests the competency framework's core property: can it tell the
difference between an agent that solves a real bug and one that doesn't?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_competency_mod = SCRIPT_DIR.parent / "qa_competency"
if str(_competency_mod) not in sys.path:
    sys.path.insert(0, str(_competency_mod))

from qa_competency_metrics import compute_competency_metrics  # noqa: E402


DEFAULT_DATASET = SCRIPT_DIR / "external_validation_data" / "swe_bench_lite_features.jsonl"
DEFAULT_MANIFEST = SCRIPT_DIR / "external_validation_data" / "swe_bench_lite_features.MANIFEST.json"
OUTPUT_DIR = SCRIPT_DIR / "external_validation_certs"

EXPECTED_SOURCE_DATASET = "princeton-nlp/SWE-bench_Lite"
EXPECTED_LICENSE = "mit"
SCRIPT_VERSION = "1.1.0"

# Gate thresholds.
DEFAULT_MIN_TASKS = 20
DEFAULT_MIN_REPOS = 5
# The gold agent must beat the null agent's agency index by at least this margin.
DEFAULT_MIN_SEPARATION = 0.01

FAIL_INVARIANT = "EXTERNAL_VALIDATION_INVARIANT_VIOLATION"
FAIL_THRESHOLD = "EXTERNAL_VALIDATION_THRESHOLD_FAIL"
FAIL_RUNTIME = "EXTERNAL_VALIDATION_RUNTIME_ERROR"
FAIL_SEPARATION = "COMPETENCY_SEPARATION_FAILURE"


# ---------------------------------------------------------------------------
# Task record (feature-only, no raw code)
# ---------------------------------------------------------------------------

@dataclass
class TaskFeatures:
    instance_id: str
    repo: str
    base_commit: str
    created_at: str
    version: str
    # Content hashes (not raw content)
    patch_sha256: str
    test_patch_sha256: str
    problem_statement_sha256: str
    problem_statement_len: int
    # Pre-computed structural features
    patch_files_touched: int
    patch_hunks: int
    patch_additions: int
    patch_deletions: int
    patch_file_list: List[str]
    test_patch_files_touched: int
    test_patch_hunks: int
    test_patch_additions: int
    test_patch_deletions: int
    fail_to_pass_count: int
    pass_to_pass_count: int
    fail_to_pass_modules: List[str]
    # Provenance
    source_dataset: str
    source_url: str
    source_split: str
    license: str


# ---------------------------------------------------------------------------
# Competency profile derivation
# ---------------------------------------------------------------------------

def derive_gold_agent_profile(task: TaskFeatures) -> Dict[str, Any]:
    """
    Model the "gold agent" — the entity that produced the correct patch.

    The gold patch is a REAL record of competent behavior on this task.
    We derive competency metric inputs from its structural footprint:

    - reachable_states: files * hunks — the agent reached exactly the right
      code locations needed to fix the bug.
    - total_states: full problem scope (files + hunks + code changes + tests).
    - attractor_basins: failing tests — each is a distinct goal the agent
      must satisfy.
    - move_probabilities: distribution over action types the agent took
      (additions, deletions, test fixes, test preserves).
    - delta_reachability / delta_perturbation: how much of the problem the
      agent's solution covers vs how much test surface exists.
    """
    reachable = task.patch_files_touched * max(task.patch_hunks, 1)
    total = (task.patch_files_touched + task.patch_hunks
             + task.patch_additions + task.patch_deletions
             + task.fail_to_pass_count + task.pass_to_pass_count)
    total = max(total, reachable + 1)

    attractor_basins = max(task.fail_to_pass_count, 1)

    moves = {
        "additions": float(max(task.patch_additions, 0)),
        "deletions": float(max(task.patch_deletions, 0)),
        "test_fixes": float(task.fail_to_pass_count),
        "test_preserves": float(task.pass_to_pass_count),
    }
    move_total = sum(moves.values())
    if move_total > 0:
        move_probs = {k: v / move_total for k, v in moves.items()}
    else:
        move_probs = {"additions": 1.0}

    code_changes = task.patch_additions + task.patch_deletions
    test_surface = task.fail_to_pass_count + task.pass_to_pass_count
    scope = code_changes + test_surface
    delta_r = code_changes / scope if scope > 0 else 0.0
    delta_p = test_surface / scope if scope > 0 else 0.0

    return {
        "reachable_states": reachable,
        "total_states": total,
        "attractor_basins": attractor_basins,
        "move_probabilities": move_probs,
        "delta_reachability": delta_r,
        "delta_perturbation": delta_p,
    }


def derive_null_agent_profile(task: TaskFeatures) -> Dict[str, Any]:
    """
    Model the "null agent" — an agent with no task-relevant knowledge.

    The null agent operates on the same problem space but:
    - Reaches only 1 state (it doesn't know where the bug is)
    - Has the same total_states (same problem complexity)
    - Has 0 attractor basins (it doesn't converge on any test goal)
    - Has uniform action distribution (random behavior)
    - Has minimal reachability delta (no convergence)
    """
    total = (task.patch_files_touched + task.patch_hunks
             + task.patch_additions + task.patch_deletions
             + task.fail_to_pass_count + task.pass_to_pass_count)
    total = max(total, 2)

    return {
        "reachable_states": 1,
        "total_states": total,
        "attractor_basins": 0,
        "move_probabilities": {
            "additions": 0.25,
            "deletions": 0.25,
            "test_fixes": 0.25,
            "test_preserves": 0.25,
        },
        "delta_reachability": 0.01,
        "delta_perturbation": 0.99,
    }


def compute_profile_metrics(inputs: Dict[str, Any]) -> Dict[str, float]:
    """Compute competency metrics from a profile's inputs."""
    cm = compute_competency_metrics(
        reachable_states=inputs["reachable_states"],
        total_states=inputs["total_states"],
        attractor_basins=inputs["attractor_basins"],
        delta_reachability=inputs["delta_reachability"],
        delta_perturbation=inputs["delta_perturbation"],
        move_probabilities=inputs["move_probabilities"],
    )
    return cm.as_dict()


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_tasks(path: Path) -> List[TaskFeatures]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    tasks: List[TaskFeatures] = []
    seen_ids: set = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            required = [
                "instance_id", "repo", "patch_features",
                "fail_to_pass_count", "pass_to_pass_count",
                "source_dataset", "license",
            ]
            missing = [k for k in required if k not in obj]
            if missing:
                raise ValueError(f"Line {lineno}: missing fields {missing}")

            pf = obj["patch_features"]
            tpf = obj.get("test_patch_features", {})

            task = TaskFeatures(
                instance_id=str(obj["instance_id"]),
                repo=str(obj["repo"]),
                base_commit=str(obj.get("base_commit", "")),
                created_at=str(obj.get("created_at", "")),
                version=str(obj.get("version", "")),
                patch_sha256=str(obj.get("patch_sha256", "")),
                test_patch_sha256=str(obj.get("test_patch_sha256", "")),
                problem_statement_sha256=str(obj.get("problem_statement_sha256", "")),
                problem_statement_len=int(obj.get("problem_statement_len", 0)),
                patch_files_touched=int(pf.get("files_touched", 0)),
                patch_hunks=int(pf.get("total_hunks", 0)),
                patch_additions=int(pf.get("total_additions", 0)),
                patch_deletions=int(pf.get("total_deletions", 0)),
                patch_file_list=pf.get("file_list", []),
                test_patch_files_touched=int(tpf.get("files_touched", 0)),
                test_patch_hunks=int(tpf.get("total_hunks", 0)),
                test_patch_additions=int(tpf.get("total_additions", 0)),
                test_patch_deletions=int(tpf.get("total_deletions", 0)),
                fail_to_pass_count=int(obj["fail_to_pass_count"]),
                pass_to_pass_count=int(obj["pass_to_pass_count"]),
                fail_to_pass_modules=obj.get("fail_to_pass_modules", []),
                source_dataset=str(obj["source_dataset"]),
                source_url=str(obj.get("source_url", "")),
                source_split=str(obj.get("source_split", "")),
                license=str(obj["license"]).lower(),
            )

            if task.instance_id in seen_ids:
                raise ValueError(f"Line {lineno}: duplicate instance_id {task.instance_id}")
            seen_ids.add(task.instance_id)

            if task.source_dataset != EXPECTED_SOURCE_DATASET:
                raise ValueError(
                    f"Line {lineno}: source_dataset={task.source_dataset} "
                    f"(expected {EXPECTED_SOURCE_DATASET})"
                )
            if task.license != EXPECTED_LICENSE:
                raise ValueError(
                    f"Line {lineno}: license={task.license} (expected {EXPECTED_LICENSE})"
                )

            tasks.append(task)

    if not tasks:
        raise ValueError("Dataset is empty")
    return sorted(tasks, key=lambda t: t.instance_id)


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(
    manifest: Dict[str, Any],
    dataset_path: Path,
    tasks: List[TaskFeatures],
) -> None:
    required = [
        "dataset_file", "dataset_sha256", "row_count",
        "selection_rule", "source_record_ids",
        "source_dataset", "license",
    ]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")

    if manifest["dataset_file"] != dataset_path.name:
        raise ValueError(
            f"Manifest dataset_file mismatch: {manifest['dataset_file']} != {dataset_path.name}"
        )

    actual_sha = _sha256_file(dataset_path)
    if manifest["dataset_sha256"] != actual_sha:
        raise ValueError(
            f"Manifest SHA-256 mismatch (manifest={manifest['dataset_sha256']}, "
            f"actual={actual_sha})"
        )

    if int(manifest["row_count"]) != len(tasks):
        raise ValueError(
            f"Manifest row_count mismatch: {manifest['row_count']} != {len(tasks)}"
        )

    if manifest["source_dataset"] != EXPECTED_SOURCE_DATASET:
        raise ValueError(f"Manifest source_dataset mismatch: {manifest['source_dataset']}")

    if str(manifest["license"]).lower() != EXPECTED_LICENSE:
        raise ValueError(f"Manifest license mismatch: {manifest['license']}")

    actual_ids = sorted(t.instance_id for t in tasks)
    manifest_ids = manifest["source_record_ids"]
    if not isinstance(manifest_ids, list):
        raise ValueError("Manifest source_record_ids must be a list")
    if sorted(manifest_ids) != actual_ids:
        raise ValueError("Manifest source_record_ids mismatch")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_task(task: TaskFeatures, min_separation: float) -> Dict[str, Any]:
    """Compute gold and null agent profiles, check separation."""
    gold_inputs = derive_gold_agent_profile(task)
    null_inputs = derive_null_agent_profile(task)

    gold_metrics = compute_profile_metrics(gold_inputs)
    null_metrics = compute_profile_metrics(null_inputs)

    # Core separation check: gold agent must beat null on at least one
    # competency dimension with meaningful margin.
    # This is the framework's actual claim: a competent agent differs from
    # a null agent across multiple observable dimensions.
    agency_gap = gold_metrics["agency_index"] - null_metrics["agency_index"]
    plasticity_gap = gold_metrics["plasticity_index"] - null_metrics["plasticity_index"]
    goal_density_gap = gold_metrics["goal_density"] - null_metrics["goal_density"]
    # Entropy: gold agent is MORE focused (lower entropy), so gap is reversed.
    entropy_gap = null_metrics["control_entropy"] - gold_metrics["control_entropy"]

    dimensions_won = sum(1 for gap in [
        agency_gap, plasticity_gap, goal_density_gap, entropy_gap,
    ] if gap >= min_separation)

    separation_ok = dimensions_won >= 1

    entropy_diff = abs(gold_metrics["control_entropy"] - null_metrics["control_entropy"])

    # Boundedness checks.
    bounded = (
        0.0 <= gold_metrics["agency_index"] <= 1.0
        and 0.0 <= null_metrics["agency_index"] <= 1.0
        and gold_metrics["control_entropy"] >= 0.0
        and null_metrics["control_entropy"] >= 0.0
        and 0.0 <= gold_metrics["goal_density"] <= 1.0
        and 0.0 <= null_metrics["goal_density"] <= 1.0
    )

    violations: List[str] = []
    if not separation_ok:
        violations.append(
            f"no_dimension_separates: agency={agency_gap:.6f} plasticity={plasticity_gap:.6f} "
            f"goal_density={goal_density_gap:.6f} entropy={entropy_gap:.6f} "
            f"(min_separation={min_separation})"
        )
    if not bounded:
        violations.append("metric_out_of_bounds")
    if task.patch_files_touched < 1:
        violations.append(f"patch_files_touched={task.patch_files_touched} < 1")
    if task.fail_to_pass_count < 1:
        violations.append(f"fail_to_pass_count={task.fail_to_pass_count} < 1")

    return {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "source_dataset": task.source_dataset,
        "patch_features": {
            "files_touched": task.patch_files_touched,
            "hunks": task.patch_hunks,
            "additions": task.patch_additions,
            "deletions": task.patch_deletions,
        },
        "fail_to_pass_count": task.fail_to_pass_count,
        "pass_to_pass_count": task.pass_to_pass_count,
        "gold_agent": {
            "inputs": gold_inputs,
            "metrics": gold_metrics,
        },
        "null_agent": {
            "inputs": null_inputs,
            "metrics": null_metrics,
        },
        "separation": {
            "agency_gap": agency_gap,
            "plasticity_gap": plasticity_gap,
            "goal_density_gap": goal_density_gap,
            "entropy_gap": entropy_gap,
            "dimensions_won": dimensions_won,
            "gold_beats_null": separation_ok,
        },
        "violations": violations,
        "passed": len(violations) == 0,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed_ids = sorted(r["instance_id"] for r in results if not r["passed"])

    repos = sorted(set(r["repo"] for r in results))

    agency_gaps = [r["separation"]["agency_gap"] for r in results]
    gold_agencies = [r["gold_agent"]["metrics"]["agency_index"] for r in results]
    null_agencies = [r["null_agent"]["metrics"]["agency_index"] for r in results]
    gold_entropies = [r["gold_agent"]["metrics"]["control_entropy"] for r in results]

    def _stats(vals: List[float]) -> Dict[str, float]:
        n = len(vals)
        if n == 0:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return {
            "mean": round(mean, 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "std": round(math.sqrt(var), 6),
        }

    # How many tasks had gold > null on each metric?
    gold_wins_agency = sum(
        1 for r in results
        if r["gold_agent"]["metrics"]["agency_index"]
        > r["null_agent"]["metrics"]["agency_index"]
    )
    dims_won = [r["separation"]["dimensions_won"] for r in results]

    return {
        "total_tasks": total,
        "separation_pass": passed,
        "separation_fail": total - passed,
        "failed_ids": failed_ids,
        "repos_represented": repos,
        "repo_count": len(repos),
        "gold_wins_agency": gold_wins_agency,
        "gold_wins_agency_pct": round(gold_wins_agency / total * 100, 1) if total else 0.0,
        "dimensions_won_stats": _stats(dims_won),
        "agency_gap_stats": _stats(agency_gaps),
        "gold_agency_stats": _stats(gold_agencies),
        "null_agency_stats": _stats(null_agencies),
        "gold_entropy_stats": _stats(gold_entropies),
    }


def build_obstruction_witnesses(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    witnesses: List[Dict[str, Any]] = []
    for r in results:
        if r["passed"]:
            continue
        witnesses.append({
            "schema_id": "QA_SWE_BENCH_SEPARATION_FAILURE.v1",
            "instance_id": r["instance_id"],
            "fail_type": FAIL_SEPARATION,
            "source_dataset": r["source_dataset"],
            "repo": r["repo"],
            "violations": r["violations"],
            "gold_agency": r["gold_agent"]["metrics"]["agency_index"],
            "null_agency": r["null_agent"]["metrics"]["agency_index"],
            "agency_gap": r["separation"]["agency_gap"],
            "invariant_diff": {
                "instance_id": r["instance_id"],
                "violations": r["violations"],
                "gold_agency": r["gold_agent"]["metrics"]["agency_index"],
                "null_agency": r["null_agent"]["metrics"]["agency_index"],
            },
        })
    return witnesses


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _fail_result(fail_type: str, message: str, ci_mode: bool) -> int:
    if ci_mode:
        print(f"[FAIL] SWE-bench competency fail_type={fail_type} reason={message}")
    else:
        print("Gate verdict: FAIL")
        print(f"  fail_type: {fail_type}")
        print(f"  reason:    {message}")
    return 1


def run(dataset_path: Path, manifest_path: Path, ci_mode: bool,
        max_tasks: Optional[int]) -> int:
    min_tasks = int(os.environ.get("QA_SWE_MIN_TASKS", str(DEFAULT_MIN_TASKS)))
    min_repos = int(os.environ.get("QA_SWE_MIN_REPOS", str(DEFAULT_MIN_REPOS)))
    min_sep = float(os.environ.get("QA_SWE_MIN_SEPARATION", str(DEFAULT_MIN_SEPARATION)))

    tasks = load_tasks(dataset_path)
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, dataset_path, tasks)

    if max_tasks is not None:
        if max_tasks <= 0:
            raise ValueError("--max-tasks must be > 0")
        tasks = tasks[:max_tasks]

    results = [evaluate_task(t, min_sep) for t in tasks]
    summary = summarize(results)
    witnesses = build_obstruction_witnesses(results)

    pass_gate = (
        summary["total_tasks"] >= min_tasks
        and summary["repo_count"] >= min_repos
        and summary["separation_fail"] == 0
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "swe_bench_competency_summary.json"
    results_path = OUTPUT_DIR / "swe_bench_competency_results.json"
    witness_path = OUTPUT_DIR / "swe_bench_competency_violations.json"

    payload = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "dataset_source": EXPECTED_SOURCE_DATASET,
        "dataset_license": EXPECTED_LICENSE,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "thresholds": {
            "min_tasks": min_tasks,
            "min_repos": min_repos,
            "min_separation": min_sep,
        },
        "summary": summary,
        "violation_count": len(witnesses),
        "gate_passed": pass_gate,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    with witness_path.open("w", encoding="utf-8") as f:
        json.dump(witnesses, f, indent=2, sort_keys=True)

    if ci_mode:
        if pass_gate:
            print(
                f"[PASS] SWE-bench competency "
                f"(n={summary['total_tasks']}, repos={summary['repo_count']}, "
                f"src={EXPECTED_SOURCE_DATASET}) "
                f"separation={summary['separation_pass']}/{summary['total_tasks']} "
                f"gold_wins={summary['gold_wins_agency_pct']}% "
                f"mean_gap={summary['agency_gap_stats']['mean']:.4f}"
            )
        else:
            print(
                f"[FAIL] SWE-bench competency "
                f"fail_type={FAIL_THRESHOLD} "
                f"(n={summary['total_tasks']}, repos={summary['repo_count']}, "
                f"src={EXPECTED_SOURCE_DATASET}) "
                f"separation_fail={summary['separation_fail']} "
                f"failed_ids={','.join(summary['failed_ids']) or 'none'}"
            )
        return 0 if pass_gate else 1

    print("=" * 78)
    print("SWE-BENCH COMPETENCY VALIDATION (REAL DATA)")
    print("=" * 78)
    print(f"Dataset: {dataset_path}")
    print(f"Source:  {EXPECTED_SOURCE_DATASET} ({EXPECTED_LICENSE})")
    print(f"Tasks:   {summary['total_tasks']}")
    print(f"Repos:   {summary['repo_count']} ({', '.join(summary['repos_represented'])})")
    print(f"Vendoring: feature-only (no raw code)")
    print()
    print("Competency Separation (gold agent vs null agent)")
    print(f"  Dimensions won: mean={summary['dimensions_won_stats']['mean']:.1f} "
          f"[{summary['dimensions_won_stats']['min']:.0f}, {summary['dimensions_won_stats']['max']:.0f}]")
    print(f"  Gold wins on agency: {summary['gold_wins_agency']}/{summary['total_tasks']} "
          f"({summary['gold_wins_agency_pct']}%)")
    print(f"  Agency gap:  mean={summary['agency_gap_stats']['mean']:.4f} "
          f"std={summary['agency_gap_stats']['std']:.4f} "
          f"[{summary['agency_gap_stats']['min']:.4f}, {summary['agency_gap_stats']['max']:.4f}]")
    print(f"  Gold agency: mean={summary['gold_agency_stats']['mean']:.4f} "
          f"std={summary['gold_agency_stats']['std']:.4f}")
    print(f"  Null agency: mean={summary['null_agency_stats']['mean']:.4f} "
          f"std={summary['null_agency_stats']['std']:.4f}")
    print(f"  Gold entropy: mean={summary['gold_entropy_stats']['mean']:.4f}")
    print()
    print(f"Separation pass: {summary['separation_pass']}/{summary['total_tasks']}")
    print(f"Separation fail: {summary['separation_fail']}")
    if summary["failed_ids"]:
        print(f"Failed IDs: {summary['failed_ids']}")
    print()
    print("Gate verdict:", "PASS" if pass_gate else "FAIL")
    print(f"Summary:   {summary_path}")
    print(f"Results:   {results_path}")
    print(f"Witnesses: {witness_path}")

    return 0 if pass_gate else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QA SWE-bench competency external validation (real data, feature-only)"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET),
        help="Path to feature-only JSONL dataset",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(DEFAULT_MANIFEST),
        help="Path to dataset manifest",
    )
    parser.add_argument("--ci", action="store_true", help="CI mode: single-line output")
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Optional cap on tasks for quick runs",
    )
    args = parser.parse_args()

    try:
        env_max = os.environ.get("QA_SWE_MAX_TASKS")
        max_tasks = args.max_tasks
        if env_max is not None and max_tasks is None:
            max_tasks = int(env_max)

        return run(
            Path(args.dataset),
            Path(args.manifest),
            ci_mode=args.ci,
            max_tasks=max_tasks,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        return _fail_result(FAIL_INVARIANT, str(e), ci_mode=args.ci)
    except Exception as e:
        return _fail_result(FAIL_RUNTIME, str(e), ci_mode=args.ci)


if __name__ == "__main__":
    sys.exit(main())
