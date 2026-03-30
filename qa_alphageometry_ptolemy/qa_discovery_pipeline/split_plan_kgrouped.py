#!/usr/bin/env python3
"""
qa_discovery_pipeline/split_plan_kgrouped.py

Deterministically split a QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1 plan into smaller
chunk plans while preserving "k-groups" by episode reference.

Rationale:
  - Multi-k sweeps (e.g. harmonic k={16,64,256}) should keep all k-level runs
    for a given episode/case together; otherwise summaries will (correctly)
    report missing k-levels / phase-law violations.

Grouping key:
  - (episode_ref.family, episode_ref.path_or_hash)

Usage:
  python3 qa_discovery_pipeline/split_plan_kgrouped.py \
    --plan qa_discovery_pipeline/plans/plan_harmonic_sweep_k3_v1.json \
    --out_dir qa_discovery_pipeline/plans/chunks/ho_k3_v1 \
    --cases-per-chunk 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(_canonical(obj) + "\n", encoding="utf-8")


def _validate_plan_shape(plan: Dict[str, Any]) -> None:
    if plan.get("schema_id") != "QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1":
        raise ValueError(f"bad schema_id={plan.get('schema_id')!r}")
    if not isinstance(plan.get("run_queue"), list) or len(plan["run_queue"]) == 0:
        raise ValueError("empty run_queue")
    det = plan.get("determinism", {})
    if det.get("canonical_json") is not True:
        raise ValueError("determinism.canonical_json must be true")
    if not plan.get("merkle_parent"):
        raise ValueError("missing merkle_parent")


def _group_by_episode_ref(run_queue: List[Dict[str, Any]]) -> List[Tuple[Tuple[str, str], List[Dict[str, Any]]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    order: List[Tuple[str, str]] = []
    for item in run_queue:
        ep = item.get("episode_ref") or {}
        key = (str(ep.get("family", "")), str(ep.get("path_or_hash", "")))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    return [(key, groups[key]) for key in order]


def _chunk_groups(
    groups: List[Tuple[Tuple[str, str], List[Dict[str, Any]]]],
    *,
    cases_per_chunk: int,
    max_runs_per_chunk: int,
) -> List[List[Dict[str, Any]]]:
    if cases_per_chunk <= 0 and max_runs_per_chunk <= 0:
        raise ValueError("must set --cases_per_chunk > 0 or --max_runs_per_chunk > 0")
    if cases_per_chunk > 0 and max_runs_per_chunk > 0:
        raise ValueError("set only one of --cases_per_chunk or --max_runs_per_chunk")

    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_cases = 0
    current_runs = 0

    def _flush() -> None:
        nonlocal current, current_cases, current_runs
        if current:
            chunks.append(current)
        current = []
        current_cases = 0
        current_runs = 0

    for _, items in groups:
        group_runs = len(items)
        if max_runs_per_chunk > 0 and group_runs > max_runs_per_chunk:
            raise ValueError(f"group too large ({group_runs} runs) for max_runs_per_chunk={max_runs_per_chunk}")

        next_cases = current_cases + 1
        next_runs = current_runs + group_runs

        should_split = False
        if cases_per_chunk > 0 and current and next_cases > cases_per_chunk:
            should_split = True
        if max_runs_per_chunk > 0 and current and next_runs > max_runs_per_chunk:
            should_split = True

        if should_split:
            _flush()

        current.extend(items)
        current_cases += 1
        current_runs += group_runs

    _flush()
    return chunks


def _plan_filename(src_plan: Path, chunk_index: int, chunk_total: int) -> str:
    stem = src_plan.name
    if stem.endswith(".json"):
        stem = stem[:-5]
    return f"{stem}__chunk_{chunk_index:03d}_of_{chunk_total:03d}.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a discovery batch plan while preserving k-groups by episode_ref.")
    ap.add_argument("--plan", required=True, help="Input QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1 plan JSON")
    ap.add_argument("--out_dir", required=True, help="Directory to write chunk plan JSONs")
    ap.add_argument(
        "--cases-per-chunk",
        "--cases_per_chunk",
        dest="cases_per_chunk",
        type=int,
        default=0,
        help="Number of episode/case groups per chunk",
    )
    ap.add_argument(
        "--max-runs-per-chunk",
        "--max_runs_per_chunk",
        dest="max_runs_per_chunk",
        type=int,
        default=0,
        help="Max runs per chunk (keeps groups intact)",
    )
    ap.add_argument(
        "--plan-id-prefix",
        "--plan_id_prefix",
        dest="plan_id_prefix",
        default="",
        help="Override base plan_id used for chunk plan_ids",
    )
    ap.add_argument(
        "--created-utc",
        "--created_utc",
        dest="created_utc",
        default="",
        help="Override created_utc (default: inherit from input plan)",
    )
    ap.add_argument(
        "--max-seconds-total",
        "--max_seconds_total",
        dest="max_seconds_total",
        type=int,
        default=0,
        help="Override budget.max_seconds_total per chunk",
    )
    ap.add_argument(
        "--merkle-parent",
        "--merkle_parent",
        dest="merkle_parent",
        default="",
        help="Override merkle_parent (default: inherit)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing chunk plan files")
    ap.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Compute chunks and print paths without writing files")
    args = ap.parse_args()

    src_plan = Path(args.plan)
    out_dir = Path(args.out_dir)

    plan = _read_json(src_plan)
    _validate_plan_shape(plan)

    run_queue = list(plan["run_queue"])
    groups = _group_by_episode_ref(run_queue)
    chunks = _chunk_groups(
        groups,
        cases_per_chunk=int(args.cases_per_chunk),
        max_runs_per_chunk=int(args.max_runs_per_chunk),
    )
    chunk_total = len(chunks)

    base_plan_id = str(args.plan_id_prefix).strip() or str(plan.get("plan_id", "PLAN-UNKNOWN"))
    created_utc = str(args.created_utc).strip() or str(plan.get("created_utc", "")) or _now_utc()
    merkle_parent = str(args.merkle_parent).strip() or str(plan.get("merkle_parent", ""))
    max_seconds_total = int(args.max_seconds_total) if int(args.max_seconds_total) > 0 else int(plan["budget"]["max_seconds_total"])

    if not merkle_parent:
        raise ValueError("merkle_parent is required (input plan missing and no override provided)")

    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[str] = []
    for idx, chunk_runs in enumerate(chunks, start=1):
        chunk_plan_id = f"{base_plan_id}__CHUNK_{idx:03d}_OF_{chunk_total:03d}"
        chunk_run_queue = sorted(chunk_runs, key=lambda item: str(item.get("run_id", "")))
        chunk_plan: Dict[str, Any] = {
            **plan,
            "plan_id": chunk_plan_id,
            "created_utc": created_utc,
            "run_queue": chunk_run_queue,
            "budget": {
                "max_runs": len(chunk_run_queue),
                "max_seconds_total": max_seconds_total,
            },
            "merkle_parent": merkle_parent,
        }

        filename = _plan_filename(src_plan, idx, chunk_total)
        out_path = out_dir / filename
        outputs.append(str(out_path))

        if not args.dry_run:
            _write_json(out_path, chunk_plan, overwrite=bool(args.overwrite))

    total_runs = len(run_queue)
    total_cases = len(groups)
    print(f"CASES total={total_cases} RUNS total={total_runs} CHUNKS={chunk_total}")
    for p in outputs:
        print(f"PLAN {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
