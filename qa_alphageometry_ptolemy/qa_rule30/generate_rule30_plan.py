#!/usr/bin/env python3
"""
Generate Rule 30 discovery pipeline batch plans.

Frozen scope v1:
  p_max = 256
  T ∈ {4096, 8192, 16384, 32768, 65536}
  k ∈ [4, 16]
  ~40 runs across 5 batch plans (one per T value)

Each batch plan contains 8 runs:
  3x PROVE at k=4,8,16 (covering P ∈ {1..64, 65..128, 129..256})
  3x REFUTE at k=4,8,16 (same P ranges)
  2x EXPLORE at k=4,8

Usage:
  python generate_rule30_plan.py --outdir qa_discovery_pipeline/plans/rule30_v1
  python generate_rule30_plan.py --outdir qa_discovery_pipeline/plans/rule30_v1 --T 16384
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Constants — frozen scope v1
# ---------------------------------------------------------------------------

T_VALUES = [4096, 8192, 16384, 32768, 65536]
P_MIN = 1
DEFAULT_P_MAX = 256

# Objective types with their episode template paths (relative to qa_alphageometry_ptolemy)
OBJECTIVE_TYPES = ["PROVE", "REFUTE", "EXPLORE"]
K_VALUES_PROVE_REFUTE = [4, 8, 16]
K_VALUES_EXPLORE = [4, 8]


def _compute_p_ranges(p_max: int) -> list:
    """Split [1, p_max] into ~3 equal sub-ranges."""
    third = max(1, p_max // 3)
    return [
        (1, third),
        (third + 1, 2 * third),
        (2 * third + 1, p_max),
    ]

PIPELINE_ID = "QA_DISCOVERY_PIPELINE.v1"
AGENT_ID = "qa-agent-ctrl-1"
CP_FAMILY = "QA_CONJECTURE_PROVE_CONTROL_LOOP.v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def generate_plan_for_T(T: int, plan_index: int, p_max: int = DEFAULT_P_MAX) -> Dict[str, Any]:
    """Generate a single batch plan for a given T value."""
    p_ranges = _compute_p_ranges(p_max)
    run_queue: List[Dict[str, Any]] = []
    run_counter = 0

    # PROVE runs: one per (P_range, k)
    for p_lo, p_hi in p_ranges:
        for k in K_VALUES_PROVE_REFUTE:
            run_counter += 1
            episode_hash = _sha256(f"rule30:prove:P{p_lo}-{p_hi}:T{T}")
            run_queue.append({
                "run_id": f"RUN-R30-T{T}-PROVE-P{p_lo}_{p_hi}-K{k}",
                "episode_ref": {
                    "family": CP_FAMILY,
                    "path_or_hash": episode_hash,
                },
                "k": k,
            })

    # REFUTE runs: one per (P_range, k)
    for p_lo, p_hi in p_ranges:
        for k in K_VALUES_PROVE_REFUTE:
            run_counter += 1
            episode_hash = _sha256(f"rule30:refute:P{p_lo}-{p_hi}:T{T}")
            run_queue.append({
                "run_id": f"RUN-R30-T{T}-REFUTE-P{p_lo}_{p_hi}-K{k}",
                "episode_ref": {
                    "family": CP_FAMILY,
                    "path_or_hash": episode_hash,
                },
                "k": k,
            })

    # EXPLORE runs: one per k
    for k in K_VALUES_EXPLORE:
        run_counter += 1
        episode_hash = _sha256(f"rule30:explore:T{T}")
        run_queue.append({
            "run_id": f"RUN-R30-T{T}-EXPLORE-K{k}",
            "episode_ref": {
                "family": CP_FAMILY,
                "path_or_hash": episode_hash,
            },
            "k": k,
        })

    # Sort lexicographically by run_id (determinism contract)
    run_queue.sort(key=lambda r: r["run_id"])

    plan = {
        "schema_id": "QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1",
        "plan_id": f"PLAN-RULE30-V1-T{T}-{plan_index:03d}",
        "created_utc": "2026-02-11T02:00:00Z",
        "agent_id": AGENT_ID,
        "pipeline_id": PIPELINE_ID,
        "run_queue": run_queue,
        "determinism": {
            "queue_ordering": "lexicographic(run_id)",
            "seed_policy": "frontier[0]",
            "canonical_json": True,
        },
        "budget": {
            "max_runs": len(run_queue),
            "max_seconds_total": max(300, T // 10),
        },
        "merkle_parent": "1111111111111111111111111111111111111111111111111111111111111111",
    }
    return plan


def main():
    parser = argparse.ArgumentParser(description="Generate Rule 30 batch plans")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for batch plans")
    parser.add_argument("--T", type=int, default=None,
                        help="Generate plan for a single T value (default: all)")
    parser.add_argument("--P_max", type=int, default=DEFAULT_P_MAX,
                        help=f"Maximum period (default: {DEFAULT_P_MAX})")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_values = [args.T] if args.T else T_VALUES
    total_runs = 0

    for i, T in enumerate(t_values):
        plan = generate_plan_for_T(T, i + 1, args.P_max)
        fname = f"batch_plan_T{T}.json"
        fpath = outdir / fname
        fpath.write_text(_canonical(plan) + "\n", encoding="utf-8")
        n_runs = len(plan["run_queue"])
        total_runs += n_runs
        print(f"  wrote {fname} ({n_runs} runs)", file=sys.stderr)

    print(f"Generated {len(t_values)} plan(s), {total_runs} total runs -> {outdir}",
          file=sys.stderr)
    # Print outdir to stdout for piping
    print(str(outdir))


if __name__ == "__main__":
    main()
