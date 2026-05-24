"""
VFS benchmark metrics.  Unified across all backends and all operation types.
"""
from __future__ import annotations
import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class OpResult:
    backend: str
    op: str                    # lookup/append/mutation/corruption_recovery
    query_id: str
    latency_ns: int
    candidate_before: int      # pre-filter candidate count
    candidate_after: int       # post-filter candidate count
    expansion_count: int       # BFS expansions during retrieval
    result_count: int          # final result set size
    mutation_cost: int         # number of write/reindex ops
    reconstruction_ops: int    # BFS steps to reconstruct a chunk
    repaired: bool             # True if corruption recovery succeeded
    storage_bytes: int         # approximate backend storage


@dataclass
class OpSummary:
    backend: str
    op: str
    query_count: int
    mean_ns: float
    p50_ns: float
    p95_ns: float
    mean_candidate_before: float
    mean_candidate_after: float
    mean_wasted_evals: float
    mean_expansion: float
    mean_mutation_cost: float
    mean_reconstruction_ops: float
    repair_rate: float          # fraction of corruption_recovery ops that succeed
    mean_storage_bytes: float


def measure(fn, *args, **kwargs):
    t0 = time.perf_counter_ns()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return result, t1 - t0


def aggregate(results: list[OpResult]) -> OpSummary:
    lats = sorted(r.latency_ns for r in results)
    n = len(lats)
    def pct(p):
        idx = max(0, min(n - 1, int(p / 100 * n)))
        return lats[idx]
    repair_results = [r for r in results if r.op == "corruption_recovery"]
    return OpSummary(
        backend=results[0].backend,
        op=results[0].op,
        query_count=n,
        mean_ns=statistics.mean(lats),
        p50_ns=pct(50),
        p95_ns=pct(95),
        mean_candidate_before=statistics.mean(r.candidate_before for r in results),
        mean_candidate_after=statistics.mean(r.candidate_after for r in results),
        mean_wasted_evals=statistics.mean(
            r.candidate_before - r.candidate_after for r in results),
        mean_expansion=statistics.mean(r.expansion_count for r in results),
        mean_mutation_cost=statistics.mean(r.mutation_cost for r in results),
        mean_reconstruction_ops=statistics.mean(r.reconstruction_ops for r in results),
        repair_rate=(
            sum(1 for r in repair_results if r.repaired) / max(1, len(repair_results))
        ),
        mean_storage_bytes=statistics.mean(r.storage_bytes for r in results),
    )


def print_comparison(op: str, summaries: list[OpSummary]):
    print(f"\n{'='*80}")
    print(f"  Operation: {op}")
    print(f"{'='*80}")
    if op == "lookup" or op.startswith("lookup_"):
        print(f"  {'backend':<20} {'mean µs':>8} {'before':>8} {'after':>8} "
              f"{'waste':>8} {'expand':>7} {'bytes':>10}")
        print(f"  {'-'*80}")
        for s in summaries:
            print(f"  {s.backend:<20} {s.mean_ns/1e3:>8.1f} "
                  f"{s.mean_candidate_before:>8.0f} {s.mean_candidate_after:>8.0f} "
                  f"{s.mean_wasted_evals:>8.0f} {s.mean_expansion:>7.1f} "
                  f"{s.mean_storage_bytes:>10.0f}")
    elif op == "mutation":
        print(f"  {'backend':<20} {'mean µs':>8} {'mut_cost':>10} {'bytes':>10}")
        print(f"  {'-'*50}")
        for s in summaries:
            print(f"  {s.backend:<20} {s.mean_ns/1e3:>8.1f} "
                  f"{s.mean_mutation_cost:>10.2f} {s.mean_storage_bytes:>10.0f}")
    elif op == "corruption_recovery":
        print(f"  {'backend':<20} {'mean µs':>8} {'repair_rate':>12} "
              f"{'recon_ops':>10} {'bytes':>10}")
        print(f"  {'-'*65}")
        for s in summaries:
            print(f"  {s.backend:<20} {s.mean_ns/1e3:>8.1f} "
                  f"{s.repair_rate:>12.3f} {s.mean_reconstruction_ops:>10.1f} "
                  f"{s.mean_storage_bytes:>10.0f}")
    elif op == "append":
        print(f"  {'backend':<20} {'mean µs':>8} {'bytes':>10}")
        print(f"  {'-'*42}")
        for s in summaries:
            print(f"  {s.backend:<20} {s.mean_ns/1e3:>8.1f} "
                  f"{s.mean_storage_bytes:>10.0f}")


def save_results(path: str, all_results: dict[str, dict[str, list[OpResult]]],
                 all_summaries: dict[str, dict[str, OpSummary]]):
    payload = {
        "summaries": {
            backend: {op: asdict(s) for op, s in ops.items()}
            for backend, ops in all_summaries.items()
        },
        "per_query": {
            backend: {
                op: [
                    {k: (v) for k, v in asdict(r).items()}
                    for r in results
                ]
                for op, results in ops.items()
            }
            for backend, ops in all_results.items()
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
