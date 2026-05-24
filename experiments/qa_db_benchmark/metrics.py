"""
Benchmark metrics collection and aggregation.
All timing in nanoseconds; storage estimates are approximate and labeled.
"""
from __future__ import annotations
import time
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class QueryResult:
    backend_name: str
    query_id: str
    latency_ns: int
    candidate_count_before: int
    candidate_count_after: int
    expansion_count: int
    results_count: int
    collapse_ratio: float          # candidate_after / candidate_before; HIGHER = tighter pre-filter (less waste)
    bytes_estimate: int            # approximate
    result_keys: frozenset         # for correctness comparison
    path_constrained: bool = False # True if QA path-legality was applied


@dataclass
class BenchmarkSummary:
    backend_name: str
    query_count: int
    p50_ns: float
    p95_ns: float
    p99_ns: float
    mean_ns: float
    mean_expansion_count: float
    mean_collapse_ratio: float          # higher = pre-filter is more precise
    mean_candidates_before: float
    mean_candidates_after: float
    mean_wasted_evals: float            # before - after; lower = less wasted work
    mean_bytes_estimate: float
    correctness_agreements: int    # vs reference backend
    correctness_mismatches: int
    path_constrained_count: int


def measure(fn, *args, **kwargs):
    t0 = time.perf_counter_ns()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return result, t1 - t0


def aggregate(results: list[QueryResult]) -> BenchmarkSummary:
    lats = sorted(r.latency_ns for r in results)
    n = len(lats)
    def pct(p):
        idx = max(0, min(n - 1, int(p / 100 * n)))
        return lats[idx]
    return BenchmarkSummary(
        backend_name=results[0].backend_name,
        query_count=n,
        p50_ns=pct(50),
        p95_ns=pct(95),
        p99_ns=pct(99),
        mean_ns=statistics.mean(lats),
        mean_expansion_count=statistics.mean(r.expansion_count for r in results),
        mean_collapse_ratio=statistics.mean(r.collapse_ratio for r in results),
        mean_candidates_before=statistics.mean(r.candidate_count_before for r in results),
        mean_candidates_after=statistics.mean(r.candidate_count_after for r in results),
        mean_wasted_evals=statistics.mean(
            r.candidate_count_before - r.candidate_count_after for r in results),
        mean_bytes_estimate=statistics.mean(r.bytes_estimate for r in results),
        correctness_agreements=0,   # filled in by run_benchmark
        correctness_mismatches=0,
        path_constrained_count=sum(1 for r in results if r.path_constrained),
    )


def correctness_check(
    ref: list[QueryResult],
    cmp: list[QueryResult],
) -> tuple[int, int, list[dict]]:
    """Compare result_keys between reference and comparison backends."""
    agreements = 0
    mismatches = 0
    mismatch_detail = []
    ref_by_id = {r.query_id: r for r in ref}
    for c in cmp:
        r = ref_by_id.get(c.query_id)
        if r is None:
            continue
        if c.path_constrained:
            # path-constrained semantics → not a hidden mismatch
            continue
        if r.result_keys == c.result_keys:
            agreements += 1
        else:
            mismatches += 1
            mismatch_detail.append({
                "query_id": c.query_id,
                "ref_backend": r.backend_name,
                "cmp_backend": c.backend_name,
                "ref_count": len(r.result_keys),
                "cmp_count": len(c.result_keys),
                "only_in_ref": len(r.result_keys - c.result_keys),
                "only_in_cmp": len(c.result_keys - r.result_keys),
            })
    return agreements, mismatches, mismatch_detail


def print_summary(summaries: list[BenchmarkSummary], mismatch_details: list[dict]):
    print("\n" + "=" * 88)
    print(f"{'Backend':<24} {'p50µs':>8} {'p95µs':>8} {'mean µs':>8} "
          f"{'expand':>7} {'before':>8} {'after':>8} {'waste':>8} {'collapse':>9} {'bytes':>10}")
    print("-" * 88)
    for s in summaries:
        print(
            f"{s.backend_name:<24} "
            f"{s.p50_ns/1e3:>8.1f} "
            f"{s.p95_ns/1e3:>8.1f} "
            f"{s.mean_ns/1e3:>8.1f} "
            f"{s.mean_expansion_count:>7.1f} "
            f"{s.mean_candidates_before:>8.0f} "
            f"{s.mean_candidates_after:>8.0f} "
            f"{s.mean_wasted_evals:>8.0f} "
            f"{s.mean_collapse_ratio:>9.3f} "
            f"{s.mean_bytes_estimate:>10.0f}"
        )
    print("=" * 88)
    print(f"\nCorrectness (vs table reference):")
    for s in summaries:
        if s.correctness_agreements + s.correctness_mismatches > 0:
            print(f"  {s.backend_name}: agree={s.correctness_agreements}  "
                  f"mismatch={s.correctness_mismatches}  "
                  f"path_constrained={s.path_constrained_count}")
    if mismatch_details:
        print(f"\n  Mismatch details (first 5):")
        for d in mismatch_details[:5]:
            print(f"    {d}")
    print()


def save_results(path: str, all_results: dict[str, list[QueryResult]],
                 summaries: list[BenchmarkSummary], mismatch_details: list[dict],
                 query_mode: str = "full_structured"):
    payload = {
        "query_mode": query_mode,
        "summaries": [asdict(s) for s in summaries],
        "mismatch_details": mismatch_details,
        "per_query": {
            backend: [
                {k: (list(v) if isinstance(v, frozenset) else v)
                 for k, v in asdict(r).items()}
                for r in results
            ]
            for backend, results in all_results.items()
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
