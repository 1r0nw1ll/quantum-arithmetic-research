"""
QA Law-Structured Memory Benchmark runner.

Usage:
  python experiments/qa_db_benchmark/run_benchmark.py --N 250 --queries 250 --seed 42
  python experiments/qa_db_benchmark/run_benchmark.py --query-mode orbit_only
  python experiments/qa_db_benchmark/run_benchmark.py --query-mode random_attribute
  python experiments/qa_db_benchmark/run_benchmark.py --query-mode range_only
  python experiments/qa_db_benchmark/run_benchmark.py --query-mode mixed_heterogeneous

Query modes:
  full_structured      All QA predicates. QA should win H1/H2.
  orbit_only           Orbit constraint only. QA advantage disappears.
  random_attribute     b_mod/e_mod predicates (non-QA-law). QA does not win.
  range_only           b/e range predicates. Neither backend gains structural edge.
  mixed_heterogeneous  50/50 structured+random. Report honestly.
"""
from __future__ import annotations
import sys
import os
import argparse
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from workload_fixture_builder import build_workload, VALID_MODES, FALSIFIER_MODES
from qa_backend import QABackend
from table_backend import TableBackend
from graph_backend import GraphBackend
from metrics import (
    aggregate, correctness_check, print_summary, save_results, BenchmarkSummary
)

# Per-mode hypothesis predictions (what we expect before running)
_MODE_PREDICTIONS: dict[str, dict] = {
    "full_structured": {
        "H1": "SUPPORTED  — rich QA buckets reduce wasted evals vs orbit-only index",
        "H2": "SUPPORTED  — generator-legal BFS eliminates most expansions",
        "H3": "SUPPORTED  — seed+law encoding <<< row+index storage",
        "H4": "HONEST     — QA not always faster; reported as win-rate",
    },
    "orbit_only": {
        "H1": "FAIL EXPECTED — no i_gap/area constraints; QA degrades to flat orbit index",
        "H2": "UNCERTAIN  — path constraint still limits BFS but filter is loose",
        "H3": "SUPPORTED  — storage independent of query mode",
        "H4": "N/A for this mode",
    },
    "random_attribute": {
        "H1": "FAIL EXPECTED — b_mod/e_mod predicates cannot be pre-bucketed by QA law",
        "H2": "UNCERTAIN  — depends on how many results survive random_attribute filter",
        "H3": "SUPPORTED  — storage independent of query mode",
        "H4": "N/A for this mode",
    },
    "range_only": {
        "H1": "FAIL EXPECTED — b/e range not in QA invariant structure; both use orbit index",
        "H2": "UNCERTAIN  — generator-legal BFS may still reduce expansion",
        "H3": "SUPPORTED  — storage independent of query mode",
        "H4": "N/A for this mode",
    },
    "mixed_heterogeneous": {
        "H1": "PARTIAL EXPECTED — 50% structured queries help; 50% random_attribute queries don't",
        "H2": "UNCERTAIN  — depends on mix",
        "H3": "SUPPORTED  — storage independent of query mode",
        "H4": "N/A for this mode",
    },
}


def main():
    parser = argparse.ArgumentParser(description="QA DB Benchmark")
    parser.add_argument("--N", type=int, default=250)
    parser.add_argument("--queries", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--query-mode", dest="query_mode", default="full_structured",
        choices=sorted(VALID_MODES),
        help="Query workload mode (default: full_structured)",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"Building workload: N={args.N}, queries={args.queries}, "
          f"seed={args.seed}, mode={args.query_mode}")
    queries = build_workload(
        N=args.N, n_queries=args.queries, seed=args.seed,
        query_mode=args.query_mode,
    )

    print("Building backends...")
    backends = {
        "table": TableBackend(N=args.N),
        "graph": GraphBackend(N=args.N),
        "qa": QABackend(N=args.N),
    }
    print("  Done.")

    all_results: dict[str, list] = {}
    for name, backend in backends.items():
        print(f"Running {name} backend ({args.queries} queries)...")
        results = [backend.run_query(q) for q in queries]
        all_results[name] = results
        print(f"  Done.")

    summaries = []
    for name, results in all_results.items():
        s = aggregate(results)
        summaries.append(s)

    ref_results = all_results["table"]
    all_mismatch_details = []
    for name in ["graph", "qa"]:
        agr, mis, details = correctness_check(ref_results, all_results[name])
        for s in summaries:
            if s.backend_name == name:
                s.correctness_agreements = agr
                s.correctness_mismatches = mis
        all_mismatch_details.extend(details)

    print(f"\n{'='*40} mode={args.query_mode} {'='*40}")
    print_summary(summaries, all_mismatch_details)
    _print_hypotheses(summaries, all_results, queries, args.query_mode)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(
        _HERE, "results", f"benchmark_{args.query_mode}_{ts}.json"
    )
    save_results(
        out_path, all_results, summaries, all_mismatch_details,
        query_mode=args.query_mode,
    )
    print(f"Results saved → {out_path}")


def _print_hypotheses(
    summaries: list[BenchmarkSummary],
    all_results: dict,
    queries: list,
    query_mode: str,
):
    by_name = {s.backend_name: s for s in summaries}
    qa_s = by_name.get("qa")
    table_s = by_name.get("table")

    preds = _MODE_PREDICTIONS.get(query_mode, {})
    is_falsifier = query_mode in FALSIFIER_MODES

    print(f"\n--- Hypothesis Verdicts  [{query_mode}] ---")

    if is_falsifier:
        print(f"  [Falsifier mode — QA structural advantage should shrink/disappear]")

    # H1: wasted evaluations
    if qa_s and table_s:
        h1_waste = qa_s.mean_wasted_evals < table_s.mean_wasted_evals
        h1_before = qa_s.mean_candidates_before < table_s.mean_candidates_before
        h1 = h1_waste and h1_before
        verdict = "SUPPORTED" if h1 else "NOT SUPPORTED"
        waste_ratio = (table_s.mean_wasted_evals / max(1, qa_s.mean_wasted_evals))
        print(f"H1 (QA reduces wasted evals): {verdict}  "
              f"waste ratio table/qa={waste_ratio:.2f}x  "
              f"before: qa={qa_s.mean_candidates_before:.0f} "
              f"table={table_s.mean_candidates_before:.0f}")
        print(f"   Prediction: {preds.get('H1', 'n/a')}")

    # H2: expansion count
    if qa_s and table_s:
        h2 = qa_s.mean_expansion_count < table_s.mean_expansion_count
        print(f"H2 (QA reduces expansion): {'SUPPORTED' if h2 else 'NOT SUPPORTED'}  "
              f"qa={qa_s.mean_expansion_count:.1f}  table={table_s.mean_expansion_count:.1f}")
        print(f"   Prediction: {preds.get('H2', 'n/a')}")

    # H3: storage
    if qa_s and table_s:
        h3 = qa_s.mean_bytes_estimate < table_s.mean_bytes_estimate
        print(f"H3 (QA compressed storage): {'SUPPORTED' if h3 else 'NOT SUPPORTED'}  "
              f"qa={qa_s.mean_bytes_estimate:.0f}  table={table_s.mean_bytes_estimate:.0f}")
        print(f"   Prediction: {preds.get('H3', 'n/a')}")

    # H4 only meaningful for full_structured
    if query_mode == "full_structured":
        qa_results = {r.query_id: r for r in all_results["qa"]}
        table_results = {r.query_id: r for r in all_results["table"]}
        broad_qs = [q for q in queries if q.get("broad_expand")]
        non_broad_qs = [q for q in queries if not q.get("broad_expand")]
        def _win_rate(qlist):
            wins = sum(
                1 for q in qlist
                if q["query_id"] in qa_results and q["query_id"] in table_results
                and qa_results[q["query_id"]].latency_ns
                < table_results[q["query_id"]].latency_ns
            )
            return wins, len(qlist)
        w_broad, n_broad = _win_rate(broad_qs)
        w_narrow, n_narrow = _win_rate(non_broad_qs)
        print(f"H4 (QA not always faster): "
              f"non-broad {w_narrow}/{n_narrow}  broad {w_broad}/{n_broad}  "
              f"(path_constrained excluded from correctness)")
        print(f"   Prediction: {preds.get('H4', 'n/a')}")

    # Falsifier summary
    if is_falsifier and qa_s and table_s:
        waste_ratio = table_s.mean_wasted_evals / max(1, qa_s.mean_wasted_evals)
        print(f"\n  Falsifier result: waste_ratio={waste_ratio:.2f}x "
              f"({'QA still wins — not a clean falsifier' if waste_ratio > 1.5 else 'QA advantage gone — falsifier confirmed'})")
    print()


if __name__ == "__main__":
    main()
