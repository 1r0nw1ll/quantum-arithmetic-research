"""
QA-VFS Benchmark runner.

Usage:
  python experiments/qa_vfs/run_vfs_benchmark.py
  python experiments/qa_vfs/run_vfs_benchmark.py --N 100 --files 200 --queries 100 --seed 42

Tests five operations across three backends:
  lookup, append, mutation, corruption_recovery
"""
from __future__ import annotations
import sys
import os
import argparse
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fixture_builder import (
    build_file_universe, build_lookup_workload,
    build_append_ops, build_mutation_ops, build_corruption_ops,
)
from qa_vfs_backend import QAVFSBackend
from sqlite_backend import SQLiteBackend
from graph_vfs_backend import GraphVFSBackend
from vfs_metrics import aggregate, print_comparison, save_results, OpResult


def main():
    parser = argparse.ArgumentParser(description="QA-VFS Benchmark")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--files", type=int, default=200)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"Building file universe: N={args.N}, files={args.files}, seed={args.seed}")
    files = build_file_universe(N=args.N, n_files=args.files, seed=args.seed)
    print(f"  {len(files)} files created.")

    print("Building backends and loading files...")
    backends = {
        "qa_vfs":    QAVFSBackend(N=args.N),
        "sqlite":    SQLiteBackend(N=args.N),
        "graph_vfs": GraphVFSBackend(N=args.N),
    }
    for b in backends.values():
        b.load_files(files)
    print("  Done.")

    # Build workloads
    struct_q  = build_lookup_workload(files, N=args.N, n_queries=args.queries,
                                      seed=args.seed, query_mode="structured")
    unstruct_q = build_lookup_workload(files, N=args.N, n_queries=args.queries,
                                       seed=args.seed + 1, query_mode="unstructured")
    append_ops = build_append_ops(files, n_ops=args.queries // 2, seed=args.seed)
    mutation_ops = build_mutation_ops(files, n_ops=args.queries, seed=args.seed)
    corrupt_ops = build_corruption_ops(files, n_ops=args.queries // 2, seed=args.seed)

    all_results: dict[str, dict[str, list[OpResult]]] = {
        name: {} for name in backends
    }

    # Run all operations
    ops_config = [
        ("lookup_structured",    "lookup",             struct_q,      "run_lookup"),
        ("lookup_unstructured",  "lookup",             unstruct_q,    "run_lookup"),
        ("append",               "append",             append_ops,    "run_append"),
        ("mutation",             "mutation",           mutation_ops,  "run_mutation"),
        ("corruption_recovery",  "corruption_recovery",corrupt_ops,   "run_corruption_recovery"),
    ]

    for op_key, op_label, workload, method in ops_config:
        print(f"\nRunning {op_key} ({len(workload)} ops)...")
        for name, backend in backends.items():
            runner = getattr(backend, method)
            results = [runner(q) for q in workload]
            for r in results:
                r.op = op_key  # tag with full key
            all_results[name][op_key] = results

    # Aggregate + print
    all_summaries = {}
    for name in backends:
        all_summaries[name] = {}
        for op_key in all_results[name]:
            all_summaries[name][op_key] = aggregate(all_results[name][op_key])

    print("\n" + "=" * 80)
    print("QA-VFS BENCHMARK RESULTS")
    for op_key in [o[0] for o in ops_config]:
        sums = [all_summaries[name][op_key] for name in backends]
        print_comparison(op_key, sums)

    _print_hypotheses(all_summaries, backends)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(_HERE, "results", f"vfs_benchmark_{ts}.json")
    save_results(out_path, all_results, all_summaries)
    print(f"\nResults saved → {out_path}")


def _print_hypotheses(all_summaries, backends):
    backend_names = list(backends.keys())
    print("\n--- Hypothesis Verdicts ---")

    def _get(name, op, field):
        try:
            return getattr(all_summaries[name][op], field)
        except (KeyError, AttributeError):
            return None

    # H1: QA-VFS structured retrieval reduces wasted evals vs SQLite/graph
    qa_waste   = _get("qa_vfs",    "lookup_structured", "mean_wasted_evals")
    sql_waste  = _get("sqlite",    "lookup_structured", "mean_wasted_evals")
    gr_waste   = _get("graph_vfs", "lookup_structured", "mean_wasted_evals")
    h1 = qa_waste is not None and sql_waste is not None and qa_waste < sql_waste
    print(f"H1 (QA structured lookup reduces wasted evals): "
          f"{'SUPPORTED' if h1 else 'NOT SUPPORTED'}  "
          f"qa={qa_waste:.0f}  sqlite={sql_waste:.0f}  graph={gr_waste:.0f}")

    # H1 unstructured: QA should NOT win
    qa_wu  = _get("qa_vfs",    "lookup_unstructured", "mean_wasted_evals")
    sql_wu = _get("sqlite",    "lookup_unstructured", "mean_wasted_evals")
    h1u = qa_wu is not None and sql_wu is not None and qa_wu >= sql_wu * 0.9
    print(f"H1-falsifier (unstructured lookup — QA should not win): "
          f"{'CONFIRMED' if h1u else 'NOT CONFIRMED'}  "
          f"qa={qa_wu:.0f}  sqlite={sql_wu:.0f}")

    # H2: QA-VFS generator repair recovers chunks with fewer stored records
    # QA-VFS: recovery is law re-derivation (0 stored records for non-deviations)
    # SQLite: recovery reads content_val column (stored at write time)
    # Metric: repaired chunks that had NO deviation → QA requires 0 stored records, SQLite requires 1
    qa_rr  = _get("qa_vfs",    "corruption_recovery", "repair_rate")
    sql_rr = _get("sqlite",    "corruption_recovery", "repair_rate")
    gr_rr  = _get("graph_vfs", "corruption_recovery", "repair_rate")
    h2 = qa_rr is not None and sql_rr is not None and qa_rr >= sql_rr
    print(f"H2 (QA corruption recovery rate >= SQLite): "
          f"{'SUPPORTED' if h2 else 'NOT SUPPORTED'}  "
          f"qa={qa_rr:.3f}  sqlite={sql_rr:.3f}  graph={gr_rr:.3f}")

    # H3: QA-VFS storage is smaller
    qa_bytes  = _get("qa_vfs",    "lookup_structured", "mean_storage_bytes")
    sql_bytes = _get("sqlite",    "lookup_structured", "mean_storage_bytes")
    gr_bytes  = _get("graph_vfs", "lookup_structured", "mean_storage_bytes")
    h3 = qa_bytes is not None and sql_bytes is not None and qa_bytes < sql_bytes
    print(f"H3 (QA storage smaller): "
          f"{'SUPPORTED' if h3 else 'NOT SUPPORTED'}  "
          f"qa={qa_bytes:.0f}  sqlite={sql_bytes:.0f}  graph={gr_bytes:.0f}")

    # H4: QA mutation cost is HIGHER for lawful mutations (must re-derive chunks if root changes)
    # Wait — for QA-VFS lawful mutation costs O(1) (just update root pointer).
    # For SQLite/graph, lawful mutation requires re-deriving ALL chunks explicitly.
    # So QA-VFS actually WINS on lawful mutation cost. The H4 falsifier is:
    # Arbitrary mutation: QA stores deviation record (same O(1) as SQLite UPDATE).
    # The honest asymmetry: QA-VFS cannot "freely mutate" content without tracking deviations.
    qa_mut_cost  = _get("qa_vfs",    "mutation", "mean_mutation_cost")
    sql_mut_cost = _get("sqlite",    "mutation", "mean_mutation_cost")
    gr_mut_cost  = _get("graph_vfs", "mutation", "mean_mutation_cost")
    print(f"H4 (Mutation costs — honest comparison): "
          f"qa={qa_mut_cost:.1f}  sqlite={sql_mut_cost:.1f}  graph={gr_mut_cost:.1f}")
    if sql_mut_cost and qa_mut_cost:
        if sql_mut_cost > qa_mut_cost:
            print(f"   SQLite lawful mutation cost is higher (re-derives all chunks explicitly)")
        else:
            print(f"   Mutation costs comparable (deviation tracking overhead)")

    print()


if __name__ == "__main__":
    main()
