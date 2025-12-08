#!/usr/bin/env python3
"""
Build a single comparison JSON merging Rust spectral modes (optionally multiple
qa_mode variants) and Louvain results for side-by-side review.

Inputs (defaults can be overridden via args):
  - --rust-json can be passed multiple times (e.g., qa21 + qa27 files)
  - --louvain-summary: codex_on_QA/out/louvain_summary.json
  - --louvain-metrics: codex_on_QA/out/louvain_metrics.json
Output:
  - codex_on_QA/out/football_comparison.json
Schema (excerpt):
{
  "graph": "...",
  "k_candidates": "2,4,6,8,10",
  "modes": {
    "qa_weight_x": {
      "qa_variants": {
        "qa21": {"Q": ..., "Purity": ..., ...},
        "qa27": {"Q": ..., "Purity": ..., ...}
      },
      "best": {"qa_mode": "qa21", "metrics": {...}}
    },
    "louvain": { ... }
  }
}
"""
from __future__ import annotations
import argparse, json, os
from typing import Dict, Any, List

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rust-json', action='append', default=[], help='Path(s) to Rust spectral JSON (repeatable)')
    ap.add_argument('--louvain-summary', default='codex_on_QA/out/louvain_summary.json')
    ap.add_argument('--louvain-metrics', default='codex_on_QA/out/louvain_metrics.json')
    ap.add_argument('--out', default='codex_on_QA/out/football_comparison.json')
    args = ap.parse_args()

    rust_paths: List[str] = args.rust_json if args.rust_json else []
    # Support comma-delimited single arg
    if len(rust_paths) == 1 and ',' in rust_paths[0]:
        rust_paths = [p.strip() for p in rust_paths[0].split(',') if p.strip()]
    if not rust_paths:
        raise SystemExit('Provide at least one --rust-json path')

    rust_docs = [load_json(p) for p in rust_paths]
    graph = rust_docs[0].get('graph')
    # Verify consistent graph
    for doc in rust_docs[1:]:
        if doc.get('graph') != graph:
            raise SystemExit('All --rust-json files must be for the same graph')
    k_candidates = rust_docs[0].get('k_candidates')

    # Merge modes across qa_mode variants
    modes: Dict[str, Any] = {}

    def push_variant(mode_name: str, qa_mode: str, run: Dict[str, Any]):
        entry = modes.setdefault(mode_name, {'qa_variants': {}})
        # Collect common metrics
        entry['qa_variants'][qa_mode] = {
            'Q': run.get('modularity_Q'),
            'Purity': run.get('purity'),
            'ARI': run.get('ARI'),
            'NMI': run.get('NMI'),
            'num_clusters': run.get('best_k'),
            'cluster_sizes': run.get('cluster_sizes'),
        }
        # Carry full-kernel extras if present
        if 'tau_selected' in run:
            entry['qa_variants'][qa_mode]['tau_selected'] = run.get('tau_selected')
            sel = run.get('selected_metrics') or {}
            entry['qa_variants'][qa_mode]['selected_metrics'] = sel

    def score_tradeoff(m: Dict[str, Any]) -> float:
        q = float(m.get('Q') or 0.0)
        pur = m.get('Purity'); ari = m.get('ARI'); nmi = m.get('NMI')
        if pur is None or ari is None or nmi is None:
            return q
        return q + 0.2 * float((pur + ari + nmi) / 3.0)

    for doc in rust_docs:
        qa_mode = doc.get('qa_mode', 'qa21')
        for run in doc.get('runs', []):
            mode_name = run.get('mode')
            push_variant(mode_name, qa_mode, run)

    # Compute best per mode (select by tradeoff score)
    for mode_name, entry in modes.items():
        variants = entry.get('qa_variants', {})
        if not variants:
            continue
        best_k = None; best_s = float('-inf')
        for qam, metrics in variants.items():
            s = score_tradeoff(metrics)
            if s > best_s:
                best_s = s; best_k = qam
        if best_k is not None:
            entry['best'] = {'qa_mode': best_k, 'metrics': variants[best_k]}

    lou_s = load_json(args.louvain_summary) if os.path.exists(args.louvain_summary) else None
    lou_m = load_json(args.louvain_metrics) if os.path.exists(args.louvain_metrics) else None
    if lou_s is not None and lou_m is not None and lou_s.get('graph') == graph:
        modes['louvain'] = {
            'Q': lou_s.get('modularity_Q'),
            'Purity': lou_m.get('purity'),
            'ARI': lou_m.get('ARI'),
            'NMI': lou_m.get('NMI'),
            'num_clusters': lou_s.get('clusters'),
            'cluster_sizes': lou_s.get('sizes'),
        }

    out = {
        'graph': graph,
        'k_candidates': k_candidates,
        'modes': modes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {args.out}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
