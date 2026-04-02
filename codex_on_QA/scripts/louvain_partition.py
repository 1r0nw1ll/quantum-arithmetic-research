#!/usr/bin/env python3
"""
Run Louvain community detection on a GraphML and write labels + summary.

Usage:
  python codex_on_QA/scripts/louvain_partition.py \
    --graph codex_on_QA/data/football.graphml \
    --outdir codex_on_QA/out

Requires: networkx, python-louvain (installable in the project venv)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

def ensure_dir(path: str):
    d = os.path.abspath(path)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', required=True)
    ap.add_argument('--outdir', default='codex_on_QA/out')
    ap.add_argument('--labels', default='labels_louvain.csv')
    ap.add_argument('--summary', default='louvain_summary.json')
    args = ap.parse_args()

    try:
        import networkx as nx  # type: ignore
        import community as community_louvain  # type: ignore
    except Exception as e:
        raise SystemExit('Missing deps. Activate venv and: pip install networkx python-louvain')

    G = nx.read_graphml(args.graph)
    G = G.to_undirected()

    part = community_louvain.best_partition(G)  # dict: node -> cluster id
    # Compute modularity
    modularity = community_louvain.modularity(part, G)

    # Write labels CSV
    ensure_dir(args.outdir)
    labels_path = os.path.join(args.outdir, args.labels)
    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write('node,label\n')
        for n in G.nodes():
            f.write(f"{n},{part[n]}\n")

    # Cluster sizes
    sizes = Counter(part.values())
    sizes_list = [sizes[c] for c in sorted(sizes)]

    # Summary JSON
    summary_path = os.path.join(args.outdir, args.summary)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'graph': args.graph,
            'clusters': len(sizes),
            'sizes': sizes_list,
            'modularity_Q': modularity,
        }, f, indent=2)

    print(f"Wrote: {labels_path}\nWrote: {summary_path}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

