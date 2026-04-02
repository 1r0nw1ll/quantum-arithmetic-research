#!/usr/bin/env python3
"""
Evaluate a clustering partition (labels CSV) against GraphML ground truth (node data 'value').
Writes a JSON with purity, ARI, NMI.

Usage:
  python codex_on_QA/scripts/eval_partition.py \
    --graph codex_on_QA/data/football.graphml \
    --labels codex_on_QA/out/labels_louvain.csv \
    --out codex_on_QA/out/louvain_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, Counter

def load_gt(graph_path: str) -> list[tuple[str, int]]:
    import networkx as nx
    G = nx.read_graphml(graph_path)
    # GraphML stored 'value' under a node data key; networkx exposes as attribute 'value'
    items = []
    for n, data in G.nodes(data=True):
        if 'value' not in data:
            raise SystemExit("GraphML missing 'value' node attribute for ground truth")
        items.append((n, int(data['value'])))
    return items

def load_pred(labels_csv: str) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(labels_csv, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row: continue
            node, lab = row[0], int(row[1])
            out[node] = lab
    return out

def contingency(pred: list[int], gt: list[int], k: int) -> list[list[int]]:
    t = max(gt) + 1
    m = [[0]*t for _ in range(k)]
    for p, g in zip(pred, gt):
        if 0 <= p < k and 0 <= g < t:
            m[p][g] += 1
    return m

def comb2(x: int) -> float:
    return 0.0 if x < 2 else x*(x-1)/2.0

def purity(cm: list[list[int]]) -> float:
    n = sum(sum(r) for r in cm)
    return sum(max(r) for r in cm) / n if n else 0.0

def ari(cm: list[list[int]]) -> float:
    n = sum(sum(r) for r in cm)
    if n < 2: return 0.0
    k = len(cm); t = len(cm[0]) if k else 0
    sum_comb = 0.0
    a_sum = 0.0
    b_sums = [0]*t
    for i in range(k):
        row = cm[i]
        row_sum = sum(row)
        a_sum += comb2(row_sum)
        for j in range(t):
            sum_comb += comb2(row[j])
            b_sums[j] += row[j]
    b_sum = sum(comb2(x) for x in b_sums)
    total = comb2(n)
    expected = (a_sum*b_sum)/total if total else 0.0
    max_idx = 0.5*(a_sum + b_sum)
    den = max_idx - expected
    return 0.0 if abs(den) < 1e-12 else (sum_comb - expected)/den

def nmi(cm: list[list[int]]) -> float:
    n = sum(sum(r) for r in cm)
    if n == 0: return 0.0
    k = len(cm); t = len(cm[0]) if k else 0
    row_sums = [sum(cm[i]) for i in range(k)]
    col_sums = [sum(cm[i][j] for i in range(k)) for j in range(t)]
    nf = float(n)
    mi = 0.0
    for i in range(k):
        for j in range(t):
            nij = cm[i][j]
            if nij == 0: continue
            p_ij = nij/nf; p_i = row_sums[i]/nf; p_j = col_sums[j]/nf
            mi += p_ij * math.log(p_ij/(p_i*p_j))
    h_u = -sum((x/nf)*math.log(x/nf) for x in row_sums if x>0)
    h_v = -sum((x/nf)*math.log(x/nf) for x in col_sums if x>0)
    den = math.sqrt(h_u*h_v)
    return 0.0 if den <= 0 else mi/den

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    gt_pairs = load_gt(args.graph)
    pred_map = load_pred(args.labels)

    # Align order of nodes
    nodes, gt_vals, pred_vals = [], [], []
    for n, g in gt_pairs:
        if n not in pred_map:
            continue
        nodes.append(n)
        gt_vals.append(int(g))
        pred_vals.append(int(pred_map[n]))

    k = max(pred_vals)+1 if pred_vals else 0
    cm = contingency(pred_vals, gt_vals, k)
    out = {
        'graph': args.graph,
        'labels': args.labels,
        'k': k,
        'purity': purity(cm),
        'ARI': ari(cm),
        'NMI': nmi(cm),
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

