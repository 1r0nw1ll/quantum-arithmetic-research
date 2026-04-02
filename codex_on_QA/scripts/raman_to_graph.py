#!/usr/bin/env python3
"""
Build a Raman sample graph (nodes = spectra) from a Raman CSV with columns id,b,e,label.

Writes GraphML compatible with qa_graph_experiments:
  - Node ground truth label under attr.name "value" (int)
  - QA tuple under keys d1,d2,d3,d4 (b,e,d=b+e,a=d+e)
  - Edges from kNN in QA feature space (qa21 on (b,e)); edge weights included
    but current Rust pipeline uses unweighted adjacency.

Usage:
  PYTHONPATH=. python codex_on_QA/scripts/raman_to_graph.py \
    --csv codex_on_QA/out/raman_qa_fundovt_bcwin_v2.csv \
    --qa-mode qa21 \
    --k 8 \
    --out codex_on_QA/data/raman_qa21.graphml
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector


def load_raman_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids: List[str] = []
    labels: List[int] = []
    b_list: List[float] = []
    e_list: List[float] = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get('id') or row.get('sample_id') or str(len(ids))
            ids.append(str(rid))
            labels.append(int(row['label']))
            b_list.append(float(row['b']))
            e_list.append(float(row['e']))
    return np.array(ids, dtype=object), np.asarray(labels, dtype=int), np.asarray(b_list, dtype=float), np.asarray(e_list, dtype=float)


def build_qa_features(b: np.ndarray, e: np.ndarray, qa_mode: str) -> Tuple[np.ndarray, List[str]]:
    feats: List[np.ndarray] = []
    names: List[str] = []
    for i, (bi, ei) in enumerate(zip(b, e)):
        v, nm = qa_feature_vector(float(bi), float(ei), mode=qa_mode)
        feats.append(v)
        if not names:
            names = nm
    return np.vstack(feats), names


def knn_edges(features: np.ndarray, k: int = 8, eps: float = 1e-9) -> List[Tuple[int, int, float]]:
    n = features.shape[0]
    # pairwise L2 distances (memory O(n^2), OK for moderate n)
    dists = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    weights: dict[Tuple[int, int], float] = {}
    for i in range(n):
        # take k nearest neighbors excluding self
        idx = np.argsort(dists[i])
        nn = idx[1 : k + 1]
        for j in nn:
            if i == j:
                continue
            w = 1.0 / (dists[i, j] + eps)
            key = (i, j) if i < j else (j, i)
            if key not in weights or w > weights[key]:
                weights[key] = w
    return [(i, j, w) for (i, j), w in weights.items()]


def write_graphml(out_path: str, ids: np.ndarray, labels: np.ndarray, b: np.ndarray, e: np.ndarray, edges: List[Tuple[int, int, float]]):
    # Conform to qa_graph_experiments expectations:
    # - node attr.name "value" used for GT labels
    # - node keys d1..d4 map to b,e,d,a
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        # keys
        f.write('  <key id="k_value" for="node" attr.name="value" attr.type="int"/>\n')
        f.write('  <key id="k_d1" for="node" attr.name="d1" attr.type="double"/>\n')
        f.write('  <key id="k_d2" for="node" attr.name="d2" attr.type="double"/>\n')
        f.write('  <key id="k_d3" for="node" attr.name="d3" attr.type="double"/>\n')
        f.write('  <key id="k_d4" for="node" attr.name="d4" attr.type="double"/>\n')
        f.write('  <key id="k_w" for="edge" attr.name="weight" attr.type="double"/>\n')
        f.write('  <graph id="raman_samples" edgedefault="undirected">\n')
        # nodes
        for nid, lab, bi, ei in zip(ids, labels, b, e):
            di = float(bi + ei)
            ai = float(di + ei)
            f.write(f'    <node id="{nid}">\n')
            f.write(f'      <data key="k_value">{int(lab)}</data>\n')
            f.write(f'      <data key="k_d1">{bi:.9f}</data>\n')
            f.write(f'      <data key="k_d2">{ei:.9f}</data>\n')
            f.write(f'      <data key="k_d3">{di:.9f}</data>\n')
            f.write(f'      <data key="k_d4">{ai:.9f}</data>\n')
            f.write('    </node>\n')
        # edges
        for (i, j, w) in edges:
            f.write(f'    <edge source="{ids[i]}" target="{ids[j]}">\n')
            f.write(f'      <data key="k_w">{w:.9f}</data>\n')
            f.write('    </edge>\n')
        f.write('  </graph>\n')
        f.write('</graphml>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Raman CSV: id,b,e,label')
    ap.add_argument('--qa-mode', choices=['qa21','qa27','qa83'], default='qa21')
    ap.add_argument('--k', type=int, default=8, help='k for kNN graph')
    ap.add_argument('--out', required=True, help='Output GraphML path')
    args = ap.parse_args()

    ids, labels, b, e = load_raman_csv(args.csv)
    qa_feats, _names = build_qa_features(b, e, qa_mode=args.qa_mode)
    edges = knn_edges(qa_feats, k=args.k)
    write_graphml(args.out, ids, labels, b, e, edges)
    print('Wrote', args.out, 'nodes:', len(ids), 'edges:', len(edges))


if __name__ == '__main__':
    main()

