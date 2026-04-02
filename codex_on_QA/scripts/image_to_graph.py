#!/usr/bin/env python3
"""
Build a GraphML of image patches in QA space.

Input: a CSV with (id,b,e,label) as produced by image_patch_csv.py
Output: GraphML with nodes=patches, edges=k‑NN in QA feature space.

Usage:
  PYTHONPATH=. python codex_on_QA/scripts/image_to_graph.py \
    --csv codex_on_QA/out/image_patches.csv \
    --qa-mode qa21 \
    --k 8 \
    --out codex_on_QA/data/image_patches.graphml

Notes:
 - The GraphML schema matches the Rust parser:
   node keys: value (int label), d1..d4 (b,e,d,a)
   edge key : weight (double)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector


def load_csv(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    ids: List[str] = []
    be: List[Tuple[float, float]] = []
    labels: List[int] = []
    with path.open('r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        req = {'id','b','e','label'}
        if not req.issubset(set(r.fieldnames or [])):
            raise SystemExit(f"Missing columns in {path}: {r.fieldnames}")
        for row in r:
            ids.append(row['id'])
            be.append((float(row['b']), float(row['e'])))
            labels.append(int(row['label']))
    return ids, np.array(be, dtype=float), np.array(labels, dtype=int)


def build_qa_matrix(be: np.ndarray, mode: str) -> np.ndarray:
    mats: List[np.ndarray] = []
    for b, e in be:
        v, _ = qa_feature_vector(float(b), float(e), mode=mode)
        mats.append(v)
    return np.stack(mats, axis=0)


def knn_pairs(X: np.ndarray, k: int) -> List[Tuple[int,int]]:
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    n = X.shape[0]
    edges = set()
    for i in range(n):
        for j in idx[i, 1:]:  # skip self
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    return sorted(edges)


def write_graphml(out_path: Path, ids: List[str], be: np.ndarray, labels: np.ndarray, edges: List[Tuple[int,int]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        f.write('  <key id="k_value" for="node" attr.name="value" attr.type="int"/>\n')
        f.write('  <key id="k_d1" for="node" attr.name="d1" attr.type="double"/>\n')
        f.write('  <key id="k_d2" for="node" attr.name="d2" attr.type="double"/>\n')
        f.write('  <key id="k_d3" for="node" attr.name="d3" attr.type="double"/>\n')
        f.write('  <key id="k_d4" for="node" attr.name="d4" attr.type="double"/>\n')
        f.write('  <key id="k_w" for="edge" attr.name="weight" attr.type="double"/>\n')
        f.write('  <graph id="image_patches" edgedefault="undirected">\n')
        # nodes
        for i, nid in enumerate(ids):
            b = float(be[i,0]); e = float(be[i,1])
            d = b + e; a = b + 2.0*e
            lab = int(labels[i])
            f.write(f'    <node id="{nid}">\n')
            f.write(f'      <data key="k_value">{lab}</data>\n')
            f.write(f'      <data key="k_d1">{b:.9f}</data>\n')
            f.write(f'      <data key="k_d2">{e:.9f}</data>\n')
            f.write(f'      <data key="k_d3">{d:.9f}</data>\n')
            f.write(f'      <data key="k_d4">{a:.9f}</data>\n')
            f.write('    </node>\n')
        # edges
        for a, b in edges:
            ida = ids[a]; idb = ids[b]
            f.write(f'    <edge source="{ida}" target="{idb}">\n')
            f.write('      <data key="k_w">1.0</data>\n')
            f.write('    </edge>\n')
        f.write('  </graph>\n')
        f.write('</graphml>\n')


def main() -> int:
    ap = argparse.ArgumentParser(description='Build GraphML from image patch QA CSV')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--qa-mode', default='qa21', choices=['qa21','qa27','qa83'])
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ids, be, labels = load_csv(Path(args.csv))
    # QA embedding for kNN (qa-mode selectable)
    X_qa = build_qa_matrix(be, args.qa_mode)
    edges = knn_pairs(X_qa, k=args.k)
    write_graphml(Path(args.out), ids, be, labels, edges)
    print(f"Wrote {args.out} nodes={len(ids)} edges={len(edges)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

