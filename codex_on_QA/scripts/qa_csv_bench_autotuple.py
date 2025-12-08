#!/usr/bin/env python3
"""
General multi-tuple QA CSV benchmark.

Input CSV schema:
  id, b1, e1, b2, e2, ..., label   (any number of (b,e) pairs)

Modes:
  - raw: concatenate all (b_i, e_i) pairs
  - qa21/qa27/qa83: for each (b_i,e_i), build QA features and concatenate

Outputs:
  codex_on_QA/out/<stem>_csv_bench.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector


OUTDIR = os.path.join('codex_on_QA', 'out')
os.makedirs(OUTDIR, exist_ok=True)

MODES = ['raw', 'qa21', 'qa27', 'qa83']


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int]]]:
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        pair_cols: List[Tuple[str,str]] = []
        # detect pairs b1/e1, b2/e2, ...; fallback to single b/e
        idx = 1
        while True:
            bcol = f'b{idx}'
            ecol = f'e{idx}'
            if bcol in cols and ecol in cols:
                pair_cols.append((bcol, ecol))
                idx += 1
            else:
                break
        if not pair_cols:
            # try single b/e
            if 'b' in cols and 'e' in cols:
                pair_cols = [('b','e')]
            else:
                raise SystemExit('No (b,e) columns found')
        X_list: List[List[float]] = []
        y_list: List[int] = []
        for row in reader:
            feats: List[float] = []
            for (bcol, ecol) in pair_cols:
                feats.extend([float(row[bcol]), float(row[ecol])])
            X_list.append(feats)
            y_list.append(int(row['label']))
    return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=int), [(i*2,(i*2)+1) for i in range(len(pair_cols))]


def build_features_multi(Xraw: np.ndarray, mode: str, pairs: List[Tuple[int,int]]) -> np.ndarray:
    if mode == 'raw':
        return Xraw
    feats: List[np.ndarray] = []
    for row in Xraw:
        parts: List[np.ndarray] = []
        for (i_b, i_e) in pairs:
            b = float(row[i_b]); e = float(row[i_e])
            v, _ = qa_feature_vector(b, e, mode=mode)
            parts.append(v)
        feats.append(np.concatenate(parts, axis=0))
    return np.vstack(feats)


def benchmark_all(X: np.ndarray, y: np.ndarray) -> dict:
    k = len(np.unique(y))
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    lab = km.fit_predict(X)
    ari = float(adjusted_rand_score(y, lab))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(Xtr, ytr)
    acc_lr = float(accuracy_score(yte, lr.predict(Xte)))
    mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=300, random_state=0)
    mlp.fit(Xtr, ytr)
    acc_mlp = float(accuracy_score(yte, mlp.predict(Xte)))
    return {'kmeans_ARI': ari, 'logreg_acc': acc_lr, 'mlp_acc': acc_mlp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()

    Xraw, y, pairs = load_csv(args.csv)
    results = {}
    for mode in MODES:
        X = build_features_multi(Xraw, mode, pairs)
        results[mode] = benchmark_all(X, y)
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = os.path.join(OUTDIR, f'{stem}_csv_bench.json')
    with open(out_path, 'w') as f:
        json.dump({'csv': args.csv, 'n': int(len(y)), 'modes': results}, f, indent=2)
    print('Wrote', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

