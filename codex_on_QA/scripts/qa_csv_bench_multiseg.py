#!/usr/bin/env python3
"""
CSV bench for multi-segment Raman encodings (3 (b,e) pairs per sample).

Input CSV schema:
  id,b1,e1,b2,e2,b3,e3,label

Modes:
  - raw: use [b1,e1,b2,e2,b3,e3]
  - qa21/qa27/qa83: per segment, build QA features via qa_feature_vector and concat

Outputs JSON summary to codex_on_QA/out/<stem>_csv_bench.json.
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


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ids: List[str] = []
    feats: List[List[float]] = []
    labels: List[int] = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        req = {'id','b1','e1','b2','e2','b3','e3','label'}
        if not req.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f'Missing columns. Found {reader.fieldnames}, need {sorted(req)}')
        for row in reader:
            ids.append(row['id'])
            feats.append([float(row['b1']), float(row['e1']), float(row['b2']), float(row['e2']), float(row['b3']), float(row['e3'])])
            labels.append(int(row['label']))
    X = np.asarray(feats, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y, ids


def build_features_multi(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'raw':
        return X
    out: List[np.ndarray] = []
    for row in X:
        segs = [(float(row[0]), float(row[1])), (float(row[2]), float(row[3])), (float(row[4]), float(row[5]))]
        parts: List[np.ndarray] = []
        for (b,e) in segs:
            vec, _ = qa_feature_vector(b, e, mode=mode)
            parts.append(vec)
        out.append(np.concatenate(parts, axis=0))
    return np.vstack(out)


def benchmark_kmeans(X: np.ndarray, y: np.ndarray) -> float:
    k = len(np.unique(y))
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    lab = km.fit_predict(X)
    return float(adjusted_rand_score(y, lab))


def benchmark_logreg(X: np.ndarray, y: np.ndarray) -> float:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))


def benchmark_mlp(X: np.ndarray, y: np.ndarray) -> float:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=300, random_state=0)
    mlp.fit(Xtr, ytr)
    return float(accuracy_score(yte, mlp.predict(Xte)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()

    Xraw, y, _ = load_csv(args.csv)
    results = {}
    for mode in MODES:
        X = build_features_multi(Xraw, mode)
        results[mode] = {
            'kmeans_ARI': benchmark_kmeans(X, y),
            'logreg_acc': benchmark_logreg(X, y),
            'mlp_acc': benchmark_mlp(X, y),
        }
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = os.path.join(OUTDIR, f'{stem}_csv_bench.json')
    with open(out_path, 'w') as f:
        json.dump({'csv': args.csv, 'n': int(len(y)), 'modes': results}, f, indent=2)
    print('Wrote', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

