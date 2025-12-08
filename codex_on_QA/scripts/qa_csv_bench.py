#!/usr/bin/env python3
"""
QA CSV bench: run clustering/classification against (b,e) CSVs.

Input CSV schema (header required):
    id,b,e,label

Outputs JSON summary to codex_on_QA/out/<stem>_csv_bench.json with
metrics for raw, qa21, qa27, qa83:
    - kmeans_ARI
    - logreg_acc
    - mlp_acc

Run:
  python codex_on_QA/scripts/qa_csv_bench.py --csv path/to/data.csv
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
from sklearn.preprocessing import StandardScaler

from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector
from codex_on_QA.feature_maps.qa_feature_map_v4 import compute_qa_features_v4


OUTDIR = os.path.join('codex_on_QA', 'out')
os.makedirs(OUTDIR, exist_ok=True)


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ids: List[str] = []
    be: List[Tuple[float, float]] = []
    labels: List[int] = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {'id', 'b', 'e', 'label'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f'Missing columns. Found {reader.fieldnames}, need {sorted(required)}')
        for row in reader:
            ids.append(str(row['id']))
            be.append((float(row['b']), float(row['e'])))
            labels.append(int(row['label']))
    X = np.array(be, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y, ids


def build_features(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'raw':
        return X
    if mode in ('qa21','qa27','qa83'):
        feats: List[np.ndarray] = []
        for b, e in X[:, :2]:
            v, _ = qa_feature_vector(float(b), float(e), mode=mode)
            feats.append(v)
        return np.stack(feats, axis=0)
    if mode in ('qa96','qa100'):
        b = X[:,0].astype(float)
        e = X[:,1].astype(float)
        feats = compute_qa_features_v4(b, e, mode=mode)
        names = sorted(feats.keys())
        return np.stack([feats[n] for n in names], axis=1)
    raise ValueError(f'Unknown mode: {mode}')


def bench_all(X: np.ndarray, y: np.ndarray, modes=('raw','qa21','qa27','qa83','qa96','qa100'), standardize: bool = False) -> dict:
    out = {}
    k = len(np.unique(y))
    for m in modes:
        Xm = build_features(X, m)
        # KMeans ARI
        km = KMeans(n_clusters=k, n_init=20, random_state=0)
        y_km = km.fit_predict(Xm)
        ari = float(adjusted_rand_score(y, y_km))
        # LogReg Acc
        Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=0.3, random_state=0, stratify=y)
        if standardize:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr.fit(Xtr, ytr)
        acc_lr = float(accuracy_score(yte, lr.predict(Xte)))
        # MLP Acc
        mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', max_iter=300, random_state=0)
        mlp.fit(Xtr, ytr)
        acc_mlp = float(accuracy_score(yte, mlp.predict(Xte)))
        out[m] = {'kmeans_ARI': ari, 'logreg_acc': acc_lr, 'mlp_acc': acc_mlp}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--modes', default='raw,qa21,qa27,qa83')
    ap.add_argument('--standardize', action='store_true', help='Standardize features before LogReg/MLP')
    ap.add_argument('--outdir', default=OUTDIR)
    args = ap.parse_args()

    X, y, ids = load_csv(args.csv)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    results = bench_all(X, y, modes=tuple(modes), standardize=args.standardize)
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = os.path.join(args.outdir, f'{stem}_csv_bench.json')
    with open(out_path, 'w') as f:
        json.dump({'csv': args.csv, 'n': int(len(y)), 'modes': results}, f, indent=2)
    print('Wrote', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
