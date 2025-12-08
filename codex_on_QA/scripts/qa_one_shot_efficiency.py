#!/usr/bin/env python3
"""
QA one-shot sample efficiency benchmark.

For dataset in {moons, circles} and encoding {first2, pca2}, compute the
minimum train size N needed to hit target accuracies (e.g., 0.90, 0.95) for
Logistic Regression and a small MLP, across feature modes {raw, qa21, qa27, qa83}.

Outputs JSON to codex_on_QA/out/qa_one_shot_efficiency_<dataset>_<encoding>.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector
from codex_on_QA.scripts.qa_one_shot_bench import be_from_encoding

OUTDIR = os.path.join('codex_on_QA', 'out')
os.makedirs(OUTDIR, exist_ok=True)

MODES = ["raw", "qa21", "qa27", "qa83"]


def dataset_xy(name: str, n_samples: int, noise: float, encoding: str):
    if name == 'moons':
        Xr, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif name == 'circles':
        Xr, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:
        raise SystemExit('dataset must be moons or circles')
    X_be = be_from_encoding(name, Xr, encoding)
    return X_be, y


def build_features(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'raw':
        return X
    feats = []
    for b, e in X[:, :2]:
        v, _ = qa_feature_vector(float(b), float(e), mode=mode)
        feats.append(v)
    return np.stack(feats, axis=0)


def min_n_for_threshold(X: np.ndarray, y: np.ndarray, mode: str, thresholds: List[float],
                        clf_kind: str = 'logreg', sizes: List[int] | None = None) -> Dict[str, int | None]:
    rng = np.random.RandomState(0)
    if sizes is None:
        sizes = [10, 20, 50, 100, 200, 400, 800, 1200]
    # Split once; we will subsample the training set deterministically
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Xt = build_features(Xtr, mode)
    Xe = build_features(Xte, mode)
    idx_all = np.arange(len(Xt))
    rng.shuffle(idx_all)
    hits = {str(th): None for th in thresholds}
    for n in sizes:
        n_use = min(n, len(idx_all))
        sel = idx_all[:n_use]
        Xn, yn = Xt[sel], ytr[sel]
        if clf_kind == 'logreg':
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        else:
            clf = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', max_iter=300, random_state=0)
        clf.fit(Xn, yn)
        acc = float((clf.predict(Xe) == yte).mean())
        for th in thresholds:
            key = str(th)
            if hits[key] is None and acc >= th:
                hits[key] = int(n_use)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='moons', choices=['moons','circles'])
    ap.add_argument('--encoding', default='first2', choices=['first2','pca2'])
    ap.add_argument('--n-samples', type=int, default=2000)
    ap.add_argument('--noise', type=float, default=0.2)
    ap.add_argument('--thresholds', default='0.9,0.95')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',') if x]
    X, y = dataset_xy(args.dataset, args.n_samples, args.noise, args.encoding)
    logreg = {}
    mlp = {}
    for mode in MODES:
        logreg[mode] = min_n_for_threshold(X, y, mode, thresholds, 'logreg')
        mlp[mode] = min_n_for_threshold(X, y, mode, thresholds, 'mlp')
    out = {
        'dataset': args.dataset,
        'encoding': args.encoding,
        'thresholds': thresholds,
        'logreg_min_n': logreg,
        'mlp_min_n': mlp,
    }
    path = os.path.join(OUTDIR, f'qa_one_shot_efficiency_{args.dataset}_{args.encoding}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

