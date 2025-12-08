#!/usr/bin/env python3
"""
Tiny one-shot QA benchmark on synthetic datasets.

- Datasets: moons, circles, swiss (swiss uses binned labels)
- Encodings for (b,e): first2, pca2, swiss_radangle, swiss_rady
- Builds QA feature maps (raw, qa21, qa27, qa83) from selected (b,e)
- Runs KMeans (ARI), LogisticRegression (Acc), MLPClassifier (Acc)
- Saves a small JSON summary under codex_on_QA/out
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split

from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector
from codex_on_QA.feature_maps.qa_feature_map_v4 import compute_qa_features_v4

OUTDIR = os.path.join('codex_on_QA','out')
os.makedirs(OUTDIR, exist_ok=True)

MODES = ["raw", "qa21", "qa27", "qa83", "qa96", "qa96lite", "qa100"]


def build_qa_features(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return X
    # v3 pointwise modes
    if mode in ("qa21", "qa27", "qa83"):
        feats = []
        for b, e in X[:, :2]:
            v, _names = qa_feature_vector(float(b), float(e), mode=mode)
            feats.append(v)
        return np.stack(feats, axis=0)
    # v4 relational modes
    if mode in ("qa96", "qa100"):
        b = X[:, 0].astype(float)
        e = X[:, 1].astype(float)
        feats = compute_qa_features_v4(b, e, mode=mode)
        names = sorted(feats.keys())
        return np.stack([feats[n] for n in names], axis=1)
    raise ValueError(f"Unknown mode: {mode}")


def benchmark_kmeans(X: np.ndarray, y: np.ndarray) -> float:
    km = KMeans(n_clusters=len(np.unique(y)), n_init=10, random_state=42)
    y_pred = km.fit_predict(X)
    return adjusted_rand_score(y, y_pred)


def benchmark_logreg(X: np.ndarray, y: np.ndarray, standardize: bool = False) -> float:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    if standardize:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="auto", n_jobs=-1)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def benchmark_mlp(X: np.ndarray, y: np.ndarray, standardize: bool = False) -> float:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    if standardize:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", max_iter=300, random_state=42)
    mlp.fit(Xtr, ytr)
    return accuracy_score(yte, mlp.predict(Xte))


def _dataset(dataset: str, n_samples: int = 1000, noise: float = 0.2):
    if dataset == 'moons':
        X_raw, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset == 'circles':
        X_raw, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset == 'swiss':
        X3, t = make_swiss_roll(n_samples=max(n_samples, 1500), noise=noise, random_state=42)
        # Use first two dims as (b,e), bin t into 5 classes
        X_raw = X3  # keep full for encoding phase
        import numpy as np
        bins = np.quantile(t, [0.2, 0.4, 0.6, 0.8])
        y = np.digitize(t, bins)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return X_raw, y


def be_from_encoding(dataset: str, X_raw, encoding: str):
    import numpy as np
    if encoding == 'first2':
        X2 = X_raw[:, :2]
        return X2
    if encoding == 'pca2':
        Xs = StandardScaler().fit_transform(X_raw)
        X2 = PCA(n_components=2, random_state=42).fit_transform(Xs)
        return X2
    if dataset == 'swiss' and encoding == 'swiss_radangle':
        # (b,e) = (radius in xz-plane, angle in xz-plane)
        x, y, z = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
        r = np.sqrt(x*x + z*z)
        ang = np.arctan2(z, x)
        return np.stack([r, ang], axis=1)
    if dataset == 'swiss' and encoding == 'swiss_rady':
        # (b,e) = (radius in xz-plane, y height)
        x, yv, z = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
        r = np.sqrt(x*x + z*z)
        return np.stack([r, yv], axis=1)
    # Fallback: first2
    return X_raw[:, :2]


def run_bench(dataset: str = 'moons', n_samples: int = 1000, noise: float = 0.2, show_plot: bool = True, encoding: str = 'first2', standardize: bool = False):
    X_raw, y = _dataset(dataset, n_samples=n_samples, noise=noise)
    X_be = be_from_encoding(dataset, X_raw, encoding)

    X_modes = {m: build_qa_features(X_be, m) for m in MODES}
    kmeans_scores = {m: benchmark_kmeans(X_modes[m], y) for m in MODES}
    logreg_scores = {m: benchmark_logreg(X_modes[m], y, standardize=standardize) for m in MODES}
    mlp_scores = {m: benchmark_mlp(X_modes[m], y, standardize=standardize) for m in MODES}

    idx = np.arange(len(MODES)); w = 0.25
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(idx - w, [kmeans_scores[m] for m in MODES], w, label="KMeans ARI")
    ax.bar(idx,     [logreg_scores[m]  for m in MODES], w, label="LogReg Acc")
    ax.bar(idx + w, [mlp_scores[m]     for m in MODES], w, label="MLP Acc")
    ax.set_xticks(idx); ax.set_xticklabels(MODES); ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score"); ax.set_title(f"QA One-Shot Benchmark: {dataset} ({encoding})")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"=== QA One-Shot Benchmark ({dataset}) ===")
    for m in MODES:
        print(f"{m:>4} | KMeans ARI={kmeans_scores[m]:.3f} | LogReg Acc={logreg_scores[m]:.3f} | MLP Acc={mlp_scores[m]:.3f}")

    # Save JSON summary
    summary = {
        'dataset': dataset,
        'encoding': encoding,
        'modes': {m: {
            'kmeans_ARI': float(kmeans_scores[m]),
            'logreg_acc': float(logreg_scores[m]),
            'mlp_acc': float(mlp_scores[m]),
        } for m in MODES}
    }
    out_path = os.path.join(OUTDIR, f'qa_one_shot_{dataset}.json') if encoding=='first2' else os.path.join(OUTDIR, f'qa_one_shot_{dataset}_{encoding}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Wrote', out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='moons', choices=['moons','circles','swiss'])
    ap.add_argument('--n-samples', type=int, default=1000)
    ap.add_argument('--noise', type=float, default=0.2)
    ap.add_argument('--encoding', default='first2', choices=['first2','pca2','swiss_radangle','swiss_rady'])
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--standardize', action='store_true', help='Standardize features before LogReg/MLP')
    args = ap.parse_args()
    run_bench(dataset=args.dataset, n_samples=args.n_samples, noise=args.noise, show_plot=not args.no_show, encoding=args.encoding, standardize=args.standardize)
