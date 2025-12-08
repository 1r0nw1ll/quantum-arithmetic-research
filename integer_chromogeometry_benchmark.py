#!/usr/bin/env python3
"""
Integer Chromogeometry Benchmark & Validation

Entry point: python integer_chromogeometry_benchmark.py

What it does
- Extracts integer-only chromogeometry features (HSI, MS) + LiDAR → 11D
- Compares against PCA+concat baseline across multiple classifiers
- Computes accuracy, latency, and estimated bandwidth per pixel
- Saves CSV and plots for tradeoff analysis next to this script

Data assumptions
- Expects Houston 2013 multimodal .mat files under multimodal_data/
  HSI_Tr.mat, LIDAR_Tr.mat, MS_Tr.mat, TrLabel.mat

Reproducibility
- Sets numpy RNG seed; deterministic sklearn configs

Notes
- Integer FFT uses fixed-point radix-2 with zero-padding for non-powers of two.
- For runtime, we subsample training points (configurable) to keep turnaround fast.
"""

import os
import time
import json
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    HAVE_SK = True
except Exception:
    HAVE_SK = False
import matplotlib.pyplot as plt

from chromogeometry_int import (
    ChromoIntConfig,
    spectral_to_chromo_int,
    chromo_fuse_int,
    fused_int_to_float,
)


# --- Config ---

RANDOM_SEED = 42
SUBSAMPLE = 40    # tiny subset to ensure quick integer run
SCALE_BITS = 14
RESULTS_DIR = Path("benchmarks")
RESULTS_DIR.mkdir(exist_ok=True)


def load_houston_data():
    base = Path('multimodal_data')
    req = [base / 'HSI_Tr.mat', base / 'LIDAR_Tr.mat', base / 'MS_Tr.mat', base / 'TrLabel.mat']
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Houston 2013 training files: " + ", ".join(missing) +
            "\nPlease place the multimodal .mat files under multimodal_data/ or adjust the paths."
        )
    hsi = sio.loadmat(base / 'HSI_Tr.mat')['Data']     # (2832, 11, 11, 144)
    lidar = sio.loadmat(base / 'LIDAR_Tr.mat')['Data'] # (2832, 11, 11, 1)
    ms = sio.loadmat(base / 'MS_Tr.mat')['Data']       # (2832, 11, 11, 8)
    labels = sio.loadmat(base / 'TrLabel.mat')['Data'].flatten()
    center = 5
    hsi_c = hsi[:, center, center, :]
    lidar_c = lidar[:, center, center, :].flatten()
    ms_c = ms[:, center, center, :]
    return hsi_c, ms_c, lidar_c, labels


def pca_numpy(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k].T  # (d,k)
    return Xc @ Vk


def compute_baselines(hsi_center: np.ndarray, ms_center: np.ndarray, lidar_center: np.ndarray):
    # PCA HSI 50 + LIDAR + MS concat -> 59D
    if HAVE_SK:
        pca = PCA(n_components=50, random_state=RANDOM_SEED)
        hsi_pca = pca.fit_transform(hsi_center)
    else:
        hsi_pca = pca_numpy(hsi_center, 50)
    concat = np.concatenate([hsi_pca, lidar_center[:, None], ms_center], axis=1)
    return hsi_pca, concat


def compute_integer_chromo(hsi_center: np.ndarray, ms_center: np.ndarray, lidar_center: np.ndarray,
                           cfg: ChromoIntConfig) -> np.ndarray:
    def downsample_to_len(vec: np.ndarray, target: int) -> np.ndarray:
        n = vec.shape[0]
        if n == target:
            return vec
        # partition vec into target groups (nearly equal) and mean
        edges = np.linspace(0, n, target + 1, dtype=int)
        out = np.empty((target,), dtype=vec.dtype)
        for i in range(target):
            a, b = edges[i], edges[i+1]
            out[i] = vec[a:b].mean() if b > a else vec[min(a, n-1)]
        return out

    feats = []
    for i in range(hsi_center.shape[0]):
        hsi_ds = downsample_to_len(hsi_center[i], 64)
        ms_ds = downsample_to_len(ms_center[i], 8)
        hsi_f = spectral_to_chromo_int(hsi_ds, cfg)
        ms_f = spectral_to_chromo_int(ms_ds, cfg)
        fused = chromo_fuse_int(hsi_f, ms_f, float(lidar_center[i]), cfg)
        feats.append(fused)
    feats = np.stack(feats, axis=0)
    # Convert to float for sklearn while preserving relative scales
    feats_float = np.vstack([fused_int_to_float(f, cfg) for f in feats])
    return feats_float


def estimate_ops_fft_radix2(n: int) -> int:
    """Rough real-op count for radix-2 FFT + magnitude scan + quadrances.

    Assumes: per complex multiply ~ 4 real mult + 2 real add; per butterfly add ~ 2 real add per complex output.
    Returns total real operations per spectrum.
    """
    import math
    s = int(math.log2(n))
    cmul = (n // 2) * s
    cadd = n * s
    real_mult = cmul * 4
    real_add = cmul * 2 + cadd * 2
    # Magnitudes for N/2+1 bins: approx max+min/2 (~3 ops/bin)
    mag_ops = (n // 2 + 1) * 3
    # Quadrances: ~5 mults + 2 adds
    quad_ops = 5 + 2
    return real_mult + real_add + mag_ops + quad_ops


def estimate_bandwidth(n_features: int, dtype_bytes: int) -> int:
    return int(n_features * dtype_bytes)


def _train_test_split_np(X, y, test_size=0.3, seed=RANDOM_SEED):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(round(test_size * n))
    te = idx[:n_test]
    tr = idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def _acc_1nn(Xtr, ytr, Xte, yte):
    correct = 0
    B = max(64, min(256, Xte.shape[0]))
    for i in range(0, Xte.shape[0], B):
        Xt = Xte[i:i+B]
        d2 = ((Xt[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2)
        nn = d2.argmin(axis=1)
        pred = ytr[nn]
        correct += (pred == yte[i:i+B]).sum()
    return correct / Xte.shape[0]


def run_classifiers(X: np.ndarray, y: np.ndarray, label: str):
    results = []
    if HAVE_SK:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
        )
        models = [
            ("RandomForest", RandomForestClassifier(n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1)),
            ("SVM-RBF", SVC(kernel='rbf', gamma='scale', C=4.0, random_state=RANDOM_SEED)),
            ("KNN-5", KNeighborsClassifier(n_neighbors=5)),
            ("LogReg", LogisticRegression(max_iter=1000, multi_class='auto', n_jobs=None, random_state=RANDOM_SEED)),
        ]
        from sklearn.metrics import accuracy_score
        for name, clf in models:
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({
                'variant': label,
                'model': name,
                'accuracy': acc,
                'train_time_s': train_time,
                'n_features': X.shape[1],
            })
    else:
        X_train, X_test, y_train, y_test = _train_test_split_np(X, y, test_size=0.3)
        t0 = time.time()
        acc = _acc_1nn(X_train, y_train, X_test, y_test)
        train_time = time.time() - t0
        results.append({
            'variant': label,
            'model': '1-NN (numpy)',
            'accuracy': acc,
            'train_time_s': train_time,
            'n_features': X.shape[1],
        })
    return results


def save_csv(rows, path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_tradeoff(rows, out_png: Path):
    # Aggregate max accuracy per n_features for clarity
    agg = {}
    for r in rows:
        key = (r['variant'], r['n_features'])
        agg[key] = max(agg.get(key, 0.0), r['accuracy'])
    xs, ys, labels = [], [], []
    for (variant, nfeat), acc in agg.items():
        xs.append(nfeat)
        ys.append(acc)
        labels.append(variant)
    plt.figure(figsize=(6,4))
    for variant in sorted(set(v for v, _ in agg.keys())):
        pts = [(n, a) for (v, n), a in agg.items() if v == variant]
        pts = sorted(pts)
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=variant)
    plt.xlabel('Feature dimension')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Feature Dimension (Houston 2013 subset)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)


def main():
    np.random.seed(RANDOM_SEED)

    # --- Section: Load data ---
    try:
        hsi_c, ms_c, lidar_c, labels = load_houston_data()
    except FileNotFoundError as e:
        print(str(e))
        print("Benchmark aborted. To proceed, add the required Houston .mat files.")
        return

    # Subsample for speed
    n = hsi_c.shape[0]
    idx = np.random.RandomState(RANDOM_SEED).permutation(n)[:min(SUBSAMPLE, n)]
    hsi_c = hsi_c[idx]
    ms_c = ms_c[idx]
    lidar_c = lidar_c[idx]
    labels_s = labels[idx]

    # --- Section: Compute features ---
    cfg = ChromoIntConfig(scale_bits=SCALE_BITS, pad_to_pow2=True, ignore_dc=True)
    t0 = time.time()
    chromo_feats = compute_integer_chromo(hsi_c, ms_c, lidar_c, cfg)
    t_chromo = time.time() - t0

    hsi_pca, concat = compute_baselines(hsi_c, ms_c, lidar_c)

    # --- Section: Bandwidth estimates ---
    bw_concat = estimate_bandwidth(concat.shape[1], 4)  # float32 per feature
    bw_chromo = estimate_bandwidth(chromo_feats.shape[1], 2)  # int16-equivalent payload
    # Ops estimate per spectrum (HSI padded length dominates)
    from chromogeometry_int import _next_power_of_two
    hsi_len = _next_power_of_two(hsi_c.shape[1])
    ms_len = _next_power_of_two(ms_c.shape[1])
    ops_est = estimate_ops_fft_radix2(hsi_len) + estimate_ops_fft_radix2(ms_len)

    # --- Section: Classifiers ---
    rows = []
    rows += run_classifiers(hsi_pca, labels_s, label=f"HSI-PCA50 ({hsi_pca.shape[1]}D)")
    rows += run_classifiers(concat, labels_s, label=f"Concat ({concat.shape[1]}D)")
    rows += run_classifiers(chromo_feats, labels_s, label=f"Chromogeometry ({chromo_feats.shape[1]}D)")

    # Annotate rows with bandwidth and latency metadata
    for r in rows:
        if 'Chromogeometry' in r['variant']:
            r['est_bandwidth_bytes_per_pixel'] = bw_chromo
            r['feature_payload_bytes'] = 2
            r['feature_count'] = chromo_feats.shape[1]
            r['feature_compute_time_s_total'] = t_chromo
            r['ops_estimate_per_sample'] = ops_est
        elif 'Concat' in r['variant']:
            r['est_bandwidth_bytes_per_pixel'] = bw_concat
            r['feature_payload_bytes'] = 4
            r['feature_count'] = concat.shape[1]
        else:
            r['est_bandwidth_bytes_per_pixel'] = estimate_bandwidth(hsi_pca.shape[1], 4)
            r['feature_payload_bytes'] = 4
            r['feature_count'] = hsi_pca.shape[1]

    # --- Section: Save outputs ---
    csv_path = RESULTS_DIR / 'integer_chromo_bench.csv'
    save_csv(rows, csv_path)

    plot_path = RESULTS_DIR / 'accuracy_vs_features.png'
    plot_tradeoff(rows, plot_path)

    # Print summary
    print("="*70)
    print("INTEGER CHROMOGEOMETRY BENCHMARK (subset)")
    print("="*70)
    print(f"Samples used: {labels_s.shape[0]}")
    print(f"Chromo feature compute time: {t_chromo:.2f}s")
    print(f"Bandwidth (per pixel): concat={bw_concat} bytes, chromo={bw_chromo} bytes")
    print(f"Estimated ops per sample (int chromo): {ops_est}")
    print()
    for r in rows:
        print(f"{r['variant']:<24} | {r['model']:<12} | acc={r['accuracy']:.4f} | train_s={r['train_time_s']:.2f}")
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == '__main__':
    main()
