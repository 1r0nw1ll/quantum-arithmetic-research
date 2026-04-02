#!/usr/bin/env python3
"""
HSI-only validation of integer chromogeometry on Indian Pines.

Purpose: sanity-check the core encoding (DFT→u,v→Qb/Qr/Qg) against a PCA baseline
on a dataset we already have locally, without any multi-modal dependencies.

Outputs under benchmarks/:
- indian_pines_chromo_hsi.csv
- indian_pines_accuracy_vs_features.png

Run:
  python integer_chromo_hsi_only_benchmark.py
"""

from pathlib import Path
import time
import csv
import numpy as np
import scipy.io as sio
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except Exception:
    HAVE_SK = False
import matplotlib.pyplot as plt

from chromogeometry_int import ChromoIntConfig, spectral_to_chromo_int, chromo_int_to_float


RANDOM_SEED = 42
SUBSAMPLE = 200  # labeled pixels, for speed
DATA_DIR = Path('hyperspectral_data')
RESULTS_DIR = Path('benchmarks')
RESULTS_DIR.mkdir(exist_ok=True)


def load_indian_pines():
    hsi_mat = sio.loadmat(DATA_DIR / 'Indian_pines_corrected.mat')
    gt_mat = sio.loadmat(DATA_DIR / 'Indian_pines_gt.mat')
    H = hsi_mat['indian_pines_corrected']  # (145,145,200)
    y = gt_mat['indian_pines_gt']          # (145,145)
    # flatten
    X = H.reshape(-1, H.shape[2]).astype(np.float64)
    yv = y.reshape(-1)
    # mask labeled
    mask = yv > 0
    X = X[mask]
    y = yv[mask]
    return X, y


def compute_chromo_features(X: np.ndarray, cfg: ChromoIntConfig) -> np.ndarray:
    feats = np.empty((X.shape[0], 5), dtype=np.float64)
    for i in range(X.shape[0]):
        f_int = spectral_to_chromo_int(X[i], cfg)
        feats[i] = chromo_int_to_float(f_int, cfg)
    return feats


def _train_test_split_np(X, y, test_size=0.3, seed=42):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(round(test_size * n))
    te = idx[:n_test]
    tr = idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def _acc_1nn(Xtr, ytr, Xte, yte):
    # Brute-force 1-NN
    # For memory safety, do in chunks
    correct = 0
    B = 256
    for i in range(0, Xte.shape[0], B):
        Xt = Xte[i:i+B]
        # distances shape: (B, n_train)
        d2 = (
            (Xt[:, None, :] - Xtr[None, :, :])
            ** 2
        ).sum(axis=2)
        nn = d2.argmin(axis=1)
        pred = ytr[nn]
        correct += (pred == yte[i:i+B]).sum()
    return correct / Xte.shape[0]


def run_models(X: np.ndarray, y: np.ndarray, label: str):
    rows = []
    if HAVE_SK:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED)
        models = [
            ("RandomForest", RandomForestClassifier(n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1)),
            ("SVM-RBF", SVC(kernel='rbf', gamma='scale', C=4.0, random_state=RANDOM_SEED)),
            ("KNN-5", KNeighborsClassifier(n_neighbors=5)),
            ("LogReg", LogisticRegression(max_iter=1000, multi_class='auto', n_jobs=None, random_state=RANDOM_SEED)),
        ]
        from sklearn.metrics import accuracy_score
        for name, clf in models:
            t0 = time.time()
            clf.fit(Xtr, ytr)
            train_time = time.time() - t0
            acc = accuracy_score(yte, clf.predict(Xte))
            rows.append({
                'variant': label,
                'model': name,
                'accuracy': acc,
                'train_time_s': train_time,
                'n_features': X.shape[1],
            })
    else:
        # Fallback: simple 1-NN
        Xtr, Xte, ytr, yte = _train_test_split_np(X, y, test_size=0.3, seed=RANDOM_SEED)
        t0 = time.time()
        acc = _acc_1nn(Xtr, ytr, Xte, yte)
        train_time = time.time() - t0
        rows.append({
            'variant': label,
            'model': '1-NN (numpy)',
            'accuracy': acc,
            'train_time_s': train_time,
            'n_features': X.shape[1],
        })
    return rows


def plot_tradeoff(rows, out_png: Path):
    # max accuracy per variant
    agg = {}
    for r in rows:
        key = (r['variant'], r['n_features'])
        agg[key] = max(agg.get(key, 0.0), r['accuracy'])
    plt.figure(figsize=(6,4))
    for variant in sorted(set(v for v, _ in agg.keys())):
        pts = sorted([(n, a) for (v, n), a in agg.items() if v == variant])
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=variant)
    plt.xlabel('Feature dimension')
    plt.ylabel('Accuracy')
    plt.title('Indian Pines: Accuracy vs Feature Dimension')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)


def main():
    np.random.seed(RANDOM_SEED)
    X, y = load_indian_pines()
    # Subsample
    if X.shape[0] > SUBSAMPLE:
        idx = np.random.RandomState(RANDOM_SEED).permutation(X.shape[0])[:SUBSAMPLE]
        X = X[idx]
        y = y[idx]

    # Baseline PCA (20D, 50D)
    rows = []
    if HAVE_SK:
        for k in [20, 50]:
            pca = PCA(n_components=k, random_state=RANDOM_SEED)
            Xp = pca.fit_transform(X)
            rows += run_models(Xp, y, label=f"PCA{k}")
    else:
        # Without sklearn, include a simple normalized-band baseline: mean/std per spectrum (2D)
        Xn = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        mean = Xn.mean(axis=1, keepdims=True)
        std = Xn.std(axis=1, keepdims=True)
        rows += run_models(np.hstack([mean, std]), y, label="MeanStd2D")

    # Chromogeometry 5D
    cfg = ChromoIntConfig(scale_bits=14)
    Xc = compute_chromo_features(X, cfg)
    rows += run_models(Xc, y, label="Chromo5D")

    # Save CSV
    csv_path = RESULTS_DIR / 'indian_pines_chromo_hsi.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant','model','accuracy','train_time_s','n_features'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot
    plot_path = RESULTS_DIR / 'indian_pines_accuracy_vs_features.png'
    plot_tradeoff(rows, plot_path)

    print("="*70)
    print("INDIAN PINES – HSI-ONLY CHROMOGEOMETRY VALIDATION")
    print("="*70)
    print(f"Samples: {X.shape[0]}, Bands: {X.shape[1]}")
    for r in rows:
        print(f"{r['variant']:<10} | {r['model']:<12} | acc={r['accuracy']:.4f} | train_s={r['train_time_s']:.2f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
