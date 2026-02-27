#!/usr/bin/env python3
"""
HSI Patch Generalization Experiment
Tests whether spatial patch statistics dominate spectral-only baselines
on standard HSI benchmarks (Indian Pines, PaviaU).

Usage:
  python test_hsi_patch_generalization.py --dataset indian_pines --patch-sweep 3,5,7 --seed 42
  python test_hsi_patch_generalization.py --dataset pavia --patch-sweep 3,5,7 --seed 42
"""

import argparse
parser = argparse.ArgumentParser(description="HSI Patch Generalization Experiment")
parser.add_argument("--dataset", choices=["indian_pines", "pavia", "salinas", "ksc"],
                    default="indian_pines")
parser.add_argument("--patch-sweep", type=str, default="3,5,7",
                    help="Comma-separated patch sizes to test (default: '3,5,7')")
parser.add_argument("--patch-size", type=int, default=5,
                    help="Single patch size when not sweeping (default: 5)")
parser.add_argument("--train-frac", type=float, default=0.1,
                    help="Fraction of labeled pixels per class for training (default: 0.1 = 10%%)")
parser.add_argument("--n-components", type=int, default=30,
                    help="PCA components for spectral baseline (default: 30)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import scipy.io as sio
import numpy as np
from pathlib import Path

DATA_DIR = Path("hyperspectral_data")

DATASET_FILES = {
    "indian_pines": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
    "pavia":        ("PaviaU.mat",                 "PaviaU_gt.mat"),
    "salinas":      ("Salinas_corrected.mat",       "Salinas_gt.mat"),
    "ksc":          ("KSC.mat",                     "KSC_gt.mat"),
}

def load_dataset(name):
    data_f, gt_f = DATASET_FILES[name]
    d = sio.loadmat(DATA_DIR / data_f)
    g = sio.loadmat(DATA_DIR / gt_f)
    dk = [k for k in d if not k.startswith("__")][0]
    gk = [k for k in g if not k.startswith("__")][0]
    image = d[dk].astype(np.float32)   # (H, W, C)
    gt    = g[gk].astype(np.int32)     # (H, W)
    # Normalize image to [0,1] per band
    mn = image.min(axis=(0,1), keepdims=True)
    mx = image.max(axis=(0,1), keepdims=True)
    image = (image - mn) / (mx - mn + 1e-8)
    return image, gt

image, gt = load_dataset(args.dataset)
H, W, C = image.shape
print(f"Dataset: {args.dataset}")
print(f"  Image: {image.shape}, GT: {gt.shape}")
classes = sorted([c for c in np.unique(gt) if c > 0])
print(f"  Classes: {len(classes)}, labeled pixels: {np.sum(gt > 0)}")

# ---------------------------------------------------------------------------
# Stratified train/test split (10% per class)
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

np.random.seed(args.seed)

labeled_rows, labeled_cols = np.where(gt > 0)
labeled_labels = gt[labeled_rows, labeled_cols]

idx_all = np.arange(len(labeled_rows))
idx_train, idx_test = train_test_split(
    idx_all,
    test_size=1.0 - args.train_frac,
    stratify=labeled_labels,
    random_state=args.seed
)
y_train = labeled_labels[idx_train]
y_test  = labeled_labels[idx_test]
rows_train = labeled_rows[idx_train]
cols_train = labeled_cols[idx_train]
rows_test  = labeled_rows[idx_test]
cols_test  = labeled_cols[idx_test]

print(f"  Train: {len(idx_train)}, Test: {len(idx_test)}")

# ---------------------------------------------------------------------------
# Patch extraction function
# ---------------------------------------------------------------------------
def extract_patches(image, rows, cols, patch_size):
    """
    Extract patch_size x patch_size patches centered at (row, col).
    Uses reflect padding at image borders.
    Returns (N, patch_size, patch_size, C).
    """
    H, W, C = image.shape
    hw = patch_size // 2
    # Pad image with reflect mode
    padded = np.pad(image, ((hw, hw), (hw, hw), (0, 0)), mode='reflect')
    N = len(rows)
    patches = np.zeros((N, patch_size, patch_size, C), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        patches[i] = padded[r:r+patch_size, c:c+patch_size, :]
    return patches

# ---------------------------------------------------------------------------
# Compute patch features (same 14-stat logic, HSI-only)
# ---------------------------------------------------------------------------
from sklearn.decomposition import PCA

def compute_hsi_patch_features(patches, pca=None):
    """
    patches: (N, ps, ps, C)
    Returns (N, ~14) feature array.
    pca: fitted PCA(3) on train patch pixels (fit on train, apply to test)
    """
    N, ps, ps2, C = patches.shape
    flat = patches.reshape(N, -1)   # (N, ps*ps*C)

    # Global spectral stats over patch
    hsi_mean = flat.mean(axis=1)    # (N,)
    hsi_std  = flat.std(axis=1)     # (N,)

    # Gradient magnitude: use first band's center region
    cx = ps // 2
    if ps >= 3:
        b0 = patches[:, :, :, 0]   # (N, ps, ps) first band
        gx = b0[:, cx, cx+1] - b0[:, cx, cx-1]
        gy = b0[:, cx+1, cx] - b0[:, cx-1, cx]
        grad_mag = np.sqrt(gx*gx + gy*gy)
    else:
        grad_mag = np.zeros(N)

    # Local variance proxy: std of per-pixel band-mean
    pixel_means = patches.mean(axis=3)  # (N, ps, ps)
    local_var = pixel_means.reshape(N, -1).std(axis=1)

    # PCA-3 patch stats
    pixels = patches.reshape(N * ps * ps, C)
    if pca is not None:
        proj = pca.transform(pixels).reshape(N, ps * ps, 3)
        pca_mean = proj.mean(axis=1)   # (N, 3)
        pca_std  = proj.std(axis=1)    # (N, 3)
    else:
        pca_mean = np.zeros((N, 3))
        pca_std  = np.zeros((N, 3))

    feats = np.column_stack([
        hsi_mean, hsi_std,      # 2
        grad_mag,               # 1
        local_var,              # 1
        pca_mean, pca_std       # 6
    ])   # total: 10
    return feats

# ---------------------------------------------------------------------------
# Spectral-only baseline features
# ---------------------------------------------------------------------------
def compute_spectral_features(image, rows, cols, pca=None):
    """
    Center-pixel spectral features, optionally PCA-reduced.
    """
    pixels = image[rows, cols, :]  # (N, C)
    if pca is not None:
        return pca.transform(pixels)
    return pixels

# ---------------------------------------------------------------------------
# Main experiment: baseline + patch sweep
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
import time

print()
print("=" * 70)
print(f"SPECTRAL BASELINE (PCA-{args.n_components} center pixel)")
print("=" * 70)

# Spectral PCA baseline
pca_spec = PCA(n_components=args.n_components, random_state=args.seed)
X_spec_train = compute_spectral_features(image, rows_train, cols_train)
pca_spec.fit(X_spec_train)
X_spec_train_r = pca_spec.transform(X_spec_train)
X_spec_test_r  = compute_spectral_features(image, rows_test, cols_test)
X_spec_test_r  = pca_spec.transform(X_spec_test_r)

t0 = time.time()
clf_spec = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
clf_spec.fit(X_spec_train_r, y_train)
pred_spec = clf_spec.predict(X_spec_test_r)
acc_spec  = accuracy_score(y_test, pred_spec)
aa_spec   = balanced_accuracy_score(y_test, pred_spec)
kap_spec  = cohen_kappa_score(y_test, pred_spec)
print(f"  PCA-{args.n_components} spectral (center pixel): OA={acc_spec:.4f}  AA={aa_spec:.4f}  κ={kap_spec:.4f}  ({time.time()-t0:.2f}s, {args.n_components}D)")

# Also raw spectral (no PCA) for reference
t0 = time.time()
clf_raw = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
clf_raw.fit(X_spec_train, y_train)
pred_raw = clf_raw.predict(compute_spectral_features(image, rows_test, cols_test))
acc_raw  = accuracy_score(y_test, pred_raw)
aa_raw   = balanced_accuracy_score(y_test, pred_raw)
kap_raw  = cohen_kappa_score(y_test, pred_raw)
print(f"  Raw spectral (center pixel, {C}D): OA={acc_raw:.4f}  AA={aa_raw:.4f}  κ={kap_raw:.4f}  ({time.time()-t0:.2f}s)")

print()
print("=" * 70)
print(f"PATCH SIZE SWEEP (seed={args.seed})")
print("=" * 70)
print(f"{'ps':<6} {'dim':<6} {'OA':<8} {'AA':<8} {'κ':<8} {'ΔOA/PCA':<10} {'ΔAA/PCA'}")
print("-" * 70)

patch_sizes = [int(x.strip()) for x in args.patch_sweep.split(',') if x.strip()]

results = {}
results_aa = {}
results_kap = {}
for ps in patch_sizes:
    # Extract patches
    patches_train = extract_patches(image, rows_train, cols_train, ps)
    patches_test  = extract_patches(image, rows_test,  cols_test,  ps)

    # Fit PCA on train patch pixels
    train_pixels = patches_train.reshape(-1, C)
    # Subsample if too large (>500k pixels)
    if len(train_pixels) > 500_000:
        idx_sub = np.random.choice(len(train_pixels), 500_000, replace=False)
        train_pixels = train_pixels[idx_sub]
    pca_patch = PCA(n_components=3, random_state=args.seed)
    pca_patch.fit(train_pixels)

    X_patch_train = compute_hsi_patch_features(patches_train, pca_patch)
    X_patch_test  = compute_hsi_patch_features(patches_test,  pca_patch)

    t0 = time.time()
    clf_patch = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
    clf_patch.fit(X_patch_train, y_train)
    pred_patch = clf_patch.predict(X_patch_test)
    acc_patch = accuracy_score(y_test, pred_patch)
    aa_patch  = balanced_accuracy_score(y_test, pred_patch)
    kap_patch = cohen_kappa_score(y_test, pred_patch)
    delta_oa  = acc_patch - acc_spec
    delta_aa  = aa_patch  - aa_spec

    feat_dim = X_patch_train.shape[1]
    print(f"{ps:<6} {feat_dim:<6} {acc_patch:<8.4f} {aa_patch:<8.4f} {kap_patch:<8.4f} {delta_oa:+.4f}{'':>4} {delta_aa:+.4f}")
    results[ps] = acc_patch
    results_aa[ps]  = aa_patch
    results_kap[ps] = kap_patch

print()
best_ps = max(results, key=results.get)
print(f"Best patch size: {best_ps}×{best_ps}")
print(f"  OA:  patch={results[best_ps]:.4f}  spectral-PCA={acc_spec:.4f}  ΔOA={results[best_ps]-acc_spec:+.4f}")
print(f"  AA:  patch={results_aa[best_ps]:.4f}  spectral-PCA={aa_spec:.4f}  ΔAA={results_aa[best_ps]-aa_spec:+.4f}")
print(f"  κ:   patch={results_kap[best_ps]:.4f}  spectral-PCA={kap_spec:.4f}  Δκ={results_kap[best_ps]-kap_spec:+.4f}")
if results[best_ps] > acc_spec:
    print("Patch-only BEATS spectral baseline -- locality dominance holds.")
else:
    print("Spectral baseline wins -- patch dominance does NOT generalize here.")
