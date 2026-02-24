#!/usr/bin/env python3
"""
Multi-Modal Fusion with Chromogeometry
Test HSI + LIDAR + MS fusion using QA chromogeometry framework
"""

import argparse
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# CLI args
parser = argparse.ArgumentParser(description="Multi-Modal Fusion with Chromogeometry")
parser.add_argument("--gated", choices=["on", "off"], default="on",
                    help="Include gated-concat variants (default: on)")
parser.add_argument("--gating-temp", type=float, default=1.0,
                    help="Temperature for entropy gating softmax (default: 1.0)")
parser.add_argument("--gating-sweep", type=str, default="",
                    help="Comma-separated temps to sweep for entropy gating (e.g., '0.5,1.0,2.5')")
parser.add_argument("--gate-signal", choices=["area", "entropy", "margin", "gini"], default="entropy",
                    help="Uncertainty signal for gated-concat: area, entropy, margin, or gini (default: entropy)")
parser.add_argument("--chromo-head", choices=["rf", "gbdt", "svm"], default="rf",
                    help="Classifier head for Chromogeometry methods: rf, gbdt, or svm (default: rf)")
parser.add_argument("--patch-stats", action="store_true", default=False,
                    help="Append patch summary statistics to chromogeometry feature vectors")
parser.add_argument("--patch-size", type=int, default=11, choices=[3, 5, 7, 11],
                    help="Spatial window size for patch stats (default: 11; requires --patch-stats)")
parser.add_argument("--patch-sweep", type=str, default="",
                    help="Comma-separated patch sizes to sweep, e.g. '3,5,7,11' (requires --patch-stats)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/test split and all classifiers (default: 42)")
parser.add_argument("--patch-only-ablation", action="store_true", default=False,
                    help="Add a 'Patch Only (no chromo)' method to compare patch stats without chromogeometry features")
args = parser.parse_args()
seed = args.seed
include_gated = (args.gated == "on")
gating_temp = float(args.gating_temp) if getattr(args, 'gating_temp', 1.0) and args.gating_temp > 0 else 1.0
gate_signal = args.gate_signal
chromo_head = args.chromo_head
use_patch_stats = args.patch_stats
patch_only_ablation = args.patch_only_ablation
patch_size = args.patch_size
patch_sweep = []
if args.patch_sweep:
    try:
        patch_sweep = [int(x.strip()) for x in args.patch_sweep.split(',') if x.strip()]
    except Exception:
        patch_sweep = []
gating_sweep = []
if getattr(args, 'gating_sweep', ""):
    try:
        gating_sweep = [float(x.strip()) for x in args.gating_sweep.split(',') if x.strip()]
    except Exception:
        gating_sweep = []

print("="*70)
print("MULTI-MODAL FUSION WITH CHROMOGEOMETRY")
print("="*70)
print()

# Load data
print("[1/6] Loading multi-modal data...")
hsi = sio.loadmat('multimodal_data/HSI_Tr.mat')['Data']     # (2832, 11, 11, 144)
lidar = sio.loadmat('multimodal_data/LIDAR_Tr.mat')['Data'] # (2832, 11, 11, 1)
ms = sio.loadmat('multimodal_data/MS_Tr.mat')['Data']       # (2832, 11, 11, 8)
labels = sio.loadmat('multimodal_data/TrLabel.mat')['Data'].flatten() # (2832,)

N_samples = hsi.shape[0]
print(f"  Samples: {N_samples}")
print(f"  HSI: {hsi.shape} (144 bands)")
print(f"  LIDAR: {lidar.shape} (elevation)")
print(f"  MS: {ms.shape} (8 bands)")
num_classes = len(np.unique(labels))
print(f"  Classes: {num_classes}")
print()

# Extract center pixel spectra (simple approach for speed)
print("[2/6] Extracting center pixel features...")
center = 5  # Center of 11×11 patch
hsi_center = hsi[:, center, center, :]  # (N_hsi, B_hsi)
lidar_center = lidar[:, center, center, :].flatten()  # (N_lidar,)
ms_center = ms[:, center, center, :]  # (N_ms, 8)

print(f"  HSI center: {hsi_center.shape}")
print(f"  LIDAR center: {lidar_center.shape}")
print(f"  MS center: {ms_center.shape}")
print()

# Detect modality sample mismatch and choose aligned mode
n_hsi = hsi_center.shape[0]
n_lidar = lidar_center.shape[0]
n_ms = ms_center.shape[0]
n_labels = labels.shape[0]
has_all = (n_hsi == n_lidar == n_ms == n_labels)
hsi_lidar_ok = (n_hsi == n_lidar == n_labels)
ms_lidar_ok = (n_ms == n_lidar == n_labels)
mode = 'ALL' if has_all else ('HSI_LIDAR' if hsi_lidar_ok else 'MS_LIDAR')
if mode != 'ALL':
    print("⚠️  Sample count mismatch across modalities.")
    print(f"    HSI: {n_hsi}, LIDAR: {n_lidar}, MS: {n_ms}, Labels: {n_labels}")
    pretty = {'HSI_LIDAR': 'HSI + LIDAR', 'MS_LIDAR': 'MS + LIDAR'}
    print(f"    Proceeding with {pretty[mode]} only based on alignment.\n")

# Amplitude anchors
if mode != 'MS_LIDAR':
    hsi_mean = hsi_center.mean(axis=1)  # (N,)
else:
    hsi_mean = None
ms_mean = ms_center.mean(axis=1)    # (N,)

# NDVI features (from HSI and/or MS center spectra)
def approx_wavelengths_144():
    return np.linspace(364.0, 1046.1, 144)

def ndvi_from_hsi_vec(vec_144: np.ndarray) -> float:
    wl = approx_wavelengths_144()
    red_mask = (wl >= 630) & (wl <= 690)
    nir_mask = (wl >= 860) & (wl <= 1040)
    red = float(vec_144[red_mask].mean()) if np.any(red_mask) else float(vec_144[70])
    nir = float(vec_144[nir_mask].mean()) if np.any(nir_mask) else float(vec_144[120])
    denom = (nir + red)
    return float((nir - red) / denom) if denom != 0 else 0.0

def ndvi_from_ms_vec(vec_8: np.ndarray) -> float:
    # WV2 order assumption: [Coastal, Blue, Green, Yellow, Red, RedEdge, NIR1, NIR2]
    red = float(vec_8[4]) if vec_8.shape[0] >= 5 else 0.0
    nir = float(vec_8[6]) if vec_8.shape[0] >= 7 else 0.0
    denom = (nir + red)
    return float((nir - red) / denom) if denom != 0 else 0.0

# Additional indices: NDRE and GNDVI
def ndre_from_hsi_vec(vec_144: np.ndarray) -> float:
    wl = approx_wavelengths_144()
    rededge_mask = (wl >= 705) & (wl <= 745)
    nir_mask = (wl >= 860) & (wl <= 1040)
    re = float(vec_144[rededge_mask].mean()) if np.any(rededge_mask) else float(vec_144[90])
    nir = float(vec_144[nir_mask].mean()) if np.any(nir_mask) else float(vec_144[120])
    denom = (nir + re)
    return float((nir - re) / denom) if denom != 0 else 0.0

def ndre_from_ms_vec(vec_8: np.ndarray) -> float:
    # RedEdge index 5, NIR1 index 6
    re = float(vec_8[5]) if vec_8.shape[0] >= 6 else 0.0
    nir = float(vec_8[6]) if vec_8.shape[0] >= 7 else 0.0
    denom = (nir + re)
    return float((nir - re) / denom) if denom != 0 else 0.0

def gndvi_from_hsi_vec(vec_144: np.ndarray) -> float:
    wl = approx_wavelengths_144()
    green_mask = (wl >= 510) & (wl <= 580)
    nir_mask = (wl >= 860) & (wl <= 1040)
    g = float(vec_144[green_mask].mean()) if np.any(green_mask) else float(vec_144[35])
    nir = float(vec_144[nir_mask].mean()) if np.any(nir_mask) else float(vec_144[120])
    denom = (nir + g)
    return float((nir - g) / denom) if denom != 0 else 0.0

def gndvi_from_ms_vec(vec_8: np.ndarray) -> float:
    # Green index 2, NIR1 index 6
    g = float(vec_8[2]) if vec_8.shape[0] >= 3 else 0.0
    nir = float(vec_8[6]) if vec_8.shape[0] >= 7 else 0.0
    denom = (nir + g)
    return float((nir - g) / denom) if denom != 0 else 0.0

# Method 1: Baseline (HSI only)
if mode != 'MS_LIDAR':
    print("[3/6] Method 1: Baseline (HSI only, PCA 50 components)...")
    pca_all = PCA(n_components=50)
    hsi_pca_all = pca_all.fit_transform(hsi_center)
    print(f"  Explained variance: {pca_all.explained_variance_ratio_[:10].sum():.3f} (first 10 PCs)")
else:
    print("[3/6] Skipping HSI-only baseline (HSI missing or misaligned)...")

# Method 2: Simple concatenation (HSI + LIDAR + MS)
if mode == 'ALL':
    print("[4/6] Method 2: Simple concatenation (HSI PCA + LIDAR + MS)...")
    concat_features = np.concatenate([hsi_pca_all, lidar_center[:, None], ms_center], axis=1)
    print(f"  Feature dimension: {concat_features.shape[1]}")
elif mode == 'MS_LIDAR':
    print("[4/6] Method 2: Simple concatenation (MS + LIDAR)...")
    concat_features = np.concatenate([lidar_center[:, None], ms_center], axis=1)
    print(f"  Feature dimension: {concat_features.shape[1]}")
elif mode == 'HSI_LIDAR':
    print("[4/6] Method 2: Simple concatenation (HSI PCA + LIDAR)...")
    concat_features = np.concatenate([hsi_pca_all, lidar_center[:, None]], axis=1)
    print(f"  Feature dimension: {concat_features.shape[1]}")

# Method 3: Chromogeometry fusion
print("[5/6] Method 3: Chromogeometry fusion...")

def spectral_to_chromo_simple(spectrum):
    """Simple chromogeometry encoding of spectrum"""
    # DFT
    fft = np.fft.rfft(spectrum - spectrum.mean())
    mag = np.abs(fft)
    phs = np.angle(fft)

    # Skip DC, find top peak
    mag[0] = 0
    if len(mag) > 1:
        top_idx = np.argmax(mag[1:]) + 1
        u = top_idx / len(mag)  # Normalized frequency
        v = phs[top_idx] / (2 * np.pi)  # Normalized phase
    else:
        u, v = 0.0, 0.0

    # Quadrances
    Qb = u**2 + v**2
    Qr = u**2 - v**2
    Qg = 2 * u * v

    return np.array([u, v, Qb, Qr, Qg])

# Apply chromogeometry to HSI and MS
if mode != 'MS_LIDAR':
    print("  Applying chromogeometry to HSI...")
    hsi_chromo = np.array([spectral_to_chromo_simple(s) for s in hsi_center])
else:
    hsi_chromo = None

print("  Applying chromogeometry to MS...")
ms_chromo = np.array([spectral_to_chromo_simple(s) for s in ms_center])

if mode == 'ALL':
    # Combine: HSI chromo + MS chromo + LIDAR
    chromo_features = np.concatenate([
        hsi_chromo,           # HSI chromogeometry
        ms_chromo,            # MS chromogeometry
        lidar_center[:, None] # LIDAR elevation
    ], axis=1)
elif mode == 'MS_LIDAR':
    # Combine: MS chromo + LIDAR
    chromo_features = np.concatenate([
        ms_chromo,            # MS chromogeometry (5)
        lidar_center[:, None] # LIDAR elevation (1)
    ], axis=1)
elif mode == 'HSI_LIDAR':
    # Combine: HSI chromo + LIDAR
    chromo_features = np.concatenate([
        hsi_chromo,           # HSI chromogeometry (5)
        lidar_center[:, None] # LIDAR elevation (1)
    ], axis=1)

print(f"  Chromogeometry feature dimension: {chromo_features.shape[1]}")
if mode != 'MS_LIDAR':
    print(f"    - HSI chromo: 5D (u, v, Qb, Qr, Qg)")
print(f"    - MS chromo: 5D (u, v, Qb, Qr, Qg)")
print(f"    - LIDAR: 1D (elevation)")
print()

# Chromo + amplitude anchors
if mode == 'ALL':
    chromo_features_amp = np.concatenate([
        chromo_features,
        hsi_mean[:, None],
        ms_mean[:, None]
    ], axis=1)
    print("  With amplitude anchors (+2D):")
    print(f"    - HSI mean, MS mean added → {chromo_features_amp.shape[1]} dims")
elif mode == 'MS_LIDAR':
    chromo_features_amp = np.concatenate([
        chromo_features,
        ms_mean[:, None]
    ], axis=1)
    print("  With amplitude anchor (+1D):")
    print(f"    - MS mean added → {chromo_features_amp.shape[1]} dims")
elif mode == 'HSI_LIDAR':
    chromo_features_amp = np.concatenate([
        chromo_features,
        hsi_mean[:, None]
    ], axis=1)
    print("  With amplitude anchor (+1D):")
    print(f"    - HSI mean added → {chromo_features_amp.shape[1]} dims")
print()

# Patch stats (computed once, applied in training loop)
if use_patch_stats:
    print("  Computing patch summary statistics...")
    from sklearn.decomposition import PCA as _PCA
    # Fit PCA(3) on ALL train pixels (idx_train not yet defined here — defer to loop)
    # Store raw arrays; apply per-method in training loop where idx_train is available
    _patch_stats_ready = True
    print(f"  Patch stats will be appended to Chromogeometry features in training loop.")
else:
    _patch_stats_ready = False

# Patch-only feature array (built once; used in training loop when --patch-only-ablation)
# Will be computed inside the training loop per-split to avoid leakage
patch_only_features = None  # placeholder; actual computation is split-safe in loop

if mode == 'ALL':
    X_methods = {
        'HSI Only (PCA 50)': None,
        'HSI+LIDAR+MS (concat)': None,
    }
    if include_gated:
        X_methods['HSI+LIDAR+MS (gated-concat)'] = None
        X_methods['HSI+LIDAR+MS (gated-concat-entropy)'] = None
    X_methods.update({
        'HSI+LIDAR+MS (concat+NDVI+std)': None,
        'HSI+LIDAR+MS (concat+spatial+std)': None,
        'HSI+LIDAR+MS (HGB)': None,
        'Late Fusion (avg-calibrated)': None,
        'Late Fusion (avg-calibrated-iso)': None,
        'Late Fusion (avg-temp)': None,
        'Late Fusion (avg)': None,
        'Late Fusion (stacking)': None,
        'HSI+LIDAR+MS (concat+LDA)': None,  # LDA fitted on train split
        'HSI+LIDAR+MS (concat+PCA15)': None,
        'HSI+LIDAR+MS (concat+PCA20)': None,
        'Chromogeometry Fusion': chromo_features,
        'Chromogeometry Fusion + Amp Anchors (+2)': chromo_features_amp
    })
    if patch_only_ablation and use_patch_stats:
        X_methods['Patch Only (no chromo)'] = None  # computed in loop
elif mode == 'MS_LIDAR':
    X_methods = {
        'MS+LIDAR (concat)': None,
    }
    if include_gated:
        X_methods['MS+LIDAR (gated-concat)'] = None
        X_methods['MS+LIDAR (gated-concat-entropy)'] = None
    X_methods.update({
        'MS+LIDAR (concat+spatial+std)': None,
        'MS+LIDAR (concat+LDA)': None,
        'MS+LIDAR (concat+PCA15)': None,
        'MS+LIDAR (concat+PCA20)': None,
        'Chromogeometry Fusion (MS only)': chromo_features,
        'Chromogeometry Fusion (MS only) + Amp Anchor (+1)': chromo_features_amp
    })
    if patch_only_ablation and use_patch_stats:
        X_methods['Patch Only (no chromo)'] = None  # computed in loop
elif mode == 'HSI_LIDAR':
    X_methods = {
        'HSI Only (PCA 50)': None,
        'HSI+LIDAR (concat)': None,
    }
    if include_gated:
        X_methods['HSI+LIDAR (gated-concat)'] = None
        X_methods['HSI+LIDAR (gated-concat-entropy)'] = None
    X_methods.update({
        'HSI+LIDAR (concat+spatial+std)': None,
        'HSI+LIDAR (concat+LDA)': None,
        'HSI+LIDAR (concat+PCA15)': None,
        'HSI+LIDAR (concat+PCA20)': None,
        'Chromogeometry Fusion (HSI only)': chromo_features,
        'Chromogeometry Fusion (HSI only) + Amp Anchor (+1)': chromo_features_amp
    })
    if patch_only_ablation and use_patch_stats:
        X_methods['Patch Only (no chromo)'] = None  # computed in loop
def compute_gate_weights(P_list, signal, T, eps=1e-8):
    """
    P_list: list of (N, C) probability arrays, one per modality (in order: hsi, ms, lidar)
    signal: 'entropy' | 'margin' | 'gini'
    T: temperature (float > 0)
    Returns: (N, M) weight array where M = len(P_list)
    """
    import numpy as np
    def _entropy(p):
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=1)
    def _margin(p):
        sorted_p = np.sort(p, axis=1)
        return sorted_p[:, -1] - sorted_p[:, -2]
    def _gini(p):
        return 1.0 - np.sum(p ** 2, axis=1)

    M = len(P_list)
    N = P_list[0].shape[0]
    U = np.zeros((N, M))
    for i, p in enumerate(P_list):
        if signal == 'entropy':
            U[:, i] = _entropy(p)          # high = uncertain
        elif signal == 'margin':
            U[:, i] = -_margin(p)          # low margin = uncertain → negate so high = uncertain
        elif signal == 'gini':
            U[:, i] = _gini(p)             # high = uncertain

    # softmax(U / T) per sample
    X = U / max(T, 1e-6)
    X = X - X.max(axis=1, keepdims=True)
    E = np.exp(X)
    W = E / (E.sum(axis=1, keepdims=True) + eps)
    return W


def print_gate_diagnostics(W_train, W_test, y_train, y_test, y_pred_train, y_pred_test, tag=""):
    """Print per-modality mean weights, saturation rate, corr(max_gate_weight, correct)."""
    import numpy as np
    mod_names = ["HSI", "MS", "LIDAR"]
    print(f"  [{tag}] Gate diagnostics:")
    for split_name, W, y_true, y_pred in [("train", W_train, y_train, y_pred_train),
                                            ("test",  W_test,  y_test,  y_pred_test)]:
        mean_w = W.mean(axis=0)
        sat = np.mean((W > 0.9).any(axis=1) | (W < 0.1).any(axis=1)) * 100
        correct = (y_pred == y_true).astype(float)
        max_gw = W.max(axis=1)
        if max_gw.std() > 1e-10 and correct.std() > 1e-10:
            corr = float(np.corrcoef(max_gw, correct)[0, 1])
        else:
            corr = float('nan')
        print(f"    {split_name}: mean_w={[f'{v:.3f}' for v in mean_w]} sat={sat:.1f}% corr(max_w,correct)={corr:.3f}")



def compute_patch_stats(hsi, lidar, ms, idx, pca_hsi_patch=None, patch_size=11):
    """
    Compute patch summary statistics for samples in idx.
    hsi: (N, 11, 11, 144)
    lidar: (N, 11, 11, 1) full array
    ms: (N, 11, 11, 8)
    idx: sample indices to process
    pca_hsi_patch: fitted PCA(3) on hsi patches (optional, fit on idx_train if None)
    patch_size: spatial window size (3, 5, 7, or 11); center-cropped from the 11x11 patch
    Returns: (len(idx), ~14) feature array
    """
    import numpy as np
    hw = patch_size // 2
    cx = 5  # center in 11x11 grid
    r0, r1 = cx - hw, cx + hw + 1

    hsi_patch = hsi[idx][:, r0:r1, r0:r1, :]   # (N, ps, ps, 144)
    ms_patch  = ms[idx][:, r0:r1, r0:r1, :]    # (N, ps, ps, 8)
    lidar_patch = lidar[idx][:, r0:r1, r0:r1, :]  # (N, ps, ps, 1)
    N = hsi_patch.shape[0]

    hsi_flat = hsi_patch.reshape(N, -1)
    hsi_mean = hsi_flat.mean(axis=1)
    hsi_std  = hsi_flat.std(axis=1)

    ms_flat = ms_patch.reshape(N, -1)
    ms_mean = ms_flat.mean(axis=1)
    ms_std  = ms_flat.std(axis=1)

    lidar_flat = lidar_patch.reshape(N, -1)
    lidar_mean = lidar_flat.mean(axis=1)
    lidar_std  = lidar_flat.std(axis=1)
    lidar_center_val = lidar[idx][:, 5, 5, 0]
    lidar_residual = lidar_center_val - lidar_mean

    # MS gradient magnitude on mean-pooled MS image
    ms_mean_img = ms_patch.mean(axis=3)  # (N, ps, ps)
    pc = hw  # center index within cropped patch
    if patch_size >= 3:
        gx = ms_mean_img[:, pc, pc+1] - ms_mean_img[:, pc, pc-1]
        gy = ms_mean_img[:, pc+1, pc] - ms_mean_img[:, pc-1, pc]
    else:
        gx = np.zeros(N)
        gy = np.zeros(N)
    ms_grad_mag = np.sqrt(gx**2 + gy**2)

    # HSI PCA-3 patch stats
    hsi_pixels = hsi_patch.reshape(N * patch_size * patch_size, 144)
    if pca_hsi_patch is not None:
        proj = pca_hsi_patch.transform(hsi_pixels).reshape(N, patch_size * patch_size, 3)
        pca_mean = proj.mean(axis=1)
        pca_std  = proj.std(axis=1)
    else:
        pca_mean = np.zeros((N, 3))
        pca_std  = np.zeros((N, 3))

    patch_feats = np.column_stack([
        hsi_mean, hsi_std,
        ms_mean, ms_std,
        lidar_mean, lidar_std,
        lidar_residual,
        ms_grad_mag,
        pca_mean, pca_std
    ])
    return patch_feats


print("[6/6] Training Random Forest classifiers...")
print()

results = {}

# Shared split for all methods
indices = np.arange({
    'ALL': N_samples,
    'MS_LIDAR': n_ms,
    'HSI_LIDAR': n_hsi
}[mode])
idx_train, idx_test, y_train, y_test = train_test_split(
    indices, labels, test_size=0.3, random_state=seed, stratify=labels
)

for name, X in X_methods.items():
    print(f"Testing: {name}")

    # Prepare features per method with train-only fitting where applicable
    if name == 'HSI Only (PCA 50)':
        pca_hsi = PCA(n_components=50)
        X_train_raw = hsi_center[idx_train]
        X_test_raw = hsi_center[idx_test]
        X_train_proc = pca_hsi.fit_transform(X_train_raw)
        X_test_proc = pca_hsi.transform(X_test_raw)
        proj_dim = X_train_proc.shape[1]
        print(f"  Post-PCA dim: {proj_dim}")

    elif name in ['HSI+LIDAR+MS (concat)', 'HSI+LIDAR+MS (gated-concat)', 'HSI+LIDAR+MS (gated-concat-entropy)', 'HSI+LIDAR+MS (concat+NDVI+std)', 'HSI+LIDAR+MS (concat+spatial+std)', 'HSI+LIDAR+MS (HGB)', 'HSI+LIDAR+MS (concat+LDA)', 'HSI+LIDAR+MS (concat+PCA15)', 'HSI+LIDAR+MS (concat+PCA20)', 'Late Fusion (avg)', 'Late Fusion (avg-temp)', 'Late Fusion (avg-calibrated)', 'Late Fusion (avg-calibrated-iso)', 'Late Fusion (stacking)']:
        # HSI PCA(50) on train only
        pca_hsi = PCA(n_components=50)
        hsi_train_pca = pca_hsi.fit_transform(hsi_center[idx_train])
        hsi_test_pca = pca_hsi.transform(hsi_center[idx_test])

        # Build concat features
        concat_train = np.concatenate([
            hsi_train_pca,
            lidar_center[idx_train, None],
            ms_center[idx_train]
        ], axis=1)
        concat_test = np.concatenate([
            hsi_test_pca,
            lidar_center[idx_test, None],
            ms_center[idx_test]
        ], axis=1)

        if name == 'HSI+LIDAR+MS (concat)':
            X_train_proc, X_test_proc = concat_train, concat_test
            proj_dim = X_train_proc.shape[1]
            print(f"  Concat dim: {proj_dim}")
        elif name == 'HSI+LIDAR+MS (gated-concat)':
            # Compute per-modality areas (proxy for confidence)
            eps = 1e-8
            A_hsi_tr = np.maximum(hsi_center[idx_train].mean(axis=1), eps)
            A_ms_tr = np.maximum(ms_center[idx_train].mean(axis=1), eps)
            A_lidar_tr = np.maximum(np.abs(lidar_center[idx_train]), eps)
            A_hsi_te = np.maximum(hsi_center[idx_test].mean(axis=1), eps)
            A_ms_te = np.maximum(ms_center[idx_test].mean(axis=1), eps)
            A_lidar_te = np.maximum(np.abs(lidar_center[idx_test]), eps)

            # Richardson-like weights: w_hsi ∝ A_ms*A_lidar, etc.
            w_hsi_tr = A_ms_tr * A_lidar_tr
            w_ms_tr = A_hsi_tr * A_lidar_tr
            w_lidar_tr = A_hsi_tr * A_ms_tr
            s_tr = w_hsi_tr + w_ms_tr + w_lidar_tr
            w_hsi_tr, w_ms_tr, w_lidar_tr = w_hsi_tr/s_tr, w_ms_tr/s_tr, w_lidar_tr/s_tr

            w_hsi_te = A_ms_te * A_lidar_te
            w_ms_te = A_hsi_te * A_lidar_te
            w_lidar_te = A_hsi_te * A_ms_te
            s_te = w_hsi_te + w_ms_te + w_lidar_te
            w_hsi_te, w_ms_te, w_lidar_te = w_hsi_te/s_te, w_ms_te/s_te, w_lidar_te/s_te

            # Apply gating per segment
            X_train_proc = concat_train.copy()
            X_test_proc = concat_test.copy()
            # segments: [0:50]=HSI, [50]=LiDAR, [51:]=MS
            X_train_proc[:, :hsi_train_pca.shape[1]] *= w_hsi_tr[:, None]
            X_train_proc[:, hsi_train_pca.shape[1]] *= w_lidar_tr
            X_train_proc[:, hsi_train_pca.shape[1]+1:] *= w_ms_tr[:, None]

            X_test_proc[:, :hsi_test_pca.shape[1]] *= w_hsi_te[:, None]
            X_test_proc[:, hsi_test_pca.shape[1]] *= w_lidar_te
            X_test_proc[:, hsi_test_pca.shape[1]+1:] *= w_ms_te[:, None]
            proj_dim = X_train_proc.shape[1]
            print(f"  Gated concat dim: {proj_dim}")
        elif name == 'HSI+LIDAR+MS (gated-concat-entropy)':
            if gate_signal == 'area':
                # area gating — reuse the area logic inline
                eps = 1e-8
                A_hsi_tr = np.maximum(hsi_center[idx_train].mean(axis=1), eps)
                A_ms_tr = np.maximum(ms_center[idx_train].mean(axis=1), eps)
                A_lidar_tr = np.maximum(np.abs(lidar_center[idx_train]), eps)
                A_hsi_te = np.maximum(hsi_center[idx_test].mean(axis=1), eps)
                A_ms_te = np.maximum(ms_center[idx_test].mean(axis=1), eps)
                A_lidar_te = np.maximum(np.abs(lidar_center[idx_test]), eps)
                w_hsi_tr = A_ms_tr * A_lidar_tr; w_ms_tr = A_hsi_tr * A_lidar_tr; w_lidar_tr = A_hsi_tr * A_ms_tr
                s_tr = w_hsi_tr + w_ms_tr + w_lidar_tr
                W_tr = np.stack([w_hsi_tr/s_tr, w_ms_tr/s_tr, w_lidar_tr/s_tr], axis=1)
                w_hsi_te = A_ms_te * A_lidar_te; w_ms_te = A_hsi_te * A_lidar_te; w_lidar_te = A_hsi_te * A_ms_te
                s_te = w_hsi_te + w_ms_te + w_lidar_te
                W_te = np.stack([w_hsi_te/s_te, w_ms_te/s_te, w_lidar_te/s_te], axis=1)
            else:
                # entropy / margin / gini: train per-modality RFs
                eps = 1e-8
                rf_hsi = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_ms = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_lidar = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_hsi.fit(hsi_train_pca, y_train)
                rf_ms.fit(ms_center[idx_train], y_train)
                rf_lidar.fit(lidar_center[idx_train, None], y_train)
                P_hsi_tr = rf_hsi.predict_proba(hsi_train_pca)
                P_ms_tr = rf_ms.predict_proba(ms_center[idx_train])
                P_ld_tr = rf_lidar.predict_proba(lidar_center[idx_train, None])
                P_hsi_te = rf_hsi.predict_proba(hsi_test_pca)
                P_ms_te = rf_ms.predict_proba(ms_center[idx_test])
                P_ld_te = rf_lidar.predict_proba(lidar_center[idx_test, None])
                W_tr = compute_gate_weights([P_hsi_tr, P_ms_tr, P_ld_tr], gate_signal, gating_temp)
                W_te = compute_gate_weights([P_hsi_te, P_ms_te, P_ld_te], gate_signal, gating_temp)

            X_train_proc = concat_train.copy()
            X_test_proc = concat_test.copy()
            X_train_proc[:, :hsi_train_pca.shape[1]] *= W_tr[:, [0]]
            X_train_proc[:, hsi_train_pca.shape[1]] *= W_tr[:, 2]
            X_train_proc[:, hsi_train_pca.shape[1]+1:] *= W_tr[:, [1]]
            X_test_proc[:, :hsi_test_pca.shape[1]] *= W_te[:, [0]]
            X_test_proc[:, hsi_test_pca.shape[1]] *= W_te[:, 2]
            X_test_proc[:, hsi_test_pca.shape[1]+1:] *= W_te[:, [1]]
            proj_dim = X_train_proc.shape[1]
            print(f"  Gated concat ({gate_signal},T={gating_temp}) dim: {proj_dim}")
            # Store W for diagnostics after classifier training
            _gate_W_tr = W_tr
            _gate_W_te = W_te
            _gate_name = f"gated-concat-{gate_signal}-T{gating_temp}"
        elif name == 'HSI+LIDAR+MS (concat+NDVI+std)':
            # Append NDVI features (HSI and MS) then standardize
            ndvi_hsi_tr = np.array([ndvi_from_hsi_vec(v) for v in hsi_center[idx_train]])[:, None]
            ndvi_ms_tr = np.array([ndvi_from_ms_vec(v) for v in ms_center[idx_train]])[:, None]
            ndre_hsi_tr = np.array([ndre_from_hsi_vec(v) for v in hsi_center[idx_train]])[:, None]
            ndre_ms_tr = np.array([ndre_from_ms_vec(v) for v in ms_center[idx_train]])[:, None]
            gndvi_hsi_tr = np.array([gndvi_from_hsi_vec(v) for v in hsi_center[idx_train]])[:, None]
            gndvi_ms_tr = np.array([gndvi_from_ms_vec(v) for v in ms_center[idx_train]])[:, None]
            ndvi_hsi_te = np.array([ndvi_from_hsi_vec(v) for v in hsi_center[idx_test]])[:, None]
            ndvi_ms_te = np.array([ndvi_from_ms_vec(v) for v in ms_center[idx_test]])[:, None]
            ndre_hsi_te = np.array([ndre_from_hsi_vec(v) for v in hsi_center[idx_test]])[:, None]
            ndre_ms_te = np.array([ndre_from_ms_vec(v) for v in ms_center[idx_test]])[:, None]
            gndvi_hsi_te = np.array([gndvi_from_hsi_vec(v) for v in hsi_center[idx_test]])[:, None]
            gndvi_ms_te = np.array([gndvi_from_ms_vec(v) for v in ms_center[idx_test]])[:, None]
            X_train_proc = np.concatenate([concat_train, ndvi_hsi_tr, ndvi_ms_tr, ndre_hsi_tr, ndre_ms_tr, gndvi_hsi_tr, gndvi_ms_tr], axis=1)
            X_test_proc = np.concatenate([concat_test, ndvi_hsi_te, ndvi_ms_te, ndre_hsi_te, ndre_ms_te, gndvi_hsi_te, gndvi_ms_te], axis=1)
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train_proc)
            X_test_proc = scaler.transform(X_test_proc)
            proj_dim = X_train_proc.shape[1]
            print(f"  Concat+NDVI+std dim: {proj_dim}")
        elif name in ['Late Fusion (avg)', 'Late Fusion (avg-calibrated)', 'Late Fusion (avg-calibrated-iso)', 'Late Fusion (stacking)']:
            # Train per-modality RFs
            rf_hsi = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf_ms = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf_lidar = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

            rf_hsi.fit(hsi_train_pca, y_train)
            rf_ms.fit(ms_center[idx_train], y_train)
            rf_lidar.fit(lidar_center[idx_train, None], y_train)

            P_hsi_tr = rf_hsi.predict_proba(hsi_train_pca)
            P_ms_tr = rf_ms.predict_proba(ms_center[idx_train])
            P_ld_tr = rf_lidar.predict_proba(lidar_center[idx_train, None])
            P_hsi_te = rf_hsi.predict_proba(hsi_test_pca)
            P_ms_te = rf_ms.predict_proba(ms_center[idx_test])
            P_ld_te = rf_lidar.predict_proba(lidar_center[idx_test, None])

            if name == 'Late Fusion (avg)':
                X_train_proc = (P_hsi_tr + P_ms_tr + P_ld_tr) / 3.0
                X_test_proc = (P_hsi_te + P_ms_te + P_ld_te) / 3.0
                proj_dim = X_train_proc.shape[1]
                print(f"  Late fusion (avg) classes: {proj_dim}")
            elif name == 'Late Fusion (avg-calibrated)':
                # Platt (sigmoid) calibration via CalibratedClassifierCV
                cal_hsi = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), method='sigmoid', cv=3)
                cal_ms = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), method='sigmoid', cv=3)
                cal_ld = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), method='sigmoid', cv=3)
                cal_hsi.fit(hsi_train_pca, y_train)
                cal_ms.fit(ms_center[idx_train], y_train)
                cal_ld.fit(lidar_center[idx_train, None], y_train)
                P_hsi_tr_c = cal_hsi.predict_proba(hsi_train_pca)
                P_ms_tr_c = cal_ms.predict_proba(ms_center[idx_train])
                P_ld_tr_c = cal_ld.predict_proba(lidar_center[idx_train, None])
                P_hsi_te_c = cal_hsi.predict_proba(hsi_test_pca)
                P_ms_te_c = cal_ms.predict_proba(ms_center[idx_test])
                P_ld_te_c = cal_ld.predict_proba(lidar_center[idx_test, None])
                X_train_proc = (P_hsi_tr_c + P_ms_tr_c + P_ld_tr_c) / 3.0
                X_test_proc = (P_hsi_te_c + P_ms_te_c + P_ld_te_c) / 3.0
                proj_dim = X_train_proc.shape[1]
                print(f"  Late fusion (avg-calibrated) classes: {proj_dim}")
            elif name == 'Late Fusion (avg-calibrated-iso)':
                # Isotonic calibration (monotonic) — may improve probability calibration
                cal_hsi = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), method='isotonic', cv=3)
                cal_ms = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), method='isotonic', cv=3)
                cal_ld = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), method='isotonic', cv=3)
                cal_hsi.fit(hsi_train_pca, y_train)
                cal_ms.fit(ms_center[idx_train], y_train)
                cal_ld.fit(lidar_center[idx_train, None], y_train)
                P_hsi_tr_c = cal_hsi.predict_proba(hsi_train_pca)
                P_ms_tr_c = cal_ms.predict_proba(ms_center[idx_train])
                P_ld_tr_c = cal_ld.predict_proba(lidar_center[idx_train, None])
                P_hsi_te_c = cal_hsi.predict_proba(hsi_test_pca)
                P_ms_te_c = cal_ms.predict_proba(ms_center[idx_test])
                P_ld_te_c = cal_ld.predict_proba(lidar_center[idx_test, None])
                X_train_proc = (P_hsi_tr_c + P_ms_tr_c + P_ld_tr_c) / 3.0
                X_test_proc = (P_hsi_te_c + P_ms_te_c + P_ld_te_c) / 3.0
                proj_dim = X_train_proc.shape[1]
                print(f"  Late fusion (avg-calibrated-iso) classes: {proj_dim}")
            else:
                Z_tr = np.concatenate([P_hsi_tr, P_ms_tr, P_ld_tr], axis=1)
                Z_te = np.concatenate([P_hsi_te, P_ms_te, P_ld_te], axis=1)
                meta = LogisticRegression(max_iter=1000, multi_class='auto')
                meta.fit(Z_tr, y_train)
                X_train_proc = meta.predict_proba(Z_tr)
                X_test_proc = meta.predict_proba(Z_te)
                proj_dim = X_train_proc.shape[1]
                print(f"  Late fusion (stacking) classes: {proj_dim}")

        elif name == 'HSI+LIDAR+MS (concat+LDA)':
            print(f"  Applying LDA (<= {num_classes - 1} dims) on train split...")
            lda = LinearDiscriminantAnalysis(solver='svd')
            X_train_proc = lda.fit_transform(concat_train, y_train)
            X_test_proc = lda.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-LDA dim: {proj_dim}")

        elif name == 'HSI+LIDAR+MS (concat+PCA15)':
            n_comp = min(15, concat_train.shape[1])
            pca_concat = PCA(n_components=n_comp)
            X_train_proc = pca_concat.fit_transform(concat_train)
            X_test_proc = pca_concat.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-PCA dim: {proj_dim}")

        elif name == 'HSI+LIDAR+MS (concat+PCA20)':
            n_comp = min(20, concat_train.shape[1])
            pca_concat = PCA(n_components=n_comp)
            X_train_proc = pca_concat.fit_transform(concat_train)
            X_test_proc = pca_concat.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-PCA dim: {proj_dim}")

    elif name in ['MS+LIDAR (concat)', 'MS+LIDAR (gated-concat)', 'MS+LIDAR (gated-concat-entropy)', 'MS+LIDAR (concat+NDVI+std)', 'MS+LIDAR (concat+spatial+std)', 'MS+LIDAR (concat+LDA)', 'MS+LIDAR (concat+PCA15)', 'MS+LIDAR (concat+PCA20)']:
        # MS+LIDAR concat (no HSI)
        concat_train = np.concatenate([
            lidar_center[idx_train, None],
            ms_center[idx_train]
        ], axis=1)
        concat_test = np.concatenate([
            lidar_center[idx_test, None],
            ms_center[idx_test]
        ], axis=1)

        if name == 'MS+LIDAR (concat)':
            X_train_proc, X_test_proc = concat_train, concat_test
            proj_dim = X_train_proc.shape[1]
            print(f"  Concat dim: {proj_dim}")
        elif name == 'MS+LIDAR (gated-concat)':
            eps = 1e-8
            A_ms_tr = np.maximum(ms_center[idx_train].mean(axis=1), eps)
            A_lidar_tr = np.maximum(np.abs(lidar_center[idx_train]), eps)
            A_ms_te = np.maximum(ms_center[idx_test].mean(axis=1), eps)
            A_lidar_te = np.maximum(np.abs(lidar_center[idx_test]), eps)

            # Two-modality weights: w_ms ∝ A_lidar, w_lidar ∝ A_ms
            w_ms_tr = A_lidar_tr / (A_ms_tr + A_lidar_tr)
            w_lidar_tr = A_ms_tr / (A_ms_tr + A_lidar_tr)
            w_ms_te = A_lidar_te / (A_ms_te + A_lidar_te)
            w_lidar_te = A_ms_te / (A_ms_te + A_lidar_te)

            X_train_proc = concat_train.copy()
            X_test_proc = concat_test.copy()
            # segments: [0]=LiDAR, [1:]=MS
            X_train_proc[:, 0] *= w_lidar_tr
            X_train_proc[:, 1:] *= w_ms_tr[:, None]
            X_test_proc[:, 0] *= w_lidar_te
            X_test_proc[:, 1:] *= w_ms_te[:, None]
            proj_dim = X_train_proc.shape[1]
            print(f"  Gated concat dim: {proj_dim}")
        elif name == 'MS+LIDAR (gated-concat-entropy)':
            if gate_signal == 'area':
                # area gating — amplitude-based weights inline
                eps = 1e-8
                A_ms_tr = np.maximum(ms_center[idx_train].mean(axis=1), eps)
                A_lidar_tr = np.maximum(np.abs(lidar_center[idx_train]), eps)
                A_ms_te = np.maximum(ms_center[idx_test].mean(axis=1), eps)
                A_lidar_te = np.maximum(np.abs(lidar_center[idx_test]), eps)
                w_lidar_tr = A_ms_tr / (A_ms_tr + A_lidar_tr)
                w_ms_tr = A_lidar_tr / (A_ms_tr + A_lidar_tr)
                w_lidar_te = A_ms_te / (A_ms_te + A_lidar_te)
                w_ms_te = A_lidar_te / (A_ms_te + A_lidar_te)
                W_tr = np.stack([w_lidar_tr, w_ms_tr], axis=1)
                W_te = np.stack([w_lidar_te, w_ms_te], axis=1)
            else:
                # entropy / margin / gini: train per-modality RFs
                eps = 1e-8
                rf_ms = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_lidar = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_ms.fit(ms_center[idx_train], y_train)
                rf_lidar.fit(lidar_center[idx_train, None], y_train)
                P_ld_tr = rf_lidar.predict_proba(lidar_center[idx_train, None])
                P_ms_tr = rf_ms.predict_proba(ms_center[idx_train])
                P_ld_te = rf_lidar.predict_proba(lidar_center[idx_test, None])
                P_ms_te = rf_ms.predict_proba(ms_center[idx_test])
                W_tr = compute_gate_weights([P_ld_tr, P_ms_tr], gate_signal, gating_temp)
                W_te = compute_gate_weights([P_ld_te, P_ms_te], gate_signal, gating_temp)

            X_train_proc = concat_train.copy()
            X_test_proc = concat_test.copy()
            X_train_proc[:, 0] *= W_tr[:, 0]
            X_train_proc[:, 1:] *= W_tr[:, [1]]
            X_test_proc[:, 0] *= W_te[:, 0]
            X_test_proc[:, 1:] *= W_te[:, [1]]
            proj_dim = X_train_proc.shape[1]
            print(f"  Gated concat ({gate_signal},T={gating_temp}) dim: {proj_dim}")
        elif name == 'MS+LIDAR (concat+NDVI+std)':
            ndvi_ms_tr = np.array([ndvi_from_ms_vec(v) for v in ms_center[idx_train]])[:, None]
            ndre_ms_tr = np.array([ndre_from_ms_vec(v) for v in ms_center[idx_train]])[:, None]
            gndvi_ms_tr = np.array([gndvi_from_ms_vec(v) for v in ms_center[idx_train]])[:, None]
            ndvi_ms_te = np.array([ndvi_from_ms_vec(v) for v in ms_center[idx_test]])[:, None]
            ndre_ms_te = np.array([ndre_from_ms_vec(v) for v in ms_center[idx_test]])[:, None]
            gndvi_ms_te = np.array([gndvi_from_ms_vec(v) for v in ms_center[idx_test]])[:, None]
            X_train_proc = np.concatenate([concat_train, ndvi_ms_tr, ndre_ms_tr, gndvi_ms_tr], axis=1)
            X_test_proc = np.concatenate([concat_test, ndvi_ms_te, ndre_ms_te, gndvi_ms_te], axis=1)
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train_proc)
            X_test_proc = scaler.transform(X_test_proc)
            proj_dim = X_train_proc.shape[1]
            print(f"  Concat+NDVI+std dim: {proj_dim}")
        elif name == 'MS+LIDAR (concat+LDA)':
            print(f"  Applying LDA (<= {num_classes - 1} dims) on train split...")
            lda = LinearDiscriminantAnalysis(solver='svd')
            X_train_proc = lda.fit_transform(concat_train, y_train)
            X_test_proc = lda.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-LDA dim: {proj_dim}")
        elif name == 'MS+LIDAR (concat+PCA15)':
            n_comp = min(15, concat_train.shape[1])
            pca_concat = PCA(n_components=n_comp)
            X_train_proc = pca_concat.fit_transform(concat_train)
            X_test_proc = pca_concat.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-PCA dim: {proj_dim}")
        elif name == 'MS+LIDAR (concat+PCA20)':
            n_comp = min(20, concat_train.shape[1])
            pca_concat = PCA(n_components=n_comp)
            X_train_proc = pca_concat.fit_transform(concat_train)
            X_test_proc = pca_concat.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-PCA dim: {proj_dim}")

    elif name in ['HSI+LIDAR (concat)', 'HSI+LIDAR (gated-concat)', 'HSI+LIDAR (gated-concat-entropy)', 'HSI+LIDAR (concat+NDVI+std)', 'HSI+LIDAR (concat+spatial+std)', 'HSI+LIDAR (concat+LDA)', 'HSI+LIDAR (concat+PCA15)', 'HSI+LIDAR (concat+PCA20)']:
        # HSI+LIDAR concat (no MS)
        pca_hsi = PCA(n_components=50)
        hsi_train_pca = pca_hsi.fit_transform(hsi_center[idx_train])
        hsi_test_pca = pca_hsi.transform(hsi_center[idx_test])

        concat_train = np.concatenate([
            hsi_train_pca,
            lidar_center[idx_train, None]
        ], axis=1)
        concat_test = np.concatenate([
            hsi_test_pca,
            lidar_center[idx_test, None]
        ], axis=1)

        if name == 'HSI+LIDAR (concat)':
            X_train_proc, X_test_proc = concat_train, concat_test
            proj_dim = X_train_proc.shape[1]
            print(f"  Concat dim: {proj_dim}")
        elif name == 'HSI+LIDAR (gated-concat)':
            eps = 1e-8
            A_hsi_tr = np.maximum(hsi_center[idx_train].mean(axis=1), eps)
            A_lidar_tr = np.maximum(np.abs(lidar_center[idx_train]), eps)
            A_hsi_te = np.maximum(hsi_center[idx_test].mean(axis=1), eps)
            A_lidar_te = np.maximum(np.abs(lidar_center[idx_test]), eps)

            # Two-modality weights: w_hsi ∝ A_lidar, w_lidar ∝ A_hsi
            w_hsi_tr = A_lidar_tr / (A_hsi_tr + A_lidar_tr)
            w_lidar_tr = A_hsi_tr / (A_hsi_tr + A_lidar_tr)
            w_hsi_te = A_lidar_te / (A_hsi_te + A_lidar_te)
            w_lidar_te = A_hsi_te / (A_hsi_te + A_lidar_te)

            X_train_proc = concat_train.copy()
            X_test_proc = concat_test.copy()
            # segments: [0:50]=HSI, [50]=LiDAR
            X_train_proc[:, :hsi_train_pca.shape[1]] *= w_hsi_tr[:, None]
            X_train_proc[:, hsi_train_pca.shape[1]] *= w_lidar_tr
            X_test_proc[:, :hsi_test_pca.shape[1]] *= w_hsi_te[:, None]
            X_test_proc[:, hsi_test_pca.shape[1]] *= w_lidar_te
            proj_dim = X_train_proc.shape[1]
            print(f"  Gated concat dim: {proj_dim}")
        elif name == 'HSI+LIDAR (gated-concat-entropy)':
            if gate_signal == 'area':
                # area gating — amplitude-based weights inline
                eps = 1e-8
                A_hsi_tr = np.maximum(hsi_center[idx_train].mean(axis=1), eps)
                A_lidar_tr = np.maximum(np.abs(lidar_center[idx_train]), eps)
                A_hsi_te = np.maximum(hsi_center[idx_test].mean(axis=1), eps)
                A_lidar_te = np.maximum(np.abs(lidar_center[idx_test]), eps)
                w_hsi_tr2 = A_lidar_tr / (A_hsi_tr + A_lidar_tr)
                w_lidar_tr2 = A_hsi_tr / (A_hsi_tr + A_lidar_tr)
                w_hsi_te2 = A_lidar_te / (A_hsi_te + A_lidar_te)
                w_lidar_te2 = A_hsi_te / (A_hsi_te + A_lidar_te)
                W_tr = np.stack([w_hsi_tr2, w_lidar_tr2], axis=1)
                W_te = np.stack([w_hsi_te2, w_lidar_te2], axis=1)
            else:
                # entropy / margin / gini: train per-modality RFs
                eps = 1e-8
                rf_hsi2 = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_lidar2 = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf_hsi2.fit(hsi_train_pca, y_train)
                rf_lidar2.fit(lidar_center[idx_train, None], y_train)
                P_hsi_tr2 = rf_hsi2.predict_proba(hsi_train_pca)
                P_ld_tr2 = rf_lidar2.predict_proba(lidar_center[idx_train, None])
                P_hsi_te2 = rf_hsi2.predict_proba(hsi_test_pca)
                P_ld_te2 = rf_lidar2.predict_proba(lidar_center[idx_test, None])
                W_tr = compute_gate_weights([P_hsi_tr2, P_ld_tr2], gate_signal, gating_temp)
                W_te = compute_gate_weights([P_hsi_te2, P_ld_te2], gate_signal, gating_temp)

            X_train_proc = concat_train.copy()
            X_test_proc = concat_test.copy()
            X_train_proc[:, :hsi_train_pca.shape[1]] *= W_tr[:, [0]]
            X_train_proc[:, hsi_train_pca.shape[1]] *= W_tr[:, 1]
            X_test_proc[:, :hsi_test_pca.shape[1]] *= W_te[:, [0]]
            X_test_proc[:, hsi_test_pca.shape[1]] *= W_te[:, 1]
            proj_dim = X_train_proc.shape[1]
            print(f"  Gated concat ({gate_signal},T={gating_temp}) dim: {proj_dim}")
        elif name == 'HSI+LIDAR (concat+NDVI+std)':
            ndvi_hsi_tr = np.array([ndvi_from_hsi_vec(v) for v in hsi_center[idx_train]])[:, None]
            ndre_hsi_tr = np.array([ndre_from_hsi_vec(v) for v in hsi_center[idx_train]])[:, None]
            gndvi_hsi_tr = np.array([gndvi_from_hsi_vec(v) for v in hsi_center[idx_train]])[:, None]
            ndvi_hsi_te = np.array([ndvi_from_hsi_vec(v) for v in hsi_center[idx_test]])[:, None]
            ndre_hsi_te = np.array([ndre_from_hsi_vec(v) for v in hsi_center[idx_test]])[:, None]
            gndvi_hsi_te = np.array([gndvi_from_hsi_vec(v) for v in hsi_center[idx_test]])[:, None]
            X_train_proc = np.concatenate([concat_train, ndvi_hsi_tr, ndre_hsi_tr, gndvi_hsi_tr], axis=1)
            X_test_proc = np.concatenate([concat_test, ndvi_hsi_te, ndre_hsi_te, gndvi_hsi_te], axis=1)
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train_proc)
            X_test_proc = scaler.transform(X_test_proc)
            proj_dim = X_train_proc.shape[1]
            print(f"  Concat+NDVI+std dim: {proj_dim}")
        elif name == 'HSI+LIDAR (concat+LDA)':
            print(f"  Applying LDA (<= {num_classes - 1} dims) on train split...")
            lda = LinearDiscriminantAnalysis(solver='svd')
            X_train_proc = lda.fit_transform(concat_train, y_train)
            X_test_proc = lda.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-LDA dim: {proj_dim}")
        elif name == 'HSI+LIDAR (concat+PCA15)':
            n_comp = min(15, concat_train.shape[1])
            pca_concat = PCA(n_components=n_comp)
            X_train_proc = pca_concat.fit_transform(concat_train)
            X_test_proc = pca_concat.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-PCA dim: {proj_dim}")
        elif name == 'HSI+LIDAR (concat+PCA20)':
            n_comp = min(20, concat_train.shape[1])
            pca_concat = PCA(n_components=n_comp)
            X_train_proc = pca_concat.fit_transform(concat_train)
            X_test_proc = pca_concat.transform(concat_test)
            proj_dim = X_train_proc.shape[1]
            print(f"  Post-PCA dim: {proj_dim}")

    elif name == 'Patch Only (no chromo)':
        # Patch stats only — no chromogeometry encoding
        hw = patch_size // 2; cx = 5; r0, r1 = cx - hw, cx + hw + 1
        hsi_tr_pix = hsi[idx_train][:, r0:r1, r0:r1, :].reshape(-1, 144)
        from sklearn.decomposition import PCA as _PCA
        pca_patch_po = _PCA(n_components=3)
        pca_patch_po.fit(hsi_tr_pix)
        X_train_proc = compute_patch_stats(hsi, lidar, ms, idx_train, pca_patch_po, patch_size=patch_size)
        X_test_proc  = compute_patch_stats(hsi, lidar, ms, idx_test,  pca_patch_po, patch_size=patch_size)
        proj_dim = X_train_proc.shape[1]
        print(f"  Feature dim: {proj_dim} (patch only, no chromo)")
    elif X is not None and name.startswith('Chromogeometry'):
        base_X = X  # chromo_features or chromo_features_amp
        X_train_proc = base_X[idx_train]
        X_test_proc = base_X[idx_test]
        if use_patch_stats:
            # Fit PCA(3) on train HSI patch pixels (patch-size-aware crop)
            hw = patch_size // 2
            cx = 5
            r0, r1 = cx - hw, cx + hw + 1
            hsi_train_pixels = hsi[idx_train][:, r0:r1, r0:r1, :].reshape(-1, 144)
            pca_patch = _PCA(n_components=3)
            pca_patch.fit(hsi_train_pixels)
            ps_train = compute_patch_stats(hsi, lidar, ms, idx_train, pca_patch, patch_size=patch_size)
            ps_test  = compute_patch_stats(hsi, lidar, ms, idx_test,  pca_patch, patch_size=patch_size)
            X_train_proc = np.concatenate([X_train_proc, ps_train], axis=1)
            X_test_proc  = np.concatenate([X_test_proc,  ps_test],  axis=1)
        proj_dim = X_train_proc.shape[1]
        print(f"  Feature dim: {proj_dim}" + (" (incl. patch stats)" if use_patch_stats else ""))
    else:
        # Methods with fixed features (no learning transform)
        X_train_proc = X[idx_train]
        X_test_proc = X[idx_test]
        proj_dim = X_train_proc.shape[1]
        print(f"  Feature dim: {proj_dim}")

    # Train
    t0 = time.time()
    if (name.startswith('Chromogeometry') or name == 'Patch Only (no chromo)') and chromo_head != 'rf':
        if chromo_head == 'gbdt':
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
        elif chromo_head == 'svm':
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            _scaler = StandardScaler()
            X_train_proc = _scaler.fit_transform(X_train_proc)
            X_test_proc  = _scaler.transform(X_test_proc)
            clf = SVC(kernel='rbf', probability=True, random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_train_proc, y_train)
    train_time = time.time() - t0

    # Test
    y_pred = clf.predict(X_test_proc)
    acc = accuracy_score(y_test, y_pred)

    # Print gate diagnostics if available
    if '_gate_W_tr' in dir() and name == 'HSI+LIDAR+MS (gated-concat-entropy)':
        y_pred_train_gate = clf.predict(X_train_proc)
        print_gate_diagnostics(_gate_W_tr, _gate_W_te, y_train, y_test,
                               y_pred_train_gate, y_pred, tag=_gate_name)
        del _gate_W_tr, _gate_W_te, _gate_name

    results[name] = {
        'accuracy': acc,
        'train_time': train_time,
        'n_features': proj_dim,
        'y_pred': y_pred
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Train time: {train_time:.2f}s")
    print()

# Summary
print("="*70)
print("RESULTS SUMMARY")
print("="*70)
print()

print(f"{'Method':<30} {'Features':<12} {'Accuracy':<12} {'Time (s)':<10}")
print("-"*70)
for name, res in results.items():
    print(f"{name:<30} {res['n_features']:<12} {res['accuracy']:<12.4f} {res['train_time']:<10.2f}")

print()

# Find best method
best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"🏆 Best method: {best_method[0]}")
print(f"   Accuracy: {best_method[1]['accuracy']:.4f}")
print()

# Check if chromogeometry helps (guard by available keys)
print()
print("Analysis:")
if mode == 'ALL':
    chromo_acc = results['Chromogeometry Fusion']['accuracy']
    baseline_acc = results['HSI Only (PCA 50)']['accuracy']
    concat_acc = results['HSI+LIDAR+MS (concat)']['accuracy']
    chromo_amp_acc = results['Chromogeometry Fusion + Amp Anchors (+2)']['accuracy']
    lda_acc = results['HSI+LIDAR+MS (concat+LDA)']['accuracy']
    concat_pca15_acc = results['HSI+LIDAR+MS (concat+PCA15)']['accuracy']
    concat_pca20_acc = results['HSI+LIDAR+MS (concat+PCA20)']['accuracy']
    if 'HSI+LIDAR+MS (gated-concat)' in results:
        gated_area_acc = results['HSI+LIDAR+MS (gated-concat)']['accuracy']
        delta = gated_area_acc - concat_acc
        print(f"  ✓ Gated-concat (area) vs concat: {delta*100:.2f} pp")
    if 'HSI+LIDAR+MS (gated-concat-entropy)' in results:
        gated_ent_acc = results['HSI+LIDAR+MS (gated-concat-entropy)']['accuracy']
        delta = gated_ent_acc - concat_acc
        print(f"  ✓ Gated-concat (entropy,T={gating_temp}) vs concat: {delta*100:.2f} pp")
    if 'Late Fusion (avg-calibrated)' in results:
        cal_acc = results['Late Fusion (avg-calibrated)']['accuracy']
        plain_acc = results['Late Fusion (avg)']['accuracy'] if 'Late Fusion (avg)' in results else cal_acc
        delta = cal_acc - plain_acc
        print(f"  ✓ Calibration (Platt) improves late-fusion avg by {delta*100:.2f} pp")

    if chromo_acc > baseline_acc:
        improvement = (chromo_acc - baseline_acc) / baseline_acc * 100
        print(f"  ✓ Chromogeometry improves over HSI-only by {improvement:.1f}%")
    else:
        decrease = (baseline_acc - chromo_acc) / baseline_acc * 100
        print(f"  ✗ Chromogeometry {decrease:.1f}% worse than HSI-only")

    if chromo_acc > concat_acc:
        improvement = (chromo_acc - concat_acc) / concat_acc * 100
        print(f"  ✓ Chromogeometry improves over concatenation by {improvement:.1f}%")
    else:
        decrease = (concat_acc - chromo_acc) / concat_acc * 100
        print(f"  ✗ Chromogeometry {decrease:.1f}% worse than concatenation")

    # Additional analysis for new methods
    print()
    print("Additional Analysis:")
    if chromo_amp_acc >= chromo_acc:
        delta = chromo_amp_acc - chromo_acc
        print(f"  ✓ Amp anchors improve Chromogeometry by {delta*100:.2f} pp")
    else:
        delta = chromo_acc - chromo_amp_acc
        print(f"  ✗ Amp anchors reduce Chromogeometry by {delta*100:.2f} pp")

    if lda_acc >= concat_acc:
        delta = lda_acc - concat_acc
        print(f"  ✓ LDA matches/beats concat by {delta*100:.2f} pp with compression")
    else:
        delta = concat_acc - lda_acc
        print(f"  ⚠️  LDA trails concat by {delta*100:.2f} pp (check class balance / solver)")

    if results.get('HSI+LIDAR+MS (concat+PCA15)'):
        concat_pca15_acc = results['HSI+LIDAR+MS (concat+PCA15)']['accuracy']
        if concat_pca15_acc >= concat_acc:
            delta = concat_pca15_acc - concat_acc
            print(f"  ✓ Concat+PCA15 beats concat by {delta*100:.2f} pp with strong compression")
        else:
            delta = concat_acc - concat_pca15_acc
            print(f"  ⚠️  Concat+PCA15 trails concat by {delta*100:.2f} pp")

    if results.get('HSI+LIDAR+MS (concat+PCA20)'):
        concat_pca20_acc = results['HSI+LIDAR+MS (concat+PCA20)']['accuracy']
        if concat_pca20_acc >= concat_acc:
            delta = concat_pca20_acc - concat_acc
            print(f"  ✓ Concat+PCA20 beats concat by {delta*100:.2f} pp with compression")
        else:
            delta = concat_acc - concat_pca20_acc
            print(f"  ⚠️  Concat+PCA20 trails concat by {delta*100:.2f} pp")

    # Per-class deltas for gating and calibration
    try:
        from sklearn.metrics import confusion_matrix
        labels_sorted = sorted(np.unique(y_test))
        def class_acc(y_true, y_hat):
            cm = confusion_matrix(y_true, y_hat, labels=labels_sorted)
            with np.errstate(divide='ignore', invalid='ignore'):
                accs = np.diag(cm) / cm.sum(axis=1)
                accs = np.nan_to_num(accs)
            return accs
        base_pred = results['HSI+LIDAR+MS (concat)']['y_pred']
        base_accs = class_acc(y_test, base_pred)
        if 'HSI+LIDAR+MS (gated-concat)' in results:
            ga_pred = results['HSI+LIDAR+MS (gated-concat)']['y_pred']
            ga_delta = class_acc(y_test, ga_pred) - base_accs
            print("  Per-class Δ (gated area vs concat):", np.round(ga_delta*100, 2))
        if 'HSI+LIDAR+MS (gated-concat-entropy)' in results:
            ge_pred = results['HSI+LIDAR+MS (gated-concat-entropy)']['y_pred']
            ge_delta = class_acc(y_test, ge_pred) - base_accs
            print(f"  Per-class Δ (gated entropy T={gating_temp} vs concat):", np.round(ge_delta*100, 2))
        if 'Late Fusion (avg-calibrated)' in results:
            lc_pred = results['Late Fusion (avg-calibrated)']['y_pred']
            lc_delta = class_acc(y_test, lc_pred) - base_accs
            print("  Per-class Δ (late-fusion calibrated vs concat):", np.round(lc_delta*100, 2))
    except Exception:
        pass
elif mode == 'MS_LIDAR':
    # MS+LIDAR analysis
    chromo_acc = results['Chromogeometry Fusion (MS only)']['accuracy']
    chromo_amp_acc = results['Chromogeometry Fusion (MS only) + Amp Anchor (+1)']['accuracy']
    concat_acc = results['MS+LIDAR (concat)']['accuracy']
    lda_acc = results['MS+LIDAR (concat+LDA)']['accuracy']
    concat_pca15_acc = results['MS+LIDAR (concat+PCA15)']['accuracy']
    concat_pca20_acc = results['MS+LIDAR (concat+PCA20)']['accuracy']

    if chromo_acc > concat_acc:
        improvement = (chromo_acc - concat_acc) / concat_acc * 100
        print(f"  ✓ Chromogeometry (MS) beats MS+LIDAR concat by {improvement:.1f}%")
    else:
        decrease = (concat_acc - chromo_acc) / concat_acc * 100
        print(f"  ✗ Chromogeometry (MS) {decrease:.1f}% worse than MS+LIDAR concat")

    print()
    print("Additional Analysis:")
    if chromo_amp_acc >= chromo_acc:
        delta = chromo_amp_acc - chromo_acc
        print(f"  ✓ Amp anchor improves Chromogeometry (MS) by {delta*100:.2f} pp")
    else:
        delta = chromo_acc - chromo_amp_acc
        print(f"  ✗ Amp anchor reduces Chromogeometry (MS) by {delta*100:.2f} pp")

    if lda_acc >= concat_acc:
        delta = lda_acc - concat_acc
        print(f"  ✓ LDA matches/beats MS+LIDAR concat by {delta*100:.2f} pp with compression")
    else:
        delta = concat_acc - lda_acc
        print(f"  ⚠️  LDA trails MS+LIDAR concat by {delta*100:.2f} pp")

    if concat_pca15_acc >= concat_acc:
        delta = concat_pca15_acc - concat_acc
        print(f"  ✓ Concat+PCA15 beats MS+LIDAR concat by {delta*100:.2f} pp")
    else:
        delta = concat_acc - concat_pca15_acc
        print(f"  ⚠️  Concat+PCA15 trails MS+LIDAR concat by {delta*100:.2f} pp")

    if concat_pca20_acc >= concat_acc:
        delta = concat_pca20_acc - concat_acc
        print(f"  ✓ Concat+PCA20 beats MS+LIDAR concat by {delta*100:.2f} pp")
    else:
        delta = concat_acc - concat_pca20_acc
        print(f"  ⚠️  Concat+PCA20 trails MS+LIDAR concat by {delta*100:.2f} pp")

elif mode == 'HSI_LIDAR':
    chromo_acc = results['Chromogeometry Fusion (HSI only)']['accuracy']
    chromo_amp_acc = results['Chromogeometry Fusion (HSI only) + Amp Anchor (+1)']['accuracy']
    baseline_acc = results['HSI Only (PCA 50)']['accuracy']
    concat_acc = results['HSI+LIDAR (concat)']['accuracy']
    lda_acc = results['HSI+LIDAR (concat+LDA)']['accuracy']

    if chromo_acc > baseline_acc:
        improvement = (chromo_acc - baseline_acc) / baseline_acc * 100
        print(f"  ✓ Chromogeometry (HSI) improves over HSI-only by {improvement:.1f}%")
    else:
        decrease = (baseline_acc - chromo_acc) / baseline_acc * 100
        print(f"  ✗ Chromogeometry (HSI) {decrease:.1f}% worse than HSI-only")

    if chromo_acc > concat_acc:
        improvement = (chromo_acc - concat_acc) / concat_acc * 100
        print(f"  ✓ Chromogeometry (HSI) beats HSI+LIDAR concat by {improvement:.1f}%")
    else:
        decrease = (concat_acc - chromo_acc) / concat_acc * 100
        print(f"  ✗ Chromogeometry (HSI) {decrease:.1f}% worse than HSI+LIDAR concat")

    print()
    print("Additional Analysis:")
    if chromo_amp_acc >= chromo_acc:
        delta = chromo_amp_acc - chromo_acc
        print(f"  ✓ Amp anchor improves Chromogeometry (HSI) by {delta*100:.2f} pp")
    else:
        delta = chromo_acc - chromo_amp_acc
        print(f"  ✗ Amp anchor reduces Chromogeometry (HSI) by {delta*100:.2f} pp")

    if lda_acc >= concat_acc:
        delta = lda_acc - concat_acc
        print(f"  ✓ LDA matches/beats HSI+LIDAR concat by {delta*100:.2f} pp with compression")
    else:
        delta = concat_acc - lda_acc
        print(f"  ⚠️  LDA trails HSI+LIDAR concat by {delta*100:.2f} pp")

    # No concat_pcaXX duplicates here; printed above in loop

if mode == 'MS_LIDAR':
    if concat_pca15_acc >= concat_acc:
        delta = concat_pca15_acc - concat_acc
        print(f"  ✓ Concat+PCA15 beats concat by {delta*100:.2f} pp with strong compression")
    else:
        delta = concat_acc - concat_pca15_acc
        print(f"  ⚠️  Concat+PCA15 trails concat by {delta*100:.2f} pp")

    if concat_pca20_acc >= concat_acc:
        delta = concat_pca20_acc - concat_acc
        print(f"  ✓ Concat+PCA20 beats concat by {delta*100:.2f} pp with compression")
    else:
        delta = concat_acc - concat_pca20_acc
        print(f"  ⚠️  Concat+PCA20 trails concat by {delta*100:.2f} pp")

print()
print("="*70)
print("CONCLUSION")
print("="*70)
print()

# Optional: gating temperature sweep plot
if mode == 'ALL' and include_gated and gating_sweep:
    try:
        import matplotlib.pyplot as plt
        # Rebuild per-modality artifacts for fair comparison using same split
        pca_hsi = PCA(n_components=50)
        hsi_train_pca = pca_hsi.fit_transform(hsi_center[idx_train])
        hsi_test_pca = pca_hsi.transform(hsi_center[idx_test])
        concat_train = np.concatenate([
            hsi_train_pca,
            lidar_center[idx_train, None],
            ms_center[idx_train]
        ], axis=1)
        concat_test = np.concatenate([
            hsi_test_pca,
            lidar_center[idx_test, None],
            ms_center[idx_test]
        ], axis=1)
        concat_acc = results['HSI+LIDAR+MS (concat)']['accuracy']
        temps = []
        deltas = []
        if gate_signal == 'area':
            print("  [sweep] gate_signal=area: weights are signal-independent; sweep skipped (constant gate).")
        else:
            # Train per-modality RFs once
            rf_hsi = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf_ms = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf_lidar = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf_hsi.fit(hsi_train_pca, y_train)
            rf_ms.fit(ms_center[idx_train], y_train)
            rf_lidar.fit(lidar_center[idx_train, None], y_train)
            P_hsi_tr = rf_hsi.predict_proba(hsi_train_pca)
            P_ms_tr = rf_ms.predict_proba(ms_center[idx_train])
            P_ld_tr = rf_lidar.predict_proba(lidar_center[idx_train, None])
            P_hsi_te = rf_hsi.predict_proba(hsi_test_pca)
            P_ms_te = rf_ms.predict_proba(ms_center[idx_test])
            P_ld_te = rf_lidar.predict_proba(lidar_center[idx_test, None])
            for T in gating_sweep:
                # build weights using gate_signal
                W_tr = compute_gate_weights([P_hsi_tr, P_ms_tr, P_ld_tr], gate_signal, T)
                W_te = compute_gate_weights([P_hsi_te, P_ms_te, P_ld_te], gate_signal, T)
                X_train_proc = concat_train.copy()
                X_test_proc = concat_test.copy()
                X_train_proc[:, :hsi_train_pca.shape[1]] *= W_tr[:, [0]]
                X_train_proc[:, hsi_train_pca.shape[1]] *= W_tr[:, 2]
                X_train_proc[:, hsi_train_pca.shape[1]+1:] *= W_tr[:, [1]]
                X_test_proc[:, :hsi_test_pca.shape[1]] *= W_te[:, [0]]
                X_test_proc[:, hsi_test_pca.shape[1]] *= W_te[:, 2]
                X_test_proc[:, hsi_test_pca.shape[1]+1:] *= W_te[:, [1]]
                clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                clf.fit(X_train_proc, y_train)
                y_pred = clf.predict(X_test_proc)
                acc = accuracy_score(y_test, y_pred)
                W_te_sweep = compute_gate_weights([P_hsi_te, P_ms_te, P_ld_te], gate_signal, T)
                mean_w = W_te_sweep.mean(axis=0)
                sat = np.mean((W_te_sweep > 0.9).any(axis=1) | (W_te_sweep < 0.1).any(axis=1)) * 100
                correct_te = (y_pred == y_test).astype(float)
                max_gw = W_te_sweep.max(axis=1)
                if max_gw.std() > 1e-10 and correct_te.std() > 1e-10:
                    corr = float(np.corrcoef(max_gw, correct_te)[0, 1])
                else:
                    corr = float('nan')
                print(f"  T={T}: acc={acc:.4f} (Δ={acc-concat_acc:.4f}) mean_w={[f'{v:.3f}' for v in mean_w]} sat={sat:.1f}% corr={corr:.3f}")
                temps.append(T)
                deltas.append((acc - concat_acc) * 100.0)
        # Plot
        out_dir = 'results'
        try:
            import os
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        if temps:
            plt.figure(figsize=(6,4))
            plt.plot(temps, deltas, marker='o')
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            plt.xlabel(f'Gating Temperature ({gate_signal})')
            plt.ylabel('Δ Accuracy vs Concat (pp)')
            plt.title(f'{gate_signal.capitalize()} Gating Sweep (Tri-modal)')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/gating_sweep.png', dpi=200)
            print(f"Saved gating sweep plot to {out_dir}/gating_sweep.png")
    except Exception as e:
        print(f"Gating sweep plotting skipped: {e}")

# Save CSV summaries
try:
    import csv, os
    os.makedirs('results', exist_ok=True)
    with open('results/fusion_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'features', 'accuracy', 'train_time'])
        for name, res in results.items():
            w.writerow([name, res['n_features'], f"{res['accuracy']:.6f}", f"{res['train_time']:.4f}"])
    if mode == 'ALL':
        from sklearn.metrics import confusion_matrix
        labels_sorted = sorted(np.unique(y_test))
        def class_acc(y_true, y_hat):
            cm = confusion_matrix(y_true, y_hat, labels=labels_sorted)
            with np.errstate(divide='ignore', invalid='ignore'):
                accs = np.diag(cm) / cm.sum(axis=1)
                accs = np.nan_to_num(accs)
            return accs
        base_pred = results['HSI+LIDAR+MS (concat)']['y_pred']
        base_accs = class_acc(y_test, base_pred)
        with open('results/per_class_deltas.csv', 'w', newline='') as f:
            w = csv.writer(f)
            header = ['method'] + [f'class_{c}' for c in labels_sorted]
            w.writerow(header)
            for m in ['HSI+LIDAR+MS (gated-concat)', 'HSI+LIDAR+MS (gated-concat-entropy)', 'Late Fusion (avg-calibrated)']:
                if m in results:
                    delta = class_acc(y_test, results[m]['y_pred']) - base_accs
                    w.writerow([m] + [f"{d:.6f}" for d in delta])
    print('Saved results to results/fusion_results.csv and results/per_class_deltas.csv')
except Exception as e:
    print(f'CSV export skipped: {e}')
if mode == 'ALL':
    if chromo_acc >= max(baseline_acc, concat_acc):
        print("✅ Chromogeometry fusion is EFFECTIVE for multi-modal data!")
        print("   The geometric encoding successfully combines HSI+LIDAR+MS.")
    else:
        print("⚠️  Chromogeometry shows promise but needs refinement.")
        print("   Consider: amplitude anchors, LDA/PCA on concat, or hybrid approaches.")
elif mode == 'MS_LIDAR':
    if chromo_acc >= concat_acc:
        print("✅ Chromogeometry (MS) is competitive with MS+LIDAR concat.")
    else:
        print("⚠️  Chromogeometry (MS) trails MS+LIDAR concat; anchors/LDA may help.")
elif mode == 'HSI_LIDAR':
    if chromo_acc >= max(baseline_acc, concat_acc):
        print("✅ Chromogeometry (HSI) is competitive with HSI+LIDAR concat.")
    else:
        print("⚠️  Chromogeometry (HSI) trails HSI+LIDAR concat; anchors/LDA may help.")

# Patch size sweep
if use_patch_stats and patch_sweep:
    print("\n" + "="*70)
    print(f"PATCH SIZE SWEEP (seed={seed}, chromo-head={chromo_head})")
    print("="*70)
    print(f"{'patch_size':<12} {'feat_dim':<10} {'Chromo+patch':<16} {'Chromo+Amp+patch':<18} {'concat':<10}")
    print("-"*70)

    concat_acc_ref = results.get('HSI+LIDAR+MS (concat)', {}).get('accuracy', float('nan'))

    for ps in patch_sweep:
        # Recompute patch features for this size
        from sklearn.decomposition import PCA as _PCA2
        hw = ps // 2; cx = 5; r0, r1 = cx - hw, cx + hw + 1
        hsi_tr_pix = hsi[idx_train][:, r0:r1, r0:r1, :].reshape(-1, 144)
        pca_p = _PCA2(n_components=3)
        pca_p.fit(hsi_tr_pix)

        ps_tr = compute_patch_stats(hsi, lidar, ms, idx_train, pca_p, patch_size=ps)
        ps_te = compute_patch_stats(hsi, lidar, ms, idx_test,  pca_p, patch_size=ps)

        accs = {}
        for cname, base_X in [('Chromogeometry Fusion', chromo_features),
                               ('Chromogeometry Fusion + Amp Anchors (+2)', chromo_features_amp)]:
            Xtr = np.concatenate([base_X[idx_train], ps_tr], axis=1)
            Xte = np.concatenate([base_X[idx_test],  ps_te], axis=1)

            if chromo_head == 'gbdt':
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf_ps = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
            elif chromo_head == 'svm':
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
                clf_ps = SVC(kernel='rbf', probability=True, random_state=seed)
            else:
                clf_ps = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

            clf_ps.fit(Xtr, y_train)
            accs[cname] = accuracy_score(y_test, clf_ps.predict(Xte))

        feat_dim = chromo_features.shape[1] + ps_tr.shape[1]
        c1 = accs['Chromogeometry Fusion']
        c2 = accs['Chromogeometry Fusion + Amp Anchors (+2)']
        print(f"{ps:<12} {feat_dim:<10} {c1:<16.4f} {c2:<18.4f} {concat_acc_ref:<10.4f}")
    print()
print()
