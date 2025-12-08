#!/usr/bin/env python3
"""
QA Hyperspectral Imaging Pipeline
Reconstructed from vault specifications (October 19, 2025 research)

Converts hyperspectral cubes (H×W×B) to QA tuples (b,e,d,a) using:
- Phase-aware DFT encoding
- Multi-peak spectral analysis
- Harmonic-aware clustering (mod-24 circular embedding)
- Sector masking by residue classes

Reference: HYPERSPECTRAL_RESEARCH_SUMMARY.md
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any

import numpy as np

TAU = 2.0 * np.pi

# ============================================================================
# Circular Statistics for Modular Arithmetic
# ============================================================================

def circular_mean(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Circular mean of angles (handles wraparound at 2π)

    Args:
        angles: Array of angles in radians
        weights: Optional weights for weighted mean

    Returns:
        Mean angle in radians
    """
    if weights is None:
        weights = np.ones_like(angles)
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    return float(np.arctan2(y, x))

def index_circular_mean(indices: np.ndarray, weights: Optional[np.ndarray], bins: int) -> float:
    """
    Circular mean over integer bin indices on modulo-bins circle

    Args:
        indices: Array of bin indices
        weights: Weights for each index
        bins: Number of bins (modulus)

    Returns:
        Mean index in continuous range [0, bins)
    """
    angles = indices * TAU / bins
    mu = circular_mean(angles, weights)
    return float((mu % TAU) * bins / TAU)

# ============================================================================
# Spectral Peak Detection
# ============================================================================

def topk_peak_indices(mag: np.ndarray, k: int, skip_dc: bool = True) -> np.ndarray:
    """
    Select top-k spectral peaks (excluding DC component if requested)

    Args:
        mag: Magnitude spectrum
        k: Number of peaks to select
        skip_dc: If True, exclude DC (index 0)

    Returns:
        Indices of top-k peaks
    """
    if skip_dc and mag.size > 0:
        mag = mag.copy()
        mag[0] = -np.inf
    k = max(1, min(k, mag.size))
    return np.argpartition(mag, -k)[-k:]

# ============================================================================
# Phase-Aware DFT Encoding
# ============================================================================

def spectrum_to_chromo_uv(
    spec: np.ndarray,
    k_peaks: int = 5,
    use_derivative: bool = True,
    derivative_order: int = 1
) -> Tuple[float, float]:
    """
    Convert spectrum to chromogeometry (u,v) coordinates using DFT peaks.

    Based on Wildberger's chromogeometry: three quadrances Qb=u²+v², Qr=u²-v², Qg=2uv
    """
    N = len(spec)
    spec_proc = spec.copy()

    # Apply derivative if requested
    if use_derivative and derivative_order > 0:
        if derivative_order == 1:
            spec_proc = np.diff(spec_proc)
        elif derivative_order == 2:
            spec_proc = np.diff(np.diff(spec_proc))
        spec_proc = np.pad(spec_proc, (0, N - len(spec_proc)), mode='edge')

    X = np.fft.rfft(spec_proc)
    mag = np.abs(X)
    phs = np.angle(X)

    # Skip DC
    mag[0] = -np.inf
    k = max(1, min(k_peaks, mag.size))
    idxs = np.argpartition(mag, -k)[-k:]
    peak_mags = mag[idxs]
    peak_phs = phs[idxs]

    # Use circular mean for frequency and phase
    angles = idxs * TAU / N
    u = index_circular_mean(idxs.astype(float), weights=peak_mags, bins=N)
    v = circular_mean(peak_phs, weights=peak_mags)

    return u, v

def chromo_quadrances(u: float, v: float) -> Dict[str, float]:
    """Compute the three chromogeometry quadrances."""
    Qb = u**2 + v**2  # Blue (Euclidean)
    Qr = u**2 - v**2  # Red (Minkowski difference)
    Qg = 2 * u * v    # Green (null product)
    return {'Qb': Qb, 'Qr': Qr, 'Qg': Qg}

def spectrum_to_be_phase_multi(
    spec: np.ndarray,
    bins: int = 24,
    k_peaks: int = 5,
    phase_mode: str = "weighted",
    use_derivative: bool = True,
    derivative_order: int = 1
) -> Tuple[int, int]:
    """
    Convert spectral vector to (b, e) QA parameters using phase-aware encoding

    IMPROVED VERSION (2025-10-31): Now uses spectral derivatives to capture
    subtle shape differences that are lost in raw spectra. This dramatically
    improves encoding variance on hyperspectral data.

    Algorithm:
    1. Optionally compute spectral derivative (1st or 2nd order)
    2. Compute DFT: X = FFT(spectrum or derivative)
    3. Extract magnitude and phase
    4. Select top-k peaks (excluding DC)
    5. b = circular mean of peak frequencies (weighted by magnitude)
    6. e = circular mean of peak phases (weighted by magnitude)

    Args:
        spec: Input spectrum (1D array)
        bins: Modular arithmetic base (default 24)
        k_peaks: Number of peaks to consider
        phase_mode: "weighted" (circular mean) or "argmax" (strongest peak)
        use_derivative: If True, use derivative instead of raw spectrum
        derivative_order: 1 (slope) or 2 (curvature) - default 2 for best variance

    Returns:
        (b, e) tuple in range [0, bins)
    """
    N = len(spec)
    spec_proc = spec.copy()

    # Apply derivative if requested (IMPROVEMENT: 5.2x variance increase)
    if use_derivative and derivative_order > 0:
        if derivative_order == 1:
            spec_proc = np.diff(spec_proc)
        elif derivative_order == 2:
            spec_proc = np.diff(np.diff(spec_proc))

        # Pad to original length
        spec_proc = np.pad(spec_proc, (0, N - len(spec_proc)), mode='edge')

    X = np.fft.rfft(spec_proc)
    mag = np.abs(X)
    phs = np.angle(X)

    # Select top-k peaks (skip DC which dominates in hyperspectral data)
    idxs = topk_peak_indices(mag, k_peaks, skip_dc=True)
    peak_mags = mag[idxs]
    peak_phs = phs[idxs]

    # b = circular mean of frequencies
    b_idx = index_circular_mean(idxs.astype(float), weights=peak_mags, bins=bins)

    # e = phase encoding
    if phase_mode == "argmax":
        i = idxs[np.argmax(peak_mags)]
        e_phase = phs[i]
    else:  # weighted
        e_phase = circular_mean(peak_phs, weights=peak_mags)

    e_val = (e_phase % TAU) * bins / TAU

    b = int(np.round(b_idx)) % bins
    e = int(np.round(e_val)) % bins

    return b, e

# ============================================================================
# QA Chromatic Fields
# ============================================================================

def qa_chromatic_fields(
    b: np.ndarray,
    e: np.ndarray,
    bins: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute QA chromatic field components from (b, e) tuples

    Eb = b mod bins          (electric / in-phase field)
    Er = (b + e) mod bins    (red / magnetic field)
    Eg = (b + 2e) mod bins   (green / coupling field)

    Args:
        b, e: QA tuple components (H × W arrays)
        bins: Modular base

    Returns:
        (Eb, Er, Eg) chromatic field tuple
    """
    Eb = b % bins
    Er = (b + e) % bins
    Eg = (b + 2 * e) % bins
    return Eb, Er, Eg

# ============================================================================
# Hyperspectral Cube → QA Fields
# ============================================================================

def cube_to_chromo_fields(
    cube: np.ndarray,
    k_peaks: int = 5,
    use_derivative: bool = True,
    derivative_order: int = 1
) -> Dict[str, np.ndarray]:
    """
    Convert hyperspectral cube to chromogeometry fields (u,v, Qb, Qr, Qg)

    Args:
        cube: Hyperspectral data (H × W × B)
        k_peaks: Number of spectral peaks to consider
        use_derivative: If True, use derivative encoding
        derivative_order: 1 (slope) or 2 (curvature)

    Returns:
        Dictionary with keys: u, v, Qb, Qr, Qg (all H × W)
    """
    H, W, B = cube.shape
    u = np.zeros((H, W), dtype=float)
    v = np.zeros((H, W), dtype=float)
    Qb = np.zeros((H, W), dtype=float)
    Qr = np.zeros((H, W), dtype=float)
    Qg = np.zeros((H, W), dtype=float)

    for i in range(H):
        row = cube[i].copy()  # W × B

        # Apply derivative if requested
        if use_derivative and derivative_order > 0:
            if derivative_order == 1:
                row = np.diff(row, axis=1)
            elif derivative_order == 2:
                row = np.diff(np.diff(row, axis=1), axis=1)

            # Pad to original length
            pad_width = ((0, 0), (0, B - row.shape[1]))
            row = np.pad(row, pad_width, mode='edge')

        for j in range(W):
            spec = row[j]
            uj, vj = spectrum_to_chromo_uv(spec, k_peaks=k_peaks, use_derivative=False, derivative_order=0)  # Already processed
            u[i, j] = uj
            v[i, j] = vj
            quad = chromo_quadrances(uj, vj)
            Qb[i, j] = quad['Qb']
            Qr[i, j] = quad['Qr']
            Qg[i, j] = quad['Qg']

    return {'u': u, 'v': v, 'Qb': Qb, 'Qr': Qr, 'Qg': Qg}

def cube_to_qa_fields_phase_multi(
    cube: np.ndarray,
    bins: int = 24,
    k_peaks: int = 5,
    phase_mode: str = "weighted",
    use_derivative: bool = True,
    derivative_order: int = 1,
    use_chromo: bool = False
) -> Dict[str, np.ndarray]:
    """
    Convert hyperspectral cube to QA field representation

    IMPROVED VERSION (2025-10-31): Now uses spectral derivatives for better variance
    CHROMO INTEGRATION: If use_chromo=True, also computes chromogeometry fields

    Args:
        cube: Hyperspectral data (H × W × B)
        bins: Modular arithmetic base
        k_peaks: Number of spectral peaks to consider
        phase_mode: Phase extraction method
        use_derivative: If True, use derivative encoding (5.2x variance improvement)
        derivative_order: 1 (slope) or 2 (curvature) - default 2 for best results
        use_chromo: If True, also compute chromogeometry fields (u,v,Qb,Qr,Qg)

    Returns:
        Dictionary with keys: b, e, d, a, Eb, Er, Eg, and if use_chromo: u, v, Qb, Qr, Qg
    """
    H, W, B = cube.shape
    b = np.zeros((H, W), dtype=np.int32)
    e = np.zeros((H, W), dtype=np.int32)

    # Vectorized row processing for efficiency
    for i in range(H):
        row = cube[i].copy()  # H×W×B → W×B

        # IMPROVEMENT: Apply derivative before DFT (5.2x variance increase!)
        if use_derivative and derivative_order > 0:
            if derivative_order == 1:
                row = np.diff(row, axis=1)
            elif derivative_order == 2:
                row = np.diff(np.diff(row, axis=1), axis=1)

            # Pad back to original length
            pad_width = ((0, 0), (0, B - row.shape[1]))
            row = np.pad(row, pad_width, mode='edge')

        X = np.fft.rfft(row, axis=1)
        mag = np.abs(X)
        phs = np.angle(X)

        for j in range(W):
            mj = mag[j].copy()
            pj = phs[j]

            if mj.size > 0:
                mj[0] = -np.inf  # Skip DC
                k_idx = max(1, min(k_peaks, mj.size))
                idxs = np.argpartition(mj, -k_idx)[-k_idx:]
                peak_mags = mj[idxs]
                peak_phs = pj[idxs]

                b_idx = index_circular_mean(idxs.astype(float), weights=peak_mags, bins=bins)

                if phase_mode == "argmax":
                    sel = idxs[np.argmax(peak_mags)]
                    e_phase = pj[sel]
                else:
                    e_phase = circular_mean(peak_phs, weights=peak_mags)

                e_val = (e_phase % TAU) * bins / TAU

                b[i, j] = int(np.round(b_idx)) % bins
                e[i, j] = int(np.round(e_val)) % bins

    # Compute derived fields
    d = (b + e) % bins
    a = (b + 2 * e) % bins
    Eb, Er, Eg = qa_chromatic_fields(b, e, bins=bins)

    result = dict(b=b, e=e, d=d, a=a, Eb=Eb, Er=Er, Eg=Eg)

    if use_chromo:
        chromo_fields = cube_to_chromo_fields(cube, k_peaks=k_peaks, use_derivative=use_derivative, derivative_order=derivative_order)
        result.update(chromo_fields)

    return result

# ============================================================================
# Harmonic-Aware Feature Embedding
# ============================================================================

def circular_embed_mod24(x: np.ndarray, bins: int = 24) -> np.ndarray:
    """
    Convert modular field to circular (cos, sin) features

    Preserves mod-24 wraparound symmetry in Euclidean space

    Args:
        x: Modular field values
        bins: Modulus

    Returns:
        (cos, sin) features of shape (..., 2)
    """
    theta = (x.astype(float) * TAU) / bins
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)

def build_harmonic_features(
    Eb: np.ndarray,
    Er: np.ndarray,
    Eg: np.ndarray,
    bins: int = 24,
    chromo_fields: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Build harmonic feature vectors from chromatic fields and optional chromogeometry

    Args:
        Eb, Er, Eg: Chromatic field components (H × W)
        bins: Modulus
        chromo_fields: Optional dict with 'u', 'v', 'Qb', 'Qr', 'Qg' arrays

    Returns:
        Feature array of shape (H*W, 6 or 11)
            Base: [cos(Eb), sin(Eb), cos(Er), sin(Er), cos(Eg), sin(Eg)]
            With chromo: + [u, v, Qb, Qr, Qg]
    """
    emb_b = circular_embed_mod24(Eb, bins=bins)
    emb_r = circular_embed_mod24(Er, bins=bins)
    emb_g = circular_embed_mod24(Eg, bins=bins)
    feats = np.concatenate([emb_b, emb_r, emb_g], axis=-1)
    H, W, _ = feats.shape

    if chromo_fields is not None:
        u = chromo_fields['u']
        v = chromo_fields['v']
        Qb = chromo_fields['Qb']
        Qr = chromo_fields['Qr']
        Qg = chromo_fields['Qg']
        chromo_stack = np.stack([u, v, Qb, Qr, Qg], axis=-1)  # (H, W, 5)
        feats = np.concatenate([feats, chromo_stack], axis=-1)  # (H, W, 11)

    return feats.reshape(H * W, -1)

# ============================================================================
# K-Means Clustering (from scratch)
# ============================================================================

def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization for better cluster seeding"""
    n, d = X.shape
    centers = np.empty((k, d), dtype=float)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    closest_dist_sq = np.sum((X - centers[0])**2, axis=1)

    for c in range(1, k):
        # Normalize probabilities carefully to avoid numerical issues
        probs = closest_dist_sq / (closest_dist_sq.sum() + 1e-12)
        probs = np.maximum(probs, 0)  # Ensure non-negative
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum  # Renormalize to exactly 1.0
        else:
            # Fallback to uniform if all distances are zero
            probs = np.ones(n) / n

        idx = rng.choice(n, p=probs)
        centers[c] = X[idx]
        new_dist_sq = np.sum((X - centers[c])**2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centers

def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering (pure NumPy implementation)

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance
        seed: Random seed

    Returns:
        (centers, labels) tuple
    """
    rng = np.random.default_rng(seed)
    centers = _kmeanspp_init(X, k, rng)

    for _ in range(max_iter):
        # Assign to nearest center
        d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(d2, axis=1)

        # Update centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = X[mask].mean(axis=0)
            else:
                new_centers[i] = X[rng.integers(0, X.shape[0])]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        if shift < tol:
            break

    return centers, labels

# ============================================================================
# DBSCAN Clustering (from scratch)
# ============================================================================

def dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    DBSCAN density-based clustering (pure NumPy implementation)

    Args:
        X: Data matrix (n_samples, n_features)
        eps: Neighborhood radius
        min_samples: Minimum samples for core point

    Returns:
        Labels array (-1 for noise)
    """
    n = X.shape[0]
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Find neighbors
        d2 = np.sum((X - X[i])**2, axis=1)
        neigh = np.where(d2 <= eps * eps)[0]

        if neigh.size < min_samples:
            labels[i] = -1  # Noise
            continue

        # Start new cluster
        labels[i] = cluster_id
        seeds = list(neigh)

        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                d2j = np.sum((X - X[j])**2, axis=1)
                neigh_j = np.where(d2j <= eps * eps)[0]

                if neigh_j.size >= min_samples:
                    for t in neigh_j:
                        if t not in seeds:
                            seeds.append(int(t))

            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels

# ============================================================================
# PCA via SVD
# ============================================================================

def pca_svd(X: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal Component Analysis via Singular Value Decomposition

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of components

    Returns:
        (Z, U_k, S_k) where:
            Z: Projected data (n_samples, k)
            U_k: Principal components (n_features, k)
            S_k: Singular values (k,)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    U_k = V[:, :k]
    S_k = S[:k]
    Z = Xc @ U_k
    return Z, U_k, S_k

# ============================================================================
# Sector Masking by Residue Classes
# ============================================================================

def sector_mask(field: np.ndarray, indices: List[int], bins: int = 24) -> np.ndarray:
    """
    Create binary mask for specified modular indices

    Args:
        field: Modular field (H × W)
        indices: List of residue classes to include
        bins: Modulus

    Returns:
        Binary mask (H × W)
    """
    return np.isin(field % bins, np.array(indices, dtype=int))

def build_sector_masks(
    field: np.ndarray,
    groups: Optional[Dict[str, List[int]]] = None,
    bins: int = 24
) -> Dict[str, np.ndarray]:
    """
    Build multiple sector masks from residue class groups

    Args:
        field: Modular field (H × W)
        groups: Dictionary of {name: [indices]} or None for defaults
        bins: Modulus

    Returns:
        Dictionary of {name: mask} arrays
    """
    if groups is None:
        # Default groups
        groups = {
            'prime_moduli': [1, 5, 7, 11, 13, 17, 19, 23],
            'quadrature': [0, 6, 12, 18],
            'thirds': [0, 8, 16]
        }

    masks = {}
    for name, indices in groups.items():
        masks[name] = sector_mask(field, indices, bins)

    return masks

# ============================================================================
# Main Pipeline Function
# ============================================================================

def qa_hyperspectral_pipeline(
    cube: np.ndarray,
    bins: int = 24,
    k_peaks: int = 3,
    phase_mode: str = "weighted",
    kmeans_k: int = 4,
    dbscan_eps: float = 0.35,
    dbscan_min_samples: int = 30,
    pca_k: int = 2,
    sector_field: str = "Er",
    sector_groups: Optional[Dict[str, List[int]]] = None,
    use_chromo: bool = True
) -> Dict[str, Any]:
    """
    Complete QA hyperspectral imaging pipeline

    Pipeline stages:
    1. Phase-aware DFT encoding → (b, e) tuples
    2. Chromatic fields → Eb, Er, Eg
    3. Harmonic circular embedding → 6D features
    4. PCA dimensionality reduction
    5. K-Means clustering
    6. DBSCAN clustering
    7. Sector mask generation

    Args:
        cube: Hyperspectral data (H × W × B)
        bins: Modular arithmetic base (default 24)
        k_peaks: Number of spectral peaks (default 3)
        phase_mode: "weighted" or "argmax"
        kmeans_k: Number of K-Means clusters
        dbscan_eps: DBSCAN epsilon
        dbscan_min_samples: DBSCAN minimum samples
        pca_k: Number of PCA components
        sector_field: Field to use for sector masks ("Er" default)
        sector_groups: Custom sector mask groups

    Returns:
        Dictionary containing:
            b, e, d, a: QA tuples (H × W)
            Eb, Er, Eg: Chromatic fields (H × W)
            features: Harmonic embeddings (H*W × 6)
            Z: PCA projection (H*W × pca_k)
            U_k, S_k: PCA components
            labels_kmeans: K-Means labels (H × W)
            labels_dbscan: DBSCAN labels (H × W)
            sector_masks: Dictionary of masks
    """
    H, W, B = cube.shape

    print(f"QA Hyperspectral Pipeline")
    print(f"  Cube shape: {cube.shape}")
    print(f"  Modulus: {bins}, Peaks: {k_peaks}, Phase: {phase_mode}")
    print()

    # Stage 1: Convert to QA fields
    print("[1/7] Phase-aware DFT encoding...")
    qa_fields = cube_to_qa_fields_phase_multi(cube, bins=bins, k_peaks=k_peaks, phase_mode=phase_mode, use_derivative=True, derivative_order=1, use_chromo=use_chromo)

    b, e, d, a = qa_fields['b'], qa_fields['e'], qa_fields['d'], qa_fields['a']
    Eb, Er, Eg = qa_fields['Eb'], qa_fields['Er'], qa_fields['Eg']

    # Stage 2: Build harmonic features
    print("[2/7] Building harmonic features...")
    chromo_fields = None
    if use_chromo:
        chromo_fields = {k: v for k, v in qa_fields.items() if k in ['u', 'v', 'Qb', 'Qr', 'Qg']}
    features = build_harmonic_features(Eb, Er, Eg, bins=bins, chromo_fields=chromo_fields)

    # Stage 3: PCA
    print("[3/7] PCA dimensionality reduction...")
    Z, U_k, S_k = pca_svd(features, k=pca_k)

    # Stage 4: K-Means
    print(f"[4/7] K-Means clustering (k={kmeans_k})...")
    _, labels_km = kmeans(features, k=kmeans_k)
    labels_kmeans = labels_km.reshape(H, W)

    # Stage 5: DBSCAN
    print(f"[5/7] DBSCAN clustering (eps={dbscan_eps})...")
    labels_db = dbscan(features, eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_dbscan = labels_db.reshape(H, W)

    # Stage 6: Sector masks
    print(f"[6/7] Generating sector masks from {sector_field}...")
    field_for_masks = qa_fields[sector_field]
    sector_masks = build_sector_masks(field_for_masks, groups=sector_groups, bins=bins)

    print("[7/7] Pipeline complete!")
    print()

    result = {
        'b': b, 'e': e, 'd': d, 'a': a,
        'Eb': Eb, 'Er': Er, 'Eg': Eg,
        'features': features,
        'Z': Z, 'U_k': U_k, 'S_k': S_k,
        'labels_kmeans': labels_kmeans,
        'labels_dbscan': labels_dbscan,
        'sector_masks': sector_masks
    }
    if chromo_fields:
        result.update(chromo_fields)
    return result

# ============================================================================
# Example Usage
# ============================================================================

def generate_synthetic_cube(shape: Tuple[int, int, int] = (50, 50, 100), seed: int = 42) -> np.ndarray:
    """
    Generate the synthetic cube used in the vault narratives (four spectral classes).
    """
    H, W, B = shape
    rng = np.random.default_rng(seed)
    cube = np.zeros((H, W, B))

    for i in range(H):
        for j in range(W):
            class_id = (i // max(1, H // 2)) * 2 + (j // max(1, W // 2))
            class_id %= 4

            peak_band = {0: 20, 1: 40, 2: 60, 3: 80}[class_id]
            cube[i, j, peak_band] = 1.0
            cube[i, j, :] += 0.1 * rng.standard_normal(B)

    return cube


def load_cube_from_file(path: Path, dataset_key: Optional[str] = None) -> np.ndarray:
    """
    Load a hyperspectral cube from npy/npz/mat formats.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cube path does not exist: {path}")

    ext = path.suffix.lower()
    if ext == ".npy":
        cube = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        key = dataset_key or (data.files[0] if data.files else None)
        if key is None:
            raise ValueError(f"No arrays stored in {path}")
        cube = data[key]
    elif ext == ".mat":
        try:
            from scipy.io import loadmat  # type: ignore
        except ImportError as exc:
            raise ImportError("scipy is required to load .mat files") from exc
        data = loadmat(path)
        if dataset_key:
            cube = data.get(dataset_key)
            if cube is None:
                raise KeyError(f"Dataset key '{dataset_key}' not found in {path}")
        else:
            cube_keys = [k for k in data.keys() if not k.startswith("__")]
            if not cube_keys:
                raise ValueError(f"No arrays found in {path}")
            cube = data[cube_keys[0]]
    else:
        raise ValueError(f"Unsupported cube format: {ext}")

    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube.shape}")
    return cube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QA hyperspectral pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to hyperspectral dataset (.mat file).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground truth labels (.mat file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results).",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        help="Dataset key for .mat files (defaults to the first array).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=24,
        help="Modular base for QA tuples (default: 24).",
    )
    parser.add_argument(
        "--k-peaks",
        type=int,
        default=3,
        help="Number of spectral peaks when encoding (default: 3).",
    )
    parser.add_argument(
        "--kmeans-k",
        type=str,
        default="auto",
        help="K-Means cluster count or 'auto' for ground truth classes (default: auto).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon (default: 0.5).",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN minimum samples (default: 5).",
    )
    parser.add_argument(
        "--sector-field",
        type=str,
        default="Er",
        choices=["Eb", "Er", "Eg"],
        help="Field used for sector masks (default: Er).",
    )
    parser.add_argument(
        "--synthetic-shape",
        type=int,
        nargs=3,
        metavar=("H", "W", "B"),
        default=(50, 50, 100),
        help="Shape for synthetic cube when no dataset is provided (default: 50 50 100).",
    )
    parser.add_argument(
        "--phase-mode",
        type=str,
        default="weighted",
        choices=["weighted", "argmax"],
        help="Phase aggregation mode (default: weighted).",
    )
    parser.add_argument(
        "--pca-k",
        type=int,
        default=2,
        help="Number of PCA components to retain (default: 2).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsampling factor for spatial dimensions (default: 1, no subsampling).",
    )
    parser.add_argument(
        "--use-chromo",
        action="store_true",
        default=True,
        help="Use chromogeometry features in addition to QA fields (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("QA Hyperspectral Pipeline - Run")
    print("=" * 70 + "\n")

    # Load dataset and ground truth
    if args.dataset:
        dataset_path = Path(args.dataset)
        print(f"Loading hyperspectral dataset from: {dataset_path}")
        cube = load_cube_from_file(dataset_path, args.dataset_key)

        gt = None
        if args.ground_truth:
            gt_path = Path(args.ground_truth)
            print(f"Loading ground truth from: {gt_path}")
            # Load ground truth (can be 2D)
            ext = gt_path.suffix.lower()
            if ext == ".mat":
                from scipy.io import loadmat
                data = loadmat(gt_path)
                if args.dataset_key:
                    gt_data = data.get(args.dataset_key)
                    if gt_data is None:
                        raise KeyError(f"Dataset key '{args.dataset_key}' not found in {gt_path}")
                else:
                    gt_keys = [k for k in data.keys() if not k.startswith("__")]
                    if not gt_keys:
                        raise ValueError(f"No arrays found in {gt_path}")
                    gt_data = data[gt_keys[0]]
            else:
                gt_data = np.load(gt_path)  # For .npy files

            gt = np.asarray(gt_data).squeeze()  # Ensure 2D
            if gt.ndim != 2:
                raise ValueError(f"Ground truth must be 2D, got shape {gt.shape}")
            if gt.shape != cube.shape[:2]:
                raise ValueError(f"Ground truth shape {gt.shape} doesn't match cube spatial dimensions {cube.shape[:2]}")
    else:
        print("Generating synthetic hyperspectral cube...")
        cube = generate_synthetic_cube(tuple(args.synthetic_shape))
        gt = None

    # Subsample if requested
    if args.subsample > 1:
        print(f"Subsampling data by factor {args.subsample}")
        cube = cube[::args.subsample, ::args.subsample, :]
        if gt is not None:
            gt = gt[::args.subsample, ::args.subsample]
        print(f"Subsampled cube shape: {cube.shape}")
        if gt is not None:
            print(f"Subsampled ground truth shape: {gt.shape}")

    print(f"Final cube shape: {cube.shape}")
    if gt is not None:
        print(f"Ground truth shape: {gt.shape}")
        print(f"Number of classes: {len(np.unique(gt)) - 1}")  # -1 for background
    print()

    # Determine k-means clusters
    kmeans_k = args.kmeans_k
    if kmeans_k == "auto":
        if gt is not None:
            kmeans_k = len(np.unique(gt)) - 1  # -1 for background
            print(f"Auto-detected {kmeans_k} clusters from ground truth")
        else:
            kmeans_k = 4  # Default for synthetic data
            print(f"Using default {kmeans_k} clusters")
    elif isinstance(kmeans_k, str):
        kmeans_k = int(kmeans_k)
    print()

    results = qa_hyperspectral_pipeline(
        cube,
        bins=args.bins,
        k_peaks=args.k_peaks,
        kmeans_k=kmeans_k,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        sector_field=args.sector_field,
        phase_mode=args.phase_mode,
        pca_k=args.pca_k,
        use_chromo=args.use_chromo,
    )

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("QA Fields:")
    print(f"  b: {results['b'].shape}, range [{results['b'].min()}, {results['b'].max()}]")
    print(f"  e: {results['e'].shape}, range [{results['e'].min()}, {results['e'].max()}]")
    print(f"  Eb unique: {np.unique(results['Eb']).size}")
    print(f"  Er unique: {np.unique(results['Er']).size}")
    print(f"  Eg unique: {np.unique(results['Eg']).size}")
    print()
    print("Clustering:")
    print(f"  K-Means clusters: {np.unique(results['labels_kmeans']).size}")
    dbscan_labels = results['labels_dbscan']
    n_noise = int(np.sum(dbscan_labels == -1))
    core_clusters = np.unique(dbscan_labels[dbscan_labels >= 0]).size
    print(f"  DBSCAN clusters: {core_clusters} (noise points: {n_noise})")
    print()
    print("Sector Masks:")
    for name, mask in results['sector_masks'].items():
        fraction = np.sum(mask) / mask.size * 100
        print(f"  {name}: {np.sum(mask)} pixels ({fraction:.1f}%)")
    print()
    print("PCA:")
    explained = results['S_k'][: args.pca_k] / np.sum(results['S_k'])
    print(f"  Explained variance (first {args.pca_k}): {explained}")
    print()

    # Ground truth evaluation
    metrics = {}
    if gt is not None:
        print("Ground Truth Evaluation:")
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        # Flatten for evaluation (only labeled pixels)
        gt_flat = gt.flatten()
        kmeans_flat = results['labels_kmeans'].flatten()
        dbscan_flat = results['labels_dbscan'].flatten()

        # Only evaluate on labeled pixels (gt > 0)
        labeled_mask = gt_flat > 0
        if np.sum(labeled_mask) > 0:
            gt_labeled = gt_flat[labeled_mask]
            kmeans_labeled = kmeans_flat[labeled_mask]
            dbscan_labeled = dbscan_flat[labeled_mask]

            metrics['kmeans_ari'] = adjusted_rand_score(gt_labeled, kmeans_labeled)
            metrics['kmeans_nmi'] = normalized_mutual_info_score(gt_labeled, kmeans_labeled)

            # DBSCAN (excluding noise points for evaluation)
            dbscan_no_noise = dbscan_labeled.copy()
            dbscan_no_noise[dbscan_no_noise == -1] = len(np.unique(gt_labeled))  # Map noise to extra class
            metrics['dbscan_ari'] = adjusted_rand_score(gt_labeled, dbscan_no_noise)
            metrics['dbscan_nmi'] = normalized_mutual_info_score(gt_labeled, dbscan_no_noise)

            print(f"  K-Means ARI: {metrics['kmeans_ari']:.3f}")
            print(f"  K-Means NMI: {metrics['kmeans_nmi']:.3f}")
            print(f"  DBSCAN ARI: {metrics['dbscan_ari']:.3f}")
            print(f"  DBSCAN NMI: {metrics['dbscan_nmi']:.3f}")
        else:
            print("  No labeled pixels found for evaluation")
        print()

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to: {output_dir}")

    # Save visualizations
    try:
        import matplotlib.pyplot as plt

        # Phase map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(results['b'], cmap='viridis')
        ax1.set_title('Phase Parameter b')
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(results['e'], cmap='plasma')
        ax2.set_title('Phase Parameter e')
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(output_dir / 'phase_map.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Chromatic fields
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(results['Eb'], cmap='RdYlBu')
        ax1.set_title('Electric Field Eb')
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(results['Er'], cmap='RdYlBu')
        ax2.set_title('Magnetic Field Er')
        plt.colorbar(im2, ax=ax2)
        im3 = ax3.imshow(results['Eg'], cmap='RdYlBu')
        ax3.set_title('Scalar Field Eg')
        plt.colorbar(im3, ax=ax3)
        plt.tight_layout()
        plt.savefig(output_dir / 'chromatic_fields.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Clustering comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(results['labels_kmeans'], cmap='tab10')
        ax1.set_title('K-Means Clustering')
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(results['labels_dbscan'], cmap='tab10')
        ax2.set_title('DBSCAN Clustering')
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(output_dir / 'clustering_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Visualizations saved")

    except ImportError:
        print("  Warning: matplotlib not available, skipping visualizations")

    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  ✓ Metrics saved")

    # Save QA parameters
    np.savez(output_dir / 'qa_parameters.npz',
             b=results['b'], e=results['e'], d=results['d'], a=results['a'],
             Eb=results['Eb'], Er=results['Er'], Eg=results['Eg'])
    print("  ✓ QA parameters saved")

    print()
    print("✓ Pipeline run complete!")
    print()


if __name__ == "__main__":
    main()
