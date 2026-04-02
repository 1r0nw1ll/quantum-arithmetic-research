#!/usr/bin/env python3
"""
Test improved encoding with DC removal
Root cause: DC component dominates DFT → all spectra look identical
Solution: Remove DC (mean-center) before DFT analysis
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter
import json

def load_indian_pines():
    """Load Indian Pines dataset"""
    mat = sio.loadmat('hyperspectral_data/Indian_pines_corrected.mat')
    keys = [k for k in mat.keys() if not k.startswith('__')]
    return mat[keys[0]]

def spectrum_to_be_original(spectrum, bins=24, num_peaks=3):
    """Original encoding (for comparison)"""
    N = len(spectrum)
    spectrum = np.asarray(spectrum, dtype=float)

    # Normalize
    spec_min = spectrum.min()
    spec_max = spectrum.max()
    if spec_max > spec_min:
        spectrum = (spectrum - spec_min) / (spec_max - spec_min)
    else:
        return 0, 0

    # DFT
    fft = np.fft.fft(spectrum)
    magnitudes = np.abs(fft[:N//2])
    phases = np.angle(fft[:N//2])

    if magnitudes.sum() == 0:
        return 0, 0

    # Top peaks
    top_indices = np.argsort(magnitudes)[-num_peaks:][::-1]

    # Weighted phase
    weights = magnitudes[top_indices]
    weights = weights / (weights.sum() + 1e-10)
    avg_phase = np.arctan2(
        np.sum(weights * np.sin(phases[top_indices])),
        np.sum(weights * np.cos(phases[top_indices]))
    )

    avg_phase_norm = (avg_phase + np.pi) / (2 * np.pi)
    b = int(avg_phase_norm * bins) % bins

    # Spectral centroid
    freqs = np.arange(len(magnitudes))
    centroid = np.sum(freqs * magnitudes) / (magnitudes.sum() + 1e-10)
    centroid_norm = centroid / len(magnitudes)
    e = int(centroid_norm * bins) % bins

    return b, e

def spectrum_to_be_dc_removed(spectrum, bins=24, num_peaks=3):
    """
    Improved encoding with DC removal

    Changes:
    1. Remove DC component (mean-center) before DFT
    2. Skip DC bin (index 0) when finding peaks
    3. Use higher-frequency information for encoding
    """
    N = len(spectrum)
    spectrum = np.asarray(spectrum, dtype=float)

    # Normalize
    spec_min = spectrum.min()
    spec_max = spectrum.max()
    if spec_max > spec_min:
        spectrum = (spectrum - spec_min) / (spec_max - spec_min)
    else:
        return 0, 0

    # **KEY CHANGE: Remove DC component (mean-center)**
    spectrum = spectrum - spectrum.mean()

    # DFT
    fft = np.fft.fft(spectrum)
    magnitudes = np.abs(fft[:N//2])
    phases = np.angle(fft[:N//2])

    if magnitudes.sum() == 0:
        return 0, 0

    # **KEY CHANGE: Skip DC component (index 0) when finding peaks**
    magnitudes_no_dc = magnitudes[1:]  # Exclude DC
    phases_no_dc = phases[1:]

    if len(magnitudes_no_dc) == 0 or magnitudes_no_dc.sum() == 0:
        return 0, 0

    # Top peaks (excluding DC)
    top_indices = np.argsort(magnitudes_no_dc)[-num_peaks:][::-1]

    # Weighted phase
    weights = magnitudes_no_dc[top_indices]
    weights = weights / (weights.sum() + 1e-10)
    avg_phase = np.arctan2(
        np.sum(weights * np.sin(phases_no_dc[top_indices])),
        np.sum(weights * np.cos(phases_no_dc[top_indices]))
    )

    avg_phase_norm = (avg_phase + np.pi) / (2 * np.pi)
    b = int(avg_phase_norm * bins) % bins

    # Spectral centroid (excluding DC)
    freqs = np.arange(1, len(magnitudes))  # Start from 1
    centroid = np.sum(freqs * magnitudes_no_dc) / (magnitudes_no_dc.sum() + 1e-10)
    centroid_norm = centroid / len(magnitudes)
    e = int(centroid_norm * bins) % bins

    return b, e

def compare_encodings(num_samples=1000):
    """Compare original vs DC-removed encoding"""
    print("="*70)
    print("COMPARING ORIGINAL VS DC-REMOVED ENCODING")
    print("="*70)
    print()

    data = load_indian_pines()
    H, W, bands = data.shape

    rng = np.random.RandomState(42)
    sample_coords = [(rng.randint(H), rng.randint(W)) for _ in range(num_samples)]

    # Original encoding
    b_orig, e_orig = [], []
    for i, j in sample_coords:
        b, e = spectrum_to_be_original(data[i, j, :], bins=24)
        b_orig.append(b)
        e_orig.append(e)

    b_orig = np.array(b_orig)
    e_orig = np.array(e_orig)

    # DC-removed encoding
    b_dc, e_dc = [], []
    for i, j in sample_coords:
        b, e = spectrum_to_be_dc_removed(data[i, j, :], bins=24)
        b_dc.append(b)
        e_dc.append(e)

    b_dc = np.array(b_dc)
    e_dc = np.array(e_dc)

    # Statistics
    print("ORIGINAL ENCODING (bins=24):")
    print(f"  b: unique={len(np.unique(b_orig))}, range=[{b_orig.min()}, {b_orig.max()}], std={b_orig.std():.2f}")
    print(f"  e: unique={len(np.unique(e_orig))}, range=[{e_orig.min()}, {e_orig.max()}], std={e_orig.std():.2f}")
    print()

    print("DC-REMOVED ENCODING (bins=24):")
    print(f"  b: unique={len(np.unique(b_dc))}, range=[{b_dc.min()}, {b_dc.max()}], std={b_dc.std():.2f}")
    print(f"  e: unique={len(np.unique(e_dc))}, range=[{e_dc.min()}, {e_dc.max()}], std={e_dc.std():.2f}")
    print()

    improvement_b = len(np.unique(b_dc)) / len(np.unique(b_orig))
    improvement_e = len(np.unique(e_dc)) / len(np.unique(e_orig))

    print(f"IMPROVEMENT:")
    print(f"  b unique values: {improvement_b:.1f}x increase")
    print(f"  e unique values: {improvement_e:.1f}x increase")
    print()

    return b_orig, e_orig, b_dc, e_dc

def test_different_bins_dc_removed():
    """Test DC-removed encoding with different bins"""
    print("="*70)
    print("TESTING DC-REMOVED ENCODING WITH DIFFERENT BINS")
    print("="*70)
    print()

    data = load_indian_pines()
    H, W, _ = data.shape

    rng = np.random.RandomState(42)
    sample_coords = [(rng.randint(H), rng.randint(W)) for _ in range(500)]

    bins_to_test = [12, 24, 36, 48]

    for bins in bins_to_test:
        b_vals, e_vals = [], []

        for i, j in sample_coords:
            b, e = spectrum_to_be_dc_removed(data[i, j, :], bins=bins)
            b_vals.append(b)
            e_vals.append(e)

        b_vals = np.array(b_vals)
        e_vals = np.array(e_vals)

        print(f"bins={bins}:")
        print(f"  b: unique={len(np.unique(b_vals))}/{bins} ({100*len(np.unique(b_vals))/bins:.1f}%), std={b_vals.std():.2f}")
        print(f"  e: unique={len(np.unique(e_vals))}/{bins} ({100*len(np.unique(e_vals))/bins:.1f}%), std={e_vals.std():.2f}")
        print()

def visualize_comparison(b_orig, e_orig, b_dc, e_dc):
    """Visualize distribution comparison"""
    print("="*70)
    print("GENERATING COMPARISON VISUALIZATION")
    print("="*70)
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original b
    ax = axes[0, 0]
    counts = Counter(b_orig)
    ax.bar(counts.keys(), counts.values(), alpha=0.7, color='blue')
    ax.set_title(f'Original Encoding: b distribution\n(unique={len(np.unique(b_orig))})', fontsize=12)
    ax.set_xlabel('b value')
    ax.set_ylabel('Count')
    ax.set_xlim(-1, 24)
    ax.grid(alpha=0.3)

    # DC-removed b
    ax = axes[0, 1]
    counts = Counter(b_dc)
    ax.bar(counts.keys(), counts.values(), alpha=0.7, color='darkblue')
    ax.set_title(f'DC-Removed Encoding: b distribution\n(unique={len(np.unique(b_dc))})', fontsize=12)
    ax.set_xlabel('b value')
    ax.set_ylabel('Count')
    ax.set_xlim(-1, 24)
    ax.grid(alpha=0.3)

    # Original e
    ax = axes[1, 0]
    counts = Counter(e_orig)
    ax.bar(counts.keys(), counts.values(), alpha=0.7, color='red')
    ax.set_title(f'Original Encoding: e distribution\n(unique={len(np.unique(e_orig))})', fontsize=12)
    ax.set_xlabel('e value')
    ax.set_ylabel('Count')
    ax.set_xlim(-1, 24)
    ax.grid(alpha=0.3)

    # DC-removed e
    ax = axes[1, 1]
    counts = Counter(e_dc)
    ax.bar(counts.keys(), counts.values(), alpha=0.7, color='darkred')
    ax.set_title(f'DC-Removed Encoding: e distribution\n(unique={len(np.unique(e_dc))})', fontsize=12)
    ax.set_xlabel('e value')
    ax.set_ylabel('Count')
    ax.set_xlim(-1, 24)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/dc_removal_comparison.png', dpi=150)
    print("✓ Saved: results/dc_removal_comparison.png")
    print()

def main():
    """Run DC removal test"""

    # Compare encodings
    b_orig, e_orig, b_dc, e_dc = compare_encodings(num_samples=1000)

    # Test different bins with DC removal
    test_different_bins_dc_removed()

    # Visualize
    visualize_comparison(b_orig, e_orig, b_dc, e_dc)

    # Summary
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()

    b_improvement = len(np.unique(b_dc)) / max(len(np.unique(b_orig)), 1)
    e_improvement = len(np.unique(e_dc)) / max(len(np.unique(e_orig)), 1)

    print(f"DC Removal Impact:")
    print(f"  b: {len(np.unique(b_orig))} → {len(np.unique(b_dc))} unique values ({b_improvement:.1f}x)")
    print(f"  e: {len(np.unique(e_orig))} → {len(np.unique(e_dc))} unique values ({e_improvement:.1f}x)")
    print()

    if b_improvement > 2 or e_improvement > 2:
        print("✓ DC REMOVAL SIGNIFICANTLY IMPROVES ENCODING VARIANCE!")
        print()
        print("Next steps:")
        print("  1. Update qa_hyperspectral_pipeline.py with DC-removed encoding")
        print("  2. Rerun clustering on Indian Pines")
        print("  3. Compare ARI/NMI vs baseline")
    else:
        print("⚠ DC removal provides modest improvement")
        print("Consider additional strategies (multi-scale, derivatives, etc.)")

    print()

if __name__ == "__main__":
    main()
