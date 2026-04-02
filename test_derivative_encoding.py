#!/usr/bin/env python3
"""
Test alternative encoding: Spectral derivatives

Hypothesis: Indian Pines spectra are similar in absolute values but differ
in their derivatives (slopes, curvature). Use derivative-based DFT encoding.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter

def load_indian_pines():
    """Load Indian Pines dataset"""
    mat = sio.loadmat('hyperspectral_data/Indian_pines_corrected.mat')
    keys = [k for k in mat.keys() if not k.startswith('__')]
    return mat[keys[0]]

def spectrum_to_be_derivative(spectrum, bins=24, num_peaks=3, derivative_order=1):
    """
    Encoding using spectral derivatives

    Args:
        derivative_order: 1 (first derivative/slope) or 2 (second derivative/curvature)
    """
    N = len(spectrum)
    spectrum = np.asarray(spectrum, dtype=float)

    # Normalize original
    spec_min = spectrum.min()
    spec_max = spectrum.max()
    if spec_max > spec_min:
        spectrum = (spectrum - spec_min) / (spec_max - spec_min)
    else:
        return 0, 0

    # Compute derivative
    if derivative_order == 1:
        spectrum_deriv = np.diff(spectrum)
    elif derivative_order == 2:
        spectrum_deriv = np.diff(np.diff(spectrum))
    else:
        spectrum_deriv = spectrum

    # Pad to original length
    spectrum_deriv = np.pad(spectrum_deriv, (0, N - len(spectrum_deriv)), mode='edge')

    # DFT on derivative
    fft = np.fft.fft(spectrum_deriv)
    magnitudes = np.abs(fft[:N//2])
    phases = np.angle(fft[:N//2])

    if magnitudes.sum() == 0:
        return 0, 0

    # Skip DC, find peaks
    magnitudes_no_dc = magnitudes[1:]
    phases_no_dc = phases[1:]

    if len(magnitudes_no_dc) == 0 or magnitudes_no_dc.sum() == 0:
        return 0, 0

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

    # Spectral centroid
    freqs = np.arange(1, len(magnitudes))
    centroid = np.sum(freqs * magnitudes_no_dc) / (magnitudes_no_dc.sum() + 1e-10)
    centroid_norm = centroid / len(magnitudes)
    e = int(centroid_norm * bins) % bins

    return b, e

def spectrum_to_be_multiscale(spectrum, bins=24):
    """
    Multi-scale encoding: combine features from original + 1st + 2nd derivatives
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

    # Get features from each scale
    features = []

    # Scale 1: Original
    fft0 = np.fft.fft(spectrum - spectrum.mean())
    mag0 = np.abs(fft0[1:N//4])  # Low frequencies only
    if mag0.sum() > 0:
        features.append(np.argmax(mag0))

    # Scale 2: First derivative
    deriv1 = np.diff(spectrum)
    deriv1 = np.pad(deriv1, (0, N - len(deriv1)), mode='edge')
    fft1 = np.fft.fft(deriv1)
    mag1 = np.abs(fft1[1:N//4])
    if mag1.sum() > 0:
        features.append(np.argmax(mag1))

    # Scale 3: Second derivative
    deriv2 = np.diff(np.diff(spectrum))
    deriv2 = np.pad(deriv2, (0, N - len(deriv2)), mode='edge')
    fft2 = np.fft.fft(deriv2)
    mag2 = np.abs(fft2[1:N//4])
    if mag2.sum() > 0:
        features.append(np.argmax(mag2))

    if len(features) == 0:
        return 0, 0

    # Combine: b from first scale, e from combination
    b = int((features[0] / (N/4)) * bins) % bins

    e_combined = np.mean(features)
    e = int((e_combined / (N/4)) * bins) % bins

    return b, e

def test_derivative_encodings():
    """Test different derivative-based encodings"""
    print("="*70)
    print("TESTING DERIVATIVE-BASED ENCODINGS")
    print("="*70)
    print()

    data = load_indian_pines()
    H, W, _ = data.shape

    rng = np.random.RandomState(42)
    sample_coords = [(rng.randint(H), rng.randint(W)) for _ in range(500)]

    methods = {
        'Original (DC-removed)': lambda s: spectrum_to_be_derivative(s, derivative_order=0),
        '1st Derivative': lambda s: spectrum_to_be_derivative(s, derivative_order=1),
        '2nd Derivative': lambda s: spectrum_to_be_derivative(s, derivative_order=2),
        'Multi-scale': lambda s: spectrum_to_be_multiscale(s),
    }

    results = {}

    for name, encoder in methods.items():
        b_vals, e_vals = [], []

        for i, j in sample_coords:
            try:
                b, e = encoder(data[i, j, :])
                b_vals.append(b)
                e_vals.append(e)
            except:
                continue

        b_vals = np.array(b_vals)
        e_vals = np.array(e_vals)

        b_unique = len(np.unique(b_vals))
        e_unique = len(np.unique(e_vals))

        results[name] = {
            'b': b_vals,
            'e': e_vals,
            'b_unique': b_unique,
            'e_unique': e_unique
        }

        print(f"{name}:")
        print(f"  b: unique={b_unique}/24 ({100*b_unique/24:.1f}%), std={b_vals.std():.2f}, range=[{b_vals.min()}, {b_vals.max()}]")
        print(f"  e: unique={e_unique}/24 ({100*e_unique/24:.1f}%), std={e_vals.std():.2f}, range=[{e_vals.min()}, {e_vals.max()}]")
        print(f"  Total unique (b,e) pairs: {len(np.unique(list(zip(b_vals, e_vals))))}")
        print()

    return results

def visualize_derivative_results(results):
    """Visualize comparison of different encodings"""
    print("="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)
    print()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for idx, (name, data) in enumerate(results.items()):
        # b distribution
        ax_b = axes[0, idx]
        counts_b = Counter(data['b'])
        ax_b.bar(counts_b.keys(), counts_b.values(), alpha=0.7, color='blue')
        ax_b.set_title(f'{name}\nb: {data["b_unique"]} unique')
        ax_b.set_xlabel('b value')
        ax_b.set_ylabel('Count')
        ax_b.set_xlim(-1, 24)
        ax_b.grid(alpha=0.3)

        # e distribution
        ax_e = axes[1, idx]
        counts_e = Counter(data['e'])
        ax_e.bar(counts_e.keys(), counts_e.values(), alpha=0.7, color='red')
        ax_e.set_title(f'e: {data["e_unique"]} unique')
        ax_e.set_xlabel('e value')
        ax_e.set_ylabel('Count')
        ax_e.set_xlim(-1, 24)
        ax_e.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/derivative_encoding_comparison.png', dpi=150)
    print("✓ Saved: results/derivative_encoding_comparison.png")
    print()

def main():
    """Run derivative encoding tests"""

    results = test_derivative_encodings()
    visualize_derivative_results(results)

    # Find best method
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()

    best_method = max(results.items(),
                      key=lambda x: x[1]['b_unique'] + x[1]['e_unique'])

    print(f"Best method: {best_method[0]}")
    print(f"  b: {best_method[1]['b_unique']} unique values")
    print(f"  e: {best_method[1]['e_unique']} unique values")
    print(f"  Total: {best_method[1]['b_unique'] + best_method[1]['e_unique']} combined")
    print()

    # Compare to original
    orig = results.get('Original (DC-removed)', results[list(results.keys())[0]])
    best = best_method[1]

    improvement = (best['b_unique'] + best['e_unique']) / (orig['b_unique'] + orig['e_unique'])
    print(f"Improvement over original: {improvement:.2f}x")
    print()

    if improvement > 1.5:
        print("✓ DERIVATIVE ENCODING SHOWS SIGNIFICANT IMPROVEMENT!")
        print()
        print("Recommendation: Use derivative-based encoding in pipeline")
    elif improvement > 1.2:
        print("⚠ Moderate improvement - may help but not dramatic")
    else:
        print("✗ No significant improvement")
        print()
        print("Fundamental issue: Indian Pines spectra are genuinely very similar")
        print("QA phase/centroid encoding may not be ideal for this dataset")
        print()
        print("Alternative strategies:")
        print("  1. Use spatial features (texture, neighbors)")
        print("  2. Combine QA with traditional features (spectral angle, absorption)")
        print("  3. Try on more diverse dataset (urban scenes, minerals)")

    print()

if __name__ == "__main__":
    main()
