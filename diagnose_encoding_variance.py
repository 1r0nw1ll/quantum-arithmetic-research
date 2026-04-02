#!/usr/bin/env python3
"""
Diagnostic script to investigate low variance in (b,e) encoding
Identified issue: b∈[2,3], e∈[17,20] - only 2-5 unique values out of 24 possible

Test different configurations to find root cause and solution.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter
import json

# Import the encoding function from qa_hyperspectral_pipeline
import sys
sys.path.append('/home/player2/signal_experiments')

def load_indian_pines():
    """Load Indian Pines dataset"""
    mat = sio.loadmat('hyperspectral_data/Indian_pines_corrected.mat')
    keys = [k for k in mat.keys() if not k.startswith('__')]
    data = mat[keys[0]]
    print(f"Loaded Indian Pines: {data.shape}")
    return data

def spectrum_to_be_phase_multi(spectrum, bins=24, num_peaks=3, phase_mode="weighted"):
    """
    Phase-aware DFT encoding
    (Copied from qa_hyperspectral_pipeline.py for debugging)
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

    # DFT
    fft = np.fft.fft(spectrum)
    magnitudes = np.abs(fft[:N//2])
    phases = np.angle(fft[:N//2])

    if magnitudes.sum() == 0:
        return 0, 0

    # Find top peaks
    top_indices = np.argsort(magnitudes)[-num_peaks:][::-1]

    if phase_mode == "weighted":
        # Weighted average by magnitude
        weights = magnitudes[top_indices]
        weights = weights / (weights.sum() + 1e-10)

        avg_phase = np.arctan2(
            np.sum(weights * np.sin(phases[top_indices])),
            np.sum(weights * np.cos(phases[top_indices]))
        )
    elif phase_mode == "first_peak":
        avg_phase = phases[top_indices[0]]
    else:
        avg_phase = np.mean(phases[top_indices])

    # Map phase to [0, 2π) → [0, bins)
    avg_phase_norm = (avg_phase + np.pi) / (2 * np.pi)
    b = int(avg_phase_norm * bins) % bins

    # Spectral centroid for e
    freqs = np.arange(len(magnitudes))
    centroid = np.sum(freqs * magnitudes) / (magnitudes.sum() + 1e-10)
    centroid_norm = centroid / len(magnitudes)
    e = int(centroid_norm * bins) % bins

    return b, e

def analyze_encoding_variance(data, bins=24, num_samples=1000):
    """
    Sample random pixels and analyze (b,e) distribution
    """
    H, W, bands = data.shape

    # Random sampling
    rng = np.random.RandomState(42)
    sample_coords = [(rng.randint(H), rng.randint(W)) for _ in range(num_samples)]

    b_values = []
    e_values = []

    for i, j in sample_coords:
        spectrum = data[i, j, :]
        b, e = spectrum_to_be_phase_multi(spectrum, bins=bins)
        b_values.append(b)
        e_values.append(e)

    return np.array(b_values), np.array(e_values)

def test_bins_parameter():
    """Test different bins values"""
    print("="*70)
    print("TESTING BINS PARAMETER EFFECT ON VARIANCE")
    print("="*70)
    print()

    data = load_indian_pines()

    bins_to_test = [12, 18, 24, 36, 48]
    results = {}

    for bins in bins_to_test:
        print(f"Testing bins={bins}...")
        b_vals, e_vals = analyze_encoding_variance(data, bins=bins, num_samples=500)

        b_unique = len(np.unique(b_vals))
        e_unique = len(np.unique(e_vals))

        b_range = (b_vals.min(), b_vals.max())
        e_range = (e_vals.min(), e_vals.max())

        b_std = b_vals.std()
        e_std = e_vals.std()

        results[bins] = {
            'b_unique': b_unique,
            'e_unique': e_unique,
            'b_range': b_range,
            'e_range': e_range,
            'b_std': b_std,
            'e_std': e_std,
            'b_values': b_vals,
            'e_values': e_vals
        }

        print(f"  b: unique={b_unique}/{bins}, range={b_range}, std={b_std:.2f}")
        print(f"  e: unique={e_unique}/{bins}, range={e_range}, std={e_std:.2f}")
        print()

    return results

def test_phase_modes():
    """Test different phase_mode settings"""
    print("="*70)
    print("TESTING PHASE_MODE PARAMETER")
    print("="*70)
    print()

    data = load_indian_pines()

    # Modified encoding function for phase_mode testing
    def test_phase_mode(spectrum, mode):
        N = len(spectrum)
        spectrum = np.asarray(spectrum, dtype=float)

        spec_min = spectrum.min()
        spec_max = spectrum.max()
        if spec_max > spec_min:
            spectrum = (spectrum - spec_min) / (spec_max - spec_min)
        else:
            return 0, 0

        fft = np.fft.fft(spectrum)
        magnitudes = np.abs(fft[:N//2])
        phases = np.angle(fft[:N//2])

        if magnitudes.sum() == 0:
            return 0, 0

        top_indices = np.argsort(magnitudes)[-3:][::-1]

        if mode == "weighted":
            weights = magnitudes[top_indices]
            weights = weights / (weights.sum() + 1e-10)
            avg_phase = np.arctan2(
                np.sum(weights * np.sin(phases[top_indices])),
                np.sum(weights * np.cos(phases[top_indices]))
            )
        elif mode == "first_peak":
            avg_phase = phases[top_indices[0]]
        elif mode == "mean":
            avg_phase = np.mean(phases[top_indices])
        else:
            avg_phase = phases[0]  # DC phase

        avg_phase_norm = (avg_phase + np.pi) / (2 * np.pi)
        b = int(avg_phase_norm * 24) % 24

        # Centroid
        freqs = np.arange(len(magnitudes))
        centroid = np.sum(freqs * magnitudes) / (magnitudes.sum() + 1e-10)
        centroid_norm = centroid / len(magnitudes)
        e = int(centroid_norm * 24) % 24

        return b, e

    modes = ["weighted", "first_peak", "mean", "dc"]

    # Sample 500 spectra
    H, W, _ = data.shape
    rng = np.random.RandomState(42)
    sample_coords = [(rng.randint(H), rng.randint(W)) for _ in range(500)]

    for mode in modes:
        b_vals = []
        e_vals = []

        for i, j in sample_coords:
            spectrum = data[i, j, :]
            b, e = test_phase_mode(spectrum, mode)
            b_vals.append(b)
            e_vals.append(e)

        b_vals = np.array(b_vals)
        e_vals = np.array(e_vals)

        print(f"Phase mode: {mode}")
        print(f"  b: unique={len(np.unique(b_vals))}, range=[{b_vals.min()}, {b_vals.max()}], std={b_vals.std():.2f}")
        print(f"  e: unique={len(np.unique(e_vals))}, range=[{e_vals.min()}, {e_vals.max()}], std={e_vals.std():.2f}")
        print()

def visualize_distributions(results):
    """Create visualization of (b,e) distributions for different bins"""
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    print()

    fig, axes = plt.subplots(2, len(results), figsize=(18, 8))

    for idx, (bins, data) in enumerate(sorted(results.items())):
        # b distribution
        ax_b = axes[0, idx]
        counts_b = Counter(data['b_values'])
        ax_b.bar(counts_b.keys(), counts_b.values(), alpha=0.7, color='blue')
        ax_b.set_title(f'bins={bins}\nb distribution')
        ax_b.set_xlabel('b value')
        ax_b.set_ylabel('Count')
        ax_b.set_xlim(-1, bins)
        ax_b.grid(alpha=0.3)

        # e distribution
        ax_e = axes[1, idx]
        counts_e = Counter(data['e_values'])
        ax_e.bar(counts_e.keys(), counts_e.values(), alpha=0.7, color='red')
        ax_e.set_title(f'bins={bins}\ne distribution')
        ax_e.set_xlabel('e value')
        ax_e.set_ylabel('Count')
        ax_e.set_xlim(-1, bins)
        ax_e.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/encoding_variance_analysis.png', dpi=150)
    print("✓ Saved: results/encoding_variance_analysis.png")
    print()

def analyze_spectral_characteristics():
    """Analyze why encoding produces low variance"""
    print("="*70)
    print("ANALYZING SPECTRAL CHARACTERISTICS")
    print("="*70)
    print()

    data = load_indian_pines()
    H, W, bands = data.shape

    # Sample 100 random spectra
    rng = np.random.RandomState(42)
    sample_coords = [(rng.randint(H), rng.randint(W)) for _ in range(100)]

    print("Analyzing 100 random spectra...")
    print()

    # Collect DFT properties
    dc_components = []
    peak_frequencies = []
    spectral_centroids = []
    phase_concentrations = []

    for i, j in sample_coords:
        spectrum = data[i, j, :]

        # Normalize
        spec_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-10)

        # DFT
        fft = np.fft.fft(spec_norm)
        magnitudes = np.abs(fft[:bands//2])
        phases = np.angle(fft[:bands//2])

        dc_components.append(magnitudes[0])
        peak_frequencies.append(np.argmax(magnitudes))

        freqs = np.arange(len(magnitudes))
        centroid = np.sum(freqs * magnitudes) / (magnitudes.sum() + 1e-10)
        spectral_centroids.append(centroid)

        # Phase concentration
        top_3_phases = phases[np.argsort(magnitudes)[-3:]]
        phase_std = np.std(top_3_phases)
        phase_concentrations.append(phase_std)

    dc_components = np.array(dc_components)
    peak_frequencies = np.array(peak_frequencies)
    spectral_centroids = np.array(spectral_centroids)
    phase_concentrations = np.array(phase_concentrations)

    print(f"DC component (magnitudes[0]):")
    print(f"  Mean: {dc_components.mean():.4f}, Std: {dc_components.std():.4f}")
    print(f"  Range: [{dc_components.min():.4f}, {dc_components.max():.4f}]")
    print()

    print(f"Peak frequencies (argmax of magnitudes):")
    print(f"  Mean: {peak_frequencies.mean():.2f}, Std: {peak_frequencies.std():.2f}")
    print(f"  Range: [{peak_frequencies.min()}, {peak_frequencies.max()}]")
    print(f"  Unique values: {len(np.unique(peak_frequencies))}")
    print()

    print(f"Spectral centroids:")
    print(f"  Mean: {spectral_centroids.mean():.2f}, Std: {spectral_centroids.std():.2f}")
    print(f"  Range: [{spectral_centroids.min():.2f}, {spectral_centroids.max():.2f}]")
    print(f"  Normalized mean: {(spectral_centroids.mean() / bands * 2):.4f}")
    print()

    print(f"Phase concentration (std of top-3 phases):")
    print(f"  Mean: {phase_concentrations.mean():.4f}, Std: {phase_concentrations.std():.4f}")
    print()

    # Check if centroids cluster
    centroid_norm = spectral_centroids / (bands / 2)
    print(f"Centroid clustering analysis:")
    print(f"  Normalized centroids range: [{centroid_norm.min():.4f}, {centroid_norm.max():.4f}]")
    print(f"  This maps to e∈[{int(centroid_norm.min()*24)}, {int(centroid_norm.max()*24)}] for bins=24")
    print()

    return {
        'dc_mean': float(dc_components.mean()),
        'centroid_range': [float(centroid_norm.min()), float(centroid_norm.max())],
        'phase_std': float(phase_concentrations.mean())
    }

def main():
    """Run complete diagnostic analysis"""

    # Test 1: Bins parameter
    bins_results = test_bins_parameter()

    # Test 2: Phase modes
    test_phase_modes()

    # Test 3: Spectral characteristics
    spectral_analysis = analyze_spectral_characteristics()

    # Visualize
    visualize_distributions(bins_results)

    # Summary report
    print("="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print()

    print("Key Findings:")
    print()

    # Find best bins
    best_bins = max(bins_results.items(),
                    key=lambda x: x[1]['b_unique'] + x[1]['e_unique'])

    print(f"1. Best bins parameter: {best_bins[0]}")
    print(f"   - b: {best_bins[1]['b_unique']} unique values (std={best_bins[1]['b_std']:.2f})")
    print(f"   - e: {best_bins[1]['e_unique']} unique values (std={best_bins[1]['e_std']:.2f})")
    print()

    print(f"2. Spectral centroid clustering:")
    print(f"   - Centroids map to narrow range: e∈{spectral_analysis['centroid_range']}")
    print(f"   - This explains low e variance!")
    print()

    print("3. Root cause hypothesis:")
    print("   - Indian Pines spectra have similar shapes (same material classes)")
    print("   - DFT centroids cluster tightly → low e variance")
    print("   - Phase information also clusters → low b variance")
    print()

    print("4. Recommendations:")
    print("   a) Increase bins to 48+ for better resolution")
    print("   b) Consider alternative encoding (peak locations, spectral derivatives)")
    print("   c) Use multi-scale DFT (combine different window sizes)")
    print("   d) Add spectral angle or shape descriptors beyond phase/centroid")
    print()

    # Save summary
    summary = {
        'bins_tested': list(bins_results.keys()),
        'best_bins': int(best_bins[0]),
        'spectral_analysis': spectral_analysis,
        'recommendations': [
            'Increase bins to 48+',
            'Consider alternative encoding methods',
            'Try multi-scale DFT',
            'Add spectral shape descriptors'
        ]
    }

    with open('results/encoding_diagnostic_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Diagnostic complete!")
    print("✓ Results saved to results/encoding_diagnostic_summary.json")
    print()

if __name__ == "__main__":
    main()
