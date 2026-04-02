#!/usr/bin/env python3
"""
Debug script to test QA pipeline on real hyperspectral data
"""

import numpy as np
from load_hyperspectral_dataset import load_dataset
from qa_hyperspectral_pipeline import spectrum_to_be_phase_multi

# Load Indian Pines data
print("Loading Indian Pines data...")
data, gt = load_dataset('Indian_pines_corrected')

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Data range: [{data.min()}, {data.max()}]")

# Test on a few spectra
print("\nTesting spectrum_to_be_phase_multi on real data...")

# Get a few sample spectra
sample_spectra = []
for i in range(min(5, data.shape[0])):
    for j in range(min(5, data.shape[1])):
        if gt[i, j] > 0:  # Only labeled pixels
            spectrum = data[i, j, :]
            sample_spectra.append((i, j, spectrum))
            if len(sample_spectra) >= 3:
                break
    if len(sample_spectra) >= 3:
        break

print(f"Testing on {len(sample_spectra)} sample spectra...")

for i, (row, col, spectrum) in enumerate(sample_spectra):
    print(f"\nSpectrum {i+1} at ({row}, {col}):")
    print(f"  Shape: {spectrum.shape}")
    print(f"  Range: [{spectrum.min():.2f}, {spectrum.max():.2f}]")
    print(f"  Mean: {spectrum.mean():.2f}")
    print(f"  Std: {spectrum.std():.2f}")

    try:
        b, e = spectrum_to_be_phase_multi(spectrum, bins=24, use_derivative=True, derivative_order=2)
        print(f"  QA encoding (derivative order 2): b={b}, e={e}")
        b1, e1 = spectrum_to_be_phase_multi(spectrum, bins=24, use_derivative=True, derivative_order=1)
        print(f"  QA encoding (derivative order 1): b={b1}, e={e1}")
        b0, e0 = spectrum_to_be_phase_multi(spectrum, bins=24, use_derivative=False)
        print(f"  QA encoding (no derivative): b={b0}, e={e0}")
    except Exception as ex:
        print(f"  ERROR: {ex}")

print("\nDebug complete.")