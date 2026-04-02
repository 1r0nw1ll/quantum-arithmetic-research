#!/usr/bin/env python3
"""Debug feature extraction - check if features differ between baseline and seizure"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from process_real_chbmit_data import RealEEGProcessor
import matplotlib.pyplot as plt

processor = RealEEGProcessor()

# Load baseline
print("Loading baseline file...")
baseline_file = Path("phase2_data/eeg/chbmit/chb01/chb01_01.edf")
baseline_results = processor.process_file(baseline_file, seizure_times=[])

# Load seizure file
print("Loading seizure file...")
seizure_file = Path("phase2_data/eeg/chbmit/chb01/chb01_03.edf")
seizure_times = [(2996, 3036)]
seizure_results = processor.process_file(seizure_file, seizure_times=seizure_times)

# Extract features
X_baseline = baseline_results['features_7d']
y_baseline = baseline_results['labels']

X_seizure_file = seizure_results['features_7d']
y_seizure_file = seizure_results['labels']

# Separate seizure vs baseline from seizure file
seizure_mask = y_seizure_file == 1
baseline_from_sz_file = X_seizure_file[~seizure_mask]
actual_seizure = X_seizure_file[seizure_mask]

print("\n" + "="*80)
print("FEATURE STATISTICS")
print("="*80)

print(f"\nBaseline samples (chb01_01): {len(X_baseline)}")
print(f"Seizure samples (chb01_03): {np.sum(seizure_mask)}")
print(f"Baseline from seizure file: {np.sum(~seizure_mask)}")

network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM']

print("\n" + "="*80)
print("MEAN FEATURE VALUES")
print("="*80)
print(f"{'Network':<10} {'Baseline':<12} {'Seizure':<12} {'Difference':<12}")
print("-"*80)

for i, name in enumerate(network_names):
    baseline_mean = np.mean(X_baseline[:, i])
    seizure_mean = np.mean(actual_seizure[:, i]) if len(actual_seizure) > 0 else 0
    diff = abs(seizure_mean - baseline_mean)
    print(f"{name:<10} {baseline_mean:>11.6f} {seizure_mean:>11.6f} {diff:>11.6f}")

print("\n" + "="*80)
print("STANDARD DEVIATION")
print("="*80)
print(f"{'Network':<10} {'Baseline':<12} {'Seizure':<12}")
print("-"*80)

for i, name in enumerate(network_names):
    baseline_std = np.std(X_baseline[:, i])
    seizure_std = np.std(actual_seizure[:, i]) if len(actual_seizure) > 0 else 0
    print(f"{name:<10} {baseline_std:>11.6f} {seizure_std:>11.6f}")

print("\n" + "="*80)
print("FEATURE RANGE")
print("="*80)
print(f"{'Network':<10} {'Min':<12} {'Max':<12} {'Range':<12}")
print("-"*80)

all_features = np.vstack([X_baseline, X_seizure_file])
for i, name in enumerate(network_names):
    feat_min = np.min(all_features[:, i])
    feat_max = np.max(all_features[:, i])
    feat_range = feat_max - feat_min
    print(f"{name:<10} {feat_min:>11.6f} {feat_max:>11.6f} {feat_range:>11.6f}")

print("\n" + "="*80)
print("ARE FEATURES ALL THE SAME?")
print("="*80)

for i, name in enumerate(network_names):
    unique_vals = len(np.unique(all_features[:, i]))
    print(f"{name:<10} Unique values: {unique_vals}")

print("\n" + "="*80)
print("SAMPLE SEIZURE VS BASELINE FEATURES")
print("="*80)

if len(actual_seizure) > 0:
    print("\nFirst seizure segment:")
    print(actual_seizure[0])
    print("\nFirst baseline segment:")
    print(X_baseline[0])
    print("\nDifference:")
    print(actual_seizure[0] - X_baseline[0])

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if features are meaningful
all_features = np.vstack([X_baseline, X_seizure_file])
feature_variance = np.var(all_features, axis=0)

print("\nFeature variance across all samples:")
for i, name in enumerate(network_names):
    print(f"  {name}: {feature_variance[i]:.8f}")

if np.all(feature_variance < 1e-6):
    print("\n❌ PROBLEM: All features have near-zero variance!")
    print("   Features are essentially identical across all samples.")
    print("   This means feature extraction is NOT capturing signal differences.")
elif np.max(feature_variance) < 1e-4:
    print("\n⚠️  WARNING: Features have very low variance!")
    print("   Features may not be discriminative enough.")
else:
    print("\n✓ Features have reasonable variance.")

# Check if seizure features differ from baseline
if len(actual_seizure) > 0:
    mean_diff = np.mean(np.abs(np.mean(actual_seizure, axis=0) - np.mean(X_baseline, axis=0)))
    print(f"\nMean absolute difference baseline vs seizure: {mean_diff:.6f}")

    if mean_diff < 0.001:
        print("❌ PROBLEM: Seizure and baseline features are nearly identical!")
        print("   Feature extraction is not capturing seizure dynamics.")
    else:
        print("✓ There is some difference between seizure and baseline features.")
