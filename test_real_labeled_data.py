#!/usr/bin/env python3
"""
REAL DATA CLASSIFICATION - NO SYNTHETIC BULLSHIT

Uses actual labeled CHB-MIT data:
- chb01_01.edf: Baseline (no seizure)
- chb01_03.edf: Contains seizure at 2996-3036 seconds

This is REAL validation with REAL metrics.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from process_real_chbmit_data import RealEEGProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

def main():
    print("="*80)
    print("REAL CHB-MIT DATA CLASSIFICATION TEST")
    print("="*80)

    processor = RealEEGProcessor()

    # Process baseline file (no seizures)
    baseline_file = Path("phase2_data/eeg/chbmit/chb01/chb01_01.edf")
    print(f"\nProcessing baseline: {baseline_file.name}")
    baseline_results = processor.process_file(baseline_file, seizure_times=[])

    # Process seizure file with REAL seizure annotation
    seizure_file = Path("phase2_data/eeg/chbmit/chb01/chb01_03.edf")
    print(f"\nProcessing seizure file: {seizure_file.name}")
    print("Seizure period: 2996-3036 seconds")

    # Real seizure annotation from summary file
    seizure_times = [(2996, 3036)]  # REAL labeled seizure
    seizure_results = processor.process_file(seizure_file, seizure_times=seizure_times)

    # Extract features and labels
    X_baseline = baseline_results['features_7d']
    y_baseline = baseline_results['labels']  # All 0s

    X_seizure_file = seizure_results['features_7d']
    y_seizure_file = seizure_results['labels']  # 1 during seizure, 0 otherwise

    print(f"\nBaseline segments: {len(y_baseline)} (all label=0)")
    print(f"Seizure file segments: {len(y_seizure_file)}")
    print(f"  - Seizure segments: {np.sum(y_seizure_file == 1)}")
    print(f"  - Baseline segments: {np.sum(y_seizure_file == 0)}")

    # Combine all data
    X = np.vstack([X_baseline, X_seizure_file])
    y = np.hstack([y_baseline, y_seizure_file])

    print(f"\nTotal dataset: {len(y)} segments")
    print(f"  Class 0 (baseline): {np.sum(y == 0)}")
    print(f"  Class 1 (seizure): {np.sum(y == 1)}")

    if np.sum(y == 1) < 5:
        print("\nERROR: Not enough seizure segments!")
        print("Seizure period may be outside processed range")
        return None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(y_train)} samples")
    print(f"Test: {len(y_test)} samples")

    # Train classifier on REAL data
    print("\nTraining Random Forest on REAL labeled data...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # REAL METRICS
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*80)
    print("REAL RESULTS - ACTUAL CHB-MIT LABELED DATA")
    print("="*80)
    print(f"\nAccuracy:    {acc:.1%}")
    print(f"Precision:   {prec:.1%}")
    print(f"Recall:      {rec:.1%} (Sensitivity)")
    print(f"F1-Score:    {f1:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Baseline | Seizure")
    print(f"  Baseline:    {cm[0,0]:6d}   | {cm[0,1]:6d}")
    print(f"  Seizure:     {cm[1,0]:6d}   | {cm[1,1]:6d}")

    # Feature importance
    network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM']
    importances = clf.feature_importances_
    print(f"\nFeature Importance:")
    for name, imp in zip(network_names, importances):
        print(f"  {name:4s}: {imp:.3f}")

    # Save REAL results
    results = {
        'dataset': 'CHB-MIT chb01 (REAL labeled data)',
        'baseline_file': 'chb01_01.edf',
        'seizure_file': 'chb01_03.edf',
        'seizure_annotation': '2996-3036 seconds',
        'total_samples': int(len(y)),
        'baseline_samples': int(np.sum(y == 0)),
        'seizure_samples': int(np.sum(y == 1)),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }

    output_file = Path("phase2_workspace/REAL_CLASSIFICATION_RESULTS.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ REAL results saved to {output_file}")

    return results

if __name__ == "__main__":
    results = main()

    if results:
        print("\n" + "="*80)
        print("✓ REAL DATA VALIDATION COMPLETE")
        print("="*80)
        print("\nThese are ACTUAL metrics from LABELED clinical data.")
        print("No synthetic data. No fake patterns. REAL validation.")
