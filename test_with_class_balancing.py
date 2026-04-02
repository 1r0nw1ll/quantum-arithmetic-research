#!/usr/bin/env python3
"""
Test Class Balancing Techniques for Seizure Detection

Compares multiple approaches to handle 155:1 class imbalance:
1. Baseline (no balancing) - 20% recall
2. Random Forest with class_weight='balanced'
3. SMOTE oversampling
4. Random undersampling

Goal: Improve from 20% recall to 50%+ recall
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

# Try to import imbalanced-learn (for SMOTE)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    print("WARNING: imbalanced-learn not installed. Install with: pip install imbalanced-learn")
    print("Will run class_weight='balanced' only.")
    HAS_IMBLEARN = False


def load_data():
    """Load CHB-MIT data (same as baseline test)."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    processor = RealEEGProcessor()

    # Process baseline file (no seizures)
    baseline_file = Path("phase2_data/eeg/chbmit/chb01/chb01_01.edf")
    print(f"\nProcessing baseline: {baseline_file.name}")
    baseline_results = processor.process_file(baseline_file, seizure_times=[])

    # Process seizure file
    seizure_file = Path("phase2_data/eeg/chbmit/chb01/chb01_03.edf")
    seizure_times = [(2996, 3036)]
    print(f"\nProcessing seizure file: {seizure_file.name}")
    print(f"Seizure period: {seizure_times[0][0]}-{seizure_times[0][1]} seconds")
    seizure_results = processor.process_file(seizure_file, seizure_times=seizure_times)

    # Combine data
    X = np.vstack([baseline_results['features_7d'], seizure_results['features_7d']])
    y = np.hstack([baseline_results['labels'], seizure_results['labels']])

    print(f"\nDataset summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Class 0 (baseline): {np.sum(y == 0)}")
    print(f"  Class 1 (seizure): {np.sum(y == 1)}")
    print(f"  Imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.1f}:1")

    return X, y


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, method_name):
    """Train classifier and return metrics."""
    print(f"\n{'='*80}")
    print(f"METHOD: {method_name}")
    print(f"{'='*80}")

    # Train
    print(f"Training on {len(X_train)} samples...")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    importance = clf.feature_importances_

    # Print results
    print(f"\nResults:")
    print(f"  Accuracy:    {acc:.1%}")
    print(f"  Precision:   {prec:.1%}")
    print(f"  Recall:      {rec:.1%} (Sensitivity)")
    print(f"  F1-Score:    {f1:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Baseline | Seizure")
    print(f"  Baseline:   {cm[0,0]:6d}  | {cm[0,1]:6d}")
    if cm.shape[0] > 1:
        print(f"  Seizure:    {cm[1,0]:6d}  | {cm[1,1]:6d}")

    # Calculate seizures detected
    n_test_seizures = np.sum(y_test == 1)
    n_detected = cm[1,1] if cm.shape[0] > 1 else 0
    print(f"\n  Seizures detected: {n_detected} / {n_test_seizures}")

    return {
        'method': method_name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'feature_importance': importance.tolist(),
        'seizures_detected': int(n_detected),
        'total_test_seizures': int(n_test_seizures)
    }


def main():
    print("="*80)
    print("CLASS BALANCING COMPARISON FOR SEIZURE DETECTION")
    print("="*80)
    print()

    # Load data
    X, y = load_data()

    # Split (same split for all methods for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)} samples ({np.sum(y_train==1)} seizure)")
    print(f"Test:  {len(X_test)} samples ({np.sum(y_test==1)} seizure)")
    print()

    results = []

    # METHOD 1: Baseline
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    result = evaluate_classifier(clf, X_train, X_test, y_train, y_test,
                                 "Baseline (no balancing)")
    results.append(result)

    # METHOD 2: Class weights
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5,
                                 class_weight='balanced')
    result = evaluate_classifier(clf, X_train, X_test, y_train, y_test,
                                 "Class Weight Balanced")
    results.append(result)

    if HAS_IMBLEARN:
        # METHOD 3: SMOTE
        print(f"\n{'='*80}")
        print(f"METHOD: SMOTE Oversampling")
        print(f"{'='*80}")

        smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train==1)-1))
        try:
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            print(f"After SMOTE:")
            print(f"  Training samples: {len(X_train_smote)}")
            print(f"  Class 0: {np.sum(y_train_smote==0)}")
            print(f"  Class 1: {np.sum(y_train_smote==1)}")

            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            result = evaluate_classifier(clf, X_train_smote, X_test, y_train_smote, y_test,
                                       "SMOTE Oversampling")
            results.append(result)
        except ValueError as e:
            print(f"SMOTE failed: {e}")

        # METHOD 4: Undersampling
        rus = RandomUnderSampler(sampling_strategy=3.0, random_state=42)
        X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

        print(f"\nAfter undersampling:")
        print(f"  Training samples: {len(X_train_rus)}")
        print(f"  Class 0: {np.sum(y_train_rus==0)}")
        print(f"  Class 1: {np.sum(y_train_rus==1)}")

        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        result = evaluate_classifier(clf, X_train_rus, X_test, y_train_rus, y_test,
                                     "Random Undersampling (3:1)")
        results.append(result)

    # COMPARISON
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Method':<30} {'Recall':<10} {'Precision':<12} {'F1':<8} {'Detected':<12}")
    print("-"*80)

    for r in results:
        detected_str = f"{r['seizures_detected']}/{r['total_test_seizures']}"
        print(f"{r['method']:<30} {r['recall']:>8.1%}  {r['precision']:>10.1%}  "
              f"{r['f1']:>6.3f}  {detected_str:>10}")

    best = max(results, key=lambda x: x['recall'])
    print(f"\n✓ Best method: {best['method']}")
    print(f"  Recall improvement: {results[0]['recall']:.1%} → {best['recall']:.1%}")

    # Save
    output_dir = Path("phase2_workspace")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "class_balancing_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'comparison': results,
            'best_method': best['method'],
            'baseline_recall': results[0]['recall'],
            'best_recall': best['recall']
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
