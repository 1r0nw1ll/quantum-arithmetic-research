#!/usr/bin/env python3
"""
QUICK VERSION: EEG Seizure Detection with HI 2.0 - BALANCED DATASET
Loads 3-4 seizure files and balances dataset for faster results.
"""

QA_COMPLIANCE = "empirical_observer — EEG signal is observer input; QA discrete orbit is the classifier state"


import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, classification_report)
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from process_real_chbmit_data import RealEEGProcessor
from qa_harmonicity_v2 import qa_tuple, compute_hi_1_0, compute_hi_2_0

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

HI2_CONFIG = {'w_ang': 0.5, 'w_rad': 0.5, 'w_fam': 0.0}


def extract_hi_features(features_7d, use_hi2=False):
    """Extract HI 1.0 or HI 2.0 features."""
    n_samples = len(features_7d)

    if use_hi2:
        hi_features = np.zeros((n_samples, 10))
        for i in range(n_samples):
            b = int(features_7d[i, 0] * 23) + 1
            e = int(features_7d[i, 1] * 23) + 1
            b, e = max(1, min(24, b)), max(1, min(24, e))

            q = qa_tuple(b, e, modulus=24)
            result = compute_hi_2_0(q, w_ang=HI2_CONFIG['w_ang'],
                                   w_rad=HI2_CONFIG['w_rad'],
                                   w_fam=HI2_CONFIG['w_fam'], modulus=24)

            C, F, G = result['pythagorean_triple']
            hi_features[i] = [result['HI_2.0'], result['H_angular'],
                             result['H_radial'], result['H_family'],
                             C/1000.0, F/1000.0, G/1000.0, result['gcd'],
                             b/24.0, e/24.0]
    else:
        hi_features = np.zeros((n_samples, 4))
        for i in range(n_samples):
            b = int(features_7d[i, 0] * 23) + 1
            e = int(features_7d[i, 1] * 23) + 1
            b, e = max(1, min(24, b)), max(1, min(24, e))

            q = qa_tuple(b, e, modulus=24)
            hi_1_0 = compute_hi_1_0(q, modulus=24)
            feat_norm = np.linalg.norm(features_7d[i])

            hi_features[i] = [hi_1_0, b/24.0, e/24.0, feat_norm]

    return hi_features


def label_segments_binary(n_segments, window_sec, overlap_sec, seizure_times):
    """Binary labeling: seizure (1) vs baseline (0)."""
    step_sec = window_sec - overlap_sec
    labels = np.zeros(n_segments, dtype=int)

    for i in range(n_segments):
        segment_start = i * step_sec
        segment_end = segment_start + window_sec

        for seizure_start, seizure_end in seizure_times:
            if not (segment_end < seizure_start or segment_start > seizure_end):
                labels[i] = 1
                break

    return labels


def balance_dataset(features, labels, method='undersample', max_per_class=500):
    """
    Balance dataset by undersampling and limiting size for speed.

    Args:
        features: Feature array
        labels: Binary labels
        method: 'undersample'
        max_per_class: Maximum samples per class

    Returns:
        Balanced features and labels
    """
    baseline_idx = np.where(labels == 0)[0]
    seizure_idx = np.where(labels == 1)[0]

    n_baseline = len(baseline_idx)
    n_seizure = len(seizure_idx)

    print(f"\nBalancing dataset:")
    print(f"  Before: Baseline={n_baseline}, Seizure={n_seizure}")

    # Undersample to match smaller class, but cap at max_per_class
    target_count = min(n_baseline, n_seizure, max_per_class)

    baseline_idx = np.random.choice(baseline_idx, target_count, replace=False)
    seizure_idx = np.random.choice(seizure_idx, target_count, replace=False)

    # Combine indices
    balanced_idx = np.concatenate([baseline_idx, seizure_idx])
    np.random.shuffle(balanced_idx)

    balanced_features = features[balanced_idx]
    balanced_labels = labels[balanced_idx]

    print(f"  After:  Baseline={np.sum(balanced_labels==0)}, Seizure={np.sum(balanced_labels==1)}")

    return balanced_features, balanced_labels


def main():
    print("="*80)
    print("QUICK BALANCED EEG HI 2.0 EXPERIMENT - 4 SEIZURE FILES")
    print("="*80)

    processor = RealEEGProcessor()

    # Process 4 seizure files for speed (instead of all 7)
    seizure_files = [
        ('chb01', 'chb01_03.edf', [(2996, 3036)]),
        ('chb01', 'chb01_04.edf', [(1467, 1494)]),
        ('chb01', 'chb01_15.edf', [(1732, 1772)]),
        ('chb01', 'chb01_16.edf', [(1015, 1066)]),
    ]

    all_features = []
    all_labels = []
    files_processed = 0

    for subject, filename, seizure_times in seizure_files:
        filepath = Path(f"phase2_data/eeg/chbmit/{subject}/{filename}")

        if not filepath.exists():
            print(f"\nSkipping {filename} (not found)")
            continue

        print(f"\nProcessing: {filename}")

        try:
            results = processor.process_file(filepath, seizure_times=seizure_times)
            labels = label_segments_binary(results['n_segments'], 4.0, 2.0, seizure_times)

            features = results['features_7d']

            # Don't subsample individual files - keep all seizure segments
            all_features.append(features)
            all_labels.append(labels)
            files_processed += 1

            print(f"  Segments: {len(labels)} (Baseline: {np.sum(labels==0)}, Seizure: {np.sum(labels==1)})")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_features) == 0:
        print("\nERROR: No data loaded!")
        return

    # Combine data
    features_7d = np.vstack(all_features)
    labels = np.hstack(all_labels)

    print(f"\n{'='*80}")
    print(f"UNBALANCED Dataset: {len(labels)} samples from {files_processed} files")
    print(f"  Baseline: {np.sum(labels==0)} ({np.sum(labels==0)/len(labels)*100:.1f}%)")
    print(f"  Seizure:  {np.sum(labels==1)} ({np.sum(labels==1)/len(labels)*100:.1f}%)")

    # Balance dataset - cap at 500 per class for statistical power
    features_7d_balanced, labels_balanced = balance_dataset(features_7d, labels,
                                                            method='undersample',
                                                            max_per_class=500)

    print(f"\nBALANCED Dataset: {len(labels_balanced)} samples")
    print(f"  Baseline: {np.sum(labels_balanced==0)} ({np.sum(labels_balanced==0)/len(labels_balanced)*100:.1f}%)")
    print(f"  Seizure:  {np.sum(labels_balanced==1)} ({np.sum(labels_balanced==1)/len(labels_balanced)*100:.1f}%)")

    # Extract HI features
    print("\nExtracting HI 1.0 features...")
    features_hi1 = extract_hi_features(features_7d_balanced, use_hi2=False)

    print("Extracting HI 2.0 features...")
    features_hi2 = extract_hi_features(features_7d_balanced, use_hi2=True)

    # Split data (stratified to maintain balance)
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        features_hi1, labels_balanced, test_size=0.25, random_state=42, stratify=labels_balanced
    )

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        features_hi2, labels_balanced, test_size=0.25, random_state=42, stratify=labels_balanced
    )

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(y1_train)} samples (Baseline: {np.sum(y1_train==0)}, Seizure: {np.sum(y1_train==1)})")
    print(f"  Test:  {len(y1_test)} samples (Baseline: {np.sum(y1_test==0)}, Seizure: {np.sum(y1_test==1)})")

    # Train classifiers
    print("\n" + "="*80)
    print("Training Classifiers")
    print("="*80)

    print("Training HI 1.0 Random Forest...")
    clf_hi1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                     class_weight='balanced')
    clf_hi1.fit(X1_train, y1_train)

    print("Training HI 2.0 Random Forest...")
    clf_hi2 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                     class_weight='balanced')
    clf_hi2.fit(X2_train, y2_train)

    # Predictions
    y1_pred = clf_hi1.predict(X1_test)
    y2_pred = clf_hi2.predict(X2_test)

    # Compute metrics
    results = {
        'HI_1.0': {
            'accuracy': float(accuracy_score(y1_test, y1_pred)),
            'precision': float(precision_score(y1_test, y1_pred, zero_division=0)),
            'recall': float(recall_score(y1_test, y1_pred, zero_division=0)),
            'f1': float(f1_score(y1_test, y1_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y1_test, y1_pred).tolist(),
            'feature_importance': clf_hi1.feature_importances_.tolist(),
            'classification_report': classification_report(y1_test, y1_pred,
                                                          target_names=['Baseline', 'Seizure'],
                                                          output_dict=True)
        },
        'HI_2.0': {
            'accuracy': float(accuracy_score(y2_test, y2_pred)),
            'precision': float(precision_score(y2_test, y2_pred, zero_division=0)),
            'recall': float(recall_score(y2_test, y2_pred, zero_division=0)),
            'f1': float(f1_score(y2_test, y2_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y2_test, y2_pred).tolist(),
            'feature_importance': clf_hi2.feature_importances_.tolist(),
            'classification_report': classification_report(y2_test, y2_pred,
                                                          target_names=['Baseline', 'Seizure'],
                                                          output_dict=True)
        }
    }

    # Statistical comparison
    y1_proba = clf_hi1.predict_proba(X1_test)[:, 1]
    y2_proba = clf_hi2.predict_proba(X2_test)[:, 1]

    t_stat, p_value = stats.ttest_rel(y1_proba, y2_proba)

    results['comparison'] = {
        'accuracy_delta': results['HI_2.0']['accuracy'] - results['HI_1.0']['accuracy'],
        'f1_delta': results['HI_2.0']['f1'] - results['HI_1.0']['f1'],
        'precision_delta': results['HI_2.0']['precision'] - results['HI_1.0']['precision'],
        'recall_delta': results['HI_2.0']['recall'] - results['HI_1.0']['recall'],
        'paired_t_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    }

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS - BALANCED DATASET")
    print(f"{'='*80}")
    print(f"\n{'Metric':<15} {'HI 1.0':>12} {'HI 2.0':>12} {'Delta':>12} {'Significance':>12}")
    print("-" * 68)
    print(f"{'Accuracy':<15} {results['HI_1.0']['accuracy']:>11.1%} "
          f"{results['HI_2.0']['accuracy']:>11.1%} "
          f"{results['comparison']['accuracy_delta']:>+11.1%} "
          f"{'p='+str(round(p_value, 4)):>12}")
    print(f"{'Precision':<15} {results['HI_1.0']['precision']:>11.1%} "
          f"{results['HI_2.0']['precision']:>11.1%} "
          f"{results['comparison']['precision_delta']:>+11.1%}")
    print(f"{'Recall':<15} {results['HI_1.0']['recall']:>11.1%} "
          f"{results['HI_2.0']['recall']:>11.1%} "
          f"{results['comparison']['recall_delta']:>+11.1%}")
    print(f"{'F1 Score':<15} {results['HI_1.0']['f1']:>12.3f} "
          f"{results['HI_2.0']['f1']:>12.3f} "
          f"{results['comparison']['f1_delta']:>+12.3f}")

    if results['comparison']['paired_t_test']['significant']:
        print(f"\n*** STATISTICALLY SIGNIFICANT difference detected (p < 0.05) ***")
    else:
        print(f"\nNo statistically significant difference (p >= 0.05)")

    # Print detailed classification reports
    print(f"\n{'='*80}")
    print("HI 1.0 Classification Report:")
    print(classification_report(y1_test, y1_pred, target_names=['Baseline', 'Seizure']))

    print(f"\n{'='*80}")
    print("HI 2.0 Classification Report:")
    print(classification_report(y2_test, y2_pred, target_names=['Baseline', 'Seizure']))

    # Save results
    results['metadata'] = {
        'dataset': f'CHB-MIT chb01 ({files_processed} seizure files - QUICK VERSION)',
        'config': HI2_CONFIG,
        'samples_unbalanced': int(len(labels)),
        'samples_balanced': int(len(labels_balanced)),
        'baseline_count': int(np.sum(labels_balanced==0)),
        'seizure_count': int(np.sum(labels_balanced==1)),
        'balance_method': 'undersample',
        'max_per_class': 500,
        'test_size': 0.25
    }

    with open("eeg_hi2_0_balanced_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: eeg_hi2_0_balanced_results.json")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Confusion matrices
    for idx, (name, res) in enumerate([('HI 1.0', results['HI_1.0']), ('HI 2.0', results['HI_2.0'])]):
        ax = axes[0, idx]
        cm = np.array(res['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Baseline', 'Seizure'],
                   yticklabels=['Baseline', 'Seizure'])
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.1%}, F1: {res['f1']:.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    # Metrics comparison
    ax = axes[0, 2]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    hi1_vals = [results['HI_1.0']['accuracy'], results['HI_1.0']['precision'],
                results['HI_1.0']['recall'], results['HI_1.0']['f1']]
    hi2_vals = [results['HI_2.0']['accuracy'], results['HI_2.0']['precision'],
                results['HI_2.0']['recall'], results['HI_2.0']['f1']]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, hi1_vals, width, label='HI 1.0', color='steelblue')
    ax.bar(x + width/2, hi2_vals, width, label='HI 2.0', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # Feature importance - HI 1.0
    ax = axes[1, 0]
    fi1 = results['HI_1.0']['feature_importance']
    feature_names_1 = ['HI', 'b', 'e', 'norm']
    ax.barh(feature_names_1, fi1, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('HI 1.0 Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')

    # Feature importance - HI 2.0
    ax = axes[1, 1]
    fi2 = results['HI_2.0']['feature_importance']
    feature_names_2 = ['HI_2.0', 'H_ang', 'H_rad', 'H_fam', 'C', 'F', 'G', 'gcd', 'b', 'e']
    ax.barh(feature_names_2, fi2, color='seagreen')
    ax.set_xlabel('Importance')
    ax.set_title('HI 2.0 Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')

    # Summary text
    ax = axes[1, 2]
    ax.axis('off')

    sig_marker = "***" if results['comparison']['paired_t_test']['significant'] else ""

    summary = f"""BALANCED EEG SEIZURE DETECTION
CHB-MIT Dataset (chb01, {files_processed} files)

Configuration:
  HI 2.0: Angular_Radial
  w_ang = {HI2_CONFIG['w_ang']:.1f}
  w_rad = {HI2_CONFIG['w_rad']:.1f}
  w_fam = {HI2_CONFIG['w_fam']:.1f}

Dataset:
  Unbalanced: {results['metadata']['samples_unbalanced']} samples
  Balanced:   {results['metadata']['samples_balanced']} samples
  Baseline:   {results['metadata']['baseline_count']}
  Seizure:    {results['metadata']['seizure_count']}
  Method:     {results['metadata']['balance_method']}
  Max/class:  {results['metadata']['max_per_class']}

Test Set Performance:
  HI 1.0:
    Accuracy:  {results['HI_1.0']['accuracy']:6.1%}
    Precision: {results['HI_1.0']['precision']:6.1%}
    Recall:    {results['HI_1.0']['recall']:6.1%}
    F1 Score:  {results['HI_1.0']['f1']:.3f}

  HI 2.0:
    Accuracy:  {results['HI_2.0']['accuracy']:6.1%}
    Precision: {results['HI_2.0']['precision']:6.1%}
    Recall:    {results['HI_2.0']['recall']:6.1%}
    F1 Score:  {results['HI_2.0']['f1']:.3f}

Improvement:
  Accuracy:  {results['comparison']['accuracy_delta']:+.1%}
  F1 Score:  {results['comparison']['f1_delta']:+.3f}

Statistical Test:
  Paired t-test: p = {p_value:.4f} {sig_marker}

Classifier: Random Forest
  n_estimators = 100
  max_depth = 8
  class_weight = balanced
"""

    ax.text(0.05, 0.5, summary, fontsize=8, family='monospace',
           verticalalignment='center', transform=ax.transAxes)

    plt.suptitle('Balanced EEG Seizure Detection: HI 1.0 vs HI 2.0 (Quick Version)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig("eeg_hi2_0_balanced_results_visualization.png", dpi=300, bbox_inches='tight')
    print("Visualization saved: eeg_hi2_0_balanced_results_visualization.png")
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"\nKey Findings:")
    print(f"  - Processed {files_processed} seizure files with balanced dataset")
    print(f"  - HI 2.0 F1 Score: {results['HI_2.0']['f1']:.3f}")
    print(f"  - HI 1.0 F1 Score: {results['HI_1.0']['f1']:.3f}")
    print(f"  - Improvement: {results['comparison']['f1_delta']:+.3f}")
    if results['comparison']['paired_t_test']['significant']:
        print(f"  - Statistical significance: YES (p={p_value:.4f})")
    else:
        print(f"  - Statistical significance: NO (p={p_value:.4f})")


if __name__ == "__main__":
    main()
