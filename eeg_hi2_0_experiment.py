#!/usr/bin/env python3
"""
EEG Seizure Detection with Harmonicity Index 2.0 (Angular_Radial Configuration)

Implements Phase 2 EEG classification using HI 2.0 to compare with HI 1.0 baseline.

Multi-class classification:
- 0: Baseline (interictal - normal brain activity)
- 1: Pre-ictal (5 minutes before seizure onset)
- 2: Ictal (during seizure)
- 3: Post-ictal (5 minutes after seizure end)

Uses CHB-MIT dataset with real seizure annotations.
"""

QA_COMPLIANCE = "empirical_observer — EEG signal is observer input; QA discrete orbit is the classifier state"


import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
from scipy import stats

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from process_real_chbmit_data import RealEEGProcessor
from qa_harmonicity_v2 import (qa_tuple, compute_hi_1_0, compute_hi_2_0,
                               pythagorean_triple, compute_angular_harmonicity,
                               compute_radial_harmonicity, compute_family_harmonicity)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Recommended HI 2.0 configuration for EEG
HI2_CONFIG = {
    'w_ang': 0.5,   # Angular harmonicity (Pisano alignment)
    'w_rad': 0.5,   # Radial harmonicity (primitivity)
    'w_fam': 0.0    # Family harmonicity (disabled for EEG)
}


class EEGHarmonyClassifier:
    """EEG seizure classification using QA + Harmonicity Index."""

    def __init__(self, data_dir: Path = Path("phase2_data/eeg/chbmit")):
        self.processor = RealEEGProcessor(data_dir)
        self.data_dir = data_dir

    def extract_hi_features(self, features_7d: np.ndarray,
                           use_hi2: bool = False) -> np.ndarray:
        """
        Extract HI 1.0 or HI 2.0 features from 7D brain network features.

        Args:
            features_7d: Array of 7D brain network features (n_samples, 7)
            use_hi2: If True, use HI 2.0; otherwise use HI 1.0

        Returns:
            HI feature vector (n_samples, n_features)
        """
        n_samples = len(features_7d)

        if use_hi2:
            # HI 2.0: Returns 10D features
            # [HI_2.0, H_angular, H_radial, H_family, C, F, G, gcd, b, e]
            hi_features = np.zeros((n_samples, 10))

            for i in range(n_samples):
                # Map to QA state using first two networks (VIS, SMN)
                b = int(features_7d[i, 0] * 23) + 1
                e = int(features_7d[i, 1] * 23) + 1
                b = max(1, min(24, b))
                e = max(1, min(24, e))

                # Generate QA tuple
                q = qa_tuple(b, e, modulus=24)

                # Compute HI 2.0 with Angular_Radial config
                result = compute_hi_2_0(q,
                                       w_ang=HI2_CONFIG['w_ang'],
                                       w_rad=HI2_CONFIG['w_rad'],
                                       w_fam=HI2_CONFIG['w_fam'],
                                       modulus=24)

                C, F, G = result['pythagorean_triple']

                hi_features[i] = [
                    result['HI_2.0'],
                    result['H_angular'],
                    result['H_radial'],
                    result['H_family'],
                    C / 1000.0,  # Normalize large values
                    F / 1000.0,
                    G / 1000.0,
                    result['gcd'],
                    b / 24.0,
                    e / 24.0
                ]
        else:
            # HI 1.0: Returns 4D features
            # [HI_1.0, b, e, norm]
            hi_features = np.zeros((n_samples, 4))

            for i in range(n_samples):
                # Map to QA state
                b = int(features_7d[i, 0] * 23) + 1
                e = int(features_7d[i, 1] * 23) + 1
                b = max(1, min(24, b))
                e = max(1, min(24, e))

                # Generate QA tuple
                q = qa_tuple(b, e, modulus=24)

                # Compute HI 1.0 (E8-only)
                hi_1_0 = compute_hi_1_0(q, modulus=24)

                # Compute norm of brain features as auxiliary feature
                feat_norm = np.linalg.norm(features_7d[i])

                hi_features[i] = [hi_1_0, b / 24.0, e / 24.0, feat_norm]

        return hi_features

    def label_segments_multiclass(self, n_segments: int, window_sec: float,
                                  overlap_sec: float,
                                  seizure_times: List[Tuple[float, float]]) -> np.ndarray:
        """
        Label segments into 4 classes: baseline, pre-ictal, ictal, post-ictal.

        Classes:
        - 0: Baseline (interictal - more than 5 min from any seizure)
        - 1: Pre-ictal (0-5 minutes before seizure)
        - 2: Ictal (during seizure)
        - 3: Post-ictal (0-5 minutes after seizure)

        Args:
            n_segments: Number of time segments
            window_sec: Window length in seconds
            overlap_sec: Overlap in seconds
            seizure_times: List of (start, end) tuples in seconds

        Returns:
            Multi-class labels (n_segments,)
        """
        step_sec = window_sec - overlap_sec
        labels = np.zeros(n_segments, dtype=int)

        pre_ictal_window = 300  # 5 minutes before seizure
        post_ictal_window = 300  # 5 minutes after seizure

        for i in range(n_segments):
            segment_start = i * step_sec
            segment_end = segment_start + window_sec
            segment_center = (segment_start + segment_end) / 2

            # Check each seizure
            for seizure_start, seizure_end in seizure_times:
                # Ictal (during seizure)
                if segment_center >= seizure_start and segment_center <= seizure_end:
                    labels[i] = 2
                    break

                # Pre-ictal (before seizure)
                elif (segment_center >= (seizure_start - pre_ictal_window) and
                      segment_center < seizure_start):
                    labels[i] = 1
                    break

                # Post-ictal (after seizure)
                elif (segment_center > seizure_end and
                      segment_center <= (seizure_end + post_ictal_window)):
                    labels[i] = 3
                    break

            # If no match, remains baseline (0)

        return labels

    def load_patient_data(self, subject: str, max_files: int = 6) -> Dict:
        """
        Load and process multiple EDF files for one patient.

        Args:
            subject: Patient ID (e.g., 'chb01')
            max_files: Maximum number of files to process

        Returns:
            Dictionary with features, labels, metadata
        """
        print(f"\n{'='*80}")
        print(f"Loading patient data: {subject}")
        print(f"{'='*80}")

        # Find EDF files for this subject
        subject_dir = self.data_dir / subject
        if not subject_dir.exists():
            raise ValueError(f"Subject directory not found: {subject_dir}")

        edf_files = sorted(list(subject_dir.glob("*.edf")))

        # Load seizure annotations
        annotations = self.processor.load_seizure_annotations(subject)

        # Build file -> seizure times mapping
        file_seizures = {}
        for ann in annotations:
            filename = ann['file']
            if filename not in file_seizures:
                file_seizures[filename] = []
            file_seizures[filename].append((ann['start_time'], ann['end_time']))

        print(f"Found {len(edf_files)} EDF files")
        print(f"Found {len(annotations)} seizure annotations")

        # Process files
        all_features_7d = []
        all_labels = []
        all_filenames = []

        processed_count = 0

        for edf_file in edf_files:
            if processed_count >= max_files:
                break

            filename = edf_file.name

            # Get seizure times for this file
            seizure_times = file_seizures.get(filename, [])

            # Skip files without seizures for balanced dataset
            # (we'll use some non-seizure files too)
            has_seizure = len(seizure_times) > 0

            print(f"\nProcessing: {filename} {'(HAS SEIZURE)' if has_seizure else ''}")

            try:
                # Process file
                results = self.processor.process_file(edf_file, seizure_times=seizure_times)

                # Multi-class labeling
                labels = self.label_segments_multiclass(
                    results['n_segments'],
                    window_sec=4.0,
                    overlap_sec=2.0,
                    seizure_times=seizure_times
                )

                print(f"  Segments: {len(labels)}")
                print(f"  Baseline: {np.sum(labels == 0)}, Pre-ictal: {np.sum(labels == 1)}, "
                      f"Ictal: {np.sum(labels == 2)}, Post-ictal: {np.sum(labels == 3)}")

                all_features_7d.append(results['features_7d'])
                all_labels.append(labels)
                all_filenames.extend([filename] * len(labels))

                processed_count += 1

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        if len(all_features_7d) == 0:
            raise ValueError("No files successfully processed!")

        # Concatenate all data
        features_7d = np.vstack(all_features_7d)
        labels = np.hstack(all_labels)

        print(f"\n{'='*80}")
        print(f"Dataset Summary: {subject}")
        print(f"{'='*80}")
        print(f"Total segments: {len(labels)}")
        print(f"  Baseline:   {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
        print(f"  Pre-ictal:  {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
        print(f"  Ictal:      {np.sum(labels == 2)} ({np.sum(labels == 2)/len(labels)*100:.1f}%)")
        print(f"  Post-ictal: {np.sum(labels == 3)} ({np.sum(labels == 3)/len(labels)*100:.1f}%)")

        return {
            'features_7d': features_7d,
            'labels': labels,
            'filenames': all_filenames,
            'subject': subject
        }

    def run_comparison_experiment(self, subjects: List[str] = ['chb01']) -> Dict:
        """
        Run HI 1.0 vs HI 2.0 comparison experiment.

        Args:
            subjects: List of patient IDs to process

        Returns:
            Results dictionary with metrics for both methods
        """
        print(f"\n{'='*80}")
        print("EEG SEIZURE DETECTION: HI 1.0 vs HI 2.0 (Angular_Radial)")
        print(f"{'='*80}")

        # Load data from all subjects
        all_features_7d = []
        all_labels = []

        for subject in subjects:
            try:
                data = self.load_patient_data(subject, max_files=6)
                all_features_7d.append(data['features_7d'])
                all_labels.append(data['labels'])
            except Exception as e:
                print(f"ERROR loading {subject}: {e}")
                continue

        if len(all_features_7d) == 0:
            raise ValueError("No data loaded!")

        # Combine all subjects
        features_7d = np.vstack(all_features_7d)
        labels = np.hstack(all_labels)

        print(f"\n{'='*80}")
        print("Combined Dataset")
        print(f"{'='*80}")
        print(f"Total samples: {len(labels)}")
        print(f"Feature shape: {features_7d.shape}")

        # Class distribution
        class_names = ['Baseline', 'Pre-ictal', 'Ictal', 'Post-ictal']
        for i, name in enumerate(class_names):
            count = np.sum(labels == i)
            print(f"  {name:12s}: {count:6d} ({count/len(labels)*100:5.1f}%)")

        # Extract HI features
        print("\nExtracting HI 1.0 features...")
        features_hi1 = self.extract_hi_features(features_7d, use_hi2=False)

        print("Extracting HI 2.0 features (Angular_Radial config)...")
        features_hi2 = self.extract_hi_features(features_7d, use_hi2=True)

        print(f"  HI 1.0 features: {features_hi1.shape}")
        print(f"  HI 2.0 features: {features_hi2.shape}")

        # Train/test split (stratified)
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            features_hi1, labels, test_size=0.25, random_state=42, stratify=labels
        )

        X2_train, X2_test, y2_train, y2_test = train_test_split(
            features_hi2, labels, test_size=0.25, random_state=42, stratify=labels
        )

        print(f"\nTrain set: {len(y1_train)} samples")
        print(f"Test set:  {len(y1_test)} samples")

        # Train classifiers
        print("\n" + "="*80)
        print("Training Classifiers")
        print("="*80)

        print("\n[1/2] Training HI 1.0 classifier...")
        clf_hi1 = RandomForestClassifier(n_estimators=200, max_depth=10,
                                         random_state=42, class_weight='balanced')
        clf_hi1.fit(X1_train, y1_train)

        print("[2/2] Training HI 2.0 classifier...")
        clf_hi2 = RandomForestClassifier(n_estimators=200, max_depth=10,
                                         random_state=42, class_weight='balanced')
        clf_hi2.fit(X2_train, y2_train)

        # Predictions
        y1_pred = clf_hi1.predict(X1_test)
        y2_pred = clf_hi2.predict(X2_test)

        # Compute metrics
        results = self.compute_metrics(y1_test, y1_pred, y2_test, y2_pred,
                                       clf_hi1, clf_hi2, features_hi1, features_hi2)

        # Visualize
        self.visualize_results(y1_test, y1_pred, y2_pred, results,
                              clf_hi1, clf_hi2, features_hi1, features_hi2)

        # Save results
        self.save_results(results)

        return results

    def compute_metrics(self, y_true, y_pred_hi1, y_true2, y_pred_hi2,
                       clf_hi1, clf_hi2, features_hi1, features_hi2) -> Dict:
        """Compute comprehensive metrics for both methods."""

        results = {
            'HI_1.0': {},
            'HI_2.0': {},
            'comparison': {}
        }

        # HI 1.0 metrics
        results['HI_1.0']['accuracy'] = float(accuracy_score(y_true, y_pred_hi1))
        results['HI_1.0']['precision_macro'] = float(precision_score(y_true, y_pred_hi1,
                                                                     average='macro', zero_division=0))
        results['HI_1.0']['recall_macro'] = float(recall_score(y_true, y_pred_hi1,
                                                               average='macro', zero_division=0))
        results['HI_1.0']['f1_macro'] = float(f1_score(y_true, y_pred_hi1,
                                                       average='macro', zero_division=0))
        results['HI_1.0']['f1_weighted'] = float(f1_score(y_true, y_pred_hi1,
                                                          average='weighted', zero_division=0))

        # Per-class metrics
        results['HI_1.0']['per_class'] = {}
        class_names = ['Baseline', 'Pre-ictal', 'Ictal', 'Post-ictal']
        for i, name in enumerate(class_names):
            mask = (y_true == i)
            if np.sum(mask) > 0:
                results['HI_1.0']['per_class'][name] = {
                    'precision': float(precision_score(y_true == i, y_pred_hi1 == i, zero_division=0)),
                    'recall': float(recall_score(y_true == i, y_pred_hi1 == i, zero_division=0)),
                    'f1': float(f1_score(y_true == i, y_pred_hi1 == i, zero_division=0))
                }

        # HI 2.0 metrics
        results['HI_2.0']['accuracy'] = float(accuracy_score(y_true2, y_pred_hi2))
        results['HI_2.0']['precision_macro'] = float(precision_score(y_true2, y_pred_hi2,
                                                                     average='macro', zero_division=0))
        results['HI_2.0']['recall_macro'] = float(recall_score(y_true2, y_pred_hi2,
                                                               average='macro', zero_division=0))
        results['HI_2.0']['f1_macro'] = float(f1_score(y_true2, y_pred_hi2,
                                                       average='macro', zero_division=0))
        results['HI_2.0']['f1_weighted'] = float(f1_score(y_true2, y_pred_hi2,
                                                          average='weighted', zero_division=0))

        # Per-class metrics
        results['HI_2.0']['per_class'] = {}
        for i, name in enumerate(class_names):
            mask = (y_true2 == i)
            if np.sum(mask) > 0:
                results['HI_2.0']['per_class'][name] = {
                    'precision': float(precision_score(y_true2 == i, y_pred_hi2 == i, zero_division=0)),
                    'recall': float(recall_score(y_true2 == i, y_pred_hi2 == i, zero_division=0)),
                    'f1': float(f1_score(y_true2 == i, y_pred_hi2 == i, zero_division=0))
                }

        # Confusion matrices
        results['HI_1.0']['confusion_matrix'] = confusion_matrix(y_true, y_pred_hi1).tolist()
        results['HI_2.0']['confusion_matrix'] = confusion_matrix(y_true2, y_pred_hi2).tolist()

        # Feature importance
        results['HI_1.0']['feature_importance'] = clf_hi1.feature_importances_.tolist()
        results['HI_2.0']['feature_importance'] = clf_hi2.feature_importances_.tolist()

        # Comparison
        results['comparison']['accuracy_delta'] = results['HI_2.0']['accuracy'] - results['HI_1.0']['accuracy']
        results['comparison']['f1_macro_delta'] = results['HI_2.0']['f1_macro'] - results['HI_1.0']['f1_macro']
        results['comparison']['f1_weighted_delta'] = results['HI_2.0']['f1_weighted'] - results['HI_1.0']['f1_weighted']

        # Statistical significance test (paired t-test on per-class F1 scores)
        hi1_f1s = [results['HI_1.0']['per_class'][name]['f1'] for name in class_names
                   if name in results['HI_1.0']['per_class']]
        hi2_f1s = [results['HI_2.0']['per_class'][name]['f1'] for name in class_names
                   if name in results['HI_2.0']['per_class']]

        if len(hi1_f1s) > 1 and len(hi2_f1s) > 1:
            t_stat, p_value = stats.ttest_rel(hi2_f1s, hi1_f1s)
            results['comparison']['ttest_statistic'] = float(t_stat)
            results['comparison']['ttest_pvalue'] = float(p_value)

        # Print summary
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Metric':<20} {'HI 1.0':>12} {'HI 2.0':>12} {'Delta':>12}")
        print("-" * 60)
        print(f"{'Accuracy':<20} {results['HI_1.0']['accuracy']:>11.1%} "
              f"{results['HI_2.0']['accuracy']:>11.1%} "
              f"{results['comparison']['accuracy_delta']:>+11.1%}")
        print(f"{'F1 (macro)':<20} {results['HI_1.0']['f1_macro']:>12.3f} "
              f"{results['HI_2.0']['f1_macro']:>12.3f} "
              f"{results['comparison']['f1_macro_delta']:>+12.3f}")
        print(f"{'F1 (weighted)':<20} {results['HI_1.0']['f1_weighted']:>12.3f} "
              f"{results['HI_2.0']['f1_weighted']:>12.3f} "
              f"{results['comparison']['f1_weighted_delta']:>+12.3f}")

        print(f"\n{'Per-Class F1 Scores':}")
        print(f"{'Class':<15} {'HI 1.0':>12} {'HI 2.0':>12} {'Delta':>12}")
        print("-" * 54)
        for name in class_names:
            if name in results['HI_1.0']['per_class'] and name in results['HI_2.0']['per_class']:
                f1_hi1 = results['HI_1.0']['per_class'][name]['f1']
                f1_hi2 = results['HI_2.0']['per_class'][name]['f1']
                print(f"{name:<15} {f1_hi1:>12.3f} {f1_hi2:>12.3f} {f1_hi2-f1_hi1:>+12.3f}")

        if 'ttest_pvalue' in results['comparison']:
            print(f"\nStatistical Significance (paired t-test):")
            print(f"  t-statistic: {results['comparison']['ttest_statistic']:.3f}")
            print(f"  p-value:     {results['comparison']['ttest_pvalue']:.4f}")
            if results['comparison']['ttest_pvalue'] < 0.05:
                print(f"  *** SIGNIFICANT at p < 0.05 ***")
            else:
                print(f"  (not significant at p < 0.05)")

        return results

    def visualize_results(self, y_true, y_pred_hi1, y_pred_hi2, results,
                         clf_hi1, clf_hi2, features_hi1, features_hi2):
        """Create comprehensive visualization comparing HI 1.0 and HI 2.0."""

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        class_names = ['Baseline', 'Pre-ictal', 'Ictal', 'Post-ictal']

        # Panel A: Confusion Matrix - HI 1.0
        ax1 = fig.add_subplot(gs[0, 0])
        cm1 = np.array(results['HI_1.0']['confusion_matrix'])
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['B', 'Pre', 'I', 'Post'],
                   yticklabels=['B', 'Pre', 'I', 'Post'])
        ax1.set_title(f"HI 1.0 Confusion Matrix\nAcc: {results['HI_1.0']['accuracy']:.1%}")
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # Panel B: Confusion Matrix - HI 2.0
        ax2 = fig.add_subplot(gs[0, 1])
        cm2 = np.array(results['HI_2.0']['confusion_matrix'])
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax2,
                   xticklabels=['B', 'Pre', 'I', 'Post'],
                   yticklabels=['B', 'Pre', 'I', 'Post'])
        ax2.set_title(f"HI 2.0 Confusion Matrix\nAcc: {results['HI_2.0']['accuracy']:.1%}")
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')

        # Panel C: Metric Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 (macro)', 'F1 (weighted)']
        hi1_values = [
            results['HI_1.0']['accuracy'],
            results['HI_1.0']['precision_macro'],
            results['HI_1.0']['recall_macro'],
            results['HI_1.0']['f1_macro'],
            results['HI_1.0']['f1_weighted']
        ]
        hi2_values = [
            results['HI_2.0']['accuracy'],
            results['HI_2.0']['precision_macro'],
            results['HI_2.0']['recall_macro'],
            results['HI_2.0']['f1_macro'],
            results['HI_2.0']['f1_weighted']
        ]

        x = np.arange(len(metrics_names))
        width = 0.35
        ax3.bar(x - width/2, hi1_values, width, label='HI 1.0', color='steelblue', edgecolor='black')
        ax3.bar(x + width/2, hi2_values, width, label='HI 2.0', color='seagreen', edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax3.set_ylabel('Score')
        ax3.set_title('Overall Metrics Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1.0)

        # Panel D: Per-Class F1 Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        classes = []
        hi1_f1 = []
        hi2_f1 = []
        for name in class_names:
            if name in results['HI_1.0']['per_class'] and name in results['HI_2.0']['per_class']:
                classes.append(name)
                hi1_f1.append(results['HI_1.0']['per_class'][name]['f1'])
                hi2_f1.append(results['HI_2.0']['per_class'][name]['f1'])

        x = np.arange(len(classes))
        ax4.bar(x - width/2, hi1_f1, width, label='HI 1.0', color='steelblue', edgecolor='black')
        ax4.bar(x + width/2, hi2_f1, width, label='HI 2.0', color='seagreen', edgecolor='black')
        ax4.set_xticks(x)
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Per-Class F1 Scores')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1.0)

        # Panel E: Feature Importance - HI 1.0
        ax5 = fig.add_subplot(gs[1, 1])
        feat_imp_hi1 = results['HI_1.0']['feature_importance']
        feat_names_hi1 = ['HI_1.0', 'b', 'e', 'norm']
        colors1 = plt.cm.Blues(np.linspace(0.4, 0.8, len(feat_imp_hi1)))
        ax5.barh(feat_names_hi1, feat_imp_hi1, color=colors1, edgecolor='black')
        ax5.set_xlabel('Importance')
        ax5.set_title('HI 1.0 Feature Importance')
        ax5.grid(True, alpha=0.3, axis='x')

        # Panel F: Feature Importance - HI 2.0
        ax6 = fig.add_subplot(gs[1, 2])
        feat_imp_hi2 = results['HI_2.0']['feature_importance']
        feat_names_hi2 = ['HI_2.0', 'H_ang', 'H_rad', 'H_fam', 'C', 'F', 'G', 'gcd', 'b', 'e']
        colors2 = plt.cm.Greens(np.linspace(0.4, 0.8, len(feat_imp_hi2)))
        ax6.barh(feat_names_hi2, feat_imp_hi2, color=colors2, edgecolor='black')
        ax6.set_xlabel('Importance')
        ax6.set_title('HI 2.0 Feature Importance')
        ax6.grid(True, alpha=0.3, axis='x')

        # Panel G: HI Distribution by Class - HI 1.0
        ax7 = fig.add_subplot(gs[2, 0])
        for i, name in enumerate(class_names):
            mask = (y_true == i)
            if np.sum(mask) > 0:
                hi_values = features_hi1[mask, 0]  # First feature is HI
                ax7.hist(hi_values, bins=20, alpha=0.5, label=name, edgecolor='black')
        ax7.set_xlabel('HI 1.0 Value')
        ax7.set_ylabel('Count')
        ax7.set_title('HI 1.0 Distribution by Class')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)

        # Panel H: HI Distribution by Class - HI 2.0
        ax8 = fig.add_subplot(gs[2, 1])
        for i, name in enumerate(class_names):
            mask = (y_true == i)
            if np.sum(mask) > 0:
                hi_values = features_hi2[mask, 0]  # First feature is HI
                ax8.hist(hi_values, bins=20, alpha=0.5, label=name, edgecolor='black')
        ax8.set_xlabel('HI 2.0 Value')
        ax8.set_ylabel('Count')
        ax8.set_title('HI 2.0 Distribution by Class')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)

        # Panel I: Summary Text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"""EEG SEIZURE DETECTION RESULTS
CHB-MIT Dataset (Real EEG)

Configuration:
  HI 2.0: Angular_Radial
  w_ang = {HI2_CONFIG['w_ang']:.1f}
  w_rad = {HI2_CONFIG['w_rad']:.1f}
  w_fam = {HI2_CONFIG['w_fam']:.1f}

Test Set Performance:
  HI 1.0 Accuracy:  {results['HI_1.0']['accuracy']:6.1%}
  HI 2.0 Accuracy:  {results['HI_2.0']['accuracy']:6.1%}

  HI 1.0 F1 (macro): {results['HI_1.0']['f1_macro']:.3f}
  HI 2.0 F1 (macro): {results['HI_2.0']['f1_macro']:.3f}

Improvement:
  Accuracy:  {results['comparison']['accuracy_delta']:+.1%}
  F1 (macro): {results['comparison']['f1_macro_delta']:+.3f}
"""

        if 'ttest_pvalue' in results['comparison']:
            p_val = results['comparison']['ttest_pvalue']
            sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            summary_text += f"\nStatistical Significance:\n  p = {p_val:.4f} {sig_marker}\n"

        summary_text += f"\nClassifier: Random Forest\n  n_estimators = 200\n  max_depth = 10"

        ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center', transform=ax9.transAxes)

        plt.suptitle('EEG Seizure Detection: HI 1.0 vs HI 2.0 (Angular_Radial)',
                    fontsize=14, fontweight='bold')

        output_path = Path("eeg_hi2_0_results_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {output_path}")
        plt.close()

    def save_results(self, results: Dict):
        """Save results to JSON file."""
        output_path = Path("eeg_hi2_0_results.json")

        results['metadata'] = {
            'dataset': 'CHB-MIT EEG',
            'task': 'Multi-class seizure detection',
            'classes': ['Baseline', 'Pre-ictal', 'Ictal', 'Post-ictal'],
            'hi2_config': HI2_CONFIG,
            'classifier': 'Random Forest (n_estimators=200, max_depth=10)',
            'test_size': 0.25
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved: {output_path}")


def main():
    """Main execution."""
    print("="*80)
    print("EEG SEIZURE DETECTION WITH HARMONICITY INDEX 2.0")
    print("Angular_Radial Configuration (w_ang=0.5, w_rad=0.5, w_fam=0.0)")
    print("="*80)

    # Initialize classifier
    classifier = EEGHarmonyClassifier()

    # Run experiment with chb01 patient (has good seizure data)
    try:
        results = classifier.run_comparison_experiment(subjects=['chb01'])

        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        print("\nKey Findings:")

        if results['comparison']['f1_macro_delta'] > 0:
            print(f"  HI 2.0 outperforms HI 1.0 by {results['comparison']['f1_macro_delta']:.3f} F1 points")
        else:
            print(f"  HI 1.0 performs {-results['comparison']['f1_macro_delta']:.3f} F1 points better")

        print(f"\nOutputs:")
        print(f"  - eeg_hi2_0_results.json")
        print(f"  - eeg_hi2_0_results_visualization.png")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
