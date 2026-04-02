#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
# SUPERSEDED — QA-noncompliant (T2-b violation: amplitude × modulus → int cast at line 94)
# Use seismic_orbit_classifier.py instead.
"""
Seismic Classification with HI 2.0 (Radial_family Configuration)

Re-runs Phase 2 seismic classification experiments using:
- HI 1.0 (baseline E8-only metric)
- HI 2.0 with Radial_family config (w_ang=0.0, w_rad=0.6, w_fam=0.4)

Goal: Compare primitive (earthquake) vs female/composite (explosion) discrimination.

The Radial_family configuration emphasizes:
- Radial harmonicity (60%): Primitivity measure via gcd(C,F,G)
- Family harmonicity (40%): Classical Pythagorean subfamily membership
- Angular harmonicity (0%): Disabled for this experiment

Hypothesis: Earthquakes (primitive signals) should have higher H_radial (gcd=1),
while explosions (composite/female signals) should have lower H_radial (gcd>1).
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

# Import seismic generator
from seismic_data_generator import SeismicWaveformGenerator

# Import HI 2.0 implementation
from qa_harmonicity_v2 import (
    qa_tuple,
    compute_hi_1_0,
    compute_hi_2_0,
    pythagorean_triple,
    classify_gender
)


class SeismicHI20Classifier:
    """
    Seismic event classifier using HI 1.0 and HI 2.0 metrics.

    Extracts QA tuples from seismic waveforms and computes harmonicity indices
    for earthquake vs explosion discrimination.
    """

    def __init__(self, modulus: int = 24, num_samples: int = 100):
        """
        Args:
            modulus: QA modulus (default 24)
            num_samples: Number of time samples to extract from waveform
        """
        self.modulus = modulus
        self.num_samples = num_samples

    def waveform_to_qa_tuples(self, waveform: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Convert seismic waveform to sequence of QA tuples.

        Strategy:
        1. Downsample/select key time points from waveform
        2. Map amplitude values to QA state pairs (b, e) via modular arithmetic
        3. Generate QA tuples (b, e, d, a)

        Args:
            waveform: Seismic time series

        Returns:
            List of QA tuples
        """
        # Downsample waveform to fixed number of samples
        if len(waveform) > self.num_samples:
            indices = np.linspace(0, len(waveform)-1, self.num_samples, dtype=int)
            sampled = waveform[indices]
        else:
            sampled = waveform

        # Normalize to [0, 1]
        wf_min, wf_max = np.min(sampled), np.max(sampled)
        if wf_max - wf_min > 1e-9:
            normalized = (sampled - wf_min) / (wf_max - wf_min)
        else:
            normalized = np.zeros_like(sampled)

        # Map to QA state pairs (b, e)
        # Use sliding window approach: (t, t+1) → (b, e)
        qa_tuples = []

        for i in range(len(normalized) - 1):
            # Map normalized amplitudes to [1, modulus] range
            b = int(normalized[i] * (self.modulus - 1)) + 1
            e = int(normalized[i+1] * (self.modulus - 1)) + 1

            # Ensure values are in valid range [1, modulus]
            b = max(1, min(self.modulus, b))
            e = max(1, min(self.modulus, e))

            # Generate QA tuple
            q = qa_tuple(b, e, modulus=self.modulus)
            qa_tuples.append(q)

        return qa_tuples

    def compute_hi_statistics(self, qa_tuples: List[Tuple[int, int, int, int]],
                             w_ang: float = 0.4, w_rad: float = 0.3, w_fam: float = 0.3) -> Dict:
        """
        Compute HI statistics over sequence of QA tuples.

        Args:
            qa_tuples: List of QA tuples from waveform
            w_ang, w_rad, w_fam: HI 2.0 weights

        Returns:
            Dictionary with aggregated statistics
        """
        hi_1_0_values = []
        hi_2_0_values = []
        h_angular_values = []
        h_radial_values = []
        h_family_values = []
        gcd_values = []
        gender_counts = {'Male (Primitive)': 0, 'Female': 0, 'Male (Composite)': 0}

        for q in qa_tuples:
            # Compute HI 1.0
            hi1 = compute_hi_1_0(q, modulus=self.modulus)
            hi_1_0_values.append(hi1)

            # Compute HI 2.0
            result = compute_hi_2_0(q, w_ang=w_ang, w_rad=w_rad, w_fam=w_fam,
                                   modulus=self.modulus)
            hi_2_0_values.append(result['HI_2.0'])
            h_angular_values.append(result['H_angular'])
            h_radial_values.append(result['H_radial'])
            h_family_values.append(result['H_family'])
            gcd_values.append(result['gcd'])

            # Gender classification
            gender = classify_gender(q)
            if 'Primitive' in gender:
                gender_counts['Male (Primitive)'] += 1
            elif 'Female' in gender:
                gender_counts['Female'] += 1
            else:
                gender_counts['Male (Composite)'] += 1

        # Compute statistics
        total = len(qa_tuples)

        return {
            'HI_1.0_mean': np.mean(hi_1_0_values),
            'HI_1.0_std': np.std(hi_1_0_values),
            'HI_1.0_max': np.max(hi_1_0_values),
            'HI_2.0_mean': np.mean(hi_2_0_values),
            'HI_2.0_std': np.std(hi_2_0_values),
            'HI_2.0_max': np.max(hi_2_0_values),
            'H_angular_mean': np.mean(h_angular_values),
            'H_radial_mean': np.mean(h_radial_values),
            'H_family_mean': np.mean(h_family_values),
            'gcd_mean': np.mean(gcd_values),
            'primitive_fraction': gender_counts['Male (Primitive)'] / total,
            'female_fraction': gender_counts['Female'] / total,
            'composite_fraction': gender_counts['Male (Composite)'] / total,
            'num_tuples': total
        }

    def process_waveform(self, waveform: np.ndarray,
                        config_name: str = 'Balanced',
                        w_ang: float = 0.4, w_rad: float = 0.3, w_fam: float = 0.3) -> Dict:
        """
        Process single waveform and extract HI features.

        Args:
            waveform: Seismic time series
            config_name: Weight configuration name
            w_ang, w_rad, w_fam: HI 2.0 weights

        Returns:
            Feature dictionary
        """
        qa_tuples = self.waveform_to_qa_tuples(waveform)
        stats = self.compute_hi_statistics(qa_tuples, w_ang, w_rad, w_fam)
        stats['config'] = config_name
        return stats

    def process_dataset(self, dataset: List[Dict],
                       config_name: str = 'Balanced',
                       w_ang: float = 0.4, w_rad: float = 0.3, w_fam: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process full dataset and extract features.

        Args:
            dataset: List of waveform dictionaries
            config_name: Weight configuration name
            w_ang, w_rad, w_fam: HI 2.0 weights

        Returns:
            (features, labels) where features is (n_samples, n_features) array
        """
        features_list = []
        labels = []

        print(f"Processing {len(dataset)} waveforms with {config_name} config...")

        for i, waveform_data in enumerate(dataset):
            stats = self.process_waveform(
                waveform_data['waveform'],
                config_name=config_name,
                w_ang=w_ang, w_rad=w_rad, w_fam=w_fam
            )

            # Extract feature vector
            features = [
                stats['HI_1.0_mean'],
                stats['HI_1.0_std'],
                stats['HI_1.0_max'],
                stats['HI_2.0_mean'],
                stats['HI_2.0_std'],
                stats['HI_2.0_max'],
                stats['H_angular_mean'],
                stats['H_radial_mean'],
                stats['H_family_mean'],
                stats['gcd_mean'],
                stats['primitive_fraction'],
                stats['female_fraction'],
                stats['composite_fraction']
            ]
            features_list.append(features)

            # Label: 1 = earthquake, 0 = explosion
            label = 1 if waveform_data['type'] == 'earthquake' else 0
            labels.append(label)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(dataset)}...")

        return np.array(features_list), np.array(labels)


def run_classification_experiment(dataset: List[Dict],
                                  config_name: str,
                                  w_ang: float, w_rad: float, w_fam: float) -> Dict:
    """
    Run classification experiment with specific HI 2.0 configuration.

    Args:
        dataset: Seismic waveform dataset
        config_name: Configuration name
        w_ang, w_rad, w_fam: HI 2.0 weights

    Returns:
        Results dictionary
    """
    print()
    print("="*80)
    print(f"Running classification with {config_name} configuration")
    print(f"  w_ang={w_ang}, w_rad={w_rad}, w_fam={w_fam}")
    print("="*80)
    print()

    # Extract features
    classifier = SeismicHI20Classifier(modulus=24, num_samples=100)
    X, y = classifier.process_dataset(dataset, config_name, w_ang, w_rad, w_fam)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.sum(y)} earthquakes, {len(y) - np.sum(y)} explosions")
    print()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train logistic regression classifier
    # Try different feature subsets
    results = {}

    # 1. HI 1.0 only (baseline)
    print("Training with HI 1.0 features only...")
    X_train_hi1 = X_train[:, :3]  # HI_1.0 mean, std, max
    X_test_hi1 = X_test[:, :3]

    clf_hi1 = LogisticRegression(random_state=42, max_iter=1000)
    clf_hi1.fit(X_train_hi1, y_train)
    y_pred_hi1 = clf_hi1.predict(X_test_hi1)
    y_proba_hi1 = clf_hi1.predict_proba(X_test_hi1)[:, 1]

    acc_hi1 = accuracy_score(y_test, y_pred_hi1)
    prec_hi1, rec_hi1, f1_hi1, _ = precision_recall_fscore_support(
        y_test, y_pred_hi1, average='binary'
    )
    auc_hi1 = roc_auc_score(y_test, y_proba_hi1)

    print(f"  Accuracy: {acc_hi1*100:.2f}%")
    print(f"  Precision: {prec_hi1:.3f}, Recall: {rec_hi1:.3f}, F1: {f1_hi1:.3f}")
    print(f"  AUC: {auc_hi1:.3f}")
    print()

    results['HI_1.0_only'] = {
        'accuracy': float(acc_hi1),
        'precision': float(prec_hi1),
        'recall': float(rec_hi1),
        'f1': float(f1_hi1),
        'auc': float(auc_hi1)
    }

    # 2. HI 2.0 only
    print("Training with HI 2.0 features only...")
    X_train_hi2 = X_train[:, 3:6]  # HI_2.0 mean, std, max
    X_test_hi2 = X_test[:, 3:6]

    clf_hi2 = LogisticRegression(random_state=42, max_iter=1000)
    clf_hi2.fit(X_train_hi2, y_train)
    y_pred_hi2 = clf_hi2.predict(X_test_hi2)
    y_proba_hi2 = clf_hi2.predict_proba(X_test_hi2)[:, 1]

    acc_hi2 = accuracy_score(y_test, y_pred_hi2)
    prec_hi2, rec_hi2, f1_hi2, _ = precision_recall_fscore_support(
        y_test, y_pred_hi2, average='binary'
    )
    auc_hi2 = roc_auc_score(y_test, y_proba_hi2)

    print(f"  Accuracy: {acc_hi2*100:.2f}%")
    print(f"  Precision: {prec_hi2:.3f}, Recall: {rec_hi2:.3f}, F1: {f1_hi2:.3f}")
    print(f"  AUC: {auc_hi2:.3f}")
    print()

    results['HI_2.0_only'] = {
        'accuracy': float(acc_hi2),
        'precision': float(prec_hi2),
        'recall': float(rec_hi2),
        'f1': float(f1_hi2),
        'auc': float(auc_hi2)
    }

    # 3. All features (HI 1.0 + HI 2.0 + components + gender)
    print("Training with all features...")
    clf_all = LogisticRegression(random_state=42, max_iter=1000)
    clf_all.fit(X_train, y_train)
    y_pred_all = clf_all.predict(X_test)
    y_proba_all = clf_all.predict_proba(X_test)[:, 1]

    acc_all = accuracy_score(y_test, y_pred_all)
    prec_all, rec_all, f1_all, _ = precision_recall_fscore_support(
        y_test, y_pred_all, average='binary'
    )
    auc_all = roc_auc_score(y_test, y_proba_all)

    print(f"  Accuracy: {acc_all*100:.2f}%")
    print(f"  Precision: {prec_all:.3f}, Recall: {rec_all:.3f}, F1: {f1_all:.3f}")
    print(f"  AUC: {auc_all:.3f}")
    print()

    results['All_features'] = {
        'accuracy': float(acc_all),
        'precision': float(prec_all),
        'recall': float(rec_all),
        'f1': float(f1_all),
        'auc': float(auc_all)
    }

    # 4. Radial + Family components only (for Radial_family config)
    if config_name == 'Radial_family':
        print("Training with Radial + Family components only...")
        X_train_rf = X_train[:, [7, 8, 10, 11, 12]]  # H_radial, H_family, gender fractions
        X_test_rf = X_test[:, [7, 8, 10, 11, 12]]

        clf_rf = LogisticRegression(random_state=42, max_iter=1000)
        clf_rf.fit(X_train_rf, y_train)
        y_pred_rf = clf_rf.predict(X_test_rf)
        y_proba_rf = clf_rf.predict_proba(X_test_rf)[:, 1]

        acc_rf = accuracy_score(y_test, y_pred_rf)
        prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(
            y_test, y_pred_rf, average='binary'
        )
        auc_rf = roc_auc_score(y_test, y_proba_rf)

        print(f"  Accuracy: {acc_rf*100:.2f}%")
        print(f"  Precision: {prec_rf:.3f}, Recall: {rec_rf:.3f}, F1: {f1_rf:.3f}")
        print(f"  AUC: {auc_rf:.3f}")
        print()

        results['Radial_Family_components'] = {
            'accuracy': float(acc_rf),
            'precision': float(prec_rf),
            'recall': float(rec_rf),
            'f1': float(f1_rf),
            'auc': float(auc_rf)
        }

    # Feature importance analysis
    feature_names = [
        'HI_1.0_mean', 'HI_1.0_std', 'HI_1.0_max',
        'HI_2.0_mean', 'HI_2.0_std', 'HI_2.0_max',
        'H_angular_mean', 'H_radial_mean', 'H_family_mean',
        'gcd_mean', 'primitive_frac', 'female_frac', 'composite_frac'
    ]

    coefficients = clf_all.coef_[0]
    importance_dict = {name: float(coef) for name, coef in zip(feature_names, coefficients)}

    # Sort by absolute value
    sorted_importance = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 most important features:")
    for i, (name, coef) in enumerate(sorted_importance[:5], 1):
        print(f"  {i}. {name}: {coef:.4f}")
    print()

    results['feature_importance'] = importance_dict
    results['config'] = config_name
    results['weights'] = {'w_ang': w_ang, 'w_rad': w_rad, 'w_fam': w_fam}

    # Analyze earthquake vs explosion feature distributions
    earthquake_mask = y == 1
    explosion_mask = y == 0

    feature_stats = {}
    for i, name in enumerate(feature_names):
        eq_vals = X[earthquake_mask, i]
        ex_vals = X[explosion_mask, i]

        feature_stats[name] = {
            'earthquake_mean': float(np.mean(eq_vals)),
            'earthquake_std': float(np.std(eq_vals)),
            'explosion_mean': float(np.mean(ex_vals)),
            'explosion_std': float(np.std(ex_vals)),
            'separation': float(abs(np.mean(eq_vals) - np.mean(ex_vals)) /
                              (np.std(eq_vals) + np.std(ex_vals) + 1e-9))
        }

    results['feature_distributions'] = feature_stats

    return results


def main():
    """
    Main experiment: Compare HI 1.0 vs HI 2.0 (Radial_family) for seismic classification.
    """
    print("="*80)
    print("SEISMIC CLASSIFICATION: HI 2.0 (Radial_family) vs HI 1.0")
    print("="*80)
    print()
    print("Experimental Setup:")
    print("  - Dataset: Synthetic earthquakes vs explosions")
    print("  - HI 1.0: E8-only baseline (w_ang=1.0, w_rad=0.0, w_fam=0.0)")
    print("  - HI 2.0 Radial_family: (w_ang=0.0, w_rad=0.6, w_fam=0.4)")
    print("  - Classification: Logistic Regression with 70/30 train/test split")
    print()

    # Generate synthetic seismic dataset
    print("Generating synthetic seismic dataset...")
    np.random.seed(42)
    generator = SeismicWaveformGenerator(sample_rate=100)
    dataset = generator.generate_dataset(n_earthquakes=100, n_explosions=100)
    print(f"  Generated {len(dataset)} waveforms (100 earthquakes, 100 explosions)")
    print()

    # Run experiments
    all_results = {}

    # 1. HI 1.0 (baseline)
    results_hi1 = run_classification_experiment(
        dataset,
        config_name='HI_1.0_baseline',
        w_ang=1.0, w_rad=0.0, w_fam=0.0
    )
    all_results['HI_1.0_baseline'] = results_hi1

    # 2. HI 2.0 Radial_family
    results_radial = run_classification_experiment(
        dataset,
        config_name='Radial_family',
        w_ang=0.0, w_rad=0.6, w_fam=0.4
    )
    all_results['Radial_family'] = results_radial

    # 3. HI 2.0 Balanced (for comparison)
    results_balanced = run_classification_experiment(
        dataset,
        config_name='Balanced',
        w_ang=0.4, w_rad=0.3, w_fam=0.3
    )
    all_results['Balanced'] = results_balanced

    # Summary comparison
    print()
    print("="*80)
    print("SUMMARY: HI 1.0 vs HI 2.0 Comparison")
    print("="*80)
    print()

    configs = ['HI_1.0_baseline', 'Radial_family', 'Balanced']

    print("Performance Comparison (using HI 2.0 features only):")
    print()
    print(f"{'Configuration':<20} {'Accuracy':<12} {'F1 Score':<12} {'AUC':<12}")
    print("-"*60)

    for config in configs:
        res = all_results[config]['HI_2.0_only']
        print(f"{config:<20} {res['accuracy']*100:>6.2f}%      "
              f"{res['f1']:>6.3f}       {res['auc']:>6.3f}")

    print()
    print("Performance Comparison (using all features):")
    print()
    print(f"{'Configuration':<20} {'Accuracy':<12} {'F1 Score':<12} {'AUC':<12}")
    print("-"*60)

    for config in configs:
        res = all_results[config]['All_features']
        print(f"{config:<20} {res['accuracy']*100:>6.2f}%      "
              f"{res['f1']:>6.3f}       {res['auc']:>6.3f}")

    print()

    # Key findings
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()

    # Compare HI 2.0 performance
    radial_hi2_acc = all_results['Radial_family']['HI_2.0_only']['accuracy']
    baseline_hi2_acc = all_results['HI_1.0_baseline']['HI_2.0_only']['accuracy']
    improvement = (radial_hi2_acc - baseline_hi2_acc) * 100

    print(f"1. HI 2.0 (Radial_family) vs HI 1.0 baseline:")
    print(f"   - Radial_family accuracy: {radial_hi2_acc*100:.2f}%")
    print(f"   - HI 1.0 baseline accuracy: {baseline_hi2_acc*100:.2f}%")
    print(f"   - Improvement: {improvement:+.2f} percentage points")
    print()

    # Analyze feature importance for Radial_family
    print("2. Most important features for Radial_family config:")
    radial_importance = all_results['Radial_family']['feature_importance']
    sorted_imp = sorted(radial_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, coef) in enumerate(sorted_imp[:5], 1):
        print(f"   {i}. {name}: {coef:.4f}")
    print()

    # Analyze primitive vs female/composite discrimination
    print("3. Primitive vs Female/Composite discrimination:")
    radial_stats = all_results['Radial_family']['feature_distributions']

    for key in ['primitive_frac', 'female_frac', 'composite_frac']:
        stats = radial_stats[key]
        print(f"   {key}:")
        print(f"     Earthquakes: {stats['earthquake_mean']:.3f} ± {stats['earthquake_std']:.3f}")
        print(f"     Explosions:  {stats['explosion_mean']:.3f} ± {stats['explosion_std']:.3f}")
        print(f"     Separation:  {stats['separation']:.3f}")
    print()

    # Save results
    output_dir = Path(".")
    results_path = output_dir / "seismic_hi2_0_results.json"

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()

    # Generate visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    configs_short = ['HI 1.0\nBaseline', 'Radial\nFamily', 'Balanced']
    hi2_accs = [all_results[c]['HI_2.0_only']['accuracy']*100 for c in configs]
    all_accs = [all_results[c]['All_features']['accuracy']*100 for c in configs]

    x = np.arange(len(configs_short))
    width = 0.35

    ax.bar(x - width/2, hi2_accs, width, label='HI 2.0 only', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, all_accs, width, label='All features', color='coral', edgecolor='black')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_short)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    # Plot 2: F1 Score comparison
    ax = axes[0, 1]
    hi2_f1s = [all_results[c]['HI_2.0_only']['f1'] for c in configs]
    all_f1s = [all_results[c]['All_features']['f1'] for c in configs]

    ax.bar(x - width/2, hi2_f1s, width, label='HI 2.0 only', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, all_f1s, width, label='All features', color='coral', edgecolor='black')

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_short)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Plot 3: Feature importance (Radial_family)
    ax = axes[1, 0]
    radial_importance = all_results['Radial_family']['feature_importance']
    sorted_features = sorted(radial_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:8]

    names = [f[0].replace('_', ' ') for f in sorted_features]
    values = [f[1] for f in sorted_features]
    colors = ['green' if v > 0 else 'red' for v in values]

    ax.barh(names, values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Coefficient', fontsize=12)
    ax.set_title('Feature Importance (Radial_family)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linewidth=1)

    # Plot 4: Primitive fraction comparison
    ax = axes[1, 1]

    # Extract primitive fractions for each config
    classifier = SeismicHI20Classifier(modulus=24, num_samples=100)

    eq_data = [d for d in dataset if d['type'] == 'earthquake']
    ex_data = [d for d in dataset if d['type'] == 'explosion']

    # Process a few samples to get distributions
    eq_prims = []
    ex_prims = []

    for wf in eq_data[:20]:
        stats = classifier.process_waveform(wf['waveform'], w_ang=0.0, w_rad=0.6, w_fam=0.4)
        eq_prims.append(stats['primitive_fraction'])

    for wf in ex_data[:20]:
        stats = classifier.process_waveform(wf['waveform'], w_ang=0.0, w_rad=0.6, w_fam=0.4)
        ex_prims.append(stats['primitive_fraction'])

    ax.hist(eq_prims, bins=15, alpha=0.6, label='Earthquakes', color='blue', edgecolor='black')
    ax.hist(ex_prims, bins=15, alpha=0.6, label='Explosions', color='red', edgecolor='black')
    ax.set_xlabel('Primitive Tuple Fraction', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Primitive vs Female/Composite Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_dir / "seismic_hi2_0_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    print()

    print("="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print()
    print(f"Summary:")
    print(f"  - Best configuration: {max(configs, key=lambda c: all_results[c]['All_features']['accuracy'])}")
    print(f"  - Radial_family leverages primitivity measure (H_radial) for discrimination")
    print(f"  - Earthquakes show {'higher' if np.mean(eq_prims) > np.mean(ex_prims) else 'lower'} primitive fraction than explosions")
    print()


if __name__ == '__main__':
    main()
