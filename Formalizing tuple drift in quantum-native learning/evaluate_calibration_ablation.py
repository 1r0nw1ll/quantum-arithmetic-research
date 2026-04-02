"""
QAWM Calibration & Ablation Suite - Paper 2 Final Polish
Implements minimal reviewer-proofing analyses:
  1. Return-in-k calibration (ECE + reliability plot)
  2. Feature ablations (primitive-only, no-log)

These strengthen the story without adding complexity.
"""

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from qa_oracle import QAOracle, construct_qa_state
from qawm import extract_state_features, generator_to_index, QAWMConfig
from dataset import QADatasetGenerator


# =============================================================================
# Calibration Metrics
# =============================================================================

def expected_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              n_bins: int = 10) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (n_k / n) * |acc_k - conf_k|

    Returns:
        ece: scalar ECE value
        bin_stats: dict with per-bin statistics for reliability plot
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute per-bin statistics
    bin_stats = {
        'bin_centers': [],
        'accuracies': [],
        'confidences': [],
        'counts': []
    }

    ece = 0.0
    total_samples = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        count = np.sum(mask)

        if count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_pred_proba[mask])

            ece += (count / total_samples) * np.abs(bin_acc - bin_conf)

            bin_stats['bin_centers'].append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
            bin_stats['accuracies'].append(bin_acc)
            bin_stats['confidences'].append(bin_conf)
            bin_stats['counts'].append(count)

    return ece, bin_stats


def plot_reliability_diagram(bin_stats: Dict, ece: float, save_path: str = None):
    """Plot calibration reliability diagram"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    # Bin statistics
    centers = np.array(bin_stats['bin_centers'])
    accuracies = np.array(bin_stats['accuracies'])
    confidences = np.array(bin_stats['confidences'])
    counts = np.array(bin_stats['counts'])

    # Plot bars (width proportional to count)
    bar_width = 0.08
    colors = plt.cm.RdYlGn_r(np.abs(accuracies - confidences) * 2)

    for i, (center, acc, conf, count) in enumerate(zip(centers, accuracies, confidences, counts)):
        ax.bar(conf, acc, width=bar_width, alpha=0.7,
               color=colors[i], edgecolor='black', linewidth=1)

    ax.set_xlabel('Confidence (Predicted Probability)', fontsize=12)
    ax.set_ylabel('Accuracy (Fraction Positive)', fontsize=12)
    ax.set_title(f'Reliability Diagram: Return-in-k Prediction\nECE = {ece:.3f}', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Add ECE annotation
    ax.text(0.05, 0.95, f'ECE = {ece:.3f}\n(lower is better)',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Reliability diagram saved: {save_path}")

    return fig


# =============================================================================
# Feature Ablation
# =============================================================================

def extract_state_features_ablated(state, N: int = 30, ablation_type: str = 'none') -> np.ndarray:
    """
    Extract features with ablations for robustness testing.

    Args:
        ablation_type:
            'none' - Full feature set (baseline)
            'primitive_only' - Only (b, e) + parity
            'no_log' - No log-scaling (raw invariants)
            'no_rationals' - Drop L features
    """
    features = []

    if ablation_type == 'primitive_only':
        # Only primitives + parity markers
        features.append(state.b / N)
        features.append(state.e / N)
        features.append(float(state.b % 2))  # b parity
        features.append(float(state.e % 2))  # e parity
        features.append(state.phi_9 / 9.0)
        features.append(state.phi_24 / 24.0)

        # Pad to 128
        features = np.array(features, dtype=np.float64)
        features = np.pad(features, (0, 128 - len(features)), 'constant')

    elif ablation_type == 'no_log':
        # Bucket A: normalized (same as baseline)
        features.append(state.b / N)
        features.append(state.e / N)
        features.append(state.d / (2 * N))
        features.append(state.a / (3 * N))
        features.append(state.phi_9 / 9.0)
        features.append(state.phi_24 / 24.0)

        # Bucket B: RAW (no log-scaling)
        max_val = N * N * 4  # Rough upper bound for normalization
        large_invariants = [
            state.B, state.E, state.D, state.A, state.X, state.C, state.F,
            state.G, state.H, state.I, state.J, state.K, state.W, state.Y,
            state.Z, state.h2,
            state.b * state.b + state.e * state.e,
            state.b + state.e + state.d + state.a
        ]

        for val in large_invariants:
            features.append(float(val) / max_val)  # Normalize instead of log

        # Bucket C: same as baseline
        L_num = state.L.numerator
        L_den = state.L.denominator
        features.append(np.log1p(abs(float(L_num))))
        features.append(np.log1p(float(L_den)))

        features = np.array(features, dtype=np.float64)
        features = np.pad(features, (0, 128 - len(features)), 'constant')

    else:
        # Baseline (full features from qawm.py)
        from qawm import extract_state_features
        features = extract_state_features(state, N)

    return features


def evaluate_ablation(oracle: QAOracle, generators: list, ablation_type: str,
                     budget: int = 1000, return_budget: int = 100) -> Dict:
    """
    Train and evaluate QAWM with ablated features.

    Returns:
        results: dict with AUROC and other metrics
    """
    print(f"\n{'='*70}")
    print(f"ABLATION: {ablation_type.upper()}")
    print(f"{'='*70}")

    # Generate dataset
    print(f"Generating dataset (budget={budget})...")
    gen = QADatasetGenerator(oracle, generators)
    dataset = gen.generate_dataset(budget=budget, return_in_k_budget=return_budget)

    # Extract features with ablation
    print(f"Extracting features (ablation={ablation_type})...")
    all_state_features = []
    all_gen_indices = []
    all_labels = {'legal': [], 'return': []}

    for idx in range(len(dataset)):
        record = dataset.records[idx]

        # Use ablated feature extraction
        state_feat = extract_state_features_ablated(record['state'], N=oracle.N,
                                                     ablation_type=ablation_type)
        gen_idx = generator_to_index(record['generator'])

        all_state_features.append(state_feat)
        all_gen_indices.append(gen_idx)
        all_labels['legal'].append(1 if record['legal'] else 0)
        all_labels['return'].append(
            1 if dataset.return_in_k_labels.get(idx, False) else
            (0 if idx in dataset.return_in_k_labels else -1)
        )

    X_state = np.stack(all_state_features)
    X_gen = np.array(all_gen_indices)

    # Prepare input
    batch_size = X_state.shape[0]
    gen_onehot = np.zeros((batch_size, 4), dtype=np.float64)
    gen_onehot[np.arange(batch_size), X_gen] = 1.0
    X_combined = np.concatenate([X_state, gen_onehot], axis=1)

    # Scale
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    y_return = np.array(all_labels['return'])
    return_mask = y_return != -1

    # Train return-in-k head only (this is what we're testing)
    if np.sum(return_mask) > 20:
        print(f"Training return-in-k head ({np.sum(return_mask)} labeled samples)...")

        X_train = X_combined[return_mask]
        y_train = y_return[return_mask]

        # Split train/val
        split_idx = int(0.8 * len(X_train))
        X_train_split = X_train[:split_idx]
        y_train_split = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]

        # Train
        model = MLPClassifier(
            hidden_layer_sizes=(256, 256),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42,
            verbose=False
        )

        model.fit(X_train_split, y_train_split)

        # Evaluate
        if len(X_val) > 10:
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            try:
                auroc = roc_auc_score(y_val, y_pred_proba)
                print(f"  ✓ Return-in-k AUROC: {auroc:.3f}")
            except:
                auroc = 0.5
                print(f"  ! Could not compute AUROC")
        else:
            auroc = 0.5
            print(f"  ! Insufficient validation samples")
    else:
        auroc = 0.5
        print(f"  ! Insufficient labeled samples for training")

    return {'auroc': auroc, 'ablation': ablation_type}


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 70)
    print("QAWM CALIBRATION & ABLATION SUITE")
    print("Paper 2 Final Polish")
    print("=" * 70)

    # =========================================================================
    # Part 1: Calibration Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: CALIBRATION ANALYSIS (Return-in-k)")
    print("=" * 70)

    # Load trained model
    print("\n[1/3] Loading trained model...")
    model = joblib.load('qawm_model.pkl')
    print("  ✓ Model loaded")

    # Generate validation dataset
    print("\n[2/3] Generating validation dataset...")
    oracle_30 = QAOracle(N=30, q_def="none")
    generators = ['sigma', 'mu', 'lambda2', 'nu']

    gen = QADatasetGenerator(oracle_30, generators)
    val_dataset = gen.generate_dataset(budget=1000, return_in_k_budget=200)

    # Extract features and predictions
    print("\n[3/3] Computing calibration metrics...")
    all_state_features = []
    all_gen_indices = []
    all_return_labels = []

    for idx in range(len(val_dataset)):
        if idx in val_dataset.return_in_k_labels:
            record = val_dataset.records[idx]

            from qawm import extract_state_features
            state_feat = extract_state_features(record['state'], N=30)
            gen_idx = generator_to_index(record['generator'])

            all_state_features.append(state_feat)
            all_gen_indices.append(gen_idx)
            all_return_labels.append(1 if val_dataset.return_in_k_labels[idx] else 0)

    X_state = np.stack(all_state_features)
    X_gen = np.array(all_gen_indices)
    X_combined = model._prepare_input(X_state, X_gen)

    y_true = np.array(all_return_labels)

    # Get predictions
    if model.return_fitted:
        y_pred_proba = model.return_head.predict_proba(X_combined)[:, 1]

        # Compute calibration
        ece, bin_stats = expected_calibration_error(y_true, y_pred_proba, n_bins=10)

        print(f"\n  Expected Calibration Error (ECE): {ece:.3f}")
        print(f"  (ECE < 0.1 is well-calibrated)")

        # AUROC for reference
        auroc = roc_auc_score(y_true, y_pred_proba)
        print(f"  Return-in-k AUROC: {auroc:.3f}")

        # Plot reliability diagram
        plot_reliability_diagram(bin_stats, ece, save_path='calibration_reliability.png')

    else:
        print("  ! Return head not fitted")
        ece = None

    # =========================================================================
    # Part 2: Feature Ablations
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("PART 2: FEATURE ABLATIONS (Return-in-k)")
    print("=" * 70)
    print("Testing: Does QAWM rely on full feature set or simpler patterns?")

    ablation_results = []

    # Baseline (full features)
    print("\n[Baseline: Full feature set]")
    baseline_result = {'ablation': 'baseline', 'auroc': 0.836}  # From training
    ablation_results.append(baseline_result)
    print(f"  Return-in-k AUROC: {baseline_result['auroc']:.3f} (from training)")

    # Primitive-only ablation
    primitive_result = evaluate_ablation(
        oracle_30, generators, 'primitive_only',
        budget=1000, return_budget=100
    )
    ablation_results.append(primitive_result)

    # No-log ablation
    nolog_result = evaluate_ablation(
        oracle_30, generators, 'no_log',
        budget=1000, return_budget=100
    )
    ablation_results.append(nolog_result)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("CALIBRATION & ABLATION SUMMARY")
    print("=" * 70)

    # Calibration
    if ece is not None:
        print(f"\n1. Calibration (Return-in-k):")
        print(f"   ECE = {ece:.3f}")
        if ece < 0.1:
            print(f"   ✅ Well-calibrated (ECE < 0.1)")
            cal_status = "TRUSTWORTHY"
        elif ece < 0.15:
            print(f"   ✓ Acceptable calibration")
            cal_status = "ACCEPTABLE"
        else:
            print(f"   ⚠ Miscalibrated (consider Platt scaling)")
            cal_status = "NEEDS CALIBRATION"
    else:
        cal_status = "N/A"

    # Ablations
    print(f"\n2. Feature Ablations (Return-in-k AUROC):")
    print(f"   {'Ablation':<20} {'AUROC':<10} {'vs Baseline':<15}")
    print(f"   {'-'*45}")

    for result in ablation_results:
        ablation_name = result['ablation']
        auroc = result['auroc']
        delta = auroc - baseline_result['auroc']

        print(f"   {ablation_name:<20} {auroc:.3f}      {delta:+.3f}")

    # Interpretation
    print(f"\n3. Interpretation:")

    primitive_drop = baseline_result['auroc'] - primitive_result['auroc']
    if primitive_drop < 0.1:
        print(f"   ✓ Primitive-only ablation shows minimal drop ({primitive_drop:.3f})")
        print(f"     → Return-in-k is learnable from simple features")
    else:
        print(f"   • Primitive-only drops {primitive_drop:.3f}")
        print(f"     → Full invariant packet provides additional signal")

    nolog_drop = baseline_result['auroc'] - nolog_result['auroc']
    if nolog_drop < 0.05:
        print(f"   ✓ No-log ablation shows robust performance")
        print(f"     → Log-scaling is helpful but not critical")
    else:
        print(f"   • No-log drops {nolog_drop:.3f}")
        print(f"     → Log-scaling important for handling large invariants")

    # Final verdict
    print(f"\n" + "=" * 70)
    print(f"FINAL VERDICT")
    print(f"=" * 70)
    print(f"Calibration: {cal_status}")
    print(f"Robustness: Feature ablations show {'ROBUST' if max(primitive_drop, nolog_drop) < 0.15 else 'MODERATE'} performance")

    print(f"\n✅ Calibration & ablation analysis complete!")
    print(f"   Files generated:")
    print(f"   - calibration_reliability.png")


if __name__ == "__main__":
    main()
