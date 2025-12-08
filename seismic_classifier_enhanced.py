#!/usr/bin/env python3
"""
Enhanced Seismic Classifier with P/S Wave Timing Ratio

Improves earthquake vs explosion discrimination by adding seismological features:
1. P-wave arrival time detection (STA/LTA)
2. S-wave arrival time detection
3. P/S timing ratio (KEY DISCRIMINATOR: ~1.7 for earthquakes, absent/low for explosions)
4. P/S amplitude ratio
5. QA Harmonic Index

Integrates these features for robust classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.ndimage import maximum_filter1d
from collections import Counter

from qa_core import QASystem
from qa_pac_bayes import (
    dqa_divergence,
    compute_pac_constants,
    pac_generalization_bound
)
from pisano_analysis import PisanoClassifier
from seismic_data_generator import SeismicWaveformGenerator


class EnhancedSeismicClassifier:
    """
    Seismic event classifier with P/S wave timing analysis.

    Combines traditional seismological features with QA framework:
    - STA/LTA for phase arrival detection
    - P/S timing and amplitude ratios
    - QA Harmonic Index and geometric features
    """

    def __init__(self, sample_rate: int = 100, modulus: int = 24, num_nodes: int = 24):
        """
        Args:
            sample_rate: Sampling rate in Hz
            modulus: QA system modulus
            num_nodes: Number of QA nodes
        """
        self.sample_rate = sample_rate
        self.modulus = modulus
        self.num_nodes = num_nodes
        self.pac_constants = compute_pac_constants(N=num_nodes, modulus=modulus)
        self.pisano_classifier = PisanoClassifier(modulus=9)

    def compute_sta_lta(self, waveform: np.ndarray,
                        sta_window: float = 0.5,
                        lta_window: float = 5.0) -> np.ndarray:
        """
        Compute Short-Term Average / Long-Term Average ratio.

        Classic seismological onset detection method.

        Args:
            waveform: Seismic trace
            sta_window: Short-term window in seconds
            lta_window: Long-term window in seconds

        Returns:
            STA/LTA ratio time series
        """
        # Convert to samples
        sta_samples = int(sta_window * self.sample_rate)
        lta_samples = int(lta_window * self.sample_rate)

        # Compute envelope (absolute value)
        envelope = np.abs(waveform)

        # Compute STA and LTA using convolution
        sta = np.convolve(envelope, np.ones(sta_samples)/sta_samples, mode='same')
        lta = np.convolve(envelope, np.ones(lta_samples)/lta_samples, mode='same')

        # Compute ratio (avoid division by zero)
        ratio = sta / (lta + 1e-9)

        return ratio

    def detect_phase_arrival(self, sta_lta: np.ndarray, threshold: float = 3.0,
                            search_start: int = 0, search_end: Optional[int] = None) -> Optional[int]:
        """
        Detect phase arrival time from STA/LTA ratio.

        Args:
            sta_lta: STA/LTA ratio time series
            threshold: Detection threshold (typical: 2-4)
            search_start: Start search at this sample
            search_end: End search at this sample

        Returns:
            Sample index of phase arrival, or None if not detected
        """
        if search_end is None:
            search_end = len(sta_lta)

        # Find first crossing of threshold
        search_region = sta_lta[search_start:search_end]
        crossings = np.where(search_region > threshold)[0]

        if len(crossings) == 0:
            return None

        return search_start + crossings[0]

    def extract_ps_features(self, waveform: np.ndarray) -> Dict:
        """
        Extract P-wave and S-wave features from seismic waveform.

        KEY DISCRIMINATORS:
        - P/S timing ratio: ~1.5-2.0 for earthquakes, undefined for explosions
        - P/S amplitude ratio: ~0.5-0.7 for earthquakes, >2.0 for explosions

        Args:
            waveform: Seismic trace

        Returns:
            Dictionary with P/S features
        """
        # Compute STA/LTA
        sta_lta = self.compute_sta_lta(waveform)

        # Detect P-wave arrival (first major arrival)
        p_arrival = self.detect_phase_arrival(sta_lta, threshold=3.0)

        if p_arrival is None:
            # No clear phases detected
            return {
                'p_arrival': None,
                's_arrival': None,
                'ps_time_ratio': 0.0,
                'ps_amplitude_ratio': 0.0,
                'has_clear_phases': False
            }

        # Search for S-wave arrival after P-wave
        # S-waves arrive ~1.5-2x later for typical distances
        s_search_start = p_arrival + int(0.5 * self.sample_rate)
        s_search_end = min(p_arrival + int(10 * self.sample_rate), len(waveform))

        # Use lower threshold for S-wave (may be weaker in noise)
        s_arrival = self.detect_phase_arrival(sta_lta, threshold=2.5,
                                             search_start=s_search_start,
                                             search_end=s_search_end)

        # Extract amplitudes around arrivals
        p_window = slice(max(0, p_arrival - 10),
                        min(len(waveform), p_arrival + 50))
        p_amplitude = np.max(np.abs(waveform[p_window]))

        if s_arrival is not None:
            s_window = slice(max(0, s_arrival - 10),
                           min(len(waveform), s_arrival + 50))
            s_amplitude = np.max(np.abs(waveform[s_window]))

            # Compute ratios
            ps_time_ratio = (s_arrival - p_arrival) / (p_arrival + 1)
            ps_amplitude_ratio = p_amplitude / (s_amplitude + 1e-9)
            has_clear_phases = True
        else:
            s_amplitude = 0.0
            ps_time_ratio = 0.0  # No S-wave detected
            ps_amplitude_ratio = p_amplitude / 0.1  # Very high (explosion-like)
            has_clear_phases = False

        return {
            'p_arrival': p_arrival,
            's_arrival': s_arrival,
            'ps_time_ratio': ps_time_ratio,
            'ps_amplitude_ratio': ps_amplitude_ratio,
            'p_amplitude': p_amplitude,
            's_amplitude': s_amplitude,
            'has_clear_phases': has_clear_phases
        }

    def classify_waveform(self, waveform: np.ndarray) -> Dict:
        """
        Classify seismic waveform using enhanced features.

        Args:
            waveform: Seismic trace

        Returns:
            Classification result with features and prediction
        """
        # Extract P/S features
        ps_features = self.extract_ps_features(waveform)

        # Run QA system
        system = QASystem(
            num_nodes=self.num_nodes,
            modulus=self.modulus,
            coupling=0.2,
            noise_base=0.1,
            noise_annealing=0.998
        )

        # Downsample if needed
        signal_input = waveform
        if len(signal_input) > 500:
            indices = np.linspace(0, len(signal_input)-1, 500, dtype=int)
            signal_input = signal_input[indices]

        system.run_simulation(len(signal_input), signal_input)

        # Extract QA features
        final_hi = system.history['hi'][-1]

        # Pisano classification
        d_vals = (system.b + system.e) % self.modulus
        a_vals = (system.b + 2 * system.e) % self.modulus

        pisano_families = []
        for j in range(self.num_nodes):
            cls = self.pisano_classifier.classify_tuple(
                system.b[j], system.e[j],
                d_vals[j], a_vals[j]
            )
            pisano_families.append(cls['family'])

        dominant_family = Counter(pisano_families).most_common(1)[0][0]

        return {
            'ps_features': ps_features,
            'harmonic_index': final_hi,
            'pisano_family': dominant_family,
            'qa_state': np.column_stack([system.b, system.e])
        }

    def classify_batch(self, waveforms: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Classify batch of waveforms using ensemble of features.

        Decision rules (based on seismological knowledge):
        1. If P/S time ratio > 0.5: likely earthquake
        2. If no S-wave detected: likely explosion
        3. If P/S amplitude ratio > 5: likely explosion
        4. Use Harmonic Index as tiebreaker

        Args:
            waveforms: List of waveform dicts from generator

        Returns:
            predictions (0=explosion, 1=earthquake), feature_matrix
        """
        results = []

        print(f"Classifying {len(waveforms)} waveforms...")
        for i, waveform_data in enumerate(waveforms):
            result = self.classify_waveform(waveform_data['waveform'])
            result['true_type'] = waveform_data['type']
            result['true_label'] = 1 if waveform_data['type'] == 'earthquake' else 0
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(waveforms)}...")

        # Extract feature matrix
        features = []
        for r in results:
            ps = r['ps_features']
            features.append([
                ps['ps_time_ratio'],
                ps['ps_amplitude_ratio'],
                1.0 if ps['has_clear_phases'] else 0.0,
                r['harmonic_index'],
                1.0 if r['pisano_family'] == 'Cosmos' else 0.0
            ])

        features = np.array(features)

        # Make predictions using weighted decision
        predictions = []

        for r in results:
            ps = r['ps_features']
            score = 0.0

            # Rule 1: P/S timing ratio (STRONGEST DISCRIMINATOR)
            if ps['ps_time_ratio'] > 0.5:
                score += 3.0  # Strong evidence for earthquake
            elif ps['ps_time_ratio'] == 0.0:
                score -= 2.0  # No S-wave → explosion

            # Rule 2: P/S amplitude ratio
            if ps['ps_amplitude_ratio'] > 5.0:
                score -= 2.0  # P >> S → explosion
            elif 0.3 < ps['ps_amplitude_ratio'] < 2.0:
                score += 1.0  # Balanced → earthquake

            # Rule 3: Clear phase presence
            if ps['has_clear_phases']:
                score += 1.0  # Earthquake-like
            else:
                score -= 1.0  # Explosion-like

            # Rule 4: QA Harmonic Index (tiebreaker)
            hi_median = np.median([res['harmonic_index'] for res in results])
            if r['harmonic_index'] > hi_median:
                score += 0.5
            else:
                score -= 0.5

            # Final prediction
            predictions.append(1 if score > 0 else 0)

        predictions = np.array(predictions)

        return predictions, results, features


def run_enhanced_validation():
    """
    Run enhanced seismic validation with P/S wave analysis.
    """
    print("="*80)
    print("ENHANCED SEISMIC CLASSIFIER - P/S WAVE TIMING ANALYSIS")
    print("="*80)
    print()

    # Generate synthetic dataset
    print("Generating synthetic seismic dataset...")
    generator = SeismicWaveformGenerator(sample_rate=100)
    dataset = generator.generate_dataset(n_earthquakes=50, n_explosions=50)
    print(f"  ✓ Generated {len(dataset)} waveforms (50 earthquakes, 50 explosions)")
    print()

    # Initialize classifier
    classifier = EnhancedSeismicClassifier(sample_rate=100, modulus=24, num_nodes=24)

    # Classify
    predictions, results, features = classifier.classify_batch(dataset)
    true_labels = np.array([r['true_label'] for r in results])

    # Evaluate
    print()
    print("="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    print()

    accuracy = np.mean(predictions == true_labels)

    # Confusion matrix
    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Overall Accuracy: {accuracy*100:.1f}%")
    print()
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Explosion | Earthquake")
    print(f"  Explosion:     {tn:3d}   |    {fp:3d}")
    print(f"  Earthquake:    {fn:3d}   |    {tp:3d}")
    print()
    print(f"Precision (Earthquake): {precision*100:.1f}%")
    print(f"Recall (Earthquake):    {recall*100:.1f}%")
    print(f"F1-Score:               {f1_score:.3f}")
    print()

    # Analyze P/S features by type
    print("="*80)
    print("P/S WAVE FEATURE ANALYSIS")
    print("="*80)
    print()

    eq_results = [r for r in results if r['true_type'] == 'earthquake']
    ex_results = [r for r in results if r['true_type'] == 'explosion']

    eq_ps_ratios = [r['ps_features']['ps_time_ratio'] for r in eq_results
                    if r['ps_features']['ps_time_ratio'] > 0]
    ex_ps_ratios = [r['ps_features']['ps_time_ratio'] for r in ex_results
                    if r['ps_features']['ps_time_ratio'] > 0]

    print("P/S Timing Ratio:")
    if eq_ps_ratios:
        print(f"  Earthquakes: {np.mean(eq_ps_ratios):.3f} ± {np.std(eq_ps_ratios):.3f}")
    else:
        print(f"  Earthquakes: No clear S-waves detected")

    if ex_ps_ratios:
        print(f"  Explosions:  {np.mean(ex_ps_ratios):.3f} ± {np.std(ex_ps_ratios):.3f}")
    else:
        print(f"  Explosions:  No clear S-waves detected")

    print()

    eq_amp_ratios = [r['ps_features']['ps_amplitude_ratio'] for r in eq_results]
    ex_amp_ratios = [r['ps_features']['ps_amplitude_ratio'] for r in ex_results]

    print("P/S Amplitude Ratio:")
    print(f"  Earthquakes: {np.mean(eq_amp_ratios):.3f} ± {np.std(eq_amp_ratios):.3f}")
    print(f"  Explosions:  {np.mean(ex_amp_ratios):.3f} ± {np.std(ex_amp_ratios):.3f}")
    print()

    # Compute PAC bounds
    print("="*80)
    print("PAC-BAYESIAN ANALYSIS")
    print("="*80)
    print()

    # Use QA states for divergence calculation
    all_qa = np.vstack([r['qa_state'] for r in results])
    correct_indices = np.where(predictions == true_labels)[0]

    if len(correct_indices) > 10:
        prior_samples = all_qa[np.random.choice(len(all_qa), 200, replace=True)]
        correct_qa = np.vstack([results[i]['qa_state'] for i in correct_indices])
        posterior_samples = correct_qa[np.random.choice(len(correct_qa), 200, replace=True)]

        dqa = dqa_divergence(posterior_samples, prior_samples, 24, method='optimal')
        empirical_risk = 1 - accuracy
        pac_bound = pac_generalization_bound(
            empirical_risk=empirical_risk,
            dqa=dqa,
            m=len(dataset),
            constants=classifier.pac_constants,
            delta=0.05
        )

        print(f"D_QA divergence:      {dqa:.3f}")
        print(f"Empirical risk:       {empirical_risk*100:.1f}%")
        print(f"PAC bound (δ=0.05):   {pac_bound*100:.1f}%")
        print(f"Generalization gap:   {(pac_bound - empirical_risk)*100:.1f}%")
        print()

    # Visualization
    workspace = Path("phase2_workspace")
    workspace.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: P/S timing ratio distribution
    ax = axes[0, 0]
    eq_ps = [r['ps_features']['ps_time_ratio'] for r in eq_results]
    ex_ps = [r['ps_features']['ps_time_ratio'] for r in ex_results]

    ax.hist(eq_ps, bins=20, alpha=0.6, label='Earthquake', color='blue', edgecolor='black')
    ax.hist(ex_ps, bins=20, alpha=0.6, label='Explosion', color='red', edgecolor='black')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
    ax.set_xlabel("P/S Timing Ratio", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("P/S Wave Timing Ratio Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: P/S amplitude ratio
    ax = axes[0, 1]
    ax.hist(eq_amp_ratios, bins=20, alpha=0.6, label='Earthquake', color='blue', edgecolor='black')
    ax.hist(ex_amp_ratios, bins=20, alpha=0.6, label='Explosion', color='red', edgecolor='black')
    ax.set_xlabel("P/S Amplitude Ratio", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("P/S Wave Amplitude Ratio Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Feature space (timing vs amplitude)
    ax = axes[1, 0]
    eq_timing = [r['ps_features']['ps_time_ratio'] for r in eq_results]
    eq_amp = [r['ps_features']['ps_amplitude_ratio'] for r in eq_results]
    ex_timing = [r['ps_features']['ps_time_ratio'] for r in ex_results]
    ex_amp = [r['ps_features']['ps_amplitude_ratio'] for r in ex_results]

    ax.scatter(eq_timing, eq_amp, c='blue', alpha=0.6, s=100, label='Earthquake', edgecolors='black')
    ax.scatter(ex_timing, ex_amp, c='red', alpha=0.6, s=100, label='Explosion', edgecolors='black')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(5.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel("P/S Timing Ratio", fontsize=12)
    ax.set_ylabel("P/S Amplitude Ratio", fontsize=12)
    ax.set_title("2D Feature Space (KEY DISCRIMINATORS)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Harmonic Index comparison
    ax = axes[1, 1]
    eq_hi = [r['harmonic_index'] for r in eq_results]
    ex_hi = [r['harmonic_index'] for r in ex_results]

    ax.hist(eq_hi, bins=20, alpha=0.6, label='Earthquake', color='blue', edgecolor='black')
    ax.hist(ex_hi, bins=20, alpha=0.6, label='Explosion', color='red', edgecolor='black')
    ax.set_xlabel("QA Harmonic Index", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("QA Harmonic Index Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = workspace / "enhanced_seismic_classifier_ps_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    print()

    print("="*80)
    print("✓ ENHANCED SEISMIC CLASSIFIER VALIDATION COMPLETE")
    print("="*80)
    print()
    print("Key Improvements:")
    print("  ✓ P-wave arrival detection using STA/LTA")
    print("  ✓ S-wave arrival detection")
    print("  ✓ P/S timing ratio extraction (KEY DISCRIMINATOR)")
    print("  ✓ P/S amplitude ratio analysis")
    print("  ✓ Combined with QA Harmonic Index")
    print()
    print(f"Performance: {accuracy*100:.1f}% accuracy, F1={f1_score:.3f}")
    print()


if __name__ == "__main__":
    run_enhanced_validation()
