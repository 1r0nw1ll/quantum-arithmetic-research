#!/usr/bin/env python3
"""
EEG → 7D Brain Network Feature Extractor

Maps EEG signals to 7D functional brain network representations:
1. VIS - Visual network
2. SMN - Somatomotor network
3. DAN - Dorsal attention network
4. VAN - Ventral attention network
5. FPN - Frontoparietal network
6. DMN - Default mode network
7. LIM - Limbic network

For Phase 2 validation: Seizure detection with Brain→QA mapping.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import signal
from scipy.stats import entropy

class EEGBrainFeatureExtractor:
    """
    Extract 7D brain-like features from EEG for QA mapping.

    Maps multi-channel EEG activity to functional brain networks
    based on spectral and connectivity patterns.
    """

    # Standard 10-20 EEG channel groups mapped to brain networks
    CHANNEL_NETWORK_MAP = {
        'VIS': ['O1', 'O2', 'Oz'],  # Occipital (visual)
        'SMN': ['C3', 'C4', 'Cz'],  # Central (motor)
        'DAN': ['P3', 'P4', 'Pz'],  # Parietal (attention)
        'VAN': ['T3', 'T4', 'T5', 'T6'],  # Temporal (attention)
        'FPN': ['F3', 'F4', 'Fz'],  # Frontal (executive)
        'DMN': ['Fp1', 'Fp2'],  # Prefrontal (default mode)
        'LIM': ['F7', 'F8']  # Temporal-frontal (limbic)
    }

    def __init__(self, sample_rate: int = 256):
        """
        Args:
            sample_rate: EEG sampling rate in Hz (256 Hz standard)
        """
        self.sample_rate = sample_rate
        self.network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM']

    def extract_band_power(self, eeg_signal: np.ndarray,
                          band: Tuple[float, float]) -> float:
        """
        Extract power in specific frequency band.

        Args:
            eeg_signal: Single channel EEG (1D array)
            band: Frequency band (low, high) in Hz

        Returns:
            Relative power in band
        """
        # Compute power spectral density
        freqs, psd = signal.welch(eeg_signal, fs=self.sample_rate,
                                  nperseg=min(256, len(eeg_signal)))

        # Find indices for band
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])

        # Compute band power
        band_power = np.trapz(psd[idx], freqs[idx])
        total_power = np.trapz(psd, freqs)

        return band_power / (total_power + 1e-9)

    def extract_network_features(self, channels_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract 7D brain network features from multi-channel EEG.

        Each network's activity is characterized by:
        - Alpha band power (8-13 Hz): Baseline/resting state
        - Beta band power (13-30 Hz): Active processing
        - Gamma band power (30-100 Hz): High-level integration

        Args:
            channels_data: Dict mapping channel names to signals

        Returns:
            7D feature vector (one value per network)
        """
        features = np.zeros(7)

        for i, network in enumerate(self.network_names):
            # Get channels for this network
            network_channels = self.CHANNEL_NETWORK_MAP.get(network, [])

            # Aggregate activity across network channels
            network_activity = 0.0
            n_channels = 0

            for ch in network_channels:
                if ch in channels_data:
                    # Multi-band characterization
                    alpha = self.extract_band_power(channels_data[ch], (8, 13))
                    beta = self.extract_band_power(channels_data[ch], (13, 30))
                    gamma = self.extract_band_power(channels_data[ch], (30, 50))

                    # Combine bands (weighted by typical network signatures)
                    if network == 'VIS':
                        # Visual: High alpha modulation
                        activity = 2.0 * alpha + beta + 0.5 * gamma
                    elif network == 'SMN':
                        # Motor: Strong beta (mu rhythm)
                        activity = alpha + 2.0 * beta + gamma
                    elif network in ['DAN', 'VAN']:
                        # Attention: Balanced alpha/beta
                        activity = 1.5 * alpha + 1.5 * beta + gamma
                    elif network == 'FPN':
                        # Executive: High gamma coherence
                        activity = alpha + beta + 2.0 * gamma
                    elif network == 'DMN':
                        # Default mode: High alpha (anti-correlated with task)
                        activity = 2.5 * alpha + 0.5 * beta + 0.5 * gamma
                    elif network == 'LIM':
                        # Limbic: Theta/alpha dominant
                        theta = self.extract_band_power(channels_data[ch], (4, 8))
                        activity = theta + 1.5 * alpha + beta

                    network_activity += activity
                    n_channels += 1

            # Average across channels
            if n_channels > 0:
                features[i] = network_activity / n_channels

        # Normalize to unit sphere (important for Brain→QA mapping)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def generate_synthetic_eeg(self, state: str = 'normal',
                               duration: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Generate synthetic EEG data for testing.

        Args:
            state: 'normal', 'pre_ictal', or 'ictal' (seizure)
            duration: Duration in seconds

        Returns:
            Dict mapping channel names to signals
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        # Collect all unique channels
        all_channels = set()
        for channels in self.CHANNEL_NETWORK_MAP.values():
            all_channels.update(channels)

        eeg_data = {}

        for ch in all_channels:
            # Base signal: Mixture of physiological rhythms
            if state == 'normal':
                # Normal EEG: Dominant alpha, some beta
                alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
                beta = 0.2 * np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
                theta = 0.1 * np.sin(2 * np.pi * 6 * t)   # 6 Hz theta
                noise = 0.1 * np.random.randn(n_samples)

            elif state == 'pre_ictal':
                # Pre-seizure: Increased beta/gamma, reduced alpha
                alpha = 0.2 * np.sin(2 * np.pi * 10 * t)
                beta = 0.4 * np.sin(2 * np.pi * 20 * t)
                gamma = 0.3 * np.sin(2 * np.pi * 40 * t)  # 40 Hz gamma
                theta = 0.2 * np.sin(2 * np.pi * 6 * t)
                noise = 0.15 * np.random.randn(n_samples)

            else:  # ictal
                # Seizure: High amplitude spike-wave patterns
                spike_freq = 3.0  # 3 Hz spike-wave (typical for absence seizures)
                spikes = 2.0 * np.sin(2 * np.pi * spike_freq * t)
                # Add harmonics
                spikes += 0.5 * np.sin(2 * np.pi * spike_freq * 2 * t)
                alpha = 0.1 * np.sin(2 * np.pi * 10 * t)
                noise = 0.2 * np.random.randn(n_samples)

                eeg_data[ch] = spikes + alpha + noise
                continue

            # Combine for normal/pre-ictal
            if state == 'pre_ictal':
                eeg_data[ch] = alpha + beta + gamma + theta + noise
            else:
                eeg_data[ch] = alpha + beta + theta + noise

        return eeg_data

    def generate_seizure_sequence(self) -> List[Dict]:
        """
        Generate a sequence: baseline → pre-ictal → ictal → post-ictal.

        Returns:
            List of (features, label) tuples
        """
        sequence = []

        # Baseline (normal) - 30 seconds
        for i in range(3):
            eeg = self.generate_synthetic_eeg('normal', duration=10.0)
            features = self.extract_network_features(eeg)
            sequence.append({
                'features': features,
                'label': 'normal',
                'time': i * 10
            })

        # Pre-ictal (warning signs) - 20 seconds
        for i in range(2):
            eeg = self.generate_synthetic_eeg('pre_ictal', duration=10.0)
            features = self.extract_network_features(eeg)
            sequence.append({
                'features': features,
                'label': 'pre_ictal',
                'time': 30 + i * 10
            })

        # Ictal (seizure) - 30 seconds
        for i in range(3):
            eeg = self.generate_synthetic_eeg('ictal', duration=10.0)
            features = self.extract_network_features(eeg)
            sequence.append({
                'features': features,
                'label': 'ictal',
                'time': 50 + i * 10
            })

        return sequence


if __name__ == "__main__":
    print("="*80)
    print("EEG → 7D BRAIN NETWORK FEATURE EXTRACTION TEST")
    print("="*80)
    print()

    extractor = EEGBrainFeatureExtractor(sample_rate=256)

    # Generate seizure sequence
    print("Generating synthetic seizure sequence...")
    print("  Phase 1: Baseline (normal) - 30s")
    print("  Phase 2: Pre-ictal (warning) - 20s")
    print("  Phase 3: Ictal (seizure) - 30s")
    print()

    sequence = extractor.generate_seizure_sequence()
    print(f"  ✓ Generated {len(sequence)} 10-second epochs")
    print()

    # Analyze features
    print("7D Brain Network Features:")
    print("  Networks: VIS | SMN | DAN | VAN | FPN | DMN | LIM")
    print("-" * 80)

    for i, epoch in enumerate(sequence):
        features = epoch['features']
        label = epoch['label']
        print(f"  Epoch {i:2d} ({label:>10s}): " +
              " | ".join([f"{f:.3f}" for f in features]))
    print()

    # Visualize feature evolution
    workspace = Path("phase2_workspace")
    workspace.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Feature evolution over time
    ax = axes[0]
    times = [ep['time'] for ep in sequence]
    features_matrix = np.array([ep['features'] for ep in sequence])

    for i, network in enumerate(extractor.network_names):
        ax.plot(times, features_matrix[:, i], '-o', label=network, linewidth=2)

    # Mark phases
    ax.axvspan(0, 30, alpha=0.2, color='green', label='Normal')
    ax.axvspan(30, 50, alpha=0.2, color='yellow', label='Pre-ictal')
    ax.axvspan(50, 80, alpha=0.2, color='red', label='Ictal')

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Network Activity (normalized)", fontsize=12)
    ax.set_title("7D Brain Network Evolution During Seizure Sequence",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Feature space trajectory (2D projection)
    ax = axes[1]
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_matrix)

    # Color by state
    colors = {'normal': 'green', 'pre_ictal': 'yellow', 'ictal': 'red'}
    for i, epoch in enumerate(sequence):
        ax.scatter(features_2d[i, 0], features_2d[i, 1],
                  c=colors[epoch['label']], s=150, alpha=0.7,
                  edgecolors='black', linewidth=1.5)
        ax.annotate(f"{i}", (features_2d[i, 0], features_2d[i, 1]),
                   ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw trajectory
    ax.plot(features_2d[:, 0], features_2d[:, 1], 'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("7D Brain Network Trajectory (PCA Projection)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Normal'),
        Patch(facecolor='yellow', alpha=0.7, label='Pre-ictal'),
        Patch(facecolor='red', alpha=0.7, label='Ictal')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()
    output_path = workspace / "eeg_7d_brain_features.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved to {output_path}")
    print()

    # Test with Brain→QA mapper
    print("Testing Brain→QA mapping integration...")
    try:
        from brain_qa_mapper import BrainQAMapper

        mapper = BrainQAMapper(modulus=24)

        # Fit mapper on all features
        mapper.fit(features_matrix)

        # Map first and last epoch
        normal_qa = mapper.map_to_qa_tuple(features_matrix[0])
        ictal_qa = mapper.map_to_qa_tuple(features_matrix[-1])

        print(f"  ✓ Brain→QA mapper initialized")
        print(f"    Normal state → Sector {normal_qa['sector']}, Magnitude {normal_qa['magnitude']:.3f}")
        print(f"    Ictal state  → Sector {ictal_qa['sector']}, Magnitude {ictal_qa['magnitude']:.3f}")
        print(f"    Sector shift: {abs(ictal_qa['sector'] - normal_qa['sector'])} (mod-24)")
        print()

    except ImportError:
        print("  ⚠ brain_qa_mapper not found (expected if testing standalone)")
        print()

    print("="*80)
    print("✓ EEG feature extractor ready for Phase 2 validation!")
    print("="*80)
