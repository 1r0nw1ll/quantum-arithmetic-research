#!/usr/bin/env python3
"""
EEG → 7D Brain Network Feature Extractor (FIXED FOR BIPOLAR MONTAGES)

Maps EEG signals to 7D functional brain network representations:
1. VIS - Visual network
2. SMN - Somatomotor network
3. DAN - Dorsal attention network
4. VAN - Ventral attention network
5. FPN - Frontoparietal network
6. DMN - Default mode network
7. LIM - Limbic network

FIXED: Now handles CHB-MIT bipolar montages (e.g., 'FP1-F7', 'F7-T7', 'P7-O1')
instead of only standard 10-20 names.
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

    FIXED: Handles bipolar montages by parsing channel names and mapping
    individual electrodes to brain networks.
    """

    # Map individual electrodes to brain networks
    ELECTRODE_NETWORK_MAP = {
        # Occipital (Visual)
        'O1': 'VIS', 'O2': 'VIS', 'Oz': 'VIS',

        # Central (Somatomotor)
        'C3': 'SMN', 'C4': 'SMN', 'Cz': 'SMN', 'C5': 'SMN', 'C6': 'SMN',

        # Parietal (Dorsal Attention)
        'P3': 'DAN', 'P4': 'DAN', 'Pz': 'DAN', 'P7': 'DAN', 'P8': 'DAN',

        # Temporal (Ventral Attention)
        'T3': 'VAN', 'T4': 'VAN', 'T5': 'VAN', 'T6': 'VAN',
        'T7': 'VAN', 'T8': 'VAN',  # Alternative naming

        # Frontal (Frontoparietal/Executive)
        'F3': 'FPN', 'F4': 'FPN', 'Fz': 'FPN',

        # Prefrontal (Default Mode)
        'Fp1': 'DMN', 'Fp2': 'DMN', 'FP1': 'DMN', 'FP2': 'DMN',  # Both cases

        # Temporal-frontal (Limbic)
        'F7': 'LIM', 'F8': 'LIM'
    }

    def __init__(self, sample_rate: int = 256):
        """
        Args:
            sample_rate: EEG sampling rate in Hz (256 Hz standard)
        """
        self.sample_rate = sample_rate
        self.network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM']

    def parse_bipolar_channel(self, channel_name: str) -> List[str]:
        """
        Parse bipolar channel name into individual electrodes.

        Examples:
            'FP1-F7' → ['FP1', 'F7']
            'P7-O1' → ['P7', 'O1']
            'C3' → ['C3'] (monopolar)

        Args:
            channel_name: EEG channel name

        Returns:
            List of electrode names
        """
        # Remove spaces
        channel_name = channel_name.strip()

        # Check if bipolar (contains hyphen or underscore)
        if '-' in channel_name:
            electrodes = channel_name.split('-')
            return [e.strip() for e in electrodes]
        elif '_' in channel_name:
            electrodes = channel_name.split('_')
            return [e.strip() for e in electrodes]
        else:
            # Monopolar or unrecognized format
            return [channel_name]

    def map_channel_to_networks(self, channel_name: str) -> List[str]:
        """
        Map a channel (bipolar or monopolar) to brain network(s).

        For bipolar channels, we consider the network(s) involved in
        the differential measurement.

        Args:
            channel_name: EEG channel name

        Returns:
            List of network names this channel contributes to
        """
        electrodes = self.parse_bipolar_channel(channel_name)
        networks = set()

        for electrode in electrodes:
            # Normalize electrode name (handle case variations)
            electrode_upper = electrode.upper()

            # Try direct lookup
            if electrode_upper in self.ELECTRODE_NETWORK_MAP:
                networks.add(self.ELECTRODE_NETWORK_MAP[electrode_upper])
            elif electrode in self.ELECTRODE_NETWORK_MAP:
                networks.add(self.ELECTRODE_NETWORK_MAP[electrode])

        return list(networks)

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

        FIXED: Now handles bipolar montages by parsing channel names and
        mapping electrodes to networks.

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

        # Build network → channels mapping
        network_channels = {net: [] for net in self.network_names}

        for channel_name in channels_data.keys():
            networks = self.map_channel_to_networks(channel_name)
            for net in networks:
                if net in network_channels:
                    network_channels[net].append(channel_name)

        # Extract features for each network
        for i, network in enumerate(self.network_names):
            channels = network_channels[network]

            if len(channels) == 0:
                # No channels for this network - feature remains 0
                continue

            # Aggregate activity across network channels
            network_activity = 0.0
            n_channels = 0

            for ch in channels:
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

        # Use standard 10-20 names for synthetic data
        standard_channels = ['O1', 'O2', 'C3', 'C4', 'P3', 'P4',
                           'T3', 'T4', 'F3', 'F4', 'Fp1', 'Fp2', 'F7', 'F8']

        eeg_data = {}

        for ch in standard_channels:
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


if __name__ == "__main__":
    print("="*80)
    print("EEG → 7D BRAIN NETWORK FEATURE EXTRACTION TEST (FIXED VERSION)")
    print("="*80)
    print()

    extractor = EEGBrainFeatureExtractor(sample_rate=256)

    # Test bipolar channel parsing
    print("Testing bipolar channel parsing:")
    test_channels = ['FP1-F7', 'F7-T7', 'P7-O1', 'F3-C3', 'C3-P3', 'O1']
    for ch in test_channels:
        electrodes = extractor.parse_bipolar_channel(ch)
        networks = extractor.map_channel_to_networks(ch)
        print(f"  {ch:12s} → electrodes: {electrodes} → networks: {networks}")
    print()

    # Test with synthetic data
    print("Testing feature extraction on synthetic data...")
    normal_eeg = extractor.generate_synthetic_eeg('normal', duration=4.0)
    seizure_eeg = extractor.generate_synthetic_eeg('ictal', duration=4.0)

    normal_features = extractor.extract_network_features(normal_eeg)
    seizure_features = extractor.extract_network_features(seizure_eeg)

    print("  Normal features:  ", [f"{f:.3f}" for f in normal_features])
    print("  Seizure features: ", [f"{f:.3f}" for f in seizure_features])
    print("  Difference:       ", [f"{f:.3f}" for f in (seizure_features - normal_features)])
    print()

    print("="*80)
    print("✓ FIXED feature extractor ready for real CHB-MIT data!")
    print("="*80)
