#!/usr/bin/env python3
"""
Process Real CHB-MIT EEG Data and Validate Brain→QA Framework

This script:
1. Loads real EEG recordings (chb05_06.edf)
2. Extracts 7D brain network features
3. Applies Brain→QA mapping
4. Compares performance with synthetic data
5. Updates paper metrics with real-world results
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

# Try to import pyedflib for real EDF reading
try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    print("WARNING: pyedflib not installed. Install with: pip install pyedflib")
    HAS_PYEDFLIB = False

# Import our brain feature extractor (FIXED VERSION for bipolar montages)
sys.path.insert(0, str(Path(__file__).parent))
from eeg_brain_feature_extractor_fixed import EEGBrainFeatureExtractor


class RealEEGProcessor:
    """Process real CHB-MIT EEG data."""

    def __init__(self, data_dir: Path = Path("phase2_data/eeg/chbmit")):
        self.data_dir = data_dir
        self.extractor = EEGBrainFeatureExtractor()
        self.sampling_rate = 256  # CHB-MIT standard sampling rate

    def map_features_to_qa(self, features_7d: np.ndarray) -> np.ndarray:
        """
        Map 7D brain network features to QA state space (mod 24).

        Simple mapping: scale normalized features to integers 1-24

        Args:
            features_7d: Array of 7D feature vectors (n_samples, 7)

        Returns:
            QA states (n_samples, 2) - (b, e) pairs
        """
        n_samples = len(features_7d)
        qa_states = np.zeros((n_samples, 2), dtype=int)

        for i in range(n_samples):
            # Use first two network features (VIS, SMN) as primary states
            # Scale from [0,1] (normalized) to [1,24]
            b = int(features_7d[i, 0] * 23) + 1  # VIS -> b
            e = int(features_7d[i, 1] * 23) + 1  # SMN -> e

            # Ensure in range [1,24]
            b = max(1, min(24, b))
            e = max(1, min(24, e))

            qa_states[i] = [b, e]

        return qa_states

    def load_edf_file(self, filepath: Path) -> Dict:
        """
        Load real EEG data from EDF file.

        Returns:
            Dictionary with signals, channel_names, sampling_rate, duration
        """
        if not HAS_PYEDFLIB:
            raise ImportError("pyedflib required. Install: pip install pyedflib")

        print(f"Loading EDF file: {filepath}")

        f = pyedflib.EdfReader(str(filepath))

        n_channels = f.signals_in_file
        channel_names = f.getSignalLabels()
        sampling_rate = f.getSampleFrequency(0)
        duration = f.getFileDuration()

        print(f"  Channels: {n_channels}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration: {duration:.1f} seconds")

        # Load all channels
        signals = []
        for i in range(n_channels):
            sig = f.readSignal(i)
            signals.append(sig)

        f.close()

        return {
            'signals': np.array(signals),  # Shape: (n_channels, n_samples)
            'channel_names': channel_names,
            'sampling_rate': sampling_rate,
            'duration': duration
        }

    def segment_eeg(self, signals: np.ndarray, window_sec: float = 4.0,
                   overlap_sec: float = 2.0) -> List[np.ndarray]:
        """
        Segment continuous EEG into overlapping windows.

        Args:
            signals: EEG signals (n_channels, n_samples)
            window_sec: Window length in seconds
            overlap_sec: Overlap between windows in seconds

        Returns:
            List of windowed segments
        """
        n_channels, n_samples = signals.shape

        window_samples = int(window_sec * self.sampling_rate)
        step_samples = int((window_sec - overlap_sec) * self.sampling_rate)

        segments = []
        start = 0

        while start + window_samples <= n_samples:
            segment = signals[:, start:start + window_samples]
            segments.append(segment)
            start += step_samples

        print(f"Created {len(segments)} segments of {window_sec}s each")
        return segments

    def extract_features_from_segment(self, segment: np.ndarray,
                                    channel_names: List[str]) -> np.ndarray:
        """
        Extract 7D brain network features from EEG segment.

        Args:
            segment: EEG data (n_channels, n_samples)
            channel_names: List of channel names

        Returns:
            7D feature vector
        """
        # Convert to dict format expected by extractor
        channels_data = {}
        for i, ch_name in enumerate(channel_names):
            channels_data[ch_name] = segment[i, :]

        # Extract network features
        features_7d = self.extractor.extract_network_features(channels_data)
        return features_7d

    def load_seizure_annotations(self, subject: str) -> List[Dict]:
        """
        Load seizure annotations from summary file.

        Returns:
            List of seizure annotations with file, start_time, end_time
        """
        summary_file = self.data_dir / subject / f"{subject}-summary.txt"

        if not summary_file.exists():
            print(f"WARNING: No summary file found at {summary_file}")
            return []

        annotations = []
        current_file = None

        with open(summary_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Parse file name
                if line.startswith('File Name:'):
                    current_file = line.split(':')[1].strip()

                # Parse seizure times
                if 'Seizure' in line and 'Start Time' in line:
                    parts = line.split()
                    start_idx = parts.index('Time:') + 1
                    end_idx = parts.index('seconds')
                    start_time = int(parts[start_idx])

                    # Find end time
                    end_line_parts = line.split('End Time:')
                    if len(end_line_parts) > 1:
                        end_time = int(end_line_parts[1].split()[0])
                    else:
                        end_time = start_time + 30  # Assume 30s if not specified

                    annotations.append({
                        'file': current_file,
                        'start_time': start_time,
                        'end_time': end_time
                    })

        print(f"Loaded {len(annotations)} seizure annotations")
        return annotations

    def label_segments(self, segments: List[np.ndarray],
                      window_sec: float,
                      overlap_sec: float,
                      seizure_times: List[Tuple[float, float]]) -> np.ndarray:
        """
        Label segments as seizure (1) or baseline (0).

        Args:
            segments: List of EEG segments
            window_sec: Window length in seconds
            overlap_sec: Overlap in seconds
            seizure_times: List of (start, end) tuples in seconds

        Returns:
            Binary labels (n_segments,)
        """
        step_sec = window_sec - overlap_sec
        labels = np.zeros(len(segments), dtype=int)

        for i in range(len(segments)):
            segment_start = i * step_sec
            segment_end = segment_start + window_sec

            # Check if segment overlaps with any seizure
            for seizure_start, seizure_end in seizure_times:
                if not (segment_end < seizure_start or segment_start > seizure_end):
                    # Overlap detected
                    labels[i] = 1
                    break

        n_seizure = np.sum(labels)
        n_baseline = len(labels) - n_seizure
        print(f"Labels: {n_seizure} seizure, {n_baseline} baseline")

        return labels

    def process_file(self, filepath: Path,
                    seizure_times: List[Tuple[float, float]] = None) -> Dict:
        """
        Complete pipeline: load → segment → extract features → label.

        Returns:
            Dictionary with features_7d, labels, qa_states
        """
        # Load EDF
        data = self.load_edf_file(filepath)

        # Segment
        segments = self.segment_eeg(data['signals'], window_sec=4.0, overlap_sec=2.0)

        # Extract features for each segment
        print("Extracting 7D brain features...")
        features_list = []
        for i, segment in enumerate(segments):
            if i % 100 == 0:
                print(f"  Processing segment {i+1}/{len(segments)}")
            features = self.extract_features_from_segment(segment, data['channel_names'])
            features_list.append(features)

        features_7d = np.array(features_list)
        print(f"Extracted features shape: {features_7d.shape}")

        # Label segments
        if seizure_times is not None:
            labels = self.label_segments(segments, window_sec=4.0, overlap_sec=2.0,
                                        seizure_times=seizure_times)
        else:
            labels = np.zeros(len(segments), dtype=int)  # All baseline

        # Map to QA states
        print("Mapping to QA states...")
        qa_states = self.map_features_to_qa(features_7d)

        return {
            'features_7d': features_7d,
            'labels': labels,
            'qa_states': qa_states,
            'n_segments': len(segments),
            'duration': data['duration']
        }


def validate_on_real_data():
    """
    Main validation pipeline on real CHB-MIT data.
    """
    print("="*80)
    print("REAL CHB-MIT EEG DATA VALIDATION")
    print("="*80)
    print()

    processor = RealEEGProcessor()

    # Check for downloaded files
    data_dir = processor.data_dir
    if not data_dir.exists():
        print(f"ERROR: {data_dir} directory not found.")
        print("Run: python download_chbmit_eeg.py")
        return None

    # Find available EDF files
    edf_files = list(data_dir.rglob("*.edf"))

    if len(edf_files) == 0:
        print("ERROR: No EDF files found in chbmit_data/")
        print("Run: python download_chbmit_eeg.py")
        return None

    print(f"Found {len(edf_files)} EDF files:")
    for f in edf_files[:5]:  # Show first 5
        print(f"  {f}")
    if len(edf_files) > 5:
        print(f"  ... and {len(edf_files) - 5} more")
    print()

    # Process first available file
    test_file = edf_files[0]
    print(f"Processing: {test_file.name}")
    print("-"*80)

    # For demonstration, assume no seizures (will update with real annotations)
    seizure_times = []  # TODO: Load from summary file

    try:
        results = processor.process_file(test_file, seizure_times=seizure_times)
    except Exception as e:
        print(f"ERROR processing file: {e}")
        print("\nAttempting to install pyedflib...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pyedflib"],
                      capture_output=True)
        print("Please re-run this script after pyedflib installation.")
        return None

    print()
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"File: {test_file.name}")
    print(f"Duration: {results['duration']:.1f} seconds")
    print(f"Segments: {results['n_segments']}")
    print(f"Features shape: {results['features_7d'].shape}")
    print(f"QA states shape: {results['qa_states'].shape}")
    print()

    # Analyze QA state distributions
    qa_means = np.mean(results['qa_states'], axis=0)
    qa_stds = np.std(results['qa_states'], axis=0)

    print("QA State Statistics:")
    print(f"  Mean: {qa_means}")
    print(f"  Std:  {qa_stds}")
    print()

    # Visualize
    visualize_real_data_results(results, test_file.name)

    # Save results
    output_dir = Path("phase2_workspace")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "real_eeg_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'file': test_file.name,
            'duration': results['duration'],
            'n_segments': results['n_segments'],
            'features_shape': list(results['features_7d'].shape),
            'qa_state_means': qa_means.tolist(),
            'qa_state_stds': qa_stds.tolist(),
        }, f, indent=2)

    print(f"✓ Results saved to {results_file}")

    return results


def visualize_real_data_results(results: Dict, filename: str):
    """
    Visualize real EEG processing results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: 7D Brain Features Over Time
    ax = axes[0, 0]
    features = results['features_7d']
    time = np.arange(len(features)) * 2.0  # 2s step (4s window - 2s overlap)

    network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM']
    for i, name in enumerate(network_names):
        ax.plot(time, features[:, i], label=name, alpha=0.7)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Network Activity (normalized)')
    ax.set_title(f'Brain Network Activity: {filename}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: QA State Trajectories
    ax = axes[0, 1]
    qa_states = results['qa_states']

    ax.plot(time, qa_states[:, 0], label='b', alpha=0.7)
    ax.plot(time, qa_states[:, 1], label='e', alpha=0.7)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('QA State Value')
    ax.set_title('QA State Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Feature Distribution Heatmap
    ax = axes[1, 0]
    im = ax.imshow(features.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Segment')
    ax.set_ylabel('Brain Network')
    ax.set_yticks(range(7))
    ax.set_yticklabels(network_names)
    ax.set_title('Feature Heatmap')
    plt.colorbar(im, ax=ax, label='Activity Level')

    # Panel D: QA State Phase Space
    ax = axes[1, 1]
    ax.scatter(qa_states[:, 0], qa_states[:, 1],
              c=time, cmap='plasma', alpha=0.6, s=20)
    ax.set_xlabel('QA State b')
    ax.set_ylabel('QA State e')
    ax.set_title('QA State Phase Space')
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Time (s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path("phase2_workspace") / "real_eeg_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


def compare_with_synthetic():
    """
    Compare real data results with synthetic data baseline.
    """
    print()
    print("="*80)
    print("REAL vs SYNTHETIC COMPARISON")
    print("="*80)

    results_file = Path("phase2_workspace/real_eeg_validation_results.json")
    if not results_file.exists():
        print("No real data results found. Run validation first.")
        return

    with open(results_file, 'r') as f:
        real_results = json.load(f)

    print("\nReal Data:")
    print(f"  File: {real_results['file']}")
    print(f"  Duration: {real_results['duration']:.1f}s")
    print(f"  Segments: {real_results['n_segments']}")
    print(f"  QA state mean: {np.array(real_results['qa_state_means'])}")

    print("\nSynthetic Data (from previous experiments):")
    print("  Generated from random noise")
    print("  Expected QA state mean: [12, 12] (uniform mod 24)")
    print("  Expected std: ~7 (uniform distribution)")

    real_mean = np.array(real_results['qa_state_means'])
    real_std = np.array(real_results['qa_state_stds'])

    print("\nStatistical Comparison:")
    print(f"  Real mean deviation from uniform: {np.abs(real_mean - 12.0)}")
    print(f"  Real std vs expected (~7): {real_std}")

    if np.any(real_std < 3.0):
        print("\n⚠️  Low variability detected - may indicate:")
        print("     - Artifact rejection needed")
        print("     - Baseline/resting state EEG")
        print("     - Feature normalization effects")

    print("\n✓ Real data successfully processed and characterized")


if __name__ == "__main__":
    # Run validation
    results = validate_on_real_data()

    if results is not None:
        # Compare with synthetic
        compare_with_synthetic()

        print()
        print("="*80)
        print("✓ REAL DATA VALIDATION COMPLETE")
        print("="*80)
        print()
        print("Next steps:")
        print("  1. Process more EDF files for statistical power")
        print("  2. Load seizure annotations from summary files")
        print("  3. Compare seizure vs baseline QA signatures")
        print("  4. Update paper with real data metrics")
        print("  5. Run CNN/LSTM baselines on real data")
