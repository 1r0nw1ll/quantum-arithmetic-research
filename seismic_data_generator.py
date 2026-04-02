#!/usr/bin/env python3
"""
Synthetic Seismic Waveform Generator

Generates realistic synthetic seismic data for testing the Phase 2 framework.
Mimics real earthquake and explosion waveforms with realistic characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

class SeismicWaveformGenerator:
    """
    Generate synthetic seismic waveforms with realistic characteristics.

    Earthquake waveforms:
    - Emergent P-wave arrival
    - Clear S-wave arrival (slower than P)
    - Long duration (30-120 seconds)
    - Rich frequency content (0.5-10 Hz)
    - Complex coda

    Explosion waveforms:
    - Impulsive P-wave arrival
    - Weak or absent S-wave
    - Short duration (10-30 seconds)
    - Higher frequency content (2-20 Hz)
    - Simple coda
    """

    def __init__(self, sample_rate: int = 100):
        """
        Args:
            sample_rate: Sampling rate in Hz (100 Hz typical for seismology)
        """
        self.sample_rate = sample_rate

    def generate_p_wave(self, duration: float, onset_style: str = 'emergent',
                       amplitude: float = 1.0) -> np.ndarray:
        """
        Generate P-wave (primary/compressional wave).

        Args:
            duration: Duration in seconds
            onset_style: 'emergent' (earthquake) or 'impulsive' (explosion)
            amplitude: Peak amplitude

        Returns:
            P-wave time series
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        # Base frequency for P-waves (6-8 Hz typical)
        f_p = 7.0

        # Generate waveform
        signal = amplitude * np.sin(2 * np.pi * f_p * t)

        # Apply envelope based on onset style
        if onset_style == 'emergent':
            # Gradual buildup (earthquake characteristic)
            envelope = 1 - np.exp(-t / 0.5)
            envelope *= np.exp(-t / (duration * 0.8))
        else:  # impulsive
            # Sharp onset (explosion characteristic)
            envelope = np.exp(-t / (duration * 0.3))

        return signal * envelope

    def generate_s_wave(self, duration: float, amplitude: float = 1.5) -> np.ndarray:
        """
        Generate S-wave (secondary/shear wave).

        Args:
            duration: Duration in seconds
            amplitude: Peak amplitude (typically 1.5-2x P-wave)

        Returns:
            S-wave time series
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        # Base frequency for S-waves (3-5 Hz, slower than P)
        f_s = 4.0

        # Generate waveform with richer harmonic content
        signal = amplitude * (np.sin(2 * np.pi * f_s * t) +
                             0.3 * np.sin(2 * np.pi * f_s * 2 * t))

        # Envelope
        envelope = 1 - np.exp(-t / 0.8)
        envelope *= np.exp(-t / (duration * 0.6))

        return signal * envelope

    def generate_coda(self, duration: float, complexity: str = 'complex') -> np.ndarray:
        """
        Generate coda (tail of seismic signal).

        Args:
            duration: Duration in seconds
            complexity: 'simple' (explosion) or 'complex' (earthquake)

        Returns:
            Coda time series
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        if complexity == 'complex':
            # Multiple scattering paths (earthquake)
            signal = np.zeros(n_samples)
            for i, (f, a) in enumerate([(2, 0.4), (3, 0.3), (5, 0.2), (7, 0.1)]):
                signal += a * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi)
        else:  # simple
            # Single dominant frequency (explosion)
            signal = 0.2 * np.sin(2 * np.pi * 3 * t)

        # Exponential decay
        envelope = np.exp(-t / (duration * 0.5))

        return signal * envelope

    def generate_earthquake(self, magnitude: float = 5.0,
                           distance_km: float = 100.0) -> Dict:
        """
        Generate synthetic earthquake waveform.

        Args:
            magnitude: Earthquake magnitude (3-7)
            distance_km: Distance from station in km (10-1000)

        Returns:
            Dictionary with waveform and metadata
        """
        # Timing based on distance
        p_arrival_time = distance_km / 6.0  # P-wave velocity ~6 km/s
        s_arrival_time = distance_km / 3.5  # S-wave velocity ~3.5 km/s

        # Durations based on magnitude
        p_duration = 5 + magnitude
        s_duration = 10 + 2 * magnitude
        coda_duration = 20 + 5 * magnitude

        total_duration = p_arrival_time + s_arrival_time + s_duration + coda_duration
        n_samples = int(total_duration * self.sample_rate)

        # Initialize waveform
        waveform = np.zeros(n_samples)

        # Add P-wave
        p_start = int(p_arrival_time * self.sample_rate)
        p_wave = self.generate_p_wave(p_duration, 'emergent',
                                      amplitude=magnitude / 10.0)
        p_end = min(p_start + len(p_wave), n_samples)
        waveform[p_start:p_end] += p_wave[:p_end-p_start]

        # Add S-wave
        s_start = int((p_arrival_time + s_arrival_time) * self.sample_rate)
        s_wave = self.generate_s_wave(s_duration, amplitude=magnitude / 5.0)
        s_end = min(s_start + len(s_wave), n_samples)
        waveform[s_start:s_end] += s_wave[:s_end-s_start]

        # Add coda
        coda_start = s_end
        if coda_start < n_samples:
            coda = self.generate_coda(coda_duration, 'complex')
            coda_end = min(coda_start + len(coda), n_samples)
            waveform[coda_start:coda_end] += coda[:coda_end-coda_start]

        # Add background noise
        noise_level = 0.05 * magnitude / 10.0
        waveform += noise_level * np.random.randn(n_samples)

        return {
            'waveform': waveform,
            'sample_rate': self.sample_rate,
            'type': 'earthquake',
            'magnitude': magnitude,
            'distance_km': distance_km,
            'p_arrival': p_arrival_time,
            's_arrival': p_arrival_time + s_arrival_time,
            'duration': total_duration
        }

    def generate_explosion(self, yield_kt: float = 10.0,
                          distance_km: float = 100.0) -> Dict:
        """
        Generate synthetic explosion waveform.

        Args:
            yield_kt: Explosion yield in kilotons (1-100)
            distance_km: Distance from station in km (10-1000)

        Returns:
            Dictionary with waveform and metadata
        """
        # Timing
        p_arrival_time = distance_km / 6.0

        # Durations (shorter than earthquakes)
        p_duration = 3 + np.log10(yield_kt)
        coda_duration = 10 + 2 * np.log10(yield_kt)

        total_duration = p_arrival_time + p_duration + coda_duration
        n_samples = int(total_duration * self.sample_rate)

        # Initialize waveform
        waveform = np.zeros(n_samples)

        # Add P-wave (impulsive)
        p_start = int(p_arrival_time * self.sample_rate)
        p_wave = self.generate_p_wave(p_duration, 'impulsive',
                                      amplitude=np.log10(yield_kt) / 2.0)
        p_end = min(p_start + len(p_wave), n_samples)
        waveform[p_start:p_end] += p_wave[:p_end-p_start]

        # Add weak or absent S-wave (characteristic of explosions)
        # Only add very weak S-wave
        s_start = int((p_arrival_time + distance_km / 3.5) * self.sample_rate)
        if s_start < n_samples:
            s_duration = 2
            s_wave = self.generate_s_wave(s_duration,
                                          amplitude=0.2 * np.log10(yield_kt) / 2.0)
            s_end = min(s_start + len(s_wave), n_samples)
            waveform[s_start:s_end] += s_wave[:s_end-s_start]

        # Add simple coda
        coda_start = p_end
        if coda_start < n_samples:
            coda = self.generate_coda(coda_duration, 'simple')
            coda_end = min(coda_start + len(coda), n_samples)
            waveform[coda_start:coda_end] += coda[:coda_end-coda_start]

        # Add background noise
        noise_level = 0.03
        waveform += noise_level * np.random.randn(n_samples)

        return {
            'waveform': waveform,
            'sample_rate': self.sample_rate,
            'type': 'explosion',
            'yield_kt': yield_kt,
            'distance_km': distance_km,
            'p_arrival': p_arrival_time,
            'duration': total_duration
        }

    def generate_dataset(self, n_earthquakes: int = 50,
                        n_explosions: int = 50) -> List[Dict]:
        """
        Generate complete seismic dataset.

        Args:
            n_earthquakes: Number of earthquake waveforms
            n_explosions: Number of explosion waveforms

        Returns:
            List of waveform dictionaries
        """
        dataset = []

        # Generate earthquakes
        for i in range(n_earthquakes):
            magnitude = np.random.uniform(4.0, 6.5)
            distance = np.random.uniform(50, 500)
            dataset.append(self.generate_earthquake(magnitude, distance))

        # Generate explosions
        for i in range(n_explosions):
            yield_kt = np.random.uniform(5, 50)
            distance = np.random.uniform(50, 500)
            dataset.append(self.generate_explosion(yield_kt, distance))

        # Shuffle
        np.random.shuffle(dataset)

        return dataset


if __name__ == "__main__":
    print("="*80)
    print("SYNTHETIC SEISMIC WAVEFORM GENERATOR TEST")
    print("="*80)
    print()

    generator = SeismicWaveformGenerator(sample_rate=100)

    # Generate examples
    print("Generating example waveforms...")
    earthquake = generator.generate_earthquake(magnitude=5.5, distance_km=150)
    explosion = generator.generate_explosion(yield_kt=20, distance_km=150)

    print(f"  ✓ Earthquake: M{earthquake['magnitude']}, {earthquake['distance_km']}km")
    print(f"    Duration: {earthquake['duration']:.1f}s, P-arrival: {earthquake['p_arrival']:.1f}s")

    print(f"  ✓ Explosion: {explosion['yield_kt']}kt, {explosion['distance_km']}km")
    print(f"    Duration: {explosion['duration']:.1f}s, P-arrival: {explosion['p_arrival']:.1f}s")
    print()

    # Generate small dataset
    print("Generating test dataset (10 earthquakes + 10 explosions)...")
    dataset = generator.generate_dataset(n_earthquakes=10, n_explosions=10)
    print(f"  ✓ Generated {len(dataset)} waveforms")

    # Save visualization
    workspace = Path("phase2_workspace")
    workspace.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot earthquake
    ax = axes[0]
    t = np.arange(len(earthquake['waveform'])) / earthquake['sample_rate']
    ax.plot(t, earthquake['waveform'], 'b-', linewidth=0.5)
    ax.axvline(earthquake['p_arrival'], color='r', linestyle='--',
               label='P-wave arrival', alpha=0.7)
    ax.axvline(earthquake['s_arrival'], color='g', linestyle='--',
               label='S-wave arrival', alpha=0.7)
    ax.set_title(f"Synthetic Earthquake (M{earthquake['magnitude']})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot explosion
    ax = axes[1]
    t = np.arange(len(explosion['waveform'])) / explosion['sample_rate']
    ax.plot(t, explosion['waveform'], 'r-', linewidth=0.5)
    ax.axvline(explosion['p_arrival'], color='r', linestyle='--',
               label='P-wave arrival', alpha=0.7)
    ax.set_title(f"Synthetic Explosion ({explosion['yield_kt']}kt)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = workspace / "synthetic_seismic_waveforms.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved to {output_path}")
    print()

    print("✓ Seismic waveform generator ready for Phase 2 validation!")
