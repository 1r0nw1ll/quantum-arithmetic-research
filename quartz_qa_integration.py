#!/usr/bin/env python3
"""
Helium-Lattice Interaction Dynamics using QA Framework

Integrates the quantum-phonon coupling with Quantum Arithmetic (QA) system
to model energy flow, resonance patterns, and harmonic coherence in the
helium-doped quartz piezoelectric system.

QA Mapping:
- Helium oscillation states → QA tuples (b, e, d, a)
- Phonon mode energies → mod-24 states
- Coupling strength → E8 alignment metric
- Piezoelectric output → Harmonic Index

Author: QA Research Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import seaborn as sns
from dataclasses import dataclass

# QA System Constants
QA_MODULUS = 24
E8_DIM = 8


class QASystem:
    """
    Core Quantum Arithmetic engine for mod-24 operations
    """

    def __init__(self, modulus: int = QA_MODULUS):
        self.modulus = modulus
        self.e8_roots = self._generate_e8_roots()

    def generate_tuple(self, b: int, e: int) -> Tuple[int, int, int, int]:
        """
        Generate QA tuple from (b, e) pair

        d = (b + e) % modulus
        a = (b + 2*e) % modulus
        """
        d = (b + e) % self.modulus
        a = (b + 2 * e) % self.modulus
        return (b, e, d, a)

    def tuple_to_8d(self, b: int, e: int) -> np.ndarray:
        """
        Project 4D QA tuple into 8D space for E8 comparison

        Pattern: [b, e, d, a, b+d, e+a, b+a, d+e]
        """
        b_e_d_a = self.generate_tuple(b, e)
        b, e, d, a = b_e_d_a

        vector_8d = np.array([
            b, e, d, a,
            (b + d) % self.modulus,
            (e + a) % self.modulus,
            (b + a) % self.modulus,
            (d + e) % self.modulus
        ], dtype=float)

        # Normalize
        norm = np.linalg.norm(vector_8d)
        if norm > 0:
            vector_8d /= norm

        return vector_8d

    def _generate_e8_roots(self) -> np.ndarray:
        """
        Generate E8 root system (240 vectors in 8D)
        Uses standard E8 root vector construction
        """
        roots = []

        # Type 1: All permutations and sign changes of (±1, ±1, 0, 0, 0, 0, 0, 0)
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(8)
                        v[i] = s1
                        v[j] = s2
                        roots.append(v)

        # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
        # with even number of minus signs
        for i in range(256):
            v = np.array([1 if (i >> j) & 1 else -1 for j in range(8)]) * 0.5
            if np.sum(v) % 2 == 0:  # Even number of -1/2
                roots.append(v)

        return np.array(roots)

    def calculate_e8_alignment(self, b: int, e: int) -> float:
        """
        Calculate maximum cosine similarity to E8 root system
        """
        vec_8d = self.tuple_to_8d(b, e)
        cosine_sims = np.dot(self.e8_roots, vec_8d)
        return np.max(np.abs(cosine_sims))


@dataclass
class HeliumQAState:
    """Represents helium oscillator state in QA framework"""
    b: int  # Base state (position quantized mod-24)
    e: int  # Excitation state (momentum quantized mod-24)
    energy: float  # Physical energy (Joules)
    phase: float  # Oscillation phase (radians)


class QuartzQAEngine:
    """
    Maps helium-phonon coupling to QA modular arithmetic
    """

    def __init__(self, num_he_atoms: int = 100, num_phonon_modes: int = 24):
        self.qa = QASystem()
        self.num_he = num_he_atoms
        self.num_modes = num_phonon_modes

        # Initialize helium states
        self.he_states = self._initialize_helium_states()

        # Initialize phonon modes
        self.phonon_modes = self._initialize_phonon_modes()

        # Coupling network (helium-to-phonon weights)
        self.coupling_matrix = np.zeros((num_he_atoms, num_phonon_modes))

    def _initialize_helium_states(self) -> List[HeliumQAState]:
        """Initialize helium atoms in random QA states"""
        states = []
        for _ in range(self.num_he):
            b = np.random.randint(1, QA_MODULUS + 1)
            e = np.random.randint(1, QA_MODULUS + 1)
            energy = np.random.uniform(0.01, 0.1) * 1.602e-19  # 0.01-0.1 eV
            phase = np.random.uniform(0, 2 * np.pi)
            states.append(HeliumQAState(b, e, energy, phase))
        return states

    def _initialize_phonon_modes(self) -> List[Dict]:
        """Initialize phonon modes with QA states"""
        modes = []
        for k in range(self.num_modes):
            b_k = (k + 1) % QA_MODULUS or QA_MODULUS
            e_k = (2 * k + 3) % QA_MODULUS or QA_MODULUS
            frequency = 0.5e12 + k * 0.1e12  # 0.5 - 3 THz
            modes.append({
                'b': b_k,
                'e': e_k,
                'frequency': frequency,
                'amplitude': 0.0,
                'energy': 0.0
            })
        return modes

    def calculate_resonance_coupling(self) -> np.ndarray:
        """
        Calculate coupling strength based on QA tuple alignment

        Coupling is strong when helium and phonon QA states are resonant:
        - Similar (b, e) values → high coupling
        - E8 alignment → coherence enhancement
        """
        for i, he_state in enumerate(self.he_states):
            for k, phonon in enumerate(self.phonon_modes):
                # QA distance metric (modular)
                b_dist = min(abs(he_state.b - phonon['b']),
                           QA_MODULUS - abs(he_state.b - phonon['b']))
                e_dist = min(abs(he_state.e - phonon['e']),
                           QA_MODULUS - abs(he_state.e - phonon['e']))

                qa_distance = np.sqrt(b_dist**2 + e_dist**2)

                # E8 alignment for helium state
                e8_he = self.qa.calculate_e8_alignment(he_state.b, he_state.e)

                # E8 alignment for phonon mode
                e8_ph = self.qa.calculate_e8_alignment(phonon['b'], phonon['e'])

                # Coupling strength: decreases with QA distance, enhanced by E8
                coupling = (e8_he + e8_ph) / (1 + qa_distance)

                self.coupling_matrix[i, k] = coupling

        return self.coupling_matrix

    def simulate_qa_dynamics(self, num_steps: int = 1000, dt: float = 1e-12):
        """
        Time evolution of helium-phonon system using QA update rules

        Update rule:
        1. Calculate resonance coupling
        2. Energy transfer proportional to coupling
        3. QA state updates via modular arithmetic
        4. Phase evolution
        """
        history = {
            'time': [],
            'he_energies': [],
            'phonon_energies': [],
            'e8_alignment': [],
            'harmonic_index': [],
            'power_output': []
        }

        for step in range(num_steps):
            t = step * dt

            # Update coupling matrix
            self.calculate_resonance_coupling()

            # Energy transfer from helium to phonons
            total_phonon_energy = 0
            e8_alignments = []

            for i, he_state in enumerate(self.he_states):
                # Calculate energy transfer to phonons
                for k, phonon in enumerate(self.phonon_modes):
                    transfer_rate = self.coupling_matrix[i, k] * 1e9  # Hz
                    energy_transfer = transfer_rate * he_state.energy * dt

                    # Update phonon energy
                    phonon['energy'] += energy_transfer
                    phonon['amplitude'] = np.sqrt(2 * phonon['energy'])

                    # Deplete helium energy
                    he_state.energy = max(0, he_state.energy - energy_transfer)

                total_phonon_energy += phonon['energy']

                # Update helium QA state based on energy
                energy_eV = he_state.energy / 1.602e-19
                he_state.b = int(energy_eV * 10) % QA_MODULUS or QA_MODULUS
                he_state.e = int(he_state.phase * QA_MODULUS / (2 * np.pi)) or 1

                # Phase evolution
                omega_he = 1e12  # 1 THz
                he_state.phase = (he_state.phase + omega_he * dt) % (2 * np.pi)

                # Track E8 alignment
                e8_alignments.append(self.qa.calculate_e8_alignment(he_state.b, he_state.e))

            # Update phonon QA states
            for k, phonon in enumerate(self.phonon_modes):
                # Energy-dependent QA state
                energy_normalized = phonon['energy'] / 1.602e-19  # eV
                phonon['b'] = (k + int(energy_normalized * 5)) % QA_MODULUS or QA_MODULUS
                phonon['e'] = (int(phonon['amplitude'] * 1e10)) % QA_MODULUS or QA_MODULUS

            # Calculate metrics
            total_he_energy = sum(s.energy for s in self.he_states)
            mean_e8 = np.mean(e8_alignments)

            # Harmonic Index
            loss = total_he_energy / (self.num_he * 0.1 * 1.602e-19)  # Normalized to initial
            harmonic_index = mean_e8 * np.exp(-0.1 * loss)

            # Power output (piezoelectric conversion)
            piezo_efficiency = 0.01  # 1% conversion efficiency
            power = total_phonon_energy / dt * piezo_efficiency

            # Store history
            history['time'].append(t)
            history['he_energies'].append(total_he_energy)
            history['phonon_energies'].append(total_phonon_energy)
            history['e8_alignment'].append(mean_e8)
            history['harmonic_index'].append(harmonic_index)
            history['power_output'].append(power)

        # Convert to arrays
        for key in history:
            history[key] = np.array(history[key])

        return history

    def visualize_qa_state_space(self, save_path: str = None):
        """Visualize helium and phonon states in QA (b,e) space"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Helium states
        he_b = [s.b for s in self.he_states]
        he_e = [s.e for s in self.he_states]
        he_energies = [s.energy / 1.602e-19 for s in self.he_states]  # eV

        scatter1 = ax1.scatter(he_b, he_e, c=he_energies, cmap='plasma',
                              s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax1.set_xlabel('b (base state)', fontsize=12)
        ax1.set_ylabel('e (excitation state)', fontsize=12)
        ax1.set_title('Helium QA State Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, QA_MODULUS + 1)
        ax1.set_ylim(0, QA_MODULUS + 1)
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Energy (eV)')

        # Phonon modes
        ph_b = [m['b'] for m in self.phonon_modes]
        ph_e = [m['e'] for m in self.phonon_modes]
        ph_freqs = [m['frequency'] / 1e12 for m in self.phonon_modes]  # THz

        scatter2 = ax2.scatter(ph_b, ph_e, c=ph_freqs, cmap='viridis',
                              s=150, alpha=0.7, marker='s', edgecolors='black', linewidth=1)
        ax2.set_xlabel('b (base state)', fontsize=12)
        ax2.set_ylabel('e (excitation state)', fontsize=12)
        ax2.set_title('Phonon Mode QA Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, QA_MODULUS + 1)
        ax2.set_ylim(0, QA_MODULUS + 1)
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Frequency (THz)')

        # Coupling matrix heatmap
        coupling = self.calculate_resonance_coupling()
        im = ax3.imshow(coupling, cmap='hot', aspect='auto', interpolation='nearest')
        ax3.set_xlabel('Phonon Mode Index', fontsize=12)
        ax3.set_ylabel('Helium Atom Index', fontsize=12)
        ax3.set_title('QA Resonance Coupling Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax3, label='Coupling Strength')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_qa_dynamics(self, history: Dict, save_path: str = None):
        """Visualize QA dynamics evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        t_ps = history['time'] * 1e12  # picoseconds

        # Energy evolution
        axes[0, 0].plot(t_ps, history['he_energies'] / 1.602e-19, linewidth=2,
                       label='Helium', color='blue')
        axes[0, 0].plot(t_ps, history['phonon_energies'] / 1.602e-19, linewidth=2,
                       label='Phonons', color='red')
        axes[0, 0].set_xlabel('Time (ps)', fontsize=11)
        axes[0, 0].set_ylabel('Energy (eV)', fontsize=11)
        axes[0, 0].set_title('Energy Transfer: He → Phonons', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # E8 alignment
        axes[0, 1].plot(t_ps, history['e8_alignment'], linewidth=2, color='green')
        axes[0, 1].set_xlabel('Time (ps)', fontsize=11)
        axes[0, 1].set_ylabel('E8 Alignment', fontsize=11)
        axes[0, 1].set_title('Geometric Coherence (E8)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        # Harmonic Index
        axes[1, 0].plot(t_ps, history['harmonic_index'], linewidth=2, color='purple')
        axes[1, 0].set_xlabel('Time (ps)', fontsize=11)
        axes[1, 0].set_ylabel('Harmonic Index', fontsize=11)
        axes[1, 0].set_title('QA Harmonic Index', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # Power output
        axes[1, 1].plot(t_ps, history['power_output'] * 1e3, linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Time (ps)', fontsize=11)
        axes[1, 1].set_ylabel('Power (mW)', fontsize=11)
        axes[1, 1].set_title('Piezoelectric Power Output', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_qa_integration_simulation():
    """Main QA integration simulation"""

    print("="*70)
    print("HELIUM-LATTICE QA INTEGRATION SIMULATION")
    print("Quantum Arithmetic Framework for Piezoelectric Quartz")
    print("="*70)

    # Initialize QA engine
    engine = QuartzQAEngine(num_he_atoms=100, num_phonon_modes=24)

    print(f"\nQA System Configuration:")
    print(f"  Modulus: {QA_MODULUS}")
    print(f"  Number of He atoms: {engine.num_he}")
    print(f"  Number of phonon modes: {engine.num_modes}")
    print(f"  E8 root vectors: {len(engine.qa.e8_roots)}")

    # Visualize initial state space
    print("\n[1/3] Visualizing QA state space...")
    engine.visualize_qa_state_space('quartz_qa_state_space.png')

    # Run dynamics
    print("[2/3] Running QA dynamics simulation...")
    history = engine.simulate_qa_dynamics(num_steps=1000, dt=1e-12)

    # Visualize dynamics
    print("[3/3] Visualizing QA dynamics...")
    engine.visualize_qa_dynamics(history, 'quartz_qa_dynamics.png')

    # Summary statistics
    print(f"\n" + "="*70)
    print("QA INTEGRATION RESULTS")
    print("="*70)
    print(f"Mean E8 alignment: {np.mean(history['e8_alignment']):.4f}")
    print(f"Peak E8 alignment: {np.max(history['e8_alignment']):.4f}")
    print(f"Mean Harmonic Index: {np.mean(history['harmonic_index']):.4f}")
    print(f"Average power output: {np.mean(history['power_output'])*1e3:.4f} mW")
    print(f"Peak power output: {np.max(history['power_output'])*1e3:.4f} mW")
    print(f"Total energy transferred to phonons: {history['phonon_energies'][-1]/1.602e-19:.4f} eV")

    # Coupling strength analysis
    coupling_stats = engine.coupling_matrix.flatten()
    print(f"\nCoupling Matrix Statistics:")
    print(f"  Mean coupling strength: {np.mean(coupling_stats):.4f}")
    print(f"  Max coupling strength: {np.max(coupling_stats):.4f}")
    print(f"  Coupling variance: {np.var(coupling_stats):.4f}")

    print("\n" + "="*70)
    print("Visualizations saved:")
    print("  - quartz_qa_state_space.png")
    print("  - quartz_qa_dynamics.png")
    print("="*70)

    return engine, history


if __name__ == "__main__":
    np.random.seed(42)
    engine, history = run_qa_integration_simulation()
