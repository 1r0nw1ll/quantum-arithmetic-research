#!/usr/bin/env python3
"""
Quantum-Phonon Coupling Simulation for Helium-Doped Quartz Piezoelectric System

This module simulates the coupling between trapped helium atoms and quartz lattice phonons,
modeling the self-oscillating piezoelectric energy generation mechanism.

Physical Model:
- Helium atoms trapped in quartz lattice interstices
- Quantum harmonic oscillator model for helium motion
- Phonon modes of quartz crystal lattice
- Helium-phonon interaction via coupling Hamiltonian
- Piezoelectric conversion from mechanical strain to electric field

Author: QA Research Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm
import seaborn as sns
from dataclasses import dataclass
from typing import Tuple, List

# Physical Constants
HBAR = 1.054571817e-34  # J·s (reduced Planck constant)
K_B = 1.380649e-23      # J/K (Boltzmann constant)
M_HE = 6.646477e-27     # kg (helium-4 mass)
Q_E = 1.602176634e-19   # C (elementary charge)

# Quartz Properties
QUARTZ_DENSITY = 2650   # kg/m³
QUARTZ_C11 = 86.74e9    # Pa (elastic constant)
QUARTZ_SOUND_SPEED = 5970  # m/s (longitudinal)
PIEZO_COEFFICIENT_D11 = 2.31e-12  # C/N (piezoelectric coefficient)

@dataclass
class QuartzLatticeParams:
    """Parameters for quartz crystal lattice"""
    lattice_constant: float = 4.913e-10  # m (a-axis)
    c_axis: float = 5.405e-10            # m (c-axis)
    phonon_cutoff_freq: float = 1.5e13   # rad/s (~2.4 THz max optical phonon)
    num_phonon_modes: int = 10           # Number of discrete modes to simulate
    temperature: float = 300.0           # K (room temperature)

@dataclass
class HeliumTrapParams:
    """Parameters for trapped helium atoms"""
    trap_frequency: float = 1e12         # rad/s (oscillation in interstitial site)
    num_atoms: int = 1e14                # Total number of trapped He atoms
    coherence_time: float = 1e-9         # s (quantum coherence lifetime)
    trap_depth: float = 0.1 * Q_E        # J (potential well depth ~0.1 eV)


class QuantumPhononSystem:
    """
    Simulates coupled quantum helium oscillators and quartz phonon modes
    """

    def __init__(self, lattice_params: QuartzLatticeParams, he_params: HeliumTrapParams):
        self.lattice = lattice_params
        self.helium = he_params

        # Calculate phonon spectrum
        self.phonon_freqs = self._generate_phonon_spectrum()

        # Calculate coupling strengths
        self.coupling_strengths = self._calculate_coupling()

        # Zero-point energies
        self.he_zero_point = 0.5 * HBAR * he_params.trap_frequency
        self.phonon_zero_points = 0.5 * HBAR * self.phonon_freqs

    def _generate_phonon_spectrum(self) -> np.ndarray:
        """
        Generate phonon dispersion relation for quartz
        Uses Debye model for acoustic phonons + optical modes
        """
        k_max = 2 * np.pi / self.lattice.lattice_constant

        # Acoustic modes (linear dispersion) - avoid zero frequency
        k_values = np.linspace(k_max/10, k_max, self.lattice.num_phonon_modes // 2)
        acoustic_freqs = QUARTZ_SOUND_SPEED * k_values

        # Optical modes (flat dispersion near cutoff)
        optical_freqs = np.linspace(
            0.8 * self.lattice.phonon_cutoff_freq,
            self.lattice.phonon_cutoff_freq,
            self.lattice.num_phonon_modes // 2
        )

        return np.concatenate([acoustic_freqs, optical_freqs])

    def _calculate_coupling(self) -> np.ndarray:
        """
        Calculate coupling strength g_k between helium and each phonon mode

        g_k = sqrt(ω_k / (2 M V)) * (helium displacement matrix element)

        Coupling is strongest near resonance (ω_He ≈ ω_k)
        """
        volume = self.lattice.lattice_constant**3

        # Base coupling strength
        g_base = np.sqrt(self.phonon_freqs / (2 * M_HE * volume))

        # Resonance enhancement factor (Lorentzian)
        detuning = self.phonon_freqs - self.helium.trap_frequency
        resonance_width = 0.1 * self.helium.trap_frequency
        resonance_factor = 1 + 10 / (1 + (detuning / resonance_width)**2)

        return g_base * resonance_factor

    def calculate_thermal_occupation(self) -> Tuple[float, np.ndarray]:
        """
        Calculate thermal phonon occupation numbers using Bose-Einstein distribution

        n_k = 1 / (exp(ℏω_k / kT) - 1)
        """
        # Avoid numerical issues with small exponents
        he_exponent = HBAR * self.helium.trap_frequency / (K_B * self.lattice.temperature)
        if he_exponent < 1e-10:
            he_occupation = K_B * self.lattice.temperature / (HBAR * self.helium.trap_frequency)
        else:
            he_occupation = 1.0 / (np.exp(he_exponent) - 1 + 1e-10)

        phonon_exponents = HBAR * self.phonon_freqs / (K_B * self.lattice.temperature)
        # Use classical limit for very low frequencies
        phonon_occupations = np.where(
            phonon_exponents < 1e-2,
            K_B * self.lattice.temperature / (HBAR * self.phonon_freqs),
            1.0 / (np.exp(phonon_exponents) - 1 + 1e-10)
        )

        return he_occupation, phonon_occupations

    def simulate_coupled_dynamics(self, t_max: float = 1e-9, num_points: int = 1000):
        """
        Simulate time evolution of coupled helium-phonon system

        Uses classical equations of motion derived from quantum Hamiltonian:
        dq_He/dt = p_He / M
        dp_He/dt = -M ω_He² q_He - Σ g_k q_k
        dq_k/dt = p_k / M_eff
        dp_k/dt = -M_eff ω_k² q_k - g_k q_He
        """
        t = np.linspace(0, t_max, num_points)

        # Initial conditions: helium displaced, phonons at thermal equilibrium
        he_thermal, phonon_thermal = self.calculate_thermal_occupation()

        # Convert occupation numbers to classical amplitudes (with bounds)
        q_he_0 = np.sqrt(2 * np.clip(he_thermal + 0.5, 0.5, 100) * HBAR / (M_HE * self.helium.trap_frequency))
        p_he_0 = 0.0

        # Effective mass for phonon modes (reduced for collective motion)
        M_eff = M_HE * self.helium.num_atoms / len(self.phonon_freqs)

        # Clip phonon thermal occupation to avoid numerical issues
        phonon_thermal_clipped = np.clip(phonon_thermal, 0.5, 100)
        q_k_0 = np.sqrt(2 * phonon_thermal_clipped * HBAR / (M_eff * self.phonon_freqs))
        p_k_0 = np.zeros_like(q_k_0)

        # State vector: [q_He, p_He, q_k1, p_k1, q_k2, p_k2, ...]
        y0 = np.concatenate([[q_he_0, p_he_0], np.ravel(np.column_stack([q_k_0, p_k_0]))])

        def coupled_equations(y, t):
            q_he, p_he = y[0], y[1]
            q_k = y[2::2]
            p_k = y[3::2]

            # Helium equations
            dq_he = p_he / M_HE
            dp_he = -M_HE * self.helium.trap_frequency**2 * q_he - np.sum(self.coupling_strengths * q_k)

            # Phonon equations
            dq_k = p_k / M_eff
            dp_k = -M_eff * self.phonon_freqs**2 * q_k - self.coupling_strengths * q_he

            # Interleave derivatives
            derivs = np.zeros_like(y)
            derivs[0] = dq_he
            derivs[1] = dp_he
            derivs[2::2] = dq_k
            derivs[3::2] = dp_k

            return derivs

        # Solve coupled ODEs with better tolerances
        from scipy.integrate import ode

        # Use scipy.integrate.ode for better control
        solver = ode(lambda t, y: coupled_equations(y, t))
        solver.set_integrator('dopri5', atol=1e-8, rtol=1e-6, nsteps=10000)
        solver.set_initial_value(y0, t[0])

        solution = np.zeros((len(t), len(y0)))
        solution[0] = y0

        for i in range(1, len(t)):
            if solver.successful():
                solver.integrate(t[i])
                solution[i] = solver.y
            else:
                solution[i] = solution[i-1]  # Hold previous value if failed

        return t, solution

    def calculate_piezoelectric_output(self, solution: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Convert phonon amplitudes to electrical output via piezoelectric effect

        Strain: ε = Σ (dq_k/dx)
        Electric field: E = d₁₁ × ε
        Power: P = ε₀ E² × Volume × f
        """
        # Extract phonon displacements
        q_k = solution[:, 2::2]

        # Calculate strain (gradient of displacement)
        strain = np.sum(q_k / self.lattice.lattice_constant, axis=1)

        # Electric field from piezoelectric coupling
        E_field = PIEZO_COEFFICIENT_D11 * strain * QUARTZ_C11

        # Energy density (J/m³)
        epsilon_0 = 8.854187817e-12  # F/m
        epsilon_r = 4.5  # Relative permittivity of quartz
        energy_density = 0.5 * epsilon_0 * epsilon_r * E_field**2

        # Estimate crystal volume (1 cm³ reference)
        volume = 1e-6  # m³

        # Average oscillation frequency
        avg_freq = np.mean(self.phonon_freqs) / (2 * np.pi)

        # Power output (Watts)
        power = energy_density * volume * avg_freq

        return power


class QuartzPiezoVisualizer:
    """Visualization tools for quantum-phonon coupling"""

    @staticmethod
    def plot_phonon_spectrum(system: QuantumPhononSystem, save_path: str = None):
        """Plot phonon dispersion relation"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Frequency spectrum
        freqs_THz = system.phonon_freqs / (2 * np.pi * 1e12)
        ax1.scatter(range(len(freqs_THz)), freqs_THz, c=system.coupling_strengths,
                   cmap='plasma', s=100, edgecolors='black', linewidth=1)
        ax1.axhline(system.helium.trap_frequency / (2 * np.pi * 1e12),
                   color='red', linestyle='--', linewidth=2, label='He trap frequency')
        ax1.set_xlabel('Mode Index', fontsize=12)
        ax1.set_ylabel('Frequency (THz)', fontsize=12)
        ax1.set_title('Quartz Phonon Spectrum', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Coupling strength
        ax2.bar(range(len(system.coupling_strengths)), system.coupling_strengths * 1e15,
               color=plt.cm.viridis(np.linspace(0, 1, len(system.coupling_strengths))))
        ax2.set_xlabel('Mode Index', fontsize=12)
        ax2.set_ylabel('Coupling Strength × 10¹⁵ (arb. units)', fontsize=12)
        ax2.set_title('He-Phonon Coupling Strengths', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_coupled_dynamics(t: np.ndarray, solution: np.ndarray,
                             system: QuantumPhononSystem, save_path: str = None):
        """Plot time evolution of coupled system"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        t_ns = t * 1e9  # Convert to nanoseconds

        # Helium oscillation
        q_he = solution[:, 0] * 1e12  # Convert to pm
        axes[0, 0].plot(t_ns, q_he, linewidth=2, color='darkblue')
        axes[0, 0].set_xlabel('Time (ns)', fontsize=11)
        axes[0, 0].set_ylabel('He Displacement (pm)', fontsize=11)
        axes[0, 0].set_title('Helium Atom Oscillation', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        # Phonon mode evolution (first 3 modes)
        q_k = solution[:, 2::2] * 1e12
        for i in range(min(3, q_k.shape[1])):
            axes[0, 1].plot(t_ns, q_k[:, i], linewidth=1.5,
                          label=f'Mode {i} ({system.phonon_freqs[i]/(2*np.pi*1e12):.2f} THz)')
        axes[0, 1].set_xlabel('Time (ns)', fontsize=11)
        axes[0, 1].set_ylabel('Phonon Amplitude (pm)', fontsize=11)
        axes[0, 1].set_title('Phonon Mode Dynamics', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Energy transfer
        E_he = 0.5 * M_HE * (solution[:, 1]**2) + 0.5 * M_HE * system.helium.trap_frequency**2 * (solution[:, 0]**2)
        E_he_eV = E_he / Q_E
        axes[1, 0].plot(t_ns, E_he_eV, linewidth=2, color='crimson')
        axes[1, 0].set_xlabel('Time (ns)', fontsize=11)
        axes[1, 0].set_ylabel('Helium Energy (eV)', fontsize=11)
        axes[1, 0].set_title('Helium Oscillator Energy', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # Phase space (He)
        axes[1, 1].plot(solution[:, 0] * 1e12, solution[:, 1] * M_HE * 1e15,
                       linewidth=0.5, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Position (pm)', fontsize=11)
        axes[1, 1].set_ylabel('Momentum (10⁻¹⁵ kg·m/s)', fontsize=11)
        axes[1, 1].set_title('Helium Phase Space Trajectory', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_power_output(t: np.ndarray, power: np.ndarray, save_path: str = None):
        """Plot piezoelectric power generation"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        t_ns = t * 1e9
        power_mW = power * 1e3

        # Time series
        ax1.plot(t_ns, power_mW, linewidth=2, color='darkorange')
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel('Power Output (mW)', fontsize=12)
        ax1.set_title('Piezoelectric Power Generation', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)

        # Power spectrum (FFT)
        dt = t[1] - t[0]
        freqs = np.fft.fftfreq(len(t), dt) / 1e12  # THz
        power_fft = np.abs(np.fft.fft(power))

        positive_freqs = freqs > 0
        ax2.semilogy(freqs[positive_freqs], power_fft[positive_freqs], linewidth=2, color='purple')
        ax2.set_xlabel('Frequency (THz)', fontsize=12)
        ax2.set_ylabel('Power Spectral Density (arb. units)', fontsize=12)
        ax2.set_title('Power Output Frequency Spectrum', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_quantum_phonon_simulation():
    """Main simulation runner"""

    print("="*70)
    print("QUANTUM-PHONON COUPLING SIMULATION")
    print("Helium-Doped Quartz Piezoelectric System")
    print("="*70)

    # Initialize system
    lattice = QuartzLatticeParams(
        num_phonon_modes=10,
        temperature=300.0
    )

    helium = HeliumTrapParams(
        trap_frequency=1e12,  # 1 THz
        num_atoms=1e14,
        coherence_time=1e-9
    )

    system = QuantumPhononSystem(lattice, helium)

    print(f"\nSystem Parameters:")
    print(f"  Helium trap frequency: {helium.trap_frequency / (2*np.pi*1e12):.3f} THz")
    print(f"  Number of He atoms: {helium.num_atoms:.2e}")
    print(f"  Number of phonon modes: {lattice.num_phonon_modes}")
    print(f"  Temperature: {lattice.temperature} K")
    print(f"  Phonon frequency range: {system.phonon_freqs[0]/(2*np.pi*1e12):.2f} - {system.phonon_freqs[-1]/(2*np.pi*1e12):.2f} THz")

    # Calculate thermal properties
    he_occ, phonon_occ = system.calculate_thermal_occupation()
    print(f"\nThermal Properties:")
    print(f"  He thermal occupation: {he_occ:.4f}")
    print(f"  Mean phonon occupation: {np.mean(phonon_occ):.4f}")
    print(f"  He zero-point energy: {system.he_zero_point/Q_E:.4f} eV")

    # Visualize phonon spectrum
    print("\n[1/4] Plotting phonon spectrum...")
    QuartzPiezoVisualizer.plot_phonon_spectrum(system, 'quartz_phonon_spectrum.png')

    # Run dynamics simulation
    print("[2/4] Running coupled dynamics simulation...")
    t, solution = system.simulate_coupled_dynamics(t_max=2e-9, num_points=2000)

    # Visualize dynamics
    print("[3/4] Plotting coupled dynamics...")
    QuartzPiezoVisualizer.plot_coupled_dynamics(t, solution, system, 'quartz_coupled_dynamics.png')

    # Calculate and plot power output
    print("[4/4] Calculating piezoelectric power output...")
    power = system.calculate_piezoelectric_output(solution, t)
    QuartzPiezoVisualizer.plot_power_output(t, power, 'quartz_power_output.png')

    # Summary statistics
    print(f"\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Average power output: {np.mean(power)*1e3:.4f} mW")
    print(f"Peak power output: {np.max(power)*1e3:.4f} mW")
    print(f"Power density (1 cm³): {np.mean(power)*1e6:.4f} W/m³ = {np.mean(power)*1e3:.4f} mW/cm³")
    print(f"Dominant frequency: {system.helium.trap_frequency/(2*np.pi*1e12):.3f} THz")

    # Energy efficiency
    total_he_atoms = helium.num_atoms
    thermal_energy_per_atom = K_B * lattice.temperature
    total_input_power = total_he_atoms * thermal_energy_per_atom * helium.trap_frequency / (2*np.pi)
    efficiency = np.mean(power) / total_input_power * 100
    print(f"Quantum efficiency: {efficiency:.4f}%")

    print("\n" + "="*70)
    print("Visualizations saved:")
    print("  - quartz_phonon_spectrum.png")
    print("  - quartz_coupled_dynamics.png")
    print("  - quartz_power_output.png")
    print("="*70)

    return system, t, solution, power


if __name__ == "__main__":
    np.random.seed(42)
    system, t, solution, power = run_quantum_phonon_simulation()
