#!/usr/bin/env python3
"""
Piezoelectric Tensor Visualizations for Quartz

Visualizes the piezoelectric coupling between mechanical strain and electric field
in quartz crystals. Includes 3D tensor representations, strain-field relationships,
and directional dependencies.

Piezoelectric Constitutive Relations:
    D_i = d_ijk * σ_jk  (Direct effect)
    ε_ij = s_ijkl * σ_kl + d_kij * E_k  (Converse effect)

Where:
    D_i = electric displacement
    σ_jk = stress tensor
    ε_ij = strain tensor
    E_k = electric field
    d_ijk = piezoelectric coefficient tensor (3rd rank)
    s_ijkl = elastic compliance tensor (4th rank)

Author: QA Research Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, List

# Quartz Piezoelectric Coefficients (C/N or m/V in SI units)
# Using Voigt notation: d_ij where i=1,2,3 (electric), j=1-6 (stress)
QUARTZ_D_TENSOR = np.array([
    [2.31e-12,  -2.31e-12,  0,        0,         0,        0],        # d_1j (x-direction)
    [0,         0,          0,        0,         0,        0],        # d_2j (y-direction)
    [0,         0,          0,        0.67e-12,  0,        0]         # d_3j (z-direction)
])

# Quartz Elastic Compliance (m²/N)
QUARTZ_COMPLIANCE = np.array([
    [12.77e-12, -1.79e-12, -1.20e-12, 4.50e-12,  0,         0],
    [-1.79e-12, 12.77e-12, -1.20e-12, -4.50e-12, 0,         0],
    [-1.20e-12, -1.20e-12, 9.65e-12,  0,         0,         0],
    [4.50e-12,  -4.50e-12, 0,         29.01e-12, 0,         0],
    [0,         0,         0,         0,         29.01e-12, 4.50e-12],
    [0,         0,         0,         0,         4.50e-12,  25.56e-12]
])

# Relative permittivity (dielectric constant)
QUARTZ_EPSILON_R = np.array([
    [4.5, 0,   0],
    [0,   4.5, 0],
    [0,   0,   4.6]
])


class PiezoelectricTensor:
    """Represents and manipulates piezoelectric tensor"""

    def __init__(self, d_matrix: np.ndarray = QUARTZ_D_TENSOR):
        """
        Initialize with piezoelectric coefficient matrix

        Args:
            d_matrix: 3x6 matrix in Voigt notation
        """
        self.d_voigt = d_matrix
        self.d_full = self._voigt_to_full_tensor()

    def _voigt_to_full_tensor(self) -> np.ndarray:
        """
        Convert 3x6 Voigt notation to full 3x3x3 tensor

        Voigt mapping: 11→1, 22→2, 33→3, 23→4, 13→5, 12→6
        """
        d_tensor = np.zeros((3, 3, 3))

        voigt_map = {
            0: (0, 0), 1: (1, 1), 2: (2, 2),
            3: (1, 2), 4: (0, 2), 5: (0, 1)
        }

        for i in range(3):
            for j in range(6):
                k, l = voigt_map[j]
                d_tensor[i, k, l] = self.d_voigt[i, j]
                if k != l:  # Symmetry
                    d_tensor[i, l, k] = self.d_voigt[i, j]

        return d_tensor

    def calculate_electric_field(self, stress: np.ndarray) -> np.ndarray:
        """
        Calculate induced electric field from stress tensor

        E_i = d_ijk * σ_jk / ε₀ε_r

        Args:
            stress: 3x3 stress tensor (Pa)

        Returns:
            Electric field vector (V/m)
        """
        # Contract tensor: sum over j,k
        E = np.einsum('ijk,jk->i', self.d_full, stress)

        # Divide by permittivity
        epsilon_0 = 8.854187817e-12
        E_normalized = E / (epsilon_0 * np.diag(QUARTZ_EPSILON_R))

        return E_normalized

    def calculate_induced_strain(self, E_field: np.ndarray) -> np.ndarray:
        """
        Calculate induced strain from electric field (converse piezo effect)

        ε_jk = d_ijk * E_i

        Args:
            E_field: Electric field vector (V/m)

        Returns:
            Strain tensor (dimensionless)
        """
        strain = np.einsum('ijk,i->jk', self.d_full, E_field)
        return strain


class QuartzPiezoVisualizer3D:
    """3D visualization tools for piezoelectric tensors"""

    def __init__(self, piezo_tensor: PiezoelectricTensor):
        self.tensor = piezo_tensor

    def plot_tensor_surface(self, save_path: str = None):
        """
        3D surface plot showing piezoelectric response as function of stress direction
        """
        fig = plt.figure(figsize=(16, 5))

        # Create spherical grid
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        THETA, PHI = np.meshgrid(theta, phi)

        # For each direction, calculate piezoelectric response
        for idx, (component, title) in enumerate([
            (0, 'X-component (d₁)'),
            (1, 'Y-component (d₂)'),
            (2, 'Z-component (d₃)')
        ]):
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')

            # Response magnitude for each direction
            response = np.zeros_like(THETA)

            for i in range(THETA.shape[0]):
                for j in range(THETA.shape[1]):
                    # Stress direction (unit vector)
                    n_x = np.sin(PHI[i,j]) * np.cos(THETA[i,j])
                    n_y = np.sin(PHI[i,j]) * np.sin(THETA[i,j])
                    n_z = np.cos(PHI[i,j])

                    # Uniaxial stress in this direction
                    stress = np.outer([n_x, n_y, n_z], [n_x, n_y, n_z])

                    # Calculate electric field
                    E = self.tensor.calculate_electric_field(stress * 1e6)  # 1 MPa stress

                    # Response magnitude for this component
                    response[i,j] = abs(E[component])

            # Normalize for visualization
            response_norm = response / (np.max(response) + 1e-20)

            # Convert to Cartesian coordinates
            R = 1 + response_norm  # Radial distance
            X = R * np.sin(PHI) * np.cos(THETA)
            Y = R * np.sin(PHI) * np.sin(THETA)
            Z = R * np.cos(PHI)

            # Plot surface
            surf = ax.plot_surface(X, Y, Z, facecolors=cm.plasma(response_norm),
                                  alpha=0.9, edgecolor='none', shade=True)

            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_coupling_modes(self, save_path: str = None):
        """
        Visualize coupling between different stress modes and electric field directions
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        stress_modes = [
            ('Tensile X', np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])),
            ('Tensile Y', np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
            ('Tensile Z', np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])),
            ('Shear XY', np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])),
            ('Shear YZ', np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])),
            ('Shear XZ', np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])),
        ]

        for idx, (name, stress_pattern) in enumerate(stress_modes):
            ax = axes[idx // 3, idx % 3]

            # Vary stress magnitude
            stress_magnitudes = np.linspace(0, 10e6, 100)  # 0 to 10 MPa
            E_x = np.zeros_like(stress_magnitudes)
            E_y = np.zeros_like(stress_magnitudes)
            E_z = np.zeros_like(stress_magnitudes)

            for i, sigma in enumerate(stress_magnitudes):
                E = self.tensor.calculate_electric_field(stress_pattern * sigma)
                E_x[i] = E[0]
                E_y[i] = E[1]
                E_z[i] = E[2]

            stress_MPa = stress_magnitudes / 1e6

            ax.plot(stress_MPa, E_x, linewidth=2, label='$E_x$', color='red')
            ax.plot(stress_MPa, E_y, linewidth=2, label='$E_y$', color='green')
            ax.plot(stress_MPa, E_z, linewidth=2, label='$E_z$', color='blue')

            ax.set_xlabel('Stress (MPa)', fontsize=10)
            ax.set_ylabel('Electric Field (V/m)', fontsize=10)
            ax.set_title(f'{name} Stress', fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_converse_effect(self, save_path: str = None):
        """
        Visualize converse piezoelectric effect (E → strain)
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        E_magnitudes = np.linspace(0, 1e7, 100)  # 0 to 10 MV/m

        field_directions = [
            ('X-field', np.array([1, 0, 0])),
            ('Y-field', np.array([0, 1, 0])),
            ('Z-field', np.array([0, 0, 1]))
        ]

        for idx, (name, E_dir) in enumerate(field_directions):
            ax = axes[idx]

            # Strain components
            strain_xx = np.zeros_like(E_magnitudes)
            strain_yy = np.zeros_like(E_magnitudes)
            strain_zz = np.zeros_like(E_magnitudes)
            strain_xy = np.zeros_like(E_magnitudes)
            strain_yz = np.zeros_like(E_magnitudes)
            strain_xz = np.zeros_like(E_magnitudes)

            for i, E_mag in enumerate(E_magnitudes):
                strain = self.tensor.calculate_induced_strain(E_dir * E_mag)
                strain_xx[i] = strain[0, 0]
                strain_yy[i] = strain[1, 1]
                strain_zz[i] = strain[2, 2]
                strain_xy[i] = strain[0, 1]
                strain_yz[i] = strain[1, 2]
                strain_xz[i] = strain[0, 2]

            E_MV = E_magnitudes / 1e6  # MV/m

            ax.plot(E_MV, strain_xx * 1e6, linewidth=2, label='$\\varepsilon_{xx}$')
            ax.plot(E_MV, strain_yy * 1e6, linewidth=2, label='$\\varepsilon_{yy}$')
            ax.plot(E_MV, strain_zz * 1e6, linewidth=2, label='$\\varepsilon_{zz}$')
            ax.plot(E_MV, strain_xy * 1e6, linewidth=2, label='$\\varepsilon_{xy}$', linestyle='--')

            ax.set_xlabel('Electric Field (MV/m)', fontsize=11)
            ax.set_ylabel('Strain (microstrain)', fontsize=11)
            ax.set_title(f'{name} Applied', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_energy_conversion_efficiency(self, save_path: str = None):
        """
        Plot energy conversion efficiency from mechanical to electrical
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Mechanical energy input (various stress levels)
        stress_range = np.linspace(0, 50e6, 100)  # 0 to 50 MPa

        # Optimal stress direction for quartz (along X)
        stress_optimal = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        mechanical_energy = np.zeros_like(stress_range)
        electrical_energy = np.zeros_like(stress_range)

        epsilon_0 = 8.854187817e-12
        volume = 1e-6  # 1 cm³

        for i, sigma in enumerate(stress_range):
            stress = stress_optimal * sigma

            # Mechanical energy density: U_m = (1/2) σ_ij ε_ij
            strain = QUARTZ_COMPLIANCE @ np.array([sigma, 0, 0, 0, 0, 0])
            mechanical_energy[i] = 0.5 * sigma * strain[0] * volume

            # Electrical energy from induced field
            E = self.tensor.calculate_electric_field(stress)
            electrical_energy[i] = 0.5 * epsilon_0 * QUARTZ_EPSILON_R[0,0] * (E[0]**2) * volume

        efficiency = (electrical_energy / (mechanical_energy + 1e-30)) * 100

        # Plot energy
        ax1.plot(stress_range / 1e6, mechanical_energy * 1e6, linewidth=2.5,
                label='Mechanical Input', color='darkblue')
        ax1.plot(stress_range / 1e6, electrical_energy * 1e6, linewidth=2.5,
                label='Electrical Output', color='darkorange')
        ax1.set_xlabel('Applied Stress (MPa)', fontsize=12)
        ax1.set_ylabel('Energy (µJ)', fontsize=12)
        ax1.set_title('Mechanical-to-Electrical Energy Conversion', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # Plot efficiency
        ax2.plot(stress_range / 1e6, efficiency, linewidth=2.5, color='green')
        ax2.set_xlabel('Applied Stress (MPa)', fontsize=12)
        ax2.set_ylabel('Conversion Efficiency (%)', fontsize=12)
        ax2.set_title('Piezoelectric Coupling Efficiency', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_piezo_tensor_visualization():
    """Main visualization runner"""

    print("="*70)
    print("PIEZOELECTRIC TENSOR VISUALIZATION")
    print("Quartz Crystal Electromechanical Coupling")
    print("="*70)

    # Initialize tensor
    piezo = PiezoelectricTensor(QUARTZ_D_TENSOR)

    print(f"\nQuartz Piezoelectric Coefficients (pC/N):")
    print(f"  d₁₁ = {QUARTZ_D_TENSOR[0, 0] * 1e12:.2f}")
    print(f"  d₁₄ = {QUARTZ_D_TENSOR[0, 3] * 1e12:.2f}")
    print(f"  d₃₆ = {QUARTZ_D_TENSOR[2, 5] * 1e12:.2f}")

    # Example calculation
    print(f"\nExample: 1 MPa tensile stress along X-axis")
    stress_x = np.array([[1e6, 0, 0], [0, 0, 0], [0, 0, 0]])
    E_field = piezo.calculate_electric_field(stress_x)
    print(f"  Induced electric field:")
    print(f"    E_x = {E_field[0]:.2f} V/m")
    print(f"    E_y = {E_field[1]:.2f} V/m")
    print(f"    E_z = {E_field[2]:.2f} V/m")

    # Initialize visualizer
    visualizer = QuartzPiezoVisualizer3D(piezo)

    print(f"\n[1/4] Generating 3D tensor surface...")
    visualizer.plot_tensor_surface('quartz_piezo_tensor_3d.png')

    print(f"[2/4] Plotting stress-field coupling modes...")
    visualizer.plot_coupling_modes('quartz_coupling_modes.png')

    print(f"[3/4] Visualizing converse piezoelectric effect...")
    visualizer.plot_converse_effect('quartz_converse_effect.png')

    print(f"[4/4] Analyzing energy conversion efficiency...")
    visualizer.plot_energy_conversion_efficiency('quartz_energy_efficiency.png')

    print(f"\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("Files saved:")
    print("  - quartz_piezo_tensor_3d.png")
    print("  - quartz_coupling_modes.png")
    print("  - quartz_converse_effect.png")
    print("  - quartz_energy_efficiency.png")
    print("="*70)

    return piezo, visualizer


if __name__ == "__main__":
    piezo, viz = run_piezo_tensor_visualization()
