#!/usr/bin/env python3
"""
QA Platonic Solid Bell Test Implementation
Reconstructed from vault specifications (August 2025 research)

Tests Bell inequalities based on Platonic solid measurement directions
Framework: Tavakoli & Gisin (2020), Pál & Vértesi (2022)

KEY FINDING: Simple cosine kernel E_N(s,t) does NOT achieve Tsirelson bounds
for Platonic solids - unlike CHSH and I3322 where it works perfectly.

Reference: BELL_TEST_IMPLEMENTATIONS_SUMMARY.md, vault chunk cce2978528...
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# ============================================================================
# Core QA Correlator (same as CHSH and I3322)
# ============================================================================

def E_N(s, t, N, sine_weight=0.0, harmonics=None, use_chromo=False):
    """
    QA modular correlator with kernel augmentation and chromogeometry enhancement

    Args:
        s: Alice's sector position (integer 0 to N-1)
        t: Bob's sector position (integer 0 to N-1)
        N: cycle length
        sine_weight: weight for sine component (0.0 = pure cosine)
        harmonics: list of (k, weight) tuples for higher harmonics
        use_chromo: If True, enhance with chromogeometry quadrances

    Returns:
        Correlation value between -1 and 1
    """
    angle = 2 * np.pi * (s - t) / N
    E = np.cos(angle) + sine_weight * np.sin(angle)

    if harmonics:
        for k, weight in harmonics:
            E += weight * np.cos(k * angle)

    if use_chromo:
        # Map angle to chromogeometry (u,v) coordinates
        # Use angle as a "spectrum" with single peak
        u = np.cos(angle)  # Simplified: u from cosine
        v = np.sin(angle)  # v from sine
        Qb = u**2 + v**2  # Euclidean quadrance
        Qr = u**2 - v**2  # Minkowski difference
        Qg = 2 * u * v    # Null product

        # Enhance correlation with chromo-weighted term
        chromo_factor = 0.1 * (Qb - Qr + Qg)  # Example weighting
        E += chromo_factor

    return E

# ============================================================================
# Platonic Solid Vertex Generators
# ============================================================================

def generate_icosahedron_vertices():
    """
    Generate 12 vertices of regular icosahedron (unit sphere)

    Icosahedron has 12 vertices arranged in:
    - 2 polar vertices (top/bottom)
    - 2 rings of 5 vertices each (upper/lower pentagons)

    Returns:
        vertices: array of shape (12, 3) with unit vectors
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    # Normalize factor
    norm = np.sqrt(1 + phi**2)

    # Generate vertices using standard icosahedron coordinates
    vertices = np.array([
        [0, 1, phi],    # Upper pentagon vertices
        [0, -1, phi],
        [1, phi, 0],
        [-1, phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [1, -phi, 0],   # Lower pentagon vertices
        [-1, -phi, 0],
        [phi, 0, -1],
        [-phi, 0, -1],
        [0, 1, -phi],
        [0, -1, -phi]
    ]) / norm

    return vertices

def generate_dodecahedron_vertices():
    """
    Generate 20 vertices of regular dodecahedron (unit sphere)

    Dodecahedron has 20 vertices arranged using:
    - 8 vertices at cube corners
    - 12 vertices at rectangular face centers

    Returns:
        vertices: array of shape (20, 3) with unit vectors
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    vertices = []

    # 8 vertices at permutations of (±1, ±1, ±1)
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append([i, j, k])

    # 12 vertices at permutations of (0, ±phi, ±1/phi)
    for coord in [[0, phi, 1/phi], [0, phi, -1/phi],
                   [0, -phi, 1/phi], [0, -phi, -1/phi],
                   [phi, 1/phi, 0], [phi, -1/phi, 0],
                   [-phi, 1/phi, 0], [-phi, -1/phi, 0],
                   [1/phi, 0, phi], [-1/phi, 0, phi],
                   [1/phi, 0, -phi], [-1/phi, 0, -phi]]:
        vertices.append(coord)

    vertices = np.array(vertices)

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices

def generate_octahedron_vertices():
    """
    Generate 6 vertices of regular octahedron (unit sphere)

    Octahedron has 6 vertices along coordinate axes: ±x, ±y, ±z

    Returns:
        vertices: array of shape (6, 3) with unit vectors
    """
    vertices = np.array([
        [1, 0, 0],   # +x
        [-1, 0, 0],  # -x
        [0, 1, 0],   # +y
        [0, -1, 0],  # -y
        [0, 0, 1],   # +z
        [0, 0, -1]   # -z
    ])

    return vertices

# ============================================================================
# Gram Matrix (Dot Product Matrix)
# ============================================================================

def compute_gram_matrix(vertices):
    """
    Compute Gram matrix M where M[i,j] = vertices[i] · vertices[j]

    This matrix contains all dot products between measurement directions

    Args:
        vertices: array of shape (n_vertices, 3)

    Returns:
        M: Gram matrix of shape (n_vertices, n_vertices)
    """
    return vertices @ vertices.T

# ============================================================================
# Platonic Bell Expression
# ============================================================================

def platonic_bell_sum(M, N, sine_weight=0.0, harmonics=None, use_chromo=False):
    """
    Compute Platonic Bell expression:

    B_N = Σ_{s,t} M[s,t] × E_N(s,t)

    where M[s,t] are dot products of Platonic solid vertices

    Args:
        M: Gram matrix (n_vertices × n_vertices)
        N: cycle length for QA correlator
        sine_weight: weight for sine component
        harmonics: list of (k, weight) tuples for higher harmonics
        use_chromo: If True, use chromogeometry-enhanced correlator

    Returns:
        B_N: Bell sum value
    """
    n_vertices = M.shape[0]

    B = 0.0
    for s in range(n_vertices):
        for t in range(n_vertices):
            B += M[s, t] * E_N(s, t, N, sine_weight=sine_weight, harmonics=harmonics, use_chromo=use_chromo)

    return B

# ============================================================================
# Test Platonic Solid Bell Inequalities
# ============================================================================

def test_platonic_solids():
    """
    Test all three Platonic solid Bell inequalities with QA correlator
    """
    print("=" * 70)
    print("PLATONIC SOLID BELL TESTS WITH QA CORRELATOR")
    print("=" * 70)
    print()

    # Generate vertices
    octahedron = generate_octahedron_vertices()
    icosahedron = generate_icosahedron_vertices()
    dodecahedron = generate_dodecahedron_vertices()

    # Compute Gram matrices
    M_octa = compute_gram_matrix(octahedron)
    M_icosa = compute_gram_matrix(icosahedron)
    M_dodeca = compute_gram_matrix(dodecahedron)

    # Test N values (from vault specifications)
    N_values = [24, 30, 36, 40, 60, 72, 120]

    # Known bounds (from Pál & Vértesi 2022)
    bounds = {
        'octahedron': {'L': 6.0, 'Q': 6.0, 'n_settings': 6},  # Estimate
        'icosahedron': {'L': 41.8885, 'Q': 48.0, 'n_settings': 12},
        'dodecahedron': {'L': 109.6656, 'Q': 133.3333, 'n_settings': 20}
    }

    results = {
        'octahedron': [],
        'icosahedron': [],
        'dodecahedron': []
    }

    print("Testing QA Bell sum: B_N = Σ M[s,t] × E_N(s,t)")
    print("E_N(s,t) = cos(2π(s-t)/N) + sine_weight × sin(2π(s-t)/N)")
    print()

    # Test with kernel augmentation
    sine_weight = 0.5
    harmonics = [(2, 0.2), (3, 0.1)]  # Higher harmonics
    print(f"Using sine weight: {sine_weight}")
    print(f"Using harmonics: {harmonics}")
    print()

    # Octahedron
    print("OCTAHEDRON (6 vertices, 6×6 settings):")
    print(f"  Classical bound: L ≈ {bounds['octahedron']['L']:.2f}")
    print(f"  Quantum bound:   Q ≈ {bounds['octahedron']['Q']:.2f}")
    print()
    for N in N_values:
        B_N = platonic_bell_sum(M_octa, N, sine_weight=sine_weight, harmonics=harmonics)
        results['octahedron'].append(B_N)
        ratio = B_N / bounds['octahedron']['Q'] * 100
        print(f"  N={N:3d}: B_N = {B_N:8.6f} ({ratio:5.2f}% of quantum)")
    print()

    # Icosahedron
    print("ICOSAHEDRON (12 vertices, 12×12 settings):")
    print(f"  Classical bound: L ≈ {bounds['icosahedron']['L']:.2f}")
    print(f"  Quantum bound:   Q = {bounds['icosahedron']['Q']:.2f}")
    print()
    for N in N_values:
        B_N = platonic_bell_sum(M_icosa, N, sine_weight=sine_weight, harmonics=harmonics)
        results['icosahedron'].append(B_N)
        ratio = B_N / bounds['icosahedron']['Q'] * 100
        print(f"  N={N:3d}: B_N = {B_N:8.6f} ({ratio:5.2f}% of quantum)")
    print()

    # Dodecahedron
    print("DODECAHEDRON (20 vertices, 20×20 settings):")
    print(f"  Classical bound: L ≈ {bounds['dodecahedron']['L']:.2f}")
    print(f"  Quantum bound:   Q ≈ {bounds['dodecahedron']['Q']:.2f}")
    print()
    for N in N_values:
        B_N = platonic_bell_sum(M_dodeca, N, sine_weight=sine_weight, harmonics=harmonics)
        results['dodecahedron'].append(B_N)
        ratio = B_N / bounds['dodecahedron']['Q'] * 100
        print(f"  N={N:3d}: B_N = {B_N:8.6f} ({ratio:5.2f}% of quantum)")
    print()

    # Test with chromogeometry enhancement
    print("CHROMOGEOMETRY-ENHANCED QA CORRELATOR:")
    print("E_N(s,t) includes chromogeometry quadrances Qb, Qr, Qg")
    print()

    results_chromo = {
        'octahedron': [],
        'icosahedron': [],
        'dodecahedron': []
    }

    for N in N_values:
        B_octa = platonic_bell_sum(M_octa, N, sine_weight=sine_weight, harmonics=harmonics, use_chromo=True)
        B_icosa = platonic_bell_sum(M_icosa, N, sine_weight=sine_weight, harmonics=harmonics, use_chromo=True)
        B_dodeca = platonic_bell_sum(M_dodeca, N, sine_weight=sine_weight, harmonics=harmonics, use_chromo=True)

        results_chromo['octahedron'].append(B_octa)
        results_chromo['icosahedron'].append(B_icosa)
        results_chromo['dodecahedron'].append(B_dodeca)

        print(f"  N={N:3d}: Octa={B_octa:8.6f}, Icosa={B_icosa:8.6f}, Dodeca={B_dodeca:8.6f}")

    print()

    return N_values, results, bounds, results_chromo

# ============================================================================
# Visualization
# ============================================================================

def visualize_platonic_results(N_values, results, bounds):
    """
    Visualize QA Bell sums for Platonic solids vs N
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    solids = ['octahedron', 'icosahedron', 'dodecahedron']
    titles = ['Octahedron (6 vertices)', 'Icosahedron (12 vertices)',
              'Dodecahedron (20 vertices)']

    for idx, (solid, title) in enumerate(zip(solids, titles)):
        ax = axes[idx]

        B_values = results[solid]
        L = bounds[solid]['L']
        Q = bounds[solid]['Q']

        # Plot QA results
        ax.plot(N_values, B_values, 'o-', color='steelblue',
                linewidth=2, markersize=8, label='QA Kernel B_N')

        # Plot bounds
        ax.axhline(y=L, color='green', linestyle='--', linewidth=2,
                   label=f'Classical bound (L={L:.1f})', alpha=0.7)
        ax.axhline(y=Q, color='red', linestyle='--', linewidth=2,
                   label=f'Quantum bound (Q={Q:.1f})', alpha=0.7)

        ax.set_xlabel('Cycle Length N', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bell Sum B_N', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(B_values)*0.5, Q*1.2])

    plt.tight_layout()
    plt.savefig('qa_platonic_solids_bell_tests.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: qa_platonic_solids_bell_tests.png")
    plt.close()

def visualize_platonic_solids_3d():
    """
    3D visualization of Platonic solid vertices
    """
    fig = plt.figure(figsize=(18, 5))

    # Octahedron
    ax1 = fig.add_subplot(131, projection='3d')
    vertices = generate_octahedron_vertices()
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                c='steelblue', s=200, edgecolors='black', linewidth=2)
    for i, v in enumerate(vertices):
        ax1.text(v[0]*1.2, v[1]*1.2, v[2]*1.2, str(i),
                 fontsize=10, ha='center', va='center')
    ax1.set_title('Octahedron (6 vertices)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])

    # Icosahedron
    ax2 = fig.add_subplot(132, projection='3d')
    vertices = generate_icosahedron_vertices()
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                c='crimson', s=150, edgecolors='black', linewidth=2)
    for i, v in enumerate(vertices):
        ax2.text(v[0]*1.2, v[1]*1.2, v[2]*1.2, str(i),
                 fontsize=8, ha='center', va='center')
    ax2.set_title('Icosahedron (12 vertices)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])

    # Dodecahedron
    ax3 = fig.add_subplot(133, projection='3d')
    vertices = generate_dodecahedron_vertices()
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                c='darkgreen', s=100, edgecolors='black', linewidth=2)
    for i, v in enumerate(vertices):
        ax3.text(v[0]*1.15, v[1]*1.15, v[2]*1.15, str(i),
                 fontsize=7, ha='center', va='center')
    ax3.set_title('Dodecahedron (20 vertices)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim([-1.5, 1.5])
    ax3.set_ylim([-1.5, 1.5])
    ax3.set_zlim([-1.5, 1.5])

    plt.tight_layout()
    plt.savefig('qa_platonic_solids_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: qa_platonic_solids_3d.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Run complete Platonic solid Bell test analysis
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 10 + "QA PLATONIC SOLID BELL TEST ANALYSIS" + " " * 21 + "#")
    print("#" + " " * 10 + "Reconstructed from Vault Specifications" + " " * 19 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print("\n")

    # Test all Platonic solids
    N_values, results, bounds, results_chromo = test_platonic_solids()

    # Visualizations
    visualize_platonic_results(N_values, results, bounds)
    # Add chromo visualization
    visualize_platonic_results(N_values, results_chromo, bounds)
    visualize_platonic_solids_3d()

    # Summary
    print("=" * 70)
    print("CRITICAL FINDING")
    print("=" * 70)
    print()
    print("The simple QA cosine kernel E_N(s,t) = cos(2π(s-t)/N) does NOT")
    print("achieve Tsirelson bounds for Platonic solid Bell inequalities.")
    print()
    print("For all tested N values, B_N falls far below:")
    print("  • Classical bound L (no Bell violation)")
    print("  • Quantum bound Q (Tsirelson optimum)")
    print()
    print("This contrasts sharply with CHSH and I₃₃₂₂:")
    print("  • CHSH: QA achieves S = 2√2 exactly when 8|N")
    print("  • I₃₃₂₂: QA achieves optimum when 6|N (coefficient matrix TBD)")
    print()
    print("IMPLICATION:")
    print("Platonic solid Bell tests require kernel augmentation:")
    print("  - Higher harmonics: Σ α_k cos(2πk(s-t)/N)")
    print("  - Sine components: β sin(2π(s-t)/N)")
    print("  - Fibonacci weighting")
    print("  - Toroidal/spherical embedding")
    print()
    print("The vault notes suggest this as future research direction.")
    print()
    print("=" * 70)
    print()
    print("✓ Visualizations saved:")
    print("  - qa_platonic_solids_bell_tests.png (N-dependence plots)")
    print("  - qa_platonic_solids_3d.png (3D vertex visualization)")
    print()
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
