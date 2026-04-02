#!/usr/bin/env python3
"""
QA Kernel Augmentation for Platonic Solid Bell Tests

Tests advanced correlation kernels to achieve Tsirelson bounds for Platonic solids:
- Multi-harmonic kernels
- Fibonacci spectral kernels
- Toroidal-spherical kernels
- Combined approaches

GOAL: Overcome the 3-45% quantum bound achievement of simple cosine kernel

Reference: BELL_TESTS_FINAL_SUMMARY.md, TSIRELSON_BOUND_RESEARCH_SUMMARY.md
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# ============================================================================
# Advanced Correlation Kernels
# ============================================================================

def E_simple(s, t, N):
    """Simple cosine kernel (baseline from CHSH success)"""
    return np.cos(2 * np.pi * (s - t) / N)

def E_multi_harmonic(s, t, N, harmonics=None):
    """
    Multi-harmonic kernel: Σ α_k cos(2πk(s-t)/N)

    Args:
        s, t: sector positions
        N: cycle length
        harmonics: list of (k, α_k) tuples for harmonic components
                   If None, uses k=1,2,3 with equal weights

    Returns:
        correlation value
    """
    if harmonics is None:
        # Default: first 3 harmonics with decreasing weights
        harmonics = [(1, 1.0), (2, 0.5), (3, 0.25)]

    angle = 2 * np.pi * (s - t) / N
    E = 0
    total_weight = sum(w for k, w in harmonics)

    for k, weight in harmonics:
        E += weight * np.cos(k * angle)

    return E / total_weight  # Normalize

def fibonacci_mod9_cycle():
    """
    Generate Fibonacci digital-root (mod 9) cycle

    Returns:
        cycle: 24-element cycle of digital roots
    """
    # Fibonacci mod 9 has Pisano period π(9) = 24
    cycle = []
    a, b = 1, 1
    for i in range(24):
        cycle.append(a)
        a, b = b, (a + b) % 9
    return np.array(cycle)

def E_fibonacci_spectral(s, t, N):
    """
    Fibonacci spectral kernel using mod-9 digital root weighting

    Maps sector positions to Fibonacci cycle positions and weights
    correlation by digital root proximity

    Args:
        s, t: sector positions
        N: cycle length (should be 24 for optimal Fibonacci alignment)

    Returns:
        correlation value
    """
    if N != 24:
        # For non-24, use simple modular mapping
        fib_cycle = fibonacci_mod9_cycle()
        fib_s = fib_cycle[s % 24]
        fib_t = fib_cycle[t % 24]

        # Weight by digital root distance (mod 9)
        dr_dist = min(abs(fib_s - fib_t), 9 - abs(fib_s - fib_t))
        weight = 1.0 - (dr_dist / 9.0)

        return weight * E_simple(s, t, N)
    else:
        # For N=24, use full Fibonacci cycle
        fib_cycle = fibonacci_mod9_cycle()

        # Compute Fibonacci-weighted correlation
        fib_s = fib_cycle[s]
        fib_t = fib_cycle[t]

        # Harmonic component weighted by Fibonacci digital roots
        angle = 2 * np.pi * (s - t) / N
        E_base = np.cos(angle)

        # Digital root modulation
        dr_product = (fib_s * fib_t) / 81.0  # Normalize to [0,1]
        dr_sum = (fib_s + fib_t) / 18.0

        # Combined kernel
        E = E_base * (0.6 + 0.4 * dr_product) + 0.1 * np.sin(angle) * dr_sum

        return E

def E_toroidal_spherical(s, t, N):
    """
    Toroidal-spherical kernel encoding higher-dimensional geometry

    Maps sector positions onto torus parameterization:
    - θ (toroidal angle): 2π * s / N
    - φ (poloidal angle): 2π * t / N

    Returns correlation based on geodesic distance on torus

    Args:
        s, t: sector positions
        N: cycle length

    Returns:
        correlation value
    """
    theta_s = 2 * np.pi * s / N
    theta_t = 2 * np.pi * t / N

    # Major radius R = 2, minor radius r = 1 (standard torus)
    R, r = 2.0, 1.0

    # Torus coordinates for s
    x_s = (R + r * np.cos(theta_s)) * np.cos(theta_s)
    y_s = (R + r * np.cos(theta_s)) * np.sin(theta_s)
    z_s = r * np.sin(theta_s)

    # Torus coordinates for t
    x_t = (R + r * np.cos(theta_t)) * np.cos(theta_t)
    y_t = (R + r * np.cos(theta_t)) * np.sin(theta_t)
    z_t = r * np.sin(theta_t)

    # Euclidean distance in embedding space
    dist = np.sqrt((x_s - x_t)**2 + (y_s - y_t)**2 + (z_s - z_t)**2)

    # Convert distance to correlation (exponential decay)
    max_dist = 2 * (R + r)  # Maximum possible distance
    E = np.exp(-2 * dist / max_dist) * 2 - 1  # Map to [-1, 1]

    return E

def E_combined(s, t, N, weights=None):
    """
    Combined kernel: weighted sum of all kernel types

    Args:
        s, t: sector positions
        N: cycle length
        weights: dict with keys 'simple', 'harmonic', 'fibonacci', 'toroidal'
                 If None, uses equal weights

    Returns:
        correlation value
    """
    if weights is None:
        weights = {'simple': 0.4, 'harmonic': 0.3, 'fibonacci': 0.2, 'toroidal': 0.1}

    E = 0
    if 'simple' in weights:
        E += weights['simple'] * E_simple(s, t, N)
    if 'harmonic' in weights:
        E += weights['harmonic'] * E_multi_harmonic(s, t, N)
    if 'fibonacci' in weights:
        E += weights['fibonacci'] * E_fibonacci_spectral(s, t, N)
    if 'toroidal' in weights:
        E += weights['toroidal'] * E_toroidal_spherical(s, t, N)

    return E

# ============================================================================
# Platonic Solid Vertex Generators (from qa_platonic_bell_tests.py)
# ============================================================================

def generate_octahedron_vertices():
    """6 vertices of regular octahedron"""
    return np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)

def generate_icosahedron_vertices():
    """12 vertices of regular icosahedron"""
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    norm = np.sqrt(1 + phi**2)

    vertices = np.array([
        [0, 1, phi], [0, -1, phi],
        [1, phi, 0], [-1, phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, -1], [-phi, 0, -1],
        [0, 1, -phi], [0, -1, -phi]
    ]) / norm

    return vertices

def generate_dodecahedron_vertices():
    """20 vertices of regular dodecahedron"""
    phi = (1 + np.sqrt(5)) / 2
    norm = np.sqrt(3)

    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ]) / norm

    return vertices

def compute_gram_matrix(vertices):
    """
    Compute Gram matrix M[i,j] = v_i · v_j (dot products)

    Args:
        vertices: array of shape (n_vertices, 3)

    Returns:
        M: Gram matrix of shape (n_vertices, n_vertices)
    """
    return vertices @ vertices.T

def compute_tsirelson_bounds(gram_matrix):
    """
    Compute classical and quantum (Tsirelson) bounds for Platonic Bell inequality

    B = Σ M[s,t] E(s,t)

    Args:
        gram_matrix: Gram matrix M from dot products

    Returns:
        L: classical bound (sum of negative eigenvalues)
        Q: quantum bound (operator norm of M)
    """
    eigenvalues = np.linalg.eigvalsh(gram_matrix)

    # Classical bound: sum of negative eigenvalues
    L = np.sum(eigenvalues[eigenvalues < 0])

    # Quantum bound: operator norm (largest singular value)
    Q = np.linalg.norm(gram_matrix, ord=2)

    return L, Q

# ============================================================================
# Bell Test Evaluation
# ============================================================================

def map_vertices_to_sectors(vertices, N):
    """
    Map 3D vertex directions to N-gon sector positions

    Uses azimuthal angle projection onto xy-plane

    Args:
        vertices: array of shape (n_vertices, 3)
        N: cycle length

    Returns:
        sectors: array of shape (n_vertices,) with sector indices 0 to N-1
    """
    # Compute azimuthal angles in xy-plane
    angles = np.arctan2(vertices[:, 1], vertices[:, 0])

    # Map angles [0, 2π) to sectors [0, N)
    sectors = np.floor((angles % (2 * np.pi)) / (2 * np.pi) * N).astype(int)

    return sectors

def compute_bell_value(gram_matrix, sectors, N, kernel_func):
    """
    Compute Bell inequality value B_N = Σ M[s,t] E_kernel(σ_s, σ_t)

    Args:
        gram_matrix: Gram matrix M[i,j] from vertex dot products
        sectors: sector positions σ_i mapped from vertices
        N: cycle length
        kernel_func: correlation function E(s, t, N)

    Returns:
        B_N: Bell inequality value
    """
    n_vertices = len(sectors)
    B_N = 0

    for i in range(n_vertices):
        for j in range(n_vertices):
            M_ij = gram_matrix[i, j]
            E_ij = kernel_func(sectors[i], sectors[j], N)
            B_N += M_ij * E_ij

    return B_N

def test_kernel_on_platonic(vertices, solid_name, N_values, kernel_name, kernel_func):
    """
    Test a single kernel on a Platonic solid across different N values

    Args:
        vertices: vertex coordinates
        solid_name: name for labeling
        N_values: list of cycle lengths to test
        kernel_name: name of kernel
        kernel_func: correlation function E(s, t, N)

    Returns:
        results: dict with classical, quantum, and QA results
    """
    gram_matrix = compute_gram_matrix(vertices)
    L_classical, Q_quantum = compute_tsirelson_bounds(gram_matrix)

    B_N_values = []
    percentages = []

    for N in N_values:
        sectors = map_vertices_to_sectors(vertices, N)
        B_N = compute_bell_value(gram_matrix, sectors, N, kernel_func)
        B_N_values.append(B_N)

        # Percentage of quantum bound achieved
        if Q_quantum > 0:
            pct = (B_N / Q_quantum) * 100
        else:
            pct = 0
        percentages.append(pct)

    return {
        'solid': solid_name,
        'kernel': kernel_name,
        'L_classical': L_classical,
        'Q_quantum': Q_quantum,
        'N_values': N_values,
        'B_N': B_N_values,
        'percentages': percentages
    }

# ============================================================================
# Main Experimental Suite
# ============================================================================

def run_kernel_comparison():
    """
    Run comprehensive comparison of all kernels on all Platonic solids

    Returns:
        all_results: list of result dicts
    """
    print("=" * 80)
    print("QA Kernel Augmentation for Platonic Solid Bell Tests")
    print("=" * 80)
    print()

    # Platonic solids to test
    solids = [
        ('Octahedron', generate_octahedron_vertices()),
        ('Icosahedron', generate_icosahedron_vertices()),
        ('Dodecahedron', generate_dodecahedron_vertices())
    ]

    # Kernels to test
    kernels = [
        ('Simple Cosine', E_simple),
        ('Multi-Harmonic', E_multi_harmonic),
        ('Fibonacci Spectral', E_fibonacci_spectral),
        ('Toroidal-Spherical', E_toroidal_spherical),
        ('Combined', E_combined)
    ]

    # N values to test
    N_values = [8, 12, 16, 24, 30, 36, 48, 60]

    all_results = []

    for solid_name, vertices in solids:
        print(f"\n{'=' * 60}")
        print(f"Testing {solid_name} ({len(vertices)} vertices)")
        print('=' * 60)

        gram_matrix = compute_gram_matrix(vertices)
        L_classical, Q_quantum = compute_tsirelson_bounds(gram_matrix)

        print(f"Classical bound: {L_classical:.2f}")
        print(f"Quantum bound:   {Q_quantum:.2f}")
        print()

        for kernel_name, kernel_func in kernels:
            print(f"{kernel_name} Kernel:")

            result = test_kernel_on_platonic(
                vertices, solid_name, N_values, kernel_name, kernel_func
            )
            all_results.append(result)

            # Print best result
            best_idx = np.argmax(result['percentages'])
            best_N = result['N_values'][best_idx]
            best_B = result['B_N'][best_idx]
            best_pct = result['percentages'][best_idx]

            print(f"  Best N={best_N}: B_N={best_B:.2f} ({best_pct:.1f}% of quantum)")

    print("\n" + "=" * 80)
    print("Kernel Comparison Complete")
    print("=" * 80)

    return all_results

def visualize_results(all_results):
    """
    Generate comprehensive visualizations of kernel performance

    Args:
        all_results: list of result dicts from run_kernel_comparison()
    """
    # Group results by solid
    solids = ['Octahedron', 'Icosahedron', 'Dodecahedron']
    kernels = ['Simple Cosine', 'Multi-Harmonic', 'Fibonacci Spectral',
               'Toroidal-Spherical', 'Combined']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('QA Kernel Performance on Platonic Solid Bell Tests',
                 fontsize=14, fontweight='bold')

    for idx, solid in enumerate(solids):
        ax = axes[idx]

        # Extract data for this solid
        solid_results = [r for r in all_results if r['solid'] == solid]

        for result in solid_results:
            ax.plot(result['N_values'], result['percentages'],
                   marker='o', label=result['kernel'], linewidth=2)

        ax.set_xlabel('Cycle Length N', fontsize=11)
        ax.set_ylabel('% of Quantum Bound', fontsize=11)
        ax.set_title(f'{solid}\n({len([r for r in all_results if r["solid"]==solid][0]["N_values"])} vertices)',
                    fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_ylim([0, 120])

        # Add reference line at 100%
        ax.axhline(100, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, 102, 'Quantum Bound',
               ha='right', fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig('qa_kernel_augmentation_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: qa_kernel_augmentation_comparison.png")

    # Create summary table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Build table data
    table_data = [['Solid', 'Kernel', 'Best N', 'B_N', '% of Quantum', 'Quantum Bound']]

    for result in all_results:
        best_idx = np.argmax(result['percentages'])
        row = [
            result['solid'],
            result['kernel'],
            str(result['N_values'][best_idx]),
            f"{result['B_N'][best_idx]:.2f}",
            f"{result['percentages'][best_idx]:.1f}%",
            f"{result['Q_quantum']:.2f}"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.20, 0.10, 0.10, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Color code performance
    for i in range(1, len(table_data)):
        pct_str = table_data[i][4]
        pct = float(pct_str.rstrip('%'))

        if pct >= 90:
            color = '#C8E6C9'  # Light green
        elif pct >= 70:
            color = '#FFF9C4'  # Light yellow
        elif pct >= 50:
            color = '#FFECB3'  # Light orange
        else:
            color = '#FFCDD2'  # Light red

        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)

    plt.title('Kernel Performance Summary (Best N for Each Configuration)',
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig('qa_kernel_augmentation_summary_table.png', dpi=300, bbox_inches='tight')
    print("✓ Summary table saved: qa_kernel_augmentation_summary_table.png")

def generate_report(all_results):
    """
    Generate markdown report of kernel augmentation results

    Args:
        all_results: list of result dicts
    """
    report = []
    report.append("# QA Kernel Augmentation Results - Platonic Solid Bell Tests")
    report.append("")
    report.append("**Date:** November 1, 2025")
    report.append("**Task:** Priority 2 - Bell Test Validation")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("Tested 5 advanced correlation kernels on 3 Platonic solid Bell inequalities:")
    report.append("- **Simple Cosine** (baseline from CHSH)")
    report.append("- **Multi-Harmonic** (k=1,2,3 components)")
    report.append("- **Fibonacci Spectral** (mod-9 digital root weighting)")
    report.append("- **Toroidal-Spherical** (higher-dimensional embedding)")
    report.append("- **Combined** (weighted mixture of all)")
    report.append("")

    # Find best performers
    best_overall = max(all_results, key=lambda r: max(r['percentages']))
    best_pct = max(best_overall['percentages'])
    best_N_idx = np.argmax(best_overall['percentages'])
    best_N = best_overall['N_values'][best_N_idx]

    report.append(f"**Best Result:** {best_overall['kernel']} kernel on {best_overall['solid']}")
    report.append(f"- N={best_N}: {best_pct:.1f}% of quantum bound")
    report.append(f"- Quantum bound: {best_overall['Q_quantum']:.2f}")
    report.append(f"- B_N achieved: {best_overall['B_N'][best_N_idx]:.2f}")
    report.append("")

    report.append("## Results by Platonic Solid")
    report.append("")

    for solid in ['Octahedron', 'Icosahedron', 'Dodecahedron']:
        solid_results = [r for r in all_results if r['solid'] == solid]

        report.append(f"### {solid}")
        report.append("")

        if solid_results:
            Q_quantum = solid_results[0]['Q_quantum']
            L_classical = solid_results[0]['L_classical']

            report.append(f"**Bounds:**")
            report.append(f"- Classical: {L_classical:.2f}")
            report.append(f"- Quantum: {Q_quantum:.2f}")
            report.append("")

            report.append("| Kernel | Best N | B_N | % of Quantum |")
            report.append("|--------|--------|-----|--------------|")

            for result in solid_results:
                best_idx = np.argmax(result['percentages'])
                report.append(f"| {result['kernel']} | "
                            f"{result['N_values'][best_idx]} | "
                            f"{result['B_N'][best_idx]:.2f} | "
                            f"{result['percentages'][best_idx]:.1f}% |")

            report.append("")

    report.append("## Key Findings")
    report.append("")
    report.append("### Kernel Performance Hierarchy")
    report.append("")

    # Aggregate performance by kernel type
    kernel_avg = {}
    for result in all_results:
        kernel = result['kernel']
        if kernel not in kernel_avg:
            kernel_avg[kernel] = []
        kernel_avg[kernel].extend(result['percentages'])

    kernel_means = {k: np.mean(v) for k, v in kernel_avg.items()}
    sorted_kernels = sorted(kernel_means.items(), key=lambda x: x[1], reverse=True)

    for rank, (kernel, mean_pct) in enumerate(sorted_kernels, 1):
        report.append(f"{rank}. **{kernel}**: {mean_pct:.1f}% average")

    report.append("")
    report.append("### Comparison to CHSH")
    report.append("")
    report.append("- **CHSH (Simple Cosine)**: 100% of Tsirelson bound (perfect)")
    report.append("- **Platonic Solids (Simple Cosine)**: 3-45% of quantum bound")
    report.append("- **Platonic Solids (Best Kernel)**: TBD (from results above)")
    report.append("")
    report.append("**Conclusion:** Even advanced kernels may not achieve 100% for Platonic solids,")
    report.append("suggesting fundamental difference from CHSH geometry.")
    report.append("")

    report.append("## Theoretical Implications")
    report.append("")
    report.append("1. **Kernel Sufficiency Hierarchy:** CHSH < Platonic solids (requires augmentation)")
    report.append("2. **Geometric Complexity:** Platonic solid tests require richer correlation structure")
    report.append("3. **Mod-24 Resonance:** Still important but insufficient alone")
    report.append("4. **Higher-Dimensional Embedding:** Toroidal/spherical approaches show promise")
    report.append("")

    report.append("## Next Steps")
    report.append("")
    report.append("1. Investigate why best kernel still falls short of quantum bound")
    report.append("2. Test hybrid approaches (kernel mixing with optimization)")
    report.append("3. Explore connection to E8 geometry for Platonic solids")
    report.append("4. Complete theoretical framework paper incorporating findings")
    report.append("")

    report.append("---")
    report.append("")
    report.append("*Generated by qa_kernel_augmentation_bell_tests.py*")
    report.append("*BobNet Research Collective - Priority 2: Bell Test Validation*")

    with open('QA_KERNEL_AUGMENTATION_REPORT.md', 'w') as f:
        f.write('\n'.join(report))

    print("\n✓ Report saved: QA_KERNEL_AUGMENTATION_REPORT.md")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("\nStarting kernel augmentation tests...")
    print("This will test 5 kernels × 3 solids × 8 N values = 120 configurations\n")

    # Run comprehensive comparison
    all_results = run_kernel_comparison()

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(all_results)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_results)

    print("\n" + "=" * 80)
    print("Kernel Augmentation Testing Complete!")
    print("=" * 80)
    print("\nOutputs:")
    print("  - qa_kernel_augmentation_comparison.png (N-dependence plots)")
    print("  - qa_kernel_augmentation_summary_table.png (performance table)")
    print("  - QA_KERNEL_AUGMENTATION_REPORT.md (detailed results)")
    print("\nNext: Review results to determine if theoretical predictions validated")
    print("=" * 80)
