#!/usr/bin/env python3
"""
Volk Toroidal Geometry Visualizer — QA ↔ Bipolar ↔ Torus

Reconstructs Greg Volk's diagrams from "Toroids, Vortices, Knots, Topology and Quanta":
1. Apollonian circles (E-circles + M-circles) in the bipolar plane
2. Inner quantum ellipse with inscribed isosceles triangle
3. 3D torus surface with (m,n) knot winding
4. Toroidal cross-section showing R=G, r=F relationship
5. Grant's sum-product triangle of means

All parameterized by QA tuple (b, e, d, a).

Volk → QA mapping:
  2a_Volk = C = 2ed     (focal separation / bipolar scale)
  a_Volk  = X = ed       (half-focal distance)
  R_torus = G = d²+e²   (major radius)
  r_torus = F = ab       (minor radius)
  E-circles → additive/AP families
  M-circles → multiplicative/GP families

Usage:
    python qa_volk_torus_visualizer.py              # default (1,2,3,5)
    python qa_volk_torus_visualizer.py 1 1 2 3
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def qa_ids(b, e, d, a):
    assert d == b + e and a == b + 2*e
    C = 2*d*e; F = a*b; G = d*d + e*e
    H = C + F; I = abs(C - F); L = (C*F)//12
    X = e*d; J = b*d; K = a*d
    return dict(b=b, e=e, d=d, a=a, C=C, F=F, G=G, H=H, I=I, L=L,
                X=X, J=J, K=K, product=b*e*d*a,
                gender='female' if b%2==0 else 'male')


def draw_apollonian(ax, ids):
    """Fig 1: Apollonian circles — E-circles (green, additive) and M-circles (red, multiplicative)."""
    volk_a = ids['X']  # = ed, bipolar scale parameter

    ax.set_title(f"Volk's Apollonian Circles\n"
                 f"Bipolar scale a = X = ed = {volk_a}",
                 fontsize=12, fontweight='bold')

    # Poles
    ax.plot([-volk_a, volk_a], [0, 0], 'ko', markersize=7, zorder=5)
    ax.text(-volk_a, -0.15*volk_a, '−X', ha='center', fontsize=10, fontweight='bold')
    ax.text(volk_a, -0.15*volk_a, '+X', ha='center', fontsize=10, fontweight='bold')

    # E-circles: pass through both poles
    # Center at (0, h), radius r = sqrt(h² + a²)
    for i, eta in enumerate([0.2, 0.5, 0.8, 1.2, 2.0, 4.0]):
        h = volk_a * eta
        r = np.sqrt(h*h + volk_a*volk_a)
        alpha = 0.5 - 0.05*i
        ax.add_patch(plt.Circle((0, h), r, fill=False, color='#228B22', alpha=max(alpha, 0.15), lw=1.2))
        ax.add_patch(plt.Circle((0, -h), r, fill=False, color='#228B22', alpha=max(alpha, 0.15), lw=1.2))

    # M-circles: orthogonal family
    for rho in [0.3, 0.5, 0.8, 1.2, 2.0, 4.0]:
        coth = np.cosh(rho) / np.sinh(rho)
        sinh = np.sinh(rho)
        cx = volk_a * coth
        cr = volk_a / sinh
        ax.add_patch(plt.Circle((cx, 0), cr, fill=False, color='#DC143C', alpha=0.4, lw=1.2))
        ax.add_patch(plt.Circle((-cx, 0), cr, fill=False, color='#DC143C', alpha=0.4, lw=1.2))

    # Base circle ρ₀ = a
    ax.add_patch(plt.Circle((0, 0), volk_a, fill=False, color='blue', lw=2.5, ls='--'))

    # Labels
    ax.text(0.03, 0.97, 'E-circles (green) = Additive/AP', transform=ax.transAxes,
            fontsize=9, color='#228B22', va='top', fontweight='bold')
    ax.text(0.03, 0.91, 'M-circles (red) = Multiplicative/GP', transform=ax.transAxes,
            fontsize=9, color='#DC143C', va='top', fontweight='bold')
    ax.text(0.03, 0.85, f'Base circle (blue) ρ₀ = X = {volk_a}', transform=ax.transAxes,
            fontsize=9, color='blue', va='top', fontweight='bold')

    lim = volk_a * 4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)


def draw_quantum_ellipse(ax, ids):
    """Fig 2: Inner quantum ellipse with inscribed isosceles triangle."""
    b, e, d, a = ids['b'], ids['e'], ids['d'], ids['a']
    F = ids['F']
    C = ids['C']

    ax.set_title('Inner Quantum Ellipse\n'
                 f'Foci ±e = ±{e}, semi-major = d = {d}',
                 fontsize=12, fontweight='bold')

    # Ellipse: foci at ±e, semi-major = d, semi-minor = √F
    a_ell = d
    b_ell = np.sqrt(F)

    theta = np.linspace(0, 2*np.pi, 300)
    ell_x = a_ell * np.cos(theta)
    ell_y = b_ell * np.sin(theta)
    ax.plot(ell_x, ell_y, 'b-', linewidth=2.5)
    ax.fill(ell_x, ell_y, alpha=0.04, color='blue')

    # Foci
    ax.plot([-e, e], [0, 0], 'ro', markersize=9, zorder=5)
    ax.text(-e, -b_ell*0.12, f'F₁(−{e},0)', ha='center', fontsize=9, color='red')
    ax.text(e, -b_ell*0.12, f'F₂(+{e},0)', ha='center', fontsize=9, color='red')

    # Inscribed isosceles triangle
    tri_x = [-e, e, 0, -e]
    tri_y = [0, 0, b_ell, 0]
    ax.fill(tri_x, tri_y, alpha=0.12, color='orange')
    ax.plot(tri_x, tri_y, 'k-', linewidth=1.5)

    # Equal sides = d
    mid_lx, mid_ly = (-e + 0)/2, b_ell/2
    mid_rx, mid_ry = (e + 0)/2, b_ell/2
    ax.text(mid_lx - a_ell*0.08, mid_ly, f'd = {d}', fontsize=11,
            fontweight='bold', color='navy',
            rotation=np.degrees(np.arctan2(b_ell, e)))
    ax.text(mid_rx + a_ell*0.04, mid_ry, f'd = {d}', fontsize=11,
            fontweight='bold', color='navy',
            rotation=-np.degrees(np.arctan2(b_ell, e)))

    # Base and height annotations
    ax.annotate('', xy=(e, -b_ell*0.2), xytext=(-e, -b_ell*0.2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(0, -b_ell*0.25, f'2e = {2*e}', ha='center', fontsize=10, color='red')

    ax.annotate('', xy=(0, b_ell), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text(a_ell*0.08, b_ell*0.5, f'√F = √{F}', fontsize=10, color='green', fontweight='bold')

    # Vertex label
    ax.plot(0, b_ell, 'go', markersize=7, zorder=5)
    ax.text(a_ell*0.1, b_ell*1.02, f'(0, √{F})', fontsize=9, color='green')

    # Key insight box
    info = (f'h² = d²−e² = {d*d}−{e*e} = {d*d - e*e} = F = ab\n'
            f'Eccentricity = e/d = {e}/{d} = {e/d:.6f}\n'
            f'Volk: 2a = C = 2ed = {C}')
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8,
            va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.margins(0.1)


def draw_torus_3d(ax, ids):
    """Fig 3: 3D torus surface with (m,n) torus knot."""
    R = ids['G']  # major radius
    r = ids['F']  # minor radius

    ax.set_title(f'Toroidal Surface\nR = G = {R}, r = F = {r}',
                 fontsize=12, fontweight='bold')

    # Torus surface
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, 2*np.pi, 80)
    u, v = np.meshgrid(u, v)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    ax.plot_surface(x, y, z, alpha=0.15, color='steelblue', edgecolor='steelblue',
                    linewidth=0.1)

    # (2,3) torus knot on the surface
    t = np.linspace(0, 2*np.pi, 500)
    m, n = 2, 3  # trefoil knot
    kx = (R + r * np.cos(n*t)) * np.cos(m*t)
    ky = (R + r * np.cos(n*t)) * np.sin(m*t)
    kz = r * np.sin(n*t)
    ax.plot(kx, ky, kz, 'r-', linewidth=2, label=f'({m},{n}) torus knot')

    # (3,5) knot — Fibonacci winding
    m2, n2 = 3, 5
    kx2 = (R + r*0.95 * np.cos(n2*t)) * np.cos(m2*t)
    ky2 = (R + r*0.95 * np.cos(n2*t)) * np.sin(m2*t)
    kz2 = r*0.95 * np.sin(n2*t)
    ax.plot(kx2, ky2, kz2, 'g-', linewidth=1.5, alpha=0.7, label=f'({m2},{n2}) Fibonacci knot')

    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Clean up viewing angle
    ax.view_init(elev=25, azim=45)
    lim = R + r * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-r*1.5, r*1.5)


def draw_sum_product_triangle(ax, ids):
    """Fig 4: Grant's sum-product triangle of means."""
    b, e, d, a = ids['b'], ids['e'], ids['d'], ids['a']
    C, F, G = ids['C'], ids['F'], ids['G']

    ax.set_title("Grant's Sum-Product Triangle\n"
                 "Triangle of Means = QA Right Triangle",
                 fontsize=12, fontweight='bold')

    # For factor pair (x, y) = (b*d, e*a) or similar
    # The triangle of means:
    #   M_A = arithmetic mean = (x+y)/2
    #   M_G = geometric mean = sqrt(xy)
    #   M_D = mean difference = |x-y|/2
    #   Pythagorean: M_A² = M_G² + M_D²
    #
    # This maps to QA: G² = C² + F² (reordered)
    # Or equivalently: the (C, F, G) right triangle IS the triangle of means

    # Draw the right triangle
    tri_x = [0, C, 0, 0]
    tri_y = [0, 0, F, 0]
    ax.fill(tri_x, tri_y, alpha=0.1, color='gold')
    ax.plot(tri_x, tri_y, 'k-', linewidth=2)
    ax.plot([C, 0], [0, F], 'b-', linewidth=2.5)

    # Labels with sum-product meaning
    ax.text(C/2, -F*0.1, f'C = 2de = {C}\n(Green = Sum structure)',
            ha='center', fontsize=9, color='green', fontweight='bold')
    ax.text(-C*0.2, F/2, f'F = ab = {F}\n(Red = Product\n structure)',
            ha='center', fontsize=9, color='red', fontweight='bold', rotation=90)
    angle = -np.degrees(np.arctan2(F, C))
    ax.text(C/2 + C*0.08, F/2 + F*0.08,
            f'G = d²+e² = {G}\n(Blue = Norm)',
            ha='center', fontsize=9, color='blue', fontweight='bold', rotation=angle)

    # Right angle
    sq = min(C, F) * 0.06
    ax.add_patch(patches.Rectangle((0, 0), sq, sq, fill=False, ec='black'))

    # Sum-product conjecture connection
    info = (f'Sum-Product Conjecture:\n'
            f'  C = 2de → additive (E-circles)\n'
            f'  F = ab  → multiplicative (M-circles)\n'
            f'  G² = C² + F² → they cannot\n'
            f'  both be small (Erdős-Szemerédi)\n'
            f'\n'
            f'  Volk: E⊥M ↔ C²+F²=G²\n'
            f'  Orthogonality of additive and\n'
            f'  multiplicative = Pythagoras!')
    ax.text(0.55, 0.95, info, transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.grid(True, alpha=0.2)


def draw_cross_section(ax, ids):
    """Fig 5: Toroidal cross-section with QA parameters labeled."""
    R = ids['G']
    r = ids['F']
    C = ids['C']
    b, e, d, a = ids['b'], ids['e'], ids['d'], ids['a']

    ax.set_title(f'Toroidal Cross-Section\nR = G = {R}, r = F = {r}, hole = R−r = {R-r}',
                 fontsize=11, fontweight='bold')

    theta = np.linspace(0, 2*np.pi, 200)

    # Right cross-section
    ax.plot(R + r*np.cos(theta), r*np.sin(theta), 'b-', lw=2)
    ax.fill(R + r*np.cos(theta), r*np.sin(theta), alpha=0.08, color='blue')

    # Left cross-section
    ax.plot(-R + r*np.cos(theta), r*np.sin(theta), 'b-', lw=2)
    ax.fill(-R + r*np.cos(theta), r*np.sin(theta), alpha=0.08, color='blue')

    # Center and radii
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)

    # R arrow
    ax.annotate('', xy=(R, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(R/2, r*0.15, f'R = G = {R}', ha='center', fontsize=10, color='red', fontweight='bold')

    # r arrow
    ax.annotate('', xy=(R + r, 0), xytext=(R, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(R + r/2, r*0.15, f'r = F = {r}', ha='center', fontsize=9, color='green', fontweight='bold')

    # Outer radius
    ax.plot([0, R+r], [0, 0], 'k--', alpha=0.2)
    ax.text(R+r, -r*0.15, f'{R+r}', fontsize=9, ha='center')

    # Inner radius (the hole)
    if R > r:
        ax.plot([0, R-r], [0, 0], 'k--', alpha=0.2)
        ax.text(R-r, -r*0.15, f'{R-r}', fontsize=9, ha='center')

    # Aspect ratio box
    info = (f'R/r = G/F = {R}/{r} = {R/r:.4f}\n'
            f'Hole = R−r = {R-r}\n'
            f'Outer = R+r = {R+r}\n'
            f'Hole/Outer = {(R-r)/(R+r):.4f}\n'
            f'Focal sep = C = {C}')
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_aspect('equal')
    lim = (R + r) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-r * 2, r * 2)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='gray', lw=0.3)


def draw_volk_qa(b, e, d, a, output_path='qa_volk_torus.png'):
    """Draw the complete Volk toroidal visualization."""
    ids = qa_ids(b, e, d, a)

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f'Volk Toroidal Geometry — QA ({b},{e},{d},{a})\n'
                 f'Triple ({ids["C"]},{ids["F"]},{ids["G"]})  |  '
                 f'R/r = G/F = {ids["G"]}/{ids["F"]} = {ids["G"]/ids["F"]:.4f}  |  '
                 f'{ids["gender"].upper()}  |  Product = {ids["product"]}',
                 fontsize=14, fontweight='bold', y=0.99)

    # Panel 1: Apollonian circles
    ax1 = fig.add_subplot(2, 3, 1)
    draw_apollonian(ax1, ids)

    # Panel 2: Quantum ellipse
    ax2 = fig.add_subplot(2, 3, 2)
    draw_quantum_ellipse(ax2, ids)

    # Panel 3: 3D torus
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    draw_torus_3d(ax3, ids)

    # Panel 4: Sum-product triangle
    ax4 = fig.add_subplot(2, 3, 4)
    draw_sum_product_triangle(ax4, ids)

    # Panel 5: Cross-section
    ax5 = fig.add_subplot(2, 3, 5)
    draw_cross_section(ax5, ids)

    # Panel 6: QA ↔ Volk mapping table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title('QA ↔ Volk Parameter Map', fontsize=12, fontweight='bold')
    ax6.axis('off')

    lines = [
        f'  QA Tuple:  ({b}, {e}, {d}, {a})',
        f'',
        f'  Volk Parameter    QA Value    Meaning',
        f'  ─────────────    ────────    ───────',
        f'  2a (focal sep)   C = {ids["C"]:>5}    Bipolar scale × 2',
        f'  a  (scale)       X = {ids["X"]:>5}    Half-focal distance',
        f'  R  (major rad)   G = {ids["G"]:>5}    Blue quadrance',
        f'  r  (minor rad)   F = {ids["F"]:>5}    Red quadrance',
        f'  Base circle      ρ₀= {ids["X"]:>5}    E/M boundary',
        f'  String ratio     G/F = {ids["G"]/ids["F"]:.4f}',
        f'  Wavelength ratio C/F = {ids["C"]/ids["F"]:.4f}',
        f'',
        f'  E-circles (green) → Additive structure',
        f'    Sum operation: C = 2de',
        f'  M-circles (red)  → Multiplicative structure',
        f'    Product operation: F = ab',
        f'  Orthogonality: E ⊥ M  ↔  C² + F² = G²',
        f'',
        f'  Torus knot (m,n): winding ratio',
        f'  QA resonance: mod-{24} × mod-{9}',
    ]

    y = 0.98
    for line in lines:
        ax6.text(0.02, y, line, transform=ax6.transAxes, fontsize=8.5,
                 fontfamily='monospace', va='top')
        y -= 0.05

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')
    return ids


if __name__ == '__main__':
    if len(sys.argv) == 5:
        b, e, d, a = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    else:
        b, e, d, a = 1, 2, 3, 5  # canonical Fibonacci seed

    print(f'\n=== Volk Toroidal Geometry: ({b},{e},{d},{a}) ===\n')
    ids = draw_volk_qa(b, e, d, a)
    print(f'Triple: ({ids["C"]}, {ids["F"]}, {ids["G"]})')
    print(f'Torus: R=G={ids["G"]}, r=F={ids["F"]}, R/r={ids["G"]/ids["F"]:.4f}')
    print(f'Volk scale: a=X={ids["X"]}, 2a=C={ids["C"]}')
