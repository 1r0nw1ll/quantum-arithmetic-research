#!/usr/bin/env python3
"""
QA Geometric Visualizer — See what Ben sees.

Renders the complete geometric picture of a QA number (b, e, d, a):
1. The Pythagorean right triangle (C, F, G) with chromogeometric coloring
2. The Inner Quantum Ellipse (foci at ±e, inscribed isosceles triangle)
3. Volk's Apollonian Circles (E-circles and M-circles from bipolar coords)
4. The Twisted Squares (I, G, H arithmetic progression)
5. Toroidal cross-section (major radius G, minor radius F)
6. The 16 QA identities + key relations

NOTE: I = |C - F| (always positive — per QA corpus, the positive difference).

Usage:
    python qa_geometric_visualizer.py              # default (1,1,2,3)
    python qa_geometric_visualizer.py 5 3 8 11     # custom tuple
    python qa_geometric_visualizer.py 2 1 3 4      # female of (1,1,2,3)
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np


def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a if a > 0 else 1


def qa_identities(b, e, d, a):
    """Compute all 16 QA identities from (b, e, d, a)."""
    assert d == b + e, f"d={d} != b+e={b+e}"
    assert a == b + 2*e, f"a={a} != b+2e={b+2*e}"

    A_val = a*a
    B_val = b*b
    C_val = 2*d*e
    D_val = d*d
    E_val = e*e
    F_val = a*b
    G_val = d*d + e*e
    H_val = C_val + F_val
    I_val = abs(C_val - F_val)   # POSITIVE difference per QA corpus
    J_val = b*d
    K_val = a*d
    L_val = (C_val * F_val) // 12
    X_val = e*d
    W_val = d*(e + a)
    Y_val = A_val - D_val
    Z_val = E_val + K_val

    # Which is larger determines the "type"
    conic_type = 'HYPERBOLA' if F_val > C_val else ('PARABOLA' if F_val == C_val else 'ELLIPSE')

    ids = {
        'b': b, 'e': e, 'd': d, 'a': a,
        'A': A_val, 'B': B_val, 'C': C_val, 'D': D_val,
        'E': E_val, 'F': F_val, 'G': G_val, 'H': H_val,
        'I': I_val, 'J': J_val, 'K': K_val, 'L': L_val,
        'X': X_val, 'W': W_val, 'Y': Y_val, 'Z': Z_val,
        'gender': 'female' if b % 2 == 0 else 'male',
        'product': b * e * d * a,
        'conic_type': conic_type,
        'C_gt_F': C_val > F_val,
    }

    # Verify
    assert C_val*C_val + F_val*F_val == G_val*G_val, "C²+F²≠G²"
    assert H_val*H_val - I_val*I_val == 4*C_val*F_val, f"H²-I²≠4CF: {H_val*H_val}-{I_val*I_val}≠{4*C_val*F_val}"

    return ids


def draw_panel_triangle(ax, ids):
    """Panel 1: The Pythagorean right triangle with chromogeometric coloring."""
    C, F, G = ids['C'], ids['F'], ids['G']
    ax.set_title('Pythagorean Triple\nC² + F² = G²', fontsize=11, fontweight='bold')

    # Right triangle at origin
    tri_x = [0, C, 0, 0]
    tri_y = [0, 0, F, 0]
    ax.fill(tri_x, tri_y, alpha=0.08, color='steelblue')
    ax.plot(tri_x, tri_y, 'k-', linewidth=1.5)
    ax.plot([C, 0], [0, F], 'b-', linewidth=2.5)

    # Side labels
    ax.text(C/2, -0.1*F, f'C = 2de = {C}', ha='center', fontsize=10,
            color='green', fontweight='bold')
    ax.text(-0.18*C, F/2, f'F = ab = {F}', ha='center', fontsize=10,
            color='red', fontweight='bold', rotation=90)
    angle = -np.degrees(np.arctan2(F, C))
    ax.text(C/2 + 0.06*C, F/2 + 0.06*F, f'G = d²+e² = {G}', ha='center',
            fontsize=10, color='blue', fontweight='bold', rotation=angle)

    # Right angle marker
    sq = min(C, F) * 0.06
    ax.add_patch(patches.Rectangle((0, 0), sq, sq, fill=False, ec='black', lw=1))

    # Chromogeometry legend
    ax.text(0.02, 0.97, 'Green Qg = C = 2de', transform=ax.transAxes,
            fontsize=8, color='green', va='top')
    ax.text(0.02, 0.90, 'Red Qr = F = d²−e² = ab', transform=ax.transAxes,
            fontsize=8, color='red', va='top')
    ax.text(0.02, 0.83, 'Blue Qb = G = d²+e²', transform=ax.transAxes,
            fontsize=8, color='blue', va='top')

    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.grid(True, alpha=0.2)


def draw_panel_quantum_ellipse(ax, ids):
    """Panel 2: The Inner Quantum Ellipse — foci at ±e, inscribed isosceles triangle."""
    b, e, d, a = ids['b'], ids['e'], ids['d'], ids['a']
    F_val = ids['F']

    ax.set_title('Inner Quantum Ellipse\nFoci at ±e, semi-minor = √F', fontsize=11, fontweight='bold')

    # Ellipse parameters:
    # Foci at (-e, 0) and (+e, 0)
    # Semi-minor axis b_ell = sqrt(F) = sqrt(a*b)
    # Semi-major axis a_ell = sqrt(e² + F) = sqrt(e² + ab)
    # This comes from: for the isosceles triangle inscribed in the ellipse,
    # the vertex is at (0, sqrt(F)) and the base vertices are at the foci.
    # Actually, the equal sides have length d (from each focus to top).

    c_ell = e  # half focal distance
    # From the isosceles triangle: each equal side = d
    # Sum of distances from foci to any point on ellipse = 2*a_ell
    # At the top vertex (0, h): dist to each focus = sqrt(e² + h²)
    # If this equals d, then h = sqrt(d² - e²) = sqrt(b² + 2be) = sqrt(b(b+2e)) = sqrt(bа)...
    # Actually h² = d² - e² = (b+e)² - e² = b² + 2be = b(b+2e) = ba = F
    # So h = sqrt(F), and 2*a_ell = 2*d (sum of distances at top = d + d = 2d)
    a_ell = d       # semi-major axis
    b_ell_sq = a_ell*a_ell - c_ell*c_ell  # = d² - e² = F
    b_ell = np.sqrt(F_val)

    # Draw ellipse
    theta = np.linspace(0, 2*np.pi, 200)
    ell_x = a_ell * np.cos(theta)
    ell_y = b_ell * np.sin(theta)
    ax.plot(ell_x, ell_y, 'b-', linewidth=2, label='Quantum Ellipse')
    ax.fill(ell_x, ell_y, alpha=0.05, color='blue')

    # Foci
    ax.plot([-e, e], [0, 0], 'ro', markersize=8, zorder=5)
    ax.text(-e, -0.15*b_ell, f'−e = −{e}', ha='center', fontsize=9, color='red')
    ax.text(e, -0.15*b_ell, f'+e = {e}', ha='center', fontsize=9, color='red')

    # Isosceles triangle: base between foci, apex at (0, sqrt(F))
    tri_x = [-e, e, 0, -e]
    tri_y = [0, 0, b_ell, 0]
    ax.plot(tri_x, tri_y, 'k-', linewidth=1.5)
    ax.fill(tri_x, tri_y, alpha=0.1, color='orange')

    # Label the equal sides (length d)
    ax.text(-e/2 - 0.1*a_ell, b_ell/2, f'd={d}', fontsize=10, fontweight='bold',
            color='darkblue', rotation=np.degrees(np.arctan2(b_ell, e)))
    ax.text(e/2 + 0.05*a_ell, b_ell/2, f'd={d}', fontsize=10, fontweight='bold',
            color='darkblue', rotation=-np.degrees(np.arctan2(b_ell, e)))

    # Label base (2e) and height (sqrt(F))
    ax.text(0, -0.25*b_ell, f'Base = 2e = {2*e}', ha='center', fontsize=9, color='red')
    ax.annotate(f'Height = √F = √{F_val}\n= √(ab) = √({a}×{b})',
                xy=(0, b_ell), xytext=(0.3*a_ell, b_ell*0.85),
                fontsize=8, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))

    # Semi-major axis label
    ax.text(0, -0.4*b_ell, f'Semi-major a = d = {d}', ha='center', fontsize=8, color='blue')
    ax.text(0, -0.52*b_ell, f'Semi-minor b = √F = {b_ell:.4f}', ha='center', fontsize=8, color='purple')

    # Eccentricity
    ecc = e / d if d > 0 else 0
    ax.text(0, -0.64*b_ell, f'Eccentricity = e/d = {e}/{d} = {ecc:.6f}',
            ha='center', fontsize=8, color='gray')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)


def draw_panel_apollonian(ax, ids):
    """Panel 3: Volk's Apollonian Circles — E-circles and M-circles."""
    b, e, d, a = ids['b'], ids['e'], ids['d'], ids['a']
    C, F, G = ids['C'], ids['F'], ids['G']
    X = ids['X']  # = ed = C/2, Volk's 'a' parameter

    ax.set_title(f"Volk's Apollonian Circles\nScale a = X = ed = {X}",
                 fontsize=11, fontweight='bold')

    volk_a = X  # bipolar scale parameter = ed

    # Poles at (-a, 0) and (+a, 0)
    ax.plot([-volk_a, volk_a], [0, 0], 'ko', markersize=6, zorder=5)
    ax.text(-volk_a, -0.12*volk_a, f'−X', ha='center', fontsize=9)
    ax.text(volk_a, -0.12*volk_a, f'+X', ha='center', fontsize=9)

    # E-circles: pass through both poles (constant eta)
    # E-circle with center on y-axis at (0, h), radius r where r² = h² + a²
    for eta_factor in [0.3, 0.6, 1.0, 1.5, 2.5]:
        h = volk_a * eta_factor
        r = np.sqrt(h*h + volk_a*volk_a)
        circle_up = plt.Circle((0, h), r, fill=False, color='green', alpha=0.4, linewidth=1)
        circle_down = plt.Circle((0, -h), r, fill=False, color='green', alpha=0.4, linewidth=1)
        ax.add_patch(circle_up)
        ax.add_patch(circle_down)

    # M-circles: orthogonal to E-circles (constant rho)
    # M-circle with center on x-axis at (c, 0) where c = a*coth(rho), radius = a/sinh(rho)
    for rho_val in [0.4, 0.7, 1.0, 1.5, 2.5]:
        coth_rho = np.cosh(rho_val) / np.sinh(rho_val)
        sinh_rho = np.sinh(rho_val)
        cx = volk_a * coth_rho
        cr = volk_a / sinh_rho
        circle_right = plt.Circle((cx, 0), cr, fill=False, color='red', alpha=0.4, linewidth=1)
        circle_left = plt.Circle((-cx, 0), cr, fill=False, color='red', alpha=0.4, linewidth=1)
        ax.add_patch(circle_right)
        ax.add_patch(circle_left)

    # Base circle (radius = a)
    base_circle = plt.Circle((0, 0), volk_a, fill=False, color='blue', linewidth=2, linestyle='--')
    ax.add_patch(base_circle)
    ax.text(volk_a*0.7, volk_a*0.7, f'Base ρ₀={X}', fontsize=8, color='blue')

    # Legend
    ax.text(0.02, 0.97, 'Green: E-circles (additive/AP)', transform=ax.transAxes,
            fontsize=8, color='green', va='top')
    ax.text(0.02, 0.90, 'Red: M-circles (multiplicative/GP)', transform=ax.transAxes,
            fontsize=8, color='red', va='top')
    ax.text(0.02, 0.83, 'Blue: Base circle (ρ₀ = X = ed)', transform=ax.transAxes,
            fontsize=8, color='blue', va='top')

    lim = volk_a * 3.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)


def draw_panel_twisted_squares(ax, ids):
    """Panel 4: Twisted Squares — I, G, H arithmetic progression."""
    C, F, G = ids['C'], ids['F'], ids['G']
    H, I_val = ids['H'], ids['I']
    L = ids['L']

    ax.set_title(f'Koenig Series: I²→G²→H²\nstep = 2CF = {2*C*F} = 24L',
                 fontsize=11, fontweight='bold')

    # Outer square (H)
    if H > 0:
        outer = patches.Rectangle((-H/2, -H/2), H, H, fill=False,
                                   ec='purple', lw=2.5)
        ax.add_patch(outer)
        ax.text(H/2 + 0.5, 0, f'H={H}', fontsize=11, color='purple', fontweight='bold')

    # Middle square (G) — rotated slightly to show nesting
    if G > 0:
        # Compute rotation angle for inscribed square
        # For the twisted square: if outer side = H and inner side = I,
        # the middle square G is at an intermediate rotation
        mid = patches.Rectangle((-G/2, -G/2), G, G, fill=False,
                                 ec='blue', lw=2, linestyle='-')
        ax.add_patch(mid)
        ax.text(-G/2 - 1, 0, f'G={G}', fontsize=10, color='blue', fontweight='bold', ha='right')

    # Inner square (I)
    if I_val > 0:
        inner = patches.Rectangle((-I_val/2, -I_val/2), I_val, I_val, fill=False,
                                   ec='orange', lw=2, linestyle='--')
        ax.add_patch(inner)
        ax.text(0, I_val/2 + 0.8, f'I=|C−F|={I_val}', fontsize=10, color='orange',
                fontweight='bold', ha='center')

    # Arithmetic progression annotation
    info = (f'I² = {I_val*I_val}\n'
            f'G² = {G*G}\n'
            f'H² = {H*H}\n'
            f'Step = 2CF = {2*C*F}\n'
            f'24L = 24×{L} = {24*L}\n'
            f'H²−I² = 4CF = {4*C*F} = 48L')
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8,
            va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Conic type
    ctype = ids['conic_type']
    ccolor = {'HYPERBOLA': 'darkred', 'ELLIPSE': 'darkblue', 'PARABOLA': 'darkgreen'}[ctype]
    reason = f'F({F})>C({C})' if F > C else (f'C({C})>F({F})' if C > F else f'C=F={C}')
    ax.text(0.5, 0.97, f'{ctype}\n({reason})', transform=ax.transAxes, fontsize=10,
            ha='center', va='top', fontweight='bold', color=ccolor)

    ax.set_aspect('equal')
    max_dim = max(H, G, I_val) * 0.65
    if max_dim == 0:
        max_dim = 1
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.grid(True, alpha=0.2)


def draw_panel_torus(ax, ids):
    """Panel 5: Toroidal cross-section — major radius R=G, minor radius r=F."""
    C, F, G = ids['C'], ids['F'], ids['G']

    ax.set_title(f'Toroidal Cross-Section\nR = G = {G}, r = F = {F}',
                 fontsize=11, fontweight='bold')

    # Torus cross-section: two circles at (±R, 0) with radius r
    # But we draw the torus profile in the (x, z) plane
    R = G  # major radius
    r = F  # minor radius

    # Outer profile of torus
    theta = np.linspace(0, 2*np.pi, 200)

    # Right cross-section circle
    cx_r = R + r * np.cos(theta)
    cy_r = r * np.sin(theta)
    ax.plot(cx_r, cy_r, 'b-', linewidth=2)
    ax.fill(cx_r, cy_r, alpha=0.1, color='blue')

    # Left cross-section circle
    cx_l = -R + r * np.cos(theta)
    cy_l = r * np.sin(theta)
    ax.plot(cx_l, cy_l, 'b-', linewidth=2)
    ax.fill(cx_l, cy_l, alpha=0.1, color='blue')

    # Outer and inner equator lines
    ax.plot([-(R+r), (R+r)], [0, 0], 'k--', alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.3)

    # Center axis
    ax.plot([0], [0], 'k+', markersize=10)

    # R labels
    ax.annotate('', xy=(R, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(R/2, -r*0.15, f'R=G={G}', ha='center', fontsize=9, color='red')

    # r labels
    ax.annotate('', xy=(R, r), xytext=(R, 0),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text(R + r*0.3, r/2, f'r=F={F}', fontsize=9, color='green')

    # Aspect ratio
    if F > 0:
        aspect = G / F
        ax.text(0.5, 0.05, f'R/r = G/F = {G}/{F} = {aspect:.4f}\n'
                f'Focal sep = 2X = C = {C}',
                transform=ax.transAxes, fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_aspect('equal')
    lim = (R + r) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.5, lim * 0.5)
    ax.grid(True, alpha=0.2)


def draw_panel_identities(ax, ids):
    """Panel 6: The 16 identities + key relations."""
    b, e, d, a = ids['b'], ids['e'], ids['d'], ids['a']
    C, F, G = ids['C'], ids['F'], ids['G']
    H, I_val, L = ids['H'], ids['I'], ids['L']

    ax.set_title(f'QA Number ({b},{e},{d},{a}) — {ids["gender"].upper()}',
                 fontsize=11, fontweight='bold')
    ax.axis('off')

    lines = [
        f'  (b,e,d,a) = ({b},{e},{d},{a})    d=b+e, a=b+2e',
        f'  Product = {ids["product"]}    Gender: {ids["gender"]}',
        f'',
        f'  A=a²={ids["A"]:>6}  B=b²={ids["B"]:>6}  C=2de={C:>6} (4-par)',
        f'  D=d²={ids["D"]:>6}  E=e²={ids["E"]:>6}  F=ab ={F:>6} (semi-latus)',
        f'  G=d²+e²={G:>4}    H=C+F={H:>4}    I=|C−F|={I_val:>4}',
        f'  J=bd={ids["J"]:>5}  K=ad={ids["K"]:>5}  L=CF/12={L}',
        f'  X=ed={ids["X"]:>5}  W=d(e+a)={ids["W"]}  Y=A−D={ids["Y"]}  Z=E+K={ids["Z"]}',
        f'',
        f'  C²+F²=G²: {C*C}+{F*F}={G*G} ✓',
        f'  G=(A+B)/2: ({ids["A"]}+{ids["B"]})/2={G} ✓',
        f'  A−B=2C: {ids["A"]}−{ids["B"]}={2*C} ✓',
        f'  H²−I²=4CF=48L: {H*H}−{I_val*I_val}={4*C*F}={48*L} ✓',
        f'',
        f'  Musical: G:F={G//gcd(G,F)}:{F//gcd(G,F)}'
        f'  G:C={G//gcd(G,C)}:{C//gcd(G,C)}'
        f'  F:X={F//gcd(F,ids["X"])}:{ids["X"]//gcd(F,ids["X"])}' if ids['X'] > 0 else '',
        f'',
        f'  Ellipse: foci=±{e}, semi-major=d={d}, semi-minor=√F=√{F}',
        f'  Eccentricity = e/d = {e}/{d} = {e/d:.6f}',
        f'  Torus: R=G={G}, r=F={F}, R/r={G/F:.4f}' if F > 0 else '',
    ]

    y = 0.98
    for line in lines:
        ax.text(0.02, y, line, transform=ax.transAxes, fontsize=8,
                fontfamily='monospace', va='top')
        y -= 0.055


def draw_qa_geometric(b, e, d, a, output_path='qa_geometric_viz.png'):
    """Draw the complete 6-panel QA geometric visualization."""
    ids = qa_identities(b, e, d, a)

    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(f'QA Geometric Visualization: ({b}, {e}, {d}, {a}) — '
                 f'Triple ({ids["C"]}, {ids["F"]}, {ids["G"]}) — '
                 f'{ids["conic_type"]}',
                 fontsize=16, fontweight='bold', y=0.99)

    ax1 = fig.add_subplot(2, 3, 1)
    draw_panel_triangle(ax1, ids)

    ax2 = fig.add_subplot(2, 3, 2)
    draw_panel_quantum_ellipse(ax2, ids)

    ax3 = fig.add_subplot(2, 3, 3)
    draw_panel_apollonian(ax3, ids)

    ax4 = fig.add_subplot(2, 3, 4)
    draw_panel_twisted_squares(ax4, ids)

    ax5 = fig.add_subplot(2, 3, 5)
    draw_panel_torus(ax5, ids)

    ax6 = fig.add_subplot(2, 3, 6)
    draw_panel_identities(ax6, ids)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')
    return ids


if __name__ == '__main__':
    if len(sys.argv) == 5:
        b, e, d, a = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    else:
        b, e, d, a = 1, 1, 2, 3

    print(f'\n=== QA Geometric Visualization ({b}, {e}, {d}, {a}) ===\n')
    ids = draw_qa_geometric(b, e, d, a)

    print(f'Triple: ({ids["C"]}, {ids["F"]}, {ids["G"]})')
    print(f'  C²+F²=G²: {ids["C"]**2}+{ids["F"]**2}={ids["G"]**2} ✓')
    print(f'Conic: {ids["conic_type"]} (I = |C−F| = {ids["I"]})')
    print(f'Ellipse: foci=±{ids["e"]}, semi-major=d={ids["d"]}, ecc={ids["e"]/ids["d"]:.6f}')
    print(f'Torus: R=G={ids["G"]}, r=F={ids["F"]}')
    print(f'Product: {ids["product"]}  Gender: {ids["gender"]}')
