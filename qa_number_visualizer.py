#!/usr/bin/env python3
"""
QA Number Visualizer — See what Ben sees.

Given a QA tuple (b, e, d, a), renders:
1. The right triangle with sides C, F, G (the three chromogeometric quadrances)
2. The twisted squares construction: inner square I, outer square H
3. The arithmetic progression I², G², H² with step 2CF = 24L
4. The three colored metrics (blue/red/green) on the direction vector (d, e)
5. The 16 QA identities annotated on the geometry

Usage:
    python qa_number_visualizer.py              # default (1,1,2,3)
    python qa_number_visualizer.py 5 3 8 11     # custom tuple
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def qa_identities(b, e, d, a):
    """Compute all 16 QA identities from (b, e, d, a)."""
    # Verify QA relations
    assert d == b + e, f"d={d} != b+e={b+e}"
    assert a == b + 2*e, f"a={a} != b+2e={b+2*e}"

    A_val = a*a
    B_val = b*b
    C_val = 2*d*e          # 4-par (green quadrance)
    D_val = d*d
    E_val = e*e
    F_val = a*b            # semi-latus (red quadrance)
    G_val = d*d + e*e      # 5-par (blue quadrance)
    H_val = C_val + F_val  # H = C + F
    I_val = C_val - F_val  # I = C - F (conic discriminant)
    J_val = b*d            # perigee
    K_val = a*d            # apogee
    L_val = (C_val * F_val) // 12  # L = CF/12 = abde/6
    X_val = e*d            # X = ed = C/2
    W_val = d*(e + a)      # W = d(e+a) = X + K
    Y_val = A_val - D_val  # Y = A - D = a² - d²
    Z_val = E_val + K_val  # Z = E + K

    ids = {
        'b': b, 'e': e, 'd': d, 'a': a,
        'A': A_val, 'B': B_val, 'C': C_val, 'D': D_val,
        'E': E_val, 'F': F_val, 'G': G_val, 'H': H_val,
        'I': I_val, 'J': J_val, 'K': K_val, 'L': L_val,
        'X': X_val, 'W': W_val, 'Y': Y_val, 'Z': Z_val,
    }

    # Verify key identities
    assert C_val*C_val + F_val*F_val == G_val*G_val, "C²+F²≠G²"
    assert G_val == (A_val + B_val) // 2 or G_val*2 == A_val + B_val, "G≠(A+B)/2"
    assert A_val - B_val == 2*C_val, "A-B≠2C"
    assert H_val*H_val - I_val*I_val == 4*C_val*F_val, "H²-I²≠4CF"

    # Par classification
    ids['C_par'] = f"{C_val % 4}-par → 4-par" if C_val % 4 == 0 else f"{C_val % 4}"
    ids['G_par'] = f"{G_val % 4}-par → 5-par" if G_val % 4 == 1 else f"{G_val % 4}"

    # Male/female
    ids['gender'] = 'female' if b % 2 == 0 else 'male'
    ids['product'] = b * e * d * a

    return ids


def draw_qa_number(b, e, d, a, output_path='qa_number_viz.png'):
    """Draw the complete QA number visualization."""
    ids = qa_identities(b, e, d, a)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'QA Number ({b}, {e}, {d}, {a}) — "{ids["gender"].upper()}"',
                 fontsize=18, fontweight='bold', y=0.98)

    # ── Panel 1: The Right Triangle (C, F, G) ──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title('The Pythagorean Triple\nC² + F² = G²', fontsize=12, fontweight='bold')

    C, F, G = ids['C'], ids['F'], ids['G']
    # Draw right triangle: right angle at origin
    tri_x = [0, C, 0, 0]
    tri_y = [0, 0, F, 0]
    ax1.fill(tri_x, tri_y, alpha=0.15, color='steelblue')
    ax1.plot(tri_x, tri_y, 'k-', linewidth=2)
    # Hypotenuse
    ax1.plot([C, 0], [0, F], 'b-', linewidth=2.5)

    # Labels
    ax1.text(C/2, -0.08*F, f'C = 2de = {C}', ha='center', fontsize=11,
             color='green', fontweight='bold')
    ax1.text(-0.15*C, F/2, f'F = ab = {F}', ha='center', fontsize=11,
             color='red', fontweight='bold', rotation=90)
    ax1.text(C/2 + 0.08*C, F/2 + 0.05*F, f'G = d²+e² = {G}', ha='center', fontsize=11,
             color='blue', fontweight='bold', rotation=-np.degrees(np.arctan2(F, C)))

    # Right angle marker
    sq_size = min(C, F) * 0.08
    ax1.add_patch(patches.Rectangle((0, 0), sq_size, sq_size, fill=False, edgecolor='black'))

    # Chromogeometric color coding
    ax1.text(0.02, 0.95, 'Green (Qg) = C = 2de', transform=ax1.transAxes,
             fontsize=9, color='green', va='top')
    ax1.text(0.02, 0.88, 'Red (Qr) = F = d²−e²= ab', transform=ax1.transAxes,
             fontsize=9, color='red', va='top')
    ax1.text(0.02, 0.81, 'Blue (Qb) = G = d²+e²', transform=ax1.transAxes,
             fontsize=9, color='blue', va='top')

    ax1.set_aspect('equal')
    ax1.margins(0.15)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: The Twisted Squares (I, G, H) ──
    ax2 = fig.add_subplot(2, 3, 2)
    H, I_val = ids['H'], ids['I']
    ax2.set_title(f'Twisted Squares\nI²={I_val*I_val}, G²={G*G}, H²={H*H}  step={2*C*F}=2CF',
                  fontsize=11, fontweight='bold')

    # Outer square (side H)
    if H > 0:
        outer = patches.Rectangle((-H/2, -H/2), H, H, fill=False,
                                   edgecolor='purple', linewidth=2.5, linestyle='-')
        ax2.add_patch(outer)
        ax2.text(H/2 + 0.5, 0, f'H={H}', fontsize=11, color='purple',
                 fontweight='bold', va='center')

    # Inner square (side |I|)
    abs_I = abs(I_val)
    if abs_I > 0:
        # Rotated 45° to show the "twist"
        inner = patches.Rectangle((-abs_I/2, -abs_I/2), abs_I, abs_I, fill=False,
                                   edgecolor='orange', linewidth=2, linestyle='--')
        ax2.add_patch(inner)
        ax2.text(0, abs_I/2 + 0.5, f'|I|={abs_I}', fontsize=11, color='orange',
                 fontweight='bold', ha='center')

    # Middle: G as the hypotenuse diagonal
    ax2.plot([0], [0], 'ko', markersize=5)
    ax2.text(0, -0.5, f'G={G}', fontsize=10, ha='center', color='blue')

    # Arithmetic progression annotation
    L_val = ids['L']
    ax2.text(0.5, 0.05, f'I²={I_val*I_val}  →  G²={G*G}  →  H²={H*H}\n'
             f'step = 2CF = {2*C*F} = 24L = 24×{L_val}\n'
             f'H²−I² = 4CF = {4*C*F} = 48L = 48×{L_val}',
             transform=ax2.transAxes, fontsize=9, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Conic classification
    if I_val > 0:
        conic = "HYPERBOLA (I > 0)"
    elif I_val < 0:
        conic = "ELLIPSE (I < 0)"
    else:
        conic = "PARABOLA (I = 0)"
    ax2.text(0.5, 0.95, conic, transform=ax2.transAxes, fontsize=11,
             ha='center', va='top', fontweight='bold',
             color='darkred' if I_val > 0 else ('darkblue' if I_val < 0 else 'darkgreen'))

    ax2.set_aspect('equal')
    max_dim = max(H, G, abs_I) * 0.7
    ax2.set_xlim(-max_dim, max_dim)
    ax2.set_ylim(-max_dim, max_dim)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Direction Vector (d, e) with three metrics ──
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title('Direction Vector (d, e)\nThree Chromogeometric Quadrances', fontsize=12, fontweight='bold')

    # Draw the direction vector
    ax3.arrow(0, 0, d, e, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax3.text(d + 0.2, e + 0.2, f'({d}, {e})', fontsize=12, fontweight='bold')

    # Blue: d²+e² (Euclidean — circle)
    theta = np.linspace(0, 2*np.pi, 100)
    r_blue = np.sqrt(G)
    ax3.plot(r_blue*np.cos(theta), r_blue*np.sin(theta), 'b-', alpha=0.3, linewidth=1.5,
             label=f'Blue Qb = {G}')

    # Red: |d²-e²| (Lorentzian — hyperbola branches)
    if F > 0:
        t = np.linspace(-2, 2, 200)
        r_red = np.sqrt(F)
        # x²-y² = F → hyperbola
        x_hyp = r_red * np.cosh(t)
        y_hyp = r_red * np.sinh(t)
        ax3.plot(x_hyp, y_hyp, 'r-', alpha=0.3, linewidth=1.5, label=f'Red Qr = {F}')
        ax3.plot(-x_hyp, -y_hyp, 'r-', alpha=0.3, linewidth=1.5)

    # Green: 2de (null — lines through origin)
    if C > 0:
        ax3.axline((0, 0), slope=0, color='green', alpha=0.2, linewidth=1.5,
                   label=f'Green Qg = {C}')
        ax3.axvline(0, color='green', alpha=0.2, linewidth=1.5)

    # Mark the projections
    ax3.plot([d, d], [0, e], 'g--', alpha=0.5)
    ax3.plot([0, d], [e, e], 'r--', alpha=0.5)

    ax3.legend(fontsize=9, loc='upper left')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    lim = max(d, e) * 1.8
    ax3.set_xlim(-lim*0.3, lim)
    ax3.set_ylim(-lim*0.3, lim)

    # ── Panel 4: The 16 Identities Table ──
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title('The 16 QA Identities', fontsize=12, fontweight='bold')
    ax4.axis('off')

    table_data = [
        [f'A = a² = {ids["A"]}', f'B = b² = {ids["B"]}'],
        [f'C = 2de = {ids["C"]}  (4-par)', f'D = d² = {ids["D"]}'],
        [f'E = e² = {ids["E"]}', f'F = ab = {ids["F"]}  (semi-latus)'],
        [f'G = d²+e² = {ids["G"]}  (5-par)', f'H = C+F = {ids["H"]}'],
        [f'I = C−F = {ids["I"]}', f'J = bd = {ids["J"]}  (perigee)'],
        [f'K = ad = {ids["K"]}  (apogee)', f'L = CF/12 = {ids["L"]}'],
        [f'X = ed = C/2 = {ids["X"]}', f'W = d(e+a) = {ids["W"]}'],
        [f'Y = A−D = {ids["Y"]}', f'Z = E+K = {ids["Z"]}'],
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='left',
                       colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # ── Panel 5: Key Relations ──
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title('Key Relations', fontsize=12, fontweight='bold')
    ax5.axis('off')

    relations = [
        f'C² + F² = G²  →  {C}² + {F}² = {G}²  →  {C*C} + {F*F} = {G*G} ✓',
        f'G = (A+B)/2  →  ({ids["A"]}+{ids["B"]})/2 = {G} ✓',
        f'A − B = 2C  →  {ids["A"]} − {ids["B"]} = {2*C} ✓',
        f'G + C = A  →  {G} + {C} = {ids["A"]} ✓',
        f'G − C = B  →  {G} − {C} = {ids["B"]} ✓',
        f'H² − I² = 4CF = 48L  →  {H*H} − {I_val*I_val} = {4*C*F} = 48×{ids["L"]} ✓',
        f'D = (G+F)/2  →  ({G}+{F})/2 = {ids["D"]} ✓' if (G+F) % 2 == 0 else f'D = (G+F)/2  →  ({G}+{F})/2 = {(G+F)/2}',
        f'',
        f'Musical intervals:',
        f'  G:F = {G}:{F} = {G//gcd(G,F)}:{F//gcd(G,F)}' if F > 0 else '',
        f'  G:C = {G}:{C} = {G//gcd(G,C)}:{C//gcd(G,C)}' if C > 0 else '',
        f'  F:X = {F}:{ids["X"]} = {F//gcd(F,ids["X"])}:{ids["X"]//gcd(F,ids["X"])}' if ids['X'] > 0 else '',
        f'',
        f'Gender: {ids["gender"].upper()}  |  Product: {ids["product"]}  |  L = {ids["L"]}',
    ]

    y_pos = 0.95
    for line in relations:
        ax5.text(0.05, y_pos, line, transform=ax5.transAxes, fontsize=9,
                 fontfamily='monospace', va='top')
        y_pos -= 0.07

    # ── Panel 6: T-operator path (first few steps) ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title(f'T-operator Path (mod 9)\n({b},{e}) → ({e},{d}) → ({d},{a}) → ...',
                  fontsize=11, fontweight='bold')

    # Generate path
    mod = 9
    state_b, state_e = b % mod, e % mod
    if state_b == 0: state_b = mod
    if state_e == 0: state_e = mod

    path_b, path_e = [state_b], [state_e]
    for _ in range(30):
        new_b = state_e
        new_e = ((state_b + state_e - 1) % mod) + 1  # A1-compliant
        state_b, state_e = new_b, new_e
        path_b.append(state_b)
        path_e.append(state_e)
        if state_b == path_b[0] and state_e == path_e[0]:
            break

    ax6.plot(path_b, path_e, 'o-', markersize=6, linewidth=1.5, color='darkblue')
    ax6.plot(path_b[0], path_e[0], 's', markersize=12, color='red',
             label=f'Start ({path_b[0]},{path_e[0]})')

    # Label first few steps
    for i in range(min(5, len(path_b))):
        ax6.annotate(f'{i}', (path_b[i], path_e[i]), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, color='gray')

    ax6.set_xlabel('b (mod 9)', fontsize=10)
    ax6.set_ylabel('e (mod 9)', fontsize=10)
    ax6.set_xlim(0, mod + 1)
    ax6.set_ylim(0, mod + 1)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

    return ids


def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a if a > 0 else 1


if __name__ == '__main__':
    if len(sys.argv) == 5:
        b, e, d, a = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    else:
        # Default: the fundamental QN (1,1,2,3) — the 3-4-5 triangle
        b, e, d, a = 1, 1, 2, 3

    print(f'\n=== QA Number ({b}, {e}, {d}, {a}) ===\n')
    ids = draw_qa_number(b, e, d, a)

    print(f'\nTuple: ({b}, {e}, {d}, {a})')
    print(f'Gender: {ids["gender"]}')
    print(f'Product: {ids["product"]}')
    print(f'Triple: C={ids["C"]}, F={ids["F"]}, G={ids["G"]}')
    print(f'  C² + F² = {ids["C"]**2} + {ids["F"]**2} = {ids["C"]**2 + ids["F"]**2} = G² = {ids["G"]**2} ✓')
    print(f'Twisted: I={ids["I"]}, G={ids["G"]}, H={ids["H"]}')
    print(f'  I²={ids["I"]**2}, G²={ids["G"]**2}, H²={ids["H"]**2}  step=2CF={2*ids["C"]*ids["F"]}')
    print(f'  Conic: {"HYPERBOLA" if ids["I"] > 0 else ("ELLIPSE" if ids["I"] < 0 else "PARABOLA")}')
    print(f'L = CF/12 = {ids["L"]}  (24L = {24*ids["L"]})')
