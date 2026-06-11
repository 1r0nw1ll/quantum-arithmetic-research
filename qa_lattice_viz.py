#!/usr/bin/env python3
QA_COMPLIANCE = "visualisation: QA lattice three-panel orbit map; integer (b,e) in mod-9/mod-24; no float state; qa_step A1-compliant"
"""
QA Lattice Visualization — three-panel orbit map.

Panels:
  1. mod-9  lattice: 5 orbit categories (Fib/Lucas/Third/Satellite/Singularity)
             ○ = male (b/e < √2)  △ = female (b/e > √2)
  2. mod-24 lattice: colored by orbit period (1/3/6/8/12/24)
  3. Algebraic landscape: f = b²+be−e² (Z[φ] norm) heatmap, g = b²−2e² = 0
             contour (√2 boundary), b×e hyperbolas (K/L quantization)

Saves: qa_lattice_viz.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import gcd as _gcd

# ── QA core (A1-compliant, no float state) ────────────────────────────────────

def qa_step(b: int, e: int, m: int):
    return e, ((b + e - 1) % m) + 1

def orbit_of(b0: int, e0: int, m: int):
    states, seen = [], set()
    b, e = b0, e0
    while (b, e) not in seen:
        seen.add((b, e))
        states.append((b, e))
        b, e = qa_step(b, e, m)
    return states

def partition(m: int):
    """Partition {1..m}² into orbits. Returns list of orbit lists."""
    remaining = {(b, e) for b in range(1, m + 1) for e in range(1, m + 1)}
    orbits = []
    while remaining:
        seed = min(remaining)
        orb = orbit_of(*seed, m)
        orbits.append(orb)
        remaining -= set(orb)
    return orbits

def is_male(b: int, e: int) -> bool:
    """b/e < √2  ↔  b*b < 2*e*e  ↔  C > F  (I > 0)."""
    return b * b < 2 * e * e

# ── Observer projections (never fed back as QA state) ─────────────────────────

def f_norm_obs(b, e):
    """Z[φ] orbit invariant: f = b²+be−e²  (observer output)."""
    return b * b + b * e - e * e

def g_norm_obs(b, e):
    """Z[√2] Pell indicator: g = b²−2e²  (observer output)."""
    return b * b - 2 * e * e

# ── Build mod-9 partition ─────────────────────────────────────────────────────

orbits9 = partition(9)

# Tag each orbit by canonical seed membership
fib9_set = frozenset(orbit_of(1, 1, 9))
luc9_set = frozenset(orbit_of(2, 1, 9))
trd9_set = frozenset(orbit_of(1, 4, 9))
sat9_set = frozenset(orbit_of(3, 3, 9))

def tag9(b, e):
    if (b, e) == (9, 9):           return 'sing'
    if (b, e) in sat9_set:         return 'sat'
    if (b, e) in fib9_set:         return 'fib'
    if (b, e) in luc9_set:         return 'luc'
    if (b, e) in trd9_set:         return 'trd'
    return 'cosmos'

# ── Build mod-24 partition ────────────────────────────────────────────────────

orbits24 = partition(24)
period24 = {}  # (b,e) → period
for orb in orbits24:
    p = len(orb)
    for s in orb:
        period24[s] = p

# ── Colours ───────────────────────────────────────────────────────────────────

C9 = {
    'fib':  '#4da6ff',
    'luc':  '#4dff91',
    'trd':  '#c06aff',
    'sat':  '#ff9a4d',
    'sing': '#ff4d4d',
}

C24 = {
    1:  '#ff4d4d',
    3:  '#ff944d',
    6:  '#ffd04d',
    8:  '#ffb04d',
    12: '#c8ff4d',
    24: '#4da6ff',
}

# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
fig.patch.set_facecolor('#0d0d0d')

for ax in axes:
    ax.set_facecolor('#111111')
    ax.spines[:].set_color('#333333')
    ax.tick_params(colors='#666666', labelsize=8)

# ── Panel 1: mod-9 ────────────────────────────────────────────────────────────

ax1 = axes[0]

for b in range(1, 10):
    for e in range(1, 10):
        tag = tag9(b, e)
        col = C9[tag]
        mk = 'o' if is_male(b, e) else '^'
        sz = 160 if tag == 'sing' else 110
        ax1.scatter(b, e, c=col, s=sz, marker=mk, zorder=3, alpha=0.92,
                    edgecolors='none')

# √2 boundary  b = e √2
e_line = np.linspace(0.4, 9.6, 300)
ax1.plot(e_line * 2**0.5, e_line,
         color='white', lw=1.3, ls='--', alpha=0.55, zorder=4,
         label='b/e = √2')

# b×e hyperbolas for K/L = 6,3,2,1  (products 1,2,3,6)
for prod, alpha in [(1, 0.35), (2, 0.5), (3, 0.35), (6, 0.25)]:
    e_h = np.linspace(max(0.4, prod / 9.6), 9.6, 300)
    b_h = prod / e_h
    mask = (b_h >= 0.4) & (b_h <= 9.6)
    ax1.plot(b_h[mask], e_h[mask], color='#80ffb0', lw=0.7, alpha=alpha, zorder=2)
    # label at left edge
    if b_h[mask].size:
        ax1.text(b_h[mask][0] + 0.05, e_h[mask][0],
                 f'be={prod}', color='#80ffb0', fontsize=6, va='center', alpha=0.7)

# seed annotations
for b, e, txt in [(1, 1, '(1,1)'), (2, 1, '(2,1)'), (1, 4, '(1,4)')]:
    ax1.annotate(txt, (b, e), xytext=(5, 5), textcoords='offset points',
                 color='white', fontsize=7, alpha=0.85)

ax1.set_xlim(0.2, 9.8)
ax1.set_ylim(0.2, 9.8)
ax1.set_xticks(range(1, 10))
ax1.set_yticks(range(1, 10))
ax1.set_xlabel('b', color='#888888', fontsize=11)
ax1.set_ylabel('e', color='#888888', fontsize=11)
ax1.set_title('mod-9 Orbit Structure\n○=male (b/e<√2)  △=female (b/e>√2)',
              color='white', fontsize=10, pad=9)
ax1.grid(True, color='#1e1e1e', lw=0.5, zorder=0)

legend_h = [
    mpatches.Patch(color=C9['fib'],  label='Fibonacci  |f|=1'),
    mpatches.Patch(color=C9['luc'],  label='Lucas      |f|=5'),
    mpatches.Patch(color=C9['trd'],  label='Third      |f|=11'),
    mpatches.Patch(color=C9['sat'],  label='Satellite  (period 8)'),
    mpatches.Patch(color=C9['sing'], label='Singularity (9,9)'),
    plt.Line2D([0], [0], color='#80ffb0', lw=1, label='b·e ∈ {1,2,3,6}  (product)'),
    plt.Line2D([0], [0], color='white', lw=1.3, ls='--', label='b/e = √2 (♂/♀)'),
]
ax1.legend(handles=legend_h, loc='upper right', fontsize=6.5,
           facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white',
           framealpha=0.85)

# ── Panel 2: mod-24 ───────────────────────────────────────────────────────────

ax2 = axes[1]

for (b, e), p in period24.items():
    col = C24.get(p, '#888888')
    mk = 'o' if is_male(b, e) else '^'
    ax2.scatter(b, e, c=col, s=14, marker=mk, zorder=3, alpha=0.82,
                edgecolors='none')

# √2 line
e_line = np.linspace(0.4, 24.6, 500)
ax2.plot(e_line * 2**0.5, e_line, color='white', lw=1.1, ls='--', alpha=0.4, zorder=4)

# b×e hyperbolas
for prod, alpha in [(2, 0.6), (6, 0.45), (24, 0.3), (72, 0.2)]:
    e_h = np.linspace(max(0.4, prod / 24.6), 24.6, 500)
    b_h = prod / e_h
    mask = (b_h >= 0.4) & (b_h <= 24.6)
    ax2.plot(b_h[mask], e_h[mask], color='#80ffb0', lw=0.6, alpha=alpha, zorder=2)

# Pell convergents (white stars) — straddle √2, alternating ♂/♀
for pb, pe in [(1, 1), (3, 2), (7, 5), (17, 12)]:
    mk = 'o' if is_male(pb, pe) else '^'
    ax2.scatter(pb, pe, c='white', s=55, marker=mk, zorder=5,
                edgecolors='#888888', lw=0.7)

ax2.text(18, 13.5, 'b/e=√2\n(♂|♀)', color='white', fontsize=7.5,
         alpha=0.7, rotation=35)
ax2.text(3, 22, '● Pell\nconvergents', color='white', fontsize=6.5, alpha=0.7)

ax2.set_xlim(0, 25.5)
ax2.set_ylim(0, 25.5)
ax2.set_xticks([1, 6, 12, 18, 24])
ax2.set_yticks([1, 6, 12, 18, 24])
ax2.set_xlabel('b', color='#888888', fontsize=11)
ax2.set_ylabel('e', color='#888888', fontsize=11)

# period counts for subtitle
from collections import Counter
pcount = Counter(period24.values())
ax2.set_title(
    f'mod-24 Orbit Periods\n'
    f'p=24: {pcount[24]}  p=8: {pcount[8]}  p=12: {pcount[12]}  '
    f'p=6: {pcount[6]}  p=3: {pcount[3]}  p=1: {pcount[1]}',
    color='white', fontsize=10, pad=9)
ax2.grid(True, color='#1a1a1a', lw=0.35, zorder=0)

legend_h2 = [mpatches.Patch(color=c, label=f'period {p}')
             for p, c in sorted(C24.items())]
ax2.legend(handles=legend_h2, loc='upper right', fontsize=7,
           facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white',
           framealpha=0.85)

# ── Panel 3: Z[φ] × Z[√2] algebraic landscape ────────────────────────────────

ax3 = axes[2]

N = 400
bv = np.linspace(0.1, 24.9, N)
ev = np.linspace(0.1, 24.9, N)
B, E = np.meshgrid(bv, ev)

F_obs = B * B + B * E - E * E          # Z[φ] norm surface (observer)
G_obs = B * B - 2 * E * E              # Z[√2] indicator (observer)

# Heatmap: f value, diverging around 0  (negative=male zone, positive=female zone)
vlim = 350
im = ax3.imshow(
    np.clip(F_obs, -vlim, vlim),
    origin='lower', extent=[0.1, 24.9, 0.1, 24.9],
    cmap='RdBu_r', aspect='equal', alpha=0.65,
    vmin=-vlim, vmax=vlim,
)

# g = 0 contour  (b/e = √2, male/female boundary) — white dashed
ax3.contour(B, E, G_obs, levels=[0],
            colors=['white'], linewidths=1.8, linestyles='--', alpha=0.8)

# f = 0 contour  (b/e = φ) — gold solid
ax3.contour(B, E, F_obs, levels=[0],
            colors=['#ffd700'], linewidths=1.5, linestyles='-', alpha=0.75)

# |f| = orbit-class isolines for |f| ∈ {1, 5, 11}  (the 3 mod-9 Cosmos norms)
ax3.contour(B, E, np.abs(F_obs), levels=[1, 5, 11],
            colors=['#4da6ff', '#4dff91', '#c06aff'],
            linewidths=0.9, linestyles=':', alpha=0.55)

# b×e hyperbolas (K/L quantization: K/L = 6/(b×e))
for prod, col, alpha, lw in [
    (1,  '#80ffb0', 0.9, 1.3),
    (2,  '#80ffb0', 0.75, 1.0),
    (3,  '#80ffb0', 0.55, 0.8),
    (6,  '#80ffb0', 0.4,  0.7),
]:
    e_h = np.linspace(max(0.1, prod / 24.9), 24.9, 600)
    b_h = prod / e_h
    mask = (b_h >= 0.1) & (b_h <= 24.9)
    ax3.plot(b_h[mask], e_h[mask], color=col, lw=lw, alpha=alpha, zorder=4)

# K/L labels near left edge
for prod, kl_label, yoff in [(1,'K/L=6',0.3),(2,'K/L=3',0.3),(3,'K/L=2',0.3),(6,'K/L=1',0.3)]:
    e_h = np.linspace(max(0.3, prod / 24.9), 24.9, 200)
    b_h = prod / e_h
    mask = (b_h >= 0.3) & (b_h <= 24.9)
    if mask.any():
        bx, ex = b_h[mask][0], e_h[mask][0]
        ax3.text(bx + 0.3, ex + yoff, kl_label,
                 color='#80ffb0', fontsize=6.5, alpha=0.85, zorder=5)

# Pell convergents — straddle g=0 line, I=1 (minimum Koenig radius)
for pb, pe in [(1, 1), (3, 2), (7, 5), (17, 12)]:
    mk = 'o' if is_male(pb, pe) else '^'
    ax3.scatter(pb, pe, c='white', s=55, marker=mk, zorder=6,
                edgecolors='#aaaaaa', lw=0.6)

# mod-9 orbit seeds (large stars)
for pb, pe, name, c in [(1,1,'Fib',C9['fib']),(2,1,'Luc',C9['luc']),(1,4,'3rd',C9['trd'])]:
    ax3.scatter(pb, pe, c=c, s=100, marker='*', zorder=7,
                edgecolors='white', lw=0.5)
    ax3.annotate(name, (pb, pe), xytext=(5, 5), textcoords='offset points',
                 color=c, fontsize=8, fontweight='bold')

# Boundary labels
ax3.text(17.5, 12.0, 'g=0\n(b/e=√2)', color='white', fontsize=7.5,
         alpha=0.85, rotation=35, zorder=6)
ax3.text(20, 14.5, 'f=0\n(b/e=φ)', color='#ffd700', fontsize=7.5,
         alpha=0.85, rotation=40, zorder=6)

# |f| contour legend inset
for fval, c, label in [(1,C9['fib'],'|f|=1'),(5,C9['luc'],'|f|=5'),(11,C9['trd'],'|f|=11')]:
    ax3.plot([], [], ':', color=c, lw=1.2, label=label, alpha=0.8)
ax3.plot([], [], '--', color='white',  lw=1.5, label='g=0 (√2)')
ax3.plot([], [], '-',  color='#ffd700', lw=1.5, label='f=0 (φ)')
ax3.plot([], [], '-',  color='#80ffb0', lw=1.0, label='b·e ∈ {1,2,3,6}  (K/L levels)')
ax3.legend(loc='upper right', fontsize=6.5,
           facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white',
           framealpha=0.85)

ax3.set_xlim(0.1, 24.9)
ax3.set_ylim(0.1, 24.9)
ax3.set_xticks([1, 6, 12, 18, 24])
ax3.set_yticks([1, 6, 12, 18, 24])
ax3.set_xlabel('b', color='#888888', fontsize=11)
ax3.set_ylabel('e', color='#888888', fontsize=11)
ax3.set_title('Z[φ] × Z[√2] Algebraic Landscape\n'
              'heatmap: f=b²+be−e²  │  contours: g=b²−2e²=0, f=0, |f|∈{1,5,11}',
              color='white', fontsize=10, pad=9)

cbar = fig.colorbar(im, ax=ax3, fraction=0.028, pad=0.025)
cbar.set_label('f = b²+be−e²  (Z[φ] norm)', color='#666666', fontsize=7.5)
cbar.ax.tick_params(colors='#666666', labelsize=7)

# ── Supertitle ────────────────────────────────────────────────────────────────

fig.suptitle('Quantum Arithmetic — (b, e) Lattice Structure',
             color='white', fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout(rect=[0, 0, 1, 0.98])

out = '/Users/player3/signal_experiments/qa_lattice_viz.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
print(f'Saved: {out}')
