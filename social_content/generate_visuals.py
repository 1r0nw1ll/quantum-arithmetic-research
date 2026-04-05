QA_COMPLIANCE = "observer_script — generates social media visualizations from QA results"
"""Generate 5 social media images for QA project.

Run: python social_content/generate_visuals.py
Output: social_content/*.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

np.random.seed(42)

OUT = "social_content"


def qa_mod(x, m=24):
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m=24):
    d = qa_mod(b + e, m)  # noqa: A2-1
    return e, d


# ── Visual 1: Orbit Structure ──────────────────────────────────────────

def generate_orbit_map():
    """576 pairs colored by orbit family."""
    m = 24
    grid = np.zeros((m, m))
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            cb, ce = b, e
            for _ in range(m + 1):
                nb, ne = qa_step(cb, ce, m)
                if (nb, ne) == (b, e):
                    break
                cb, ce = nb, ne
            period = 1
            cb, ce = qa_step(b, e, m)
            while (cb, ce) != (b, e):
                cb, ce = qa_step(cb, ce, m)
                period += 1
                if period > m * m:
                    break
            if period == 1:
                grid[b - 1, e - 1] = 3  # singularity
            elif period <= 8:
                grid[b - 1, e - 1] = 2  # satellite
            else:
                grid[b - 1, e - 1] = 1  # cosmos

    fig, ax = plt.subplots(figsize=(10, 10))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#1a1a2e', '#e94560', '#0f3460', '#ffd700'])
    ax.imshow(grid, cmap=cmap, origin='lower', aspect='equal')
    ax.set_xlabel('e', fontsize=14)
    ax.set_ylabel('b', fontsize=14)
    ax.set_title('QA Orbit Structure (mod 24)\n552 Cosmos | 23 Satellite | 1 Singularity',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks([0, 7, 15, 23])
    ax.set_xticklabels([1, 8, 16, 24])
    ax.set_yticks([0, 7, 15, 23])
    ax.set_yticklabels([1, 8, 16, 24])
    fig.tight_layout()
    fig.savefig(f'{OUT}/01_orbit_structure.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a', edgecolor='none')
    plt.close()
    print("  01_orbit_structure.png")


# ── Visual 2: Fibonacci Spiral in QA ───────────────────────────────────

def generate_fibonacci_spiral():
    """Fibonacci orbit trajectory on mod-24 state space."""
    m = 24
    b, e = 1, 1
    path_b, path_e = [b], [e]
    for _ in range(24):
        b, e = qa_step(b, e, m)
        path_b.append(b)
        path_e.append(e)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')

    # Plot trajectory with gradient color
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(path_b) - 1))
    for i in range(len(path_b) - 1):
        ax.plot([path_e[i], path_e[i + 1]], [path_b[i], path_b[i + 1]],
                color=colors[i], linewidth=2.5, alpha=0.8)
        ax.scatter(path_e[i], path_b[i], color=colors[i], s=80, zorder=5)

    # Annotate first few steps
    labels = ['F1', 'F1', 'F2', 'F3', 'F5', 'F8', 'F13', 'F21']
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (path_e[i], path_b[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, color='white', fontweight='bold')

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_xlabel('e', fontsize=14, color='white')
    ax.set_ylabel('b', fontsize=14, color='white')
    ax.set_title('Fibonacci Orbit in QA State Space\n(1,1) → (1,2) → (2,3) → (3,5) → (5,8) → ...',
                 fontsize=16, fontweight='bold', color='white', pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    fig.tight_layout()
    fig.savefig(f'{OUT}/02_fibonacci_spiral.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a', edgecolor='none')
    plt.close()
    print("  02_fibonacci_spiral.png")


# ── Visual 3: Domain Results Scorecard ─────────────────────────────────

def generate_scorecard():
    """6-domain empirical results as a visual scorecard."""
    domains = ['EEG\nSeizure', 'EMG\nPathology', 'Climate\nENSO',
               'Finance\nVolatility', 'Audio\nClassif.', 'ERA5\nAtmos.']
    deltas = [0.21, 0.61, 0.97, 0.22, 0.75, 0.20]
    labels = ['+0.21 R²', '+0.61 R²', '97% sat.', 'r=-0.22', 'r=+0.75', 'r=-0.20']
    colors = ['#e94560', '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#9b59b6']

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    bars = ax.bar(domains, deltas, color=colors, width=0.6, edgecolor='none')
    for bar, lbl in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                lbl, ha='center', va='bottom', fontsize=13, fontweight='bold',
                color='white')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Effect Size', fontsize=14, color='white')
    ax.set_title('QA Coherence Index — 6 Domains, All Significant\nEvery result controls for the best conventional predictor',
                 fontsize=16, fontweight='bold', color='white', pad=15)
    ax.tick_params(colors='white', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    fig.tight_layout()
    fig.savefig(f'{OUT}/03_domain_scorecard.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a', edgecolor='none')
    plt.close()
    print("  03_domain_scorecard.png")


# ── Visual 4: Chromogeometry Identity ──────────────────────────────────

def generate_chromogeometry():
    """Visual proof: C² + F² = G² for Fibonacci directions."""
    directions = [(2, 1), (3, 2), (5, 3), (8, 5), (13, 8), (21, 13)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0a1a')
    fig.suptitle('Chromogeometry: C² + F² = G² for Fibonacci Directions',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    for idx, (d, e) in enumerate(directions):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor('#0a0a1a')
        C = 2 * d * e
        F = d * d - e * e
        G = d * d + e * e

        # Draw squares proportional to C², F², G²
        vals = [C * C, F * F, G * G]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        labels = [f'C²={C*C}', f'F²={F*F}', f'G²={G*G}']
        ax.bar(range(3), vals, color=colors, width=0.7)
        for i, (v, l) in enumerate(zip(vals, labels)):
            ax.text(i, v + max(vals) * 0.03, l, ha='center', fontsize=9,
                    color='white', fontweight='bold')

        ax.set_title(f'(d,e) = ({d},{e})\n{C}² + {F}² = {G}²',
                     fontsize=12, color='white', fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Green', 'Red', 'Blue'], fontsize=10, color='white')
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/04_chromogeometry.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a', edgecolor='none')
    plt.close()
    print("  04_chromogeometry.png")


# ── Visual 5: QCI Structured vs Noise ──────────────────────────────────

def generate_qci_separation():
    """QCI time series: structured orbit vs random noise."""
    m = 24
    # Structured
    structured = []
    b, e = 1, 1
    for _ in range(300):
        structured.append(b)
        b, e = qa_step(b, e, m)

    # Noise
    noise = [int(x) for x in np.random.randint(1, m + 1, size=300)]

    def rolling_qci(seq, window=40):
        matches = []
        for t in range(len(seq) - 2):
            pred = qa_mod(seq[t] + seq[t + 1], m)
            matches.append(1 if pred == seq[t + 2] else 0)
        out = []
        for i in range(len(matches)):
            start = max(0, i - window + 1)
            out.append(np.mean(matches[start:i + 1]))
        return out

    s_qci = rolling_qci(structured)
    n_qci = rolling_qci(noise)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    ax.plot(s_qci, color='#ffd700', linewidth=2, label='Fibonacci orbit (QCI = 1.0)')
    ax.plot(n_qci, color='#e94560', linewidth=1.5, alpha=0.7, label='Random noise (QCI = chance)')
    ax.axhline(y=1.0 / m, color='#555', linestyle='--', linewidth=1, label=f'Chance = 1/{m}')

    ax.set_xlabel('Time step', fontsize=13, color='white')
    ax.set_ylabel('QCI (T-operator match rate)', fontsize=13, color='white')
    ax.set_title('QA Coherence Index Separates Structure from Noise',
                 fontsize=16, fontweight='bold', color='white', pad=15)
    ax.legend(fontsize=12, loc='center right', facecolor='#1a1a2e', edgecolor='#333',
              labelcolor='white')
    ax.set_ylim(-0.05, 1.1)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    fig.tight_layout()
    fig.savefig(f'{OUT}/05_qci_separation.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a', edgecolor='none')
    plt.close()
    print("  05_qci_separation.png")


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    print("Generating social content visuals...")
    generate_orbit_map()
    generate_fibonacci_spiral()
    generate_scorecard()
    generate_chromogeometry()
    generate_qci_separation()
    print(f"\nDone. 5 images in {OUT}/")
