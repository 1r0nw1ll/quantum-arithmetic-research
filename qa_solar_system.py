#!/usr/bin/env python3
"""
QA Solar System — Quantum Numbers for all planets, major moons, and comets.

For each body, finds the best QN (b,e,d,a) such that eccentricity e/d matches
the orbital eccentricity. Then maps the prime factor sharing network to reveal
harmonic resonances (Ben's Law of Harmonics).

Outputs:
  1. Table of all QNs with triples
  2. Prime factor sharing matrix
  3. Harmonic network graph
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# ═══════════════════════════════════════════
# SOLAR SYSTEM DATA
# ═══════════════════════════════════════════

BODIES = {
    # Planets — orbital eccentricity around Sun
    'Mercury':   {'ecc': 0.20563, 'type': 'planet', 'parent': 'Sun'},
    'Venus':     {'ecc': 0.00677, 'type': 'planet', 'parent': 'Sun'},
    'Earth':     {'ecc': 0.01671, 'type': 'planet', 'parent': 'Sun'},
    'Mars':      {'ecc': 0.09339, 'type': 'planet', 'parent': 'Sun'},
    'Jupiter':   {'ecc': 0.04839, 'type': 'planet', 'parent': 'Sun'},
    'Saturn':    {'ecc': 0.05386, 'type': 'planet', 'parent': 'Sun'},
    'Uranus':    {'ecc': 0.04726, 'type': 'planet', 'parent': 'Sun'},
    'Neptune':   {'ecc': 0.00860, 'type': 'planet', 'parent': 'Sun'},

    # Major moons — orbital eccentricity around parent
    'Moon':      {'ecc': 0.0549, 'type': 'moon', 'parent': 'Earth'},
    'Io':        {'ecc': 0.0041, 'type': 'moon', 'parent': 'Jupiter'},
    'Europa':    {'ecc': 0.0094, 'type': 'moon', 'parent': 'Jupiter'},
    'Ganymede':  {'ecc': 0.0013, 'type': 'moon', 'parent': 'Jupiter'},
    'Callisto':  {'ecc': 0.0074, 'type': 'moon', 'parent': 'Jupiter'},
    'Titan':     {'ecc': 0.0288, 'type': 'moon', 'parent': 'Saturn'},
    'Triton':    {'ecc': 0.000016, 'type': 'moon', 'parent': 'Neptune'},
    'Phobos':    {'ecc': 0.0151, 'type': 'moon', 'parent': 'Mars'},
    'Deimos':    {'ecc': 0.0002, 'type': 'moon', 'parent': 'Mars'},
    'Enceladus': {'ecc': 0.0047, 'type': 'moon', 'parent': 'Saturn'},
    'Mimas':     {'ecc': 0.0196, 'type': 'moon', 'parent': 'Saturn'},

    # Dwarf planets
    'Pluto':     {'ecc': 0.2488, 'type': 'dwarf', 'parent': 'Sun'},
    'Ceres':     {'ecc': 0.0758, 'type': 'dwarf', 'parent': 'Sun'},
    'Eris':      {'ecc': 0.4407, 'type': 'dwarf', 'parent': 'Sun'},
    'Haumea':    {'ecc': 0.1912, 'type': 'dwarf', 'parent': 'Sun'},
    'Makemake':  {'ecc': 0.1559, 'type': 'dwarf', 'parent': 'Sun'},

    # Notable comets
    'Halley':    {'ecc': 0.96714, 'type': 'comet', 'parent': 'Sun'},
    'Hale-Bopp': {'ecc': 0.99510, 'type': 'comet', 'parent': 'Sun'},
    'Encke':     {'ecc': 0.8483, 'type': 'comet', 'parent': 'Sun'},
    'Hyakutake': {'ecc': 0.99990, 'type': 'comet', 'parent': 'Sun'},
}


def find_best_qn(target_ecc, max_d=500, top_n=3):
    """Find best primitive QN(s) matching target eccentricity."""
    matches = []
    for d in range(2, max_d):
        for e in range(1, d):
            if math.gcd(e, d) > 1:
                continue
            ecc = e / d
            diff = abs(ecc - target_ecc)
            if diff < 0.002:
                b = d - e
                a = d + e
                matches.append((diff, b, e, d, a, ecc))
    matches.sort()
    return matches[:top_n]


def prime_factors(n):
    """Return set of prime factors."""
    n = abs(n)
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def all_factors_from_qn(b, e, d, a):
    """Get all prime factors appearing in the QN tuple."""
    factors = set()
    for val in [b, e, d, a]:
        factors |= prime_factors(val)
    return factors


def main():
    print('=' * 100)
    print('QA SOLAR SYSTEM — Quantum Numbers for Celestial Bodies')
    print('=' * 100)
    print()

    # Find QNs for all bodies
    results = {}
    for name, data in BODIES.items():
        ecc = data['ecc']

        # Special handling for very small eccentricities
        if ecc < 0.001:
            matches = find_best_qn(ecc, max_d=2000, top_n=3)
        else:
            matches = find_best_qn(ecc, max_d=500, top_n=3)

        if matches:
            diff, b, e, d, a, found_ecc = matches[0]
            F = a * b
            C = 2 * d * e
            G = d*d + e*e
            results[name] = {
                'qn': (b, e, d, a),
                'ecc': found_ecc,
                'target_ecc': data['ecc'],
                'err': diff,
                'triple': (C, F, G),
                'type': data['type'],
                'parent': data['parent'],
                'factors': all_factors_from_qn(b, e, d, a),
            }

    # ═══ PRINT TABLE ═══
    print(f'{"Body":<12} {"Type":<7} {"QN (b,e,d,a)":<22} {"ecc=e/d":<12} '
          f'{"actual":<10} {"err%":<8} {"Triple (C,F,G)":<24} {"Key primes"}')
    print('─' * 130)

    for name in sorted(results.keys(), key=lambda n: (
            {'planet': 0, 'moon': 1, 'dwarf': 2, 'comet': 3}[results[n]['type']],
            results[n]['target_ecc'])):
        r = results[name]
        b, e, d, a = r['qn']
        C, F, G = r['triple']
        err_pct = r['err'] / r['target_ecc'] * 100 if r['target_ecc'] > 0 else 0
        primes = sorted(r['factors'])
        prime_str = ','.join(str(p) for p in primes if p > 2)
        print(f'{name:<12} {r["type"]:<7} ({b:>3},{e:>3},{d:>3},{a:>3}) '
              f'{r["ecc"]:<12.6f} {r["target_ecc"]:<10.5f} {err_pct:<8.3f} '
              f'({C},{F},{G})'[:24].ljust(24) + f' {prime_str}')

    # ═══ PRIME FACTOR SHARING MATRIX ═══
    print()
    print('=' * 100)
    print('PRIME FACTOR SHARING (Ben\'s Law of Harmonics)')
    print('=' * 100)
    print()

    # Collect all primes > 2
    all_primes = set()
    for r in results.values():
        all_primes |= {p for p in r['factors'] if p > 2}

    # For each pair of bodies, find shared primes
    names = sorted(results.keys())
    shared_pairs = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            shared = results[n1]['factors'] & results[n2]['factors']
            shared_big = {p for p in shared if p > 2}
            if shared_big:
                shared_pairs.append((n1, n2, shared_big))

    # Print most interesting connections (shared primes > 5)
    shared_pairs.sort(key=lambda x: -max(x[2]))
    print(f'{"Body 1":<12} {"Body 2":<12} {"Shared primes":<30} {"Resonance strength"}')
    print('─' * 80)
    for n1, n2, shared in shared_pairs[:50]:
        big_shared = {p for p in shared if p > 5}
        if big_shared:
            strength = sum(1/p for p in big_shared)  # lower ratio = stronger harmony
            primes_str = ', '.join(str(p) for p in sorted(big_shared))
            print(f'{n1:<12} {n2:<12} {primes_str:<30} {strength:.4f}')

    # ═══ HARMONIC NETWORK GRAPH ═══
    build_harmonic_graph(results, shared_pairs)

    # ═══ PRIME FAMILIES ═══
    print()
    print('=' * 100)
    print('PRIME HARMONIC FAMILIES')
    print('=' * 100)
    print()

    # Group bodies by shared primes
    prime_to_bodies = defaultdict(list)
    for name, r in results.items():
        for p in r['factors']:
            if p > 5:
                prime_to_bodies[p].append(name)

    for p in sorted(prime_to_bodies.keys()):
        bodies = prime_to_bodies[p]
        if len(bodies) >= 2:
            print(f'  Prime {p:>4}: {", ".join(sorted(bodies))}')


def build_harmonic_graph(results, shared_pairs):
    """Build a network graph showing harmonic resonances."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_title('QA Solar System Harmonic Network\n'
                 'Bodies connected by shared prime factors (Law of Harmonics)',
                 fontsize=14, fontweight='bold')

    # Position bodies in concentric rings by type
    type_colors = {'planet': '#2196F3', 'moon': '#4CAF50', 'dwarf': '#FF9800', 'comet': '#F44336'}
    type_radii = {'planet': 3.0, 'moon': 5.5, 'dwarf': 7.5, 'comet': 9.5}

    positions = {}
    type_groups = defaultdict(list)
    for name, r in results.items():
        type_groups[r['type']].append(name)

    for body_type, names in type_groups.items():
        radius = type_radii[body_type]
        n = len(names)
        for i, name in enumerate(sorted(names)):
            angle = 2 * np.pi * i / n - np.pi/2
            # Offset each ring so they don't overlap
            offset = {'planet': 0, 'moon': 0.3, 'dwarf': 0.6, 'comet': 0.9}[body_type]
            positions[name] = (radius * np.cos(angle + offset),
                               radius * np.sin(angle + offset))

    # Draw connections (shared primes > 5)
    for n1, n2, shared in shared_pairs:
        big_shared = {p for p in shared if p > 5}
        if big_shared and n1 in positions and n2 in positions:
            x1, y1 = positions[n1]
            x2, y2 = positions[n2]
            max_prime = max(big_shared)
            # Thicker line for sharing larger primes
            lw = min(3, len(big_shared) * 0.8)
            alpha = min(0.8, 0.2 + len(big_shared) * 0.15)
            ax.plot([x1, x2], [y1, y2], '-', color='gray', lw=lw, alpha=alpha, zorder=1)
            # Label the shared prime on the midpoint
            mx, my = (x1+x2)/2, (y1+y2)/2
            label = ','.join(str(p) for p in sorted(big_shared) if p > 10)
            if label:
                ax.text(mx, my, label, fontsize=6, ha='center', va='center',
                        color='purple', alpha=0.7)

    # Draw nodes
    for name, r in results.items():
        if name not in positions:
            continue
        x, y = positions[name]
        color = type_colors[r['type']]
        size = 12 if r['type'] == 'planet' else (10 if r['type'] == 'moon' else 8)

        ax.plot(x, y, 'o', color=color, markersize=size, zorder=3,
                markeredgecolor='black', markeredgewidth=0.5)

        b, e, d, a = r['qn']
        ax.text(x, y - 0.45, name, fontsize=8, ha='center', va='top', fontweight='bold')
        ax.text(x, y - 0.8, f'({b},{e},{d},{a})', fontsize=6, ha='center',
                va='top', color='gray')

    # Legend
    for btype, color in type_colors.items():
        ax.plot([], [], 'o', color=color, markersize=10, label=btype.capitalize())
    ax.legend(loc='upper left', fontsize=10)

    ax.set_aspect('equal')
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('qa_solar_system_harmonics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved: qa_solar_system_harmonics.png')


if __name__ == '__main__':
    main()
