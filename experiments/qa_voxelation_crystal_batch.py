#!/usr/bin/env python3
QA_COMPLIANCE = "observer=experiment_script, state_alphabet=mod{9,24}"
"""
QA Voxelation × Crystallography — Batch Analysis
=================================================

Uses known crystal structure data to generate Miller index families
for 19 minerals across multiple crystal systems. Tests whether
QA orbit classification predicts intensity/d-spacing structure
at scale.

Rather than downloading individual DIF files, we use the known
lattice parameters and space groups to enumerate all allowed
reflections up to Q_M_max and compute d-spacings algebraically.

For cubic: d² = a²/Q_M  (exact rational)
For hexagonal: 1/d² = (4/3)(h²+hk+k²)/a² + l²/c²
For tetragonal: 1/d² = (h²+k²)/a² + l²/c²
For orthorhombic: 1/d² = h²/a² + k²/b² + l²/c²

Will Dale, 2026-04-03
"""

QA_COMPLIANCE = "observer_projection — crystal lattice parameters are observer-layer measurements; QA orbit classification applied to integer Miller indices; no float state in QA logic"

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from qa_orbit_rules import orbit_family  # noqa: ORBIT-4,ORBIT-5

np.random.seed(42)

# ─── QA Core ─────────────────────────────────────────────────────

def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1

# ─── Crystal Database ────────────────────────────────────────────
# Real lattice parameters from RRUFF, AMCSD, and standard references

CRYSTALS = [
    # Cubic (a only)
    {'name': 'Silicon',    'system': 'cubic', 'a': 5.4299, 'sg': 'Fd3m'},
    {'name': 'Diamond',    'system': 'cubic', 'a': 3.5668, 'sg': 'Fd3m'},
    {'name': 'Halite',     'system': 'cubic', 'a': 5.6402, 'sg': 'Fm3m'},  # noqa: A2-2
    {'name': 'Magnetite',  'system': 'cubic', 'a': 8.3970, 'sg': 'Fd3m'},  # noqa: A2-2
    {'name': 'Fluorite',   'system': 'cubic', 'a': 5.4626, 'sg': 'Fm3m'},  # noqa: A2-2
    {'name': 'Pyrite',     'system': 'cubic', 'a': 5.4166, 'sg': 'Pa3'},  # noqa: A2-2
    {'name': 'Galena',     'system': 'cubic', 'a': 5.9360, 'sg': 'Fm3m'},  # noqa: A2-2
    {'name': 'Garnet',     'system': 'cubic', 'a': 11.459, 'sg': 'Ia3d'},  # noqa: A2-2

    # Hexagonal (a, c)
    {'name': 'Quartz',     'system': 'hexagonal', 'a': 4.9134, 'c': 5.4042, 'sg': 'P3_221'},  # noqa: A2-2
    {'name': 'Calcite',    'system': 'hexagonal', 'a': 4.9892, 'c': 17.062, 'sg': 'R-3c'},  # noqa: A2-2
    {'name': 'Corundum',   'system': 'hexagonal', 'a': 4.7589, 'c': 12.991, 'sg': 'R-3c'},  # noqa: A2-2
    {'name': 'Hematite',   'system': 'hexagonal', 'a': 5.0356, 'c': 13.749, 'sg': 'R-3c'},  # noqa: A2-2
    {'name': 'Apatite',    'system': 'hexagonal', 'a': 9.4180, 'c': 6.8840, 'sg': 'P6_3/m'},  # noqa: A2-2
    {'name': 'Beryl',      'system': 'hexagonal', 'a': 9.2100, 'c': 9.1900, 'sg': 'P6/mcc'},  # noqa: A2-2

    # Tetragonal (a, c)
    {'name': 'Rutile',     'system': 'tetragonal', 'a': 4.5937, 'c': 2.9587, 'sg': 'P4_2/mnm'},  # noqa: A2-2
    {'name': 'Anatase',    'system': 'tetragonal', 'a': 3.7852, 'c': 9.5139, 'sg': 'I4_1/amd'},  # noqa: A2-2
    {'name': 'Zircon',     'system': 'tetragonal', 'a': 6.6042, 'c': 5.9796, 'sg': 'I4_1/amd'},  # noqa: A2-2

    # Orthorhombic (a, b, c)
    {'name': 'Forsterite', 'system': 'orthorhombic', 'a': 4.756, 'b': 10.207, 'c': 5.980, 'sg': 'Pbnm'},  # noqa: A2-2
    {'name': 'Anhydrite',  'system': 'orthorhombic', 'a': 6.993, 'b': 6.995, 'c': 6.245, 'sg': 'Amma'},  # noqa: A2-2
    {'name': 'Enstatite',  'system': 'orthorhombic', 'a': 18.228, 'b': 8.819, 'c': 5.185, 'sg': 'Pbca'},  # noqa: A2-2
    {'name': 'Olivine',    'system': 'orthorhombic', 'a': 4.822, 'b': 10.477, 'c': 6.105, 'sg': 'Pbnm'},  # noqa: A2-2
]


def compute_d_spacing(h, k, l, crystal):
    """Compute d-spacing from Miller indices and lattice parameters."""
    sys = crystal['system']
    if sys == 'cubic':
        q = h*h + k*k + l*l  # noqa: A2-2
        if q == 0:
            return None
        return crystal['a'] / np.sqrt(q)  # noqa: A2-2
    elif sys == 'hexagonal':
        inv_d2 = (4.0/3.0) * (h*h + h*k + k*k) / (crystal['a'] * crystal['a']) + \
                 (l*l) / (crystal['c'] * crystal['c'])  # noqa: A2-2
        if inv_d2 == 0:
            return None
        return 1.0 / np.sqrt(inv_d2)
    elif sys == 'tetragonal':
        inv_d2 = (h*h + k*k) / (crystal['a'] * crystal['a']) + \
                 (l*l) / (crystal['c'] * crystal['c'])  # noqa: A2-2
        if inv_d2 == 0:
            return None
        return 1.0 / np.sqrt(inv_d2)
    elif sys == 'orthorhombic':
        inv_d2 = (h*h) / (crystal['a'] * crystal['a']) + \
                 (k*k) / (crystal['b'] * crystal['b']) + \
                 (l*l) / (crystal['c'] * crystal['c'])  # noqa: A2-2
        if inv_d2 == 0:
            return None
        return 1.0 / np.sqrt(inv_d2)
    return None


def enumerate_reflections(crystal, q_max=50, d_min=0.8):
    """Enumerate all (h,k,l) with Q_M <= q_max and d >= d_min."""
    h_max = int(np.sqrt(q_max)) + 1
    refs = []
    seen = set()
    for h in range(0, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                q = h*h + k*k + l*l
                if q > q_max:
                    continue
                # Canonical form: skip duplicates (h,k,l) ~ (-h,-k,-l)
                canon = tuple(sorted([(h,k,l), (-h,-k,-l)]))[1]
                if canon in seen:
                    continue
                seen.add(canon)

                d = compute_d_spacing(h, k, l, crystal)
                if d is None or d < d_min:
                    continue

                refs.append({
                    'h': h, 'k': k, 'l': l,
                    'Q_M': q,
                    'd_spacing': d,
                })
    return refs


# ─── Generate reflections for all crystals ───────────────────────

print("=" * 70)
print("QA VOXELATION × CRYSTALLOGRAPHY — BATCH ANALYSIS")
print(f"  {len(CRYSTALS)} minerals, 4 crystal systems")
print("=" * 70)

all_data = []

for crystal in CRYSTALS:
    refs = enumerate_reflections(crystal, q_max=50, d_min=0.8)

    for r in refs:
        h, k = abs(r['h']), abs(r['k'])
        # Chromogeometric quadrances of (|h|,|k|) direction
        r['Qr'] = h*h - k*k
        r['Qg'] = 2 * h * k
        r['Qb'] = h*h + k*k

        # QA orbit classification
        h9 = qa_mod(h, 9) if h > 0 else 9
        k9 = qa_mod(k, 9) if k > 0 else 9
        r['orbit_9'] = orbit_family(h9, k9, 9)

        h24 = qa_mod(h, 24) if h > 0 else 24
        k24 = qa_mod(k, 24) if k > 0 else 24
        r['orbit_24'] = orbit_family(h24, k24, 24)

        # Chromogeometric channel
        abs_channels = [abs(r['Qr']), r['Qg'], r['Qb']]
        mx = max(abs_channels)
        if mx == 0:
            r['channel'] = 'degenerate'
        elif mx == r['Qb'] and r['Qb'] > r['Qg']:
            r['channel'] = 'blue'
        elif mx == r['Qg'] and r['Qg'] > abs(r['Qr']):
            r['channel'] = 'green'
        elif mx == abs(r['Qr']):
            r['channel'] = 'red'
        else:
            r['channel'] = 'tie'

        r['mineral'] = crystal['name']
        r['system'] = crystal['system']
        all_data.append(r)

    print(f"  {crystal['name']:15s} ({crystal['system']:13s}): {len(refs):4d} reflections")

total = len(all_data)
print(f"\n  TOTAL: {total} reflections across {len(CRYSTALS)} minerals")

# ─── Aggregate Analysis ──────────────────────────────────────────

print(f"\n{'─' * 70}")
print("ORBIT DISTRIBUTION (mod-9)")
print(f"{'─' * 70}")

orbit_counts = Counter(r['orbit_9'] for r in all_data)
for orb in ['cosmos', 'satellite', 'singularity']:
    c = orbit_counts.get(orb, 0)
    print(f"  {orb:12s}: {c:5d} ({c/total:.4f})")

# By crystal system
print(f"\nBy crystal system:")
systems = sorted(set(r['system'] for r in all_data))
for sys in systems:
    sys_data = [r for r in all_data if r['system'] == sys]
    n = len(sys_data)
    counts = Counter(r['orbit_9'] for r in sys_data)
    parts = [f"{orb}={counts.get(orb,0)}" for orb in ['cosmos', 'satellite', 'singularity']]
    print(f"  {sys:15s} (n={n:4d}): {', '.join(parts)}")

# ─── KEY TEST: d-spacing by orbit ────────────────────────────────

print(f"\n{'─' * 70}")
print("D-SPACING BY ORBIT TYPE")
print(f"{'─' * 70}")

from scipy import stats

for orb in ['cosmos', 'satellite', 'singularity']:
    orb_data = [r for r in all_data if r['orbit_9'] == orb]
    if orb_data:
        ds = [r['d_spacing'] for r in orb_data]
        print(f"\n  {orb} (n={len(orb_data)}):")
        print(f"    mean_d = {np.mean(ds):.4f} Å")
        print(f"    med_d  = {np.median(ds):.4f} Å")
        print(f"    std_d  = {np.std(ds):.4f} Å")
        print(f"    range  = [{min(ds):.4f}, {max(ds):.4f}]")

# Statistical test: cosmos vs satellite d-spacing
cosmos_d = [r['d_spacing'] for r in all_data if r['orbit_9'] == 'cosmos']
sat_d = [r['d_spacing'] for r in all_data if r['orbit_9'] == 'satellite']
if len(cosmos_d) > 1 and len(sat_d) > 1:
    t_stat, p_val = stats.ttest_ind(cosmos_d, sat_d)
    ks_stat, ks_p = stats.ks_2samp(cosmos_d, sat_d)
    mw_stat, mw_p = stats.mannwhitneyu(cosmos_d, sat_d, alternative='two-sided')
    print(f"\n  Cosmos vs Satellite:")
    print(f"    t-test:        t={t_stat:.3f}, p={p_val:.6f}")
    print(f"    KS test:       D={ks_stat:.3f}, p={ks_p:.6f}")
    print(f"    Mann-Whitney:  U={mw_stat:.0f}, p={mw_p:.6f}")

# ─── KEY TEST: Q_M by orbit ─────────────────────────────────────

print(f"\n{'─' * 70}")
print("MILLER QUADRANCE Q_M BY ORBIT TYPE")
print(f"{'─' * 70}")

for orb in ['cosmos', 'satellite', 'singularity']:
    orb_data = [r for r in all_data if r['orbit_9'] == orb]
    if orb_data:
        qms = [r['Q_M'] for r in orb_data]
        print(f"\n  {orb} (n={len(orb_data)}):")
        print(f"    mean_Q_M = {np.mean(qms):.2f}")
        print(f"    med_Q_M  = {np.median(qms):.2f}")
        print(f"    Q_M values = {sorted(set(qms))}")

# ─── KEY TEST: chromogeometric channel distribution by orbit ─────

print(f"\n{'─' * 70}")
print("CHROMOGEOMETRIC CHANNEL BY ORBIT TYPE")
print(f"{'─' * 70}")

for orb in ['cosmos', 'satellite', 'singularity']:
    orb_data = [r for r in all_data if r['orbit_9'] == orb]
    if not orb_data:
        continue
    n = len(orb_data)
    ch_counts = Counter(r['channel'] for r in orb_data)
    parts = [f"{ch}={ch_counts.get(ch,0)} ({ch_counts.get(ch,0)/n:.3f})" for ch in ['blue', 'green', 'red', 'tie', 'degenerate']]
    print(f"  {orb:12s} (n={n:4d}): {', '.join(parts)}")

# ─── KEY TEST: Q_M mod structure ─────────────────────────────────

print(f"\n{'─' * 70}")
print("Q_M MOD-9 RESIDUE BY ORBIT TYPE")
print(f"{'─' * 70}")

for orb in ['cosmos', 'satellite', 'singularity']:
    orb_data = [r for r in all_data if r['orbit_9'] == orb]
    if orb_data:
        residues = Counter(r['Q_M'] % 9 for r in orb_data)
        print(f"\n  {orb}:")
        for res in range(9):
            c = residues.get(res, 0)
            bar = '█' * (c // 2) if c > 0 else ''
            print(f"    Q_M ≡ {res} (mod 9): {c:4d} {bar}")

# ─── Per-mineral breakdown ───────────────────────────────────────

print(f"\n{'─' * 70}")
print("PER-MINERAL D-SPACING BY ORBIT (mod-9)")
print(f"{'─' * 70}")

for crystal in CRYSTALS:
    name = crystal['name']
    mineral_data = [r for r in all_data if r['mineral'] == name]
    cosmos_d = [r['d_spacing'] for r in mineral_data if r['orbit_9'] == 'cosmos']
    sat_d = [r['d_spacing'] for r in mineral_data if r['orbit_9'] == 'satellite']
    sing_d = [r['d_spacing'] for r in mineral_data if r['orbit_9'] == 'singularity']

    parts = []
    if cosmos_d:
        parts.append(f"cos={np.mean(cosmos_d):.3f}({len(cosmos_d)})")
    if sat_d:
        parts.append(f"sat={np.mean(sat_d):.3f}({len(sat_d)})")
    if sing_d:
        parts.append(f"sing={np.mean(sing_d):.3f}({len(sing_d)})")

    # Check ordering: cosmos > satellite > singularity (mean d)
    ordering = "?"
    if cosmos_d and sat_d:
        if np.mean(cosmos_d) > np.mean(sat_d):
            ordering = "cos>sat ✓"
        else:
            ordering = "cos<sat ✗"
    print(f"  {name:15s}: {', '.join(parts):50s}  {ordering}")

# ─── Visualization ───────────────────────────────────────────────

print(f"\n{'─' * 70}")
print("GENERATING VISUALIZATIONS...")
print(f"{'─' * 70}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
colors_orbit = {'cosmos': '#2196F3', 'satellite': '#FF5722', 'singularity': '#4CAF50'}
colors_system = {'cubic': '#E91E63', 'hexagonal': '#FF9800',
                 'tetragonal': '#9C27B0', 'orthorhombic': '#00BCD4'}

# Panel 1: d-spacing distribution by orbit
ax = axes[0, 0]
for orb in ['cosmos', 'satellite', 'singularity']:
    ds = [r['d_spacing'] for r in all_data if r['orbit_9'] == orb]
    if ds:
        ax.hist(ds, bins=30, alpha=0.5, color=colors_orbit[orb],
                label=f'{orb} (n={len(ds)})', density=True)
ax.set_xlabel('d-spacing (Å)')
ax.set_ylabel('Density')
ax.set_title('d-spacing by QA Orbit (mod-9)', fontsize=11)
ax.legend(fontsize=8)

# Panel 2: Q_M vs d-spacing by orbit
ax = axes[0, 1]
for orb in ['cosmos', 'satellite', 'singularity']:
    orb_data = [r for r in all_data if r['orbit_9'] == orb]
    if orb_data:
        qms = [r['Q_M'] for r in orb_data]
        ds = [r['d_spacing'] for r in orb_data]
        s = 5 if orb == 'cosmos' else (30 if orb == 'satellite' else 80)
        ax.scatter(qms, ds, c=colors_orbit[orb], s=s, alpha=0.4, label=orb)
ax.set_xlabel('Q_M = h² + k² + l²')
ax.set_ylabel('d-spacing (Å)')
ax.set_title('Q_M vs d-spacing by Orbit', fontsize=11)
ax.legend(fontsize=8)

# Panel 3: Orbit distribution by crystal system
ax = axes[0, 2]
sys_names = []
cos_fracs = []
sat_fracs = []
sing_fracs = []
for sys in systems:
    sys_data = [r for r in all_data if r['system'] == sys]
    n = len(sys_data)
    counts = Counter(r['orbit_9'] for r in sys_data)
    sys_names.append(sys[:4])
    cos_fracs.append(counts.get('cosmos', 0) / n)
    sat_fracs.append(counts.get('satellite', 0) / n)
    sing_fracs.append(counts.get('singularity', 0) / n)

x = np.arange(len(sys_names))
w = 0.25
ax.bar(x - w, cos_fracs, w, label='cosmos', color=colors_orbit['cosmos'])
ax.bar(x, sat_fracs, w, label='satellite', color=colors_orbit['satellite'])
ax.bar(x + w, sing_fracs, w, label='singularity', color=colors_orbit['singularity'])
ax.set_xticks(x)
ax.set_xticklabels(sys_names)
ax.set_ylabel('Fraction')
ax.set_title('Orbit Distribution by Crystal System', fontsize=11)
ax.legend(fontsize=8)

# Panel 4: Chromogeometric channel by orbit
ax = axes[1, 0]
channel_names = ['blue', 'green', 'red', 'tie', 'degenerate']
channel_colors = ['#2196F3', '#4CAF50', '#F44336', '#9E9E9E', '#000000']
for i, orb in enumerate(['cosmos', 'satellite', 'singularity']):
    orb_data = [r for r in all_data if r['orbit_9'] == orb]
    if not orb_data:
        continue
    n = len(orb_data)
    ch_counts = Counter(r['channel'] for r in orb_data)
    fracs = [ch_counts.get(ch, 0) / n for ch in channel_names]
    bottom = 0
    for j, (ch, frac) in enumerate(zip(channel_names, fracs)):
        ax.bar(i, frac, bottom=bottom, color=channel_colors[j],
               label=ch if i == 0 else None, alpha=0.7)
        bottom += frac
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['cosmos', 'satellite', 'singularity'])
ax.set_ylabel('Fraction')
ax.set_title('Chromogeometric Channel by Orbit', fontsize=11)
ax.legend(fontsize=7)

# Panel 5: Q_M mod-9 histogram
ax = axes[1, 1]
qm_mod9 = [r['Q_M'] % 9 for r in all_data]
ax.hist(qm_mod9, bins=range(10), alpha=0.7, color='#FF9800',
        edgecolor='k', linewidth=0.5, align='left')
ax.set_xlabel('Q_M mod 9')
ax.set_ylabel('Count')
ax.set_title('Q_M mod-9 Residue Distribution', fontsize=11)
ax.set_xticks(range(9))

# Panel 6: Per-mineral ordering check
ax = axes[1, 2]
mineral_names = []
cosmos_means = []
sat_means = []
for crystal in CRYSTALS:
    mineral_data = [r for r in all_data if r['mineral'] == crystal['name']]
    c_d = [r['d_spacing'] for r in mineral_data if r['orbit_9'] == 'cosmos']
    s_d = [r['d_spacing'] for r in mineral_data if r['orbit_9'] == 'satellite']
    if c_d and s_d:
        mineral_names.append(crystal['name'][:6])
        cosmos_means.append(np.mean(c_d))
        sat_means.append(np.mean(s_d))

x = np.arange(len(mineral_names))
ax.bar(x - 0.2, cosmos_means, 0.4, label='cosmos', color=colors_orbit['cosmos'], alpha=0.7)
ax.bar(x + 0.2, sat_means, 0.4, label='satellite', color=colors_orbit['satellite'], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(mineral_names, rotation=45, fontsize=7)
ax.set_ylabel('Mean d-spacing (Å)')
ax.set_title('Cosmos vs Satellite d-spacing by Mineral', fontsize=11)
ax.legend(fontsize=8)

plt.suptitle(f'QA Voxelation × Crystallography Batch\n{len(CRYSTALS)} minerals, {total} reflections',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('qa_voxelation_crystal_batch.png', dpi=150, bbox_inches='tight')
print("\nSaved: qa_voxelation_crystal_batch.png")

# ─── Summary statistics ──────────────────────────────────────────

print(f"\n{'=' * 70}")
print("BATCH CRYSTALLOGRAPHY SUMMARY")
print(f"{'=' * 70}")

# Count minerals where cosmos_mean_d > satellite_mean_d
n_ordered = 0
n_testable = 0
for crystal in CRYSTALS:
    mineral_data = [r for r in all_data if r['mineral'] == crystal['name']]
    c_d = [r['d_spacing'] for r in mineral_data if r['orbit_9'] == 'cosmos']
    s_d = [r['d_spacing'] for r in mineral_data if r['orbit_9'] == 'satellite']
    if c_d and s_d:
        n_testable += 1
        if np.mean(c_d) > np.mean(s_d):
            n_ordered += 1

print(f"""
RESULTS ({total} reflections, {len(CRYSTALS)} minerals, 4 crystal systems):

1. ORBIT DISTRIBUTION:
   cosmos:      {orbit_counts.get('cosmos', 0):5d} ({orbit_counts.get('cosmos', 0)/total:.3f})
   satellite:   {orbit_counts.get('satellite', 0):5d} ({orbit_counts.get('satellite', 0)/total:.3f})
   singularity: {orbit_counts.get('singularity', 0):5d} ({orbit_counts.get('singularity', 0)/total:.3f})

2. D-SPACING ORDERING (cosmos > satellite):
   {n_ordered}/{n_testable} minerals show cosmos_mean_d > satellite_mean_d
   (Satellite reflections have higher Q_M → shorter d-spacing)

3. STRUCTURAL EXPLANATION:
   Satellite requires |h|,|k| both ≡ 0 mod 3 → h,k ∈ {{0,3,6,...}}
   This forces Q_M = h²+k²+l² ≥ 9 (minimum: h=3,k=0,l=0)
   Higher Q_M → smaller d → satellite = high-angle diffraction

4. This is NOT a statistical finding — it's a GEOMETRIC CONSEQUENCE
   of the satellite divisibility condition on Miller indices.
""")
print("Done.")
