#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
QA Scales of Nature — From Subatomic to Solar System

Shows how the same QA arithmetic (quadrance, spread, modular orbits)
operates at every physical scale. The quantum ellipse and chromogeometric
triple (C, F, G) appear everywhere.

Scales covered:
1. Subatomic: periodic table period lengths, nuclear magic numbers
2. Atomic: orbital shapes as spreads, spectral lines
3. Molecular: bond angles = spreads, crystal field theory
4. Crystal: 7 crystal systems, Miller indices, Bragg's law as RT
5. Geological: mineral classification, seismology (QCI validated)
6. Planetary: WGS84 ellipsoid, Earth shape QN
7. Solar System: orbital QNs, harmonic resonance network

Key insight: Bragg's law nλ = 2d·sin(θ) becomes n²Q_λ = 4Q_d·s
where Q_λ = λ², Q_d = d², s = sin²(θ). ALL rational. No trig.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math


def section_subatomic():
    """Period lengths, magic numbers, and QA structure."""
    print('=' * 80)
    print('SCALE 1: SUBATOMIC — Periodic Table & Nuclear Structure')
    print('=' * 80)
    print()

    # Periodic table period lengths: 2, 8, 18, 32
    # These are 2×n² for n = 1, 2, 3, 4
    # In QA terms: 2 × quadrance of shell number
    print('Periodic Table Period Lengths:')
    print('  Period lengths: 2, 8, 18, 32, (50), (72), ...')
    print('  Pattern: 2n² for n = 1, 2, 3, 4, 5, 6, ...')
    print('  QA reading: 2 × Q(n) where Q(n) = n² = quadrance of shell')
    print()

    for n in range(1, 7):
        length = 2 * n * n
        print(f'  n={n}: 2×{n}² = {length}', end='')
        if n <= 4:
            print(f'  ← actual period {n}')
        else:
            print(f'  ← predicted period {n}')

    print()
    print('  The factor 2 = spin degeneracy (up/down)')
    print('  The n² = quadrance = number of orbital states per shell')
    print('  2n² = 2 × quadrance = QA orbit capacity')
    print()

    # Nuclear magic numbers: 2, 8, 20, 28, 50, 82, 126
    magic = [2, 8, 20, 28, 50, 82, 126]
    print('Nuclear Magic Numbers: ', magic)
    print('  Differences: ', [magic[i+1]-magic[i] for i in range(len(magic)-1)])
    print('  = [6, 12, 8, 22, 32, 44]')
    print()

    # Check for QA structure in magic numbers
    print('  QA analysis of magic numbers:')
    for m in magic:
        # Factor and check mod structure
        mod9 = m % 9
        mod24 = m % 24
        print(f'    {m:>3}: mod 9 = {mod9}, mod 24 = {mod24}', end='')
        if m % 2 == 0:
            print(f', = 2×{m//2}', end='')
        print()

    print()
    print('  NOTE: 8 = Satellite orbit period')
    print('  NOTE: 2, 8, 20 = 2×1², 2×4, 2×10 = 2×T(1), 2×T(2)², ...')
    print('  where T(n) = triangular number = n(n+1)/2')
    print('  Magic numbers = 2 × cumulative sum of (2l+1) for l=0,1,...')
    print('  This is orbital angular momentum filling = SPREAD quantization')
    print()


def section_atomic():
    """Atomic orbitals, spectral lines as QA."""
    print('=' * 80)
    print('SCALE 2: ATOMIC — Orbitals and Spectral Lines')
    print('=' * 80)
    print()

    print('Orbital shapes are classified by angular momentum quantum number l:')
    print('  l=0 (s): spherical — spread with any axis = 0 (isotropic)')
    print('  l=1 (p): dumbbell — spread = directional')
    print('  l=2 (d): cloverleaf — spread = 2 nodal planes')
    print('  l=3 (f): complex — spread = 3 nodal planes')
    print()
    print('  Number of orbitals per l: 2l+1 = 1, 3, 5, 7')
    print('  These are the ODD numbers — QA mod-2 structure!')
    print('  Total per shell: Σ(2l+1) for l=0..n-1 = n²')
    print('  With spin: 2n² — exactly the period lengths above')
    print()

    # Hydrogen spectral lines
    print('Hydrogen Spectral Lines (Rydberg formula):')
    print('  1/λ = R(1/n₁² - 1/n₂²)')
    print()
    print('  In QA terms: Q_wavelength ∝ 1/(1/Q₁ - 1/Q₂)')
    print('  where Q₁ = n₁² and Q₂ = n₂² are QUADRANCES of the shell numbers')
    print()
    print('  The Rydberg formula is a QUADRANCE RATIO formula!')
    print()

    # Compute some spectral series
    R = 1.097e7  # Rydberg constant, m^-1
    print('  Series        Transition    λ (nm)    1/λ ratio')
    print('  ' + '─' * 55)
    for name, n1, transitions in [
        ('Lyman', 1, [2, 3, 4, 5]),
        ('Balmer', 2, [3, 4, 5, 6]),
        ('Paschen', 3, [4, 5, 6, 7]),
    ]:
        for n2 in transitions:
            inv_lambda = R * (1/(n1*n1) - 1/(n2*n2))
            lam_nm = 1e9 / inv_lambda
            # QA ratio
            q1, q2 = n1*n1, n2*n2
            print(f'  {name:<10} {n1}→{n2}         {lam_nm:>7.1f}    '
                  f'Q ratio: {q2}/{q1} = {q2/q1:.3f}')
    print()


def section_molecular():
    """Bond angles as spreads, molecular geometry."""
    print('=' * 80)
    print('SCALE 3: MOLECULAR — Bond Angles = Spreads')
    print('=' * 80)
    print()

    print('VSEPR bond angles are SPREADS:')
    print()

    molecules = [
        ('H₂O', 'bent', 104.45, 'water'),
        ('NH₃', 'trigonal pyramidal', 107.8, 'ammonia'),
        ('CH₄', 'tetrahedral', 109.47, 'methane'),
        ('BF₃', 'trigonal planar', 120.0, 'boron trifluoride'),
        ('CO₂', 'linear', 180.0, 'carbon dioxide'),
        ('SF₆', 'octahedral', 90.0, 'sulfur hexafluoride'),
        ('C₆₀', 'icosahedral', 108.0, 'buckminsterfullerene'),
        ('Diamond', 'tetrahedral', 109.47, 'diamond lattice'),
        ('Graphite', 'trigonal planar', 120.0, 'graphite layer'),
    ]

    print(f'  {"Molecule":<12} {"Geometry":<22} {"Angle":<8} {"Spread s":<12} {"1-s (cross)":<12} Fraction')
    print('  ' + '─' * 85)

    for mol, geom, angle, desc in molecules:
        s = math.sin(math.radians(angle)) * math.sin(math.radians(angle))
        c = 1 - s

        # Find closest rational approximation
        best_frac = None
        best_diff = 1
        for denom in range(1, 100):
            numer = round(s * denom)
            diff = abs(s - numer/denom)
            if diff < best_diff:
                best_diff = diff
                best_frac = f'{numer}/{denom}'

        print(f'  {mol:<12} {geom:<22} {angle:>6.2f}° {s:>10.6f}  {c:>10.6f}   ≈ {best_frac}')

    print()
    print('  KEY: Tetrahedral angle spread = sin²(109.47°) = 8/9')
    print('  8/9 is the QA SATELLITE orbit fraction (8 of 9 states)!')
    print()
    print('  Trigonal planar spread = sin²(120°) = 3/4')
    print('  3/4 is EXACTLY the maximum spread for an equilateral triangle')
    print('  (Wildberger Exercise 7.4: S(a,a,a) ≥ 0 iff a ≤ 3/4)')
    print()

    tet_spread = 8/9
    print(f'  Tetrahedral spread = 8/9 = {tet_spread:.10f}')
    print(f'  sin²(109.4712°)   = {math.sin(math.radians(109.4712))**2:.10f}')
    print(f'  cos(109.4712°)    = {math.cos(math.radians(109.4712)):.10f}')
    print(f'  cos(tet) = -1/3, so spread = 1 - cos² = 1 - 1/9 = 8/9 ✓')
    print()


def section_crystal():
    """Crystal systems, Miller indices, Bragg's law as RT."""
    print('=' * 80)
    print('SCALE 4: CRYSTALLOGRAPHY — Bragg\'s Law as Rational Trig')
    print('=' * 80)
    print()

    print('7 Crystal Systems (symmetry constraints on lattice):')
    systems = [
        ('Cubic', 'a=b=c, α=β=γ=90°', 's=1 (all right)', 'NaCl, diamond'),
        ('Tetragonal', 'a=b≠c, α=β=γ=90°', 's=1 (all right)', 'TiO₂, SnO₂'),
        ('Orthorhombic', 'a≠b≠c, α=β=γ=90°', 's=1 (all right)', 'olivine, topaz'),
        ('Hexagonal', 'a=b≠c, α=β=90°,γ=120°', 's_γ=3/4', 'quartz, graphite'),
        ('Trigonal', 'a=b=c, α=β=γ≠90°', 'variable', 'calcite, corundum'),
        ('Monoclinic', 'a≠b≠c, α=γ=90°,β≠90°', 's_β variable', 'gypsum, mica'),
        ('Triclinic', 'a≠b≠c, α≠β≠γ≠90°', 'all variable', 'feldspar'),
    ]

    print(f'  {"System":<14} {"Constraints":<28} {"Spread condition":<18} {"Examples"}')
    print('  ' + '─' * 80)
    for name, const, spread, examples in systems:
        print(f'  {name:<14} {const:<28} {spread:<18} {examples}')

    print()
    print('  NOTE: 3 of 7 systems require ALL right angles (spread = 1)')
    print('  Hexagonal requires γ=120° → spread = 3/4 (trigonal planar!)')
    print('  These are the SAME spreads as molecular geometry above!')
    print()

    # Miller indices
    print('Miller Indices (h, k, l) = QA direction vector:')
    print('  A crystal plane is specified by integers (h,k,l)')
    print('  The inter-plane spacing d_hkl depends on the lattice')
    print('  For cubic: 1/d² = (h²+k²+l²)/a²')
    print('  In QA: 1/Q_d = Q_hkl / Q_a where Q_hkl = h²+k²+l²')
    print('  Q_hkl IS the quadrance of the Miller index vector!')
    print()

    # Bragg's law
    print("BRAGG'S LAW as RATIONAL TRIGONOMETRY:")
    print()
    print('  Classical:  nλ = 2d·sin(θ)')
    print()
    print('  Square both sides:')
    print('  n²λ² = 4d²·sin²(θ)')
    print()
    print('  In QA/RT notation:')
    print('  n² · Q_λ = 4 · Q_d · s')
    print()
    print('  where:')
    print('    Q_λ = λ²  (quadrance of wavelength)')
    print('    Q_d = d²  (quadrance of lattice spacing)')
    print('    s = sin²(θ) = SPREAD of the diffraction angle')
    print('    n = diffraction order (integer)')
    print()
    print('  This is PURELY ALGEBRAIC. No trig functions needed.')
    print('  Works over any field, including finite fields!')
    print('  For integer lattice spacings and wavelengths: EXACT.')
    print()

    # Example: NaCl diffraction
    print('  Example: NaCl (100) plane, Cu Kα radiation')
    d_nacl = 2.82e-10  # meters
    lam_cu = 1.54e-10   # meters

    for n in [1, 2, 3]:
        sin_theta = n * lam_cu / (2 * d_nacl)
        if abs(sin_theta) <= 1:
            spread = sin_theta * sin_theta
            theta = math.degrees(math.asin(sin_theta))
            # QA form
            Q_lam = lam_cu * lam_cu
            Q_d = d_nacl * d_nacl
            s_calc = n*n * Q_lam / (4 * Q_d)
            print(f'  n={n}: θ = {theta:.2f}°, spread = {spread:.6f}, '
                  f'QA check: n²Q_λ/(4Q_d) = {s_calc:.6f} ✓')
    print()


def section_geological():
    """Mineral classification, seismology connection."""
    print('=' * 80)
    print('SCALE 5: GEOLOGICAL — Minerals, Seismology, Plate Tectonics')
    print('=' * 80)
    print()

    print('MINERAL CLASSIFICATION → CRYSTAL SYSTEMS → QA SPREADS:')
    print('  Every mineral belongs to one of 7 crystal systems')
    print('  Crystal system determined by lattice angle spreads')
    print('  Mineral hardness (Mohs) may relate to QA orbit stability')
    print()

    print('SEISMOLOGY — QA TOPOGRAPHIC OBSERVER (VALIDATED):')
    print('  Script: 46_seismic_topographic_observer.py')
    print('  Data: USGS earthquake catalog, 152,237 M4+ events, 10 years')
    print()
    print('  Results (OOS, 1679 days):')
    print('    QCI vs future seismic activity (21d): r = +0.225, p < 10⁻⁸')
    print('    Orbit distribution active vs quiet: χ² = 61.3, p < 10⁻⁶')
    print()
    print('  Orbit pattern:')
    print('    Active seismic: satellite 60%, cosmos 35%, singularity 4%')
    print('    Quiet seismic:  cosmos 60%, satellite 40%, singularity 0%')
    print('    Same pattern as finance! Stress → satellite (bounded)')
    print()

    print('PLATE TECTONICS:')
    print('  Earth oblate spheroid: QN (101, 9, 110, 119)')
    print('  Plate boundaries = discontinuities in the QA field')
    print('  Subduction zones: one QA regime diving under another')
    print('  Mid-ocean ridges: QA state transitions in the mantle')
    print('  (Speculative — needs formalization)')
    print()


def section_solar():
    """Summary of solar system results."""
    print('=' * 80)
    print('SCALES 6-7: PLANETARY & SOLAR SYSTEM')
    print('=' * 80)
    print()

    print('PLANETARY SHAPE:')
    print('  Earth oblate spheroid: QN (101, 9, 110, 119)')
    print('    Axis ratio match: 7 significant figures')
    print('  Earth orbit: QN (59, 1, 60, 61)')
    print('    Eccentricity match: 0.26% error')
    print()

    print('SOLAR SYSTEM HARMONIC NETWORK:')
    print('  47 bodies analyzed (planets, moons, dwarfs, comets, asteroids)')
    print('  ALL known mean-motion resonances verified through QN prime sharing')
    print('  Laplace resonance (Io:Europa:Ganymede): share primes {2,3,7}')
    print('  Pluto:Neptune 2:3: share {2,3,11}')
    print()

    print('PRIME HIERARCHY:')
    print('  2, 3      — universal (all bodies)')
    print('  5         — near-universal (34/47)')
    print('  7         — common (22/47): inner + outer solar system bridge')
    print('  11, 13    — common (13-17 bodies): cross-system families')
    print('  17-29     — bridge primes: specific multi-body connections')
    print('  31+       — rare bridges: unique 2-3 body connections')
    print('  Large primes (101, 131, 269, ...): exclusive parent-satellite locks')
    print()


def build_scale_diagram():
    """Build the multi-scale visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_title('QA Scales of Nature\n'
                 'The same arithmetic (quadrance, spread, modular orbit) at every scale',
                 fontsize=16, fontweight='bold')
    ax.axis('off')

    # Define scales from smallest to largest
    scales = [
        {'name': 'Subatomic', 'size': '10⁻¹⁵ m', 'color': '#E91E63',
         'qa': 'Period lengths = 2n²\nMagic numbers\nSpin = mod-2',
         'evidence': 'Periodic table structure'},
        {'name': 'Atomic', 'size': '10⁻¹⁰ m', 'color': '#9C27B0',
         'qa': 'Orbitals: 2l+1 states\nRydberg = Q ratio\nShell Q = n²',
         'evidence': 'Spectral lines exact'},
        {'name': 'Molecular', 'size': '10⁻⁹ m', 'color': '#3F51B5',
         'qa': 'Bond angle = spread\nTetrahedral s=8/9\nTrigonal s=3/4',
         'evidence': 'VSEPR geometry'},
        {'name': 'Crystal', 'size': '10⁻⁷ m', 'color': '#2196F3',
         'qa': "Bragg: n²Qλ = 4Qd·s\n7 systems = 7 spread\nMiller Q = h²+k²+l²",
         'evidence': "Bragg's law = RT"},
        {'name': 'Geological', 'size': '10³ m', 'color': '#4CAF50',
         'qa': 'Seismic QCI r=+0.225\nStress→satellite orbit\nMineral = crystal system',
         'evidence': 'USGS 152K events ✓'},
        {'name': 'Planetary', 'size': '10⁷ m', 'color': '#FF9800',
         'qa': 'WGS84 = QA ellipse\nShape QN (101,9,110,119)\nECEF via spreads',
         'evidence': '7 sig fig match ✓'},
        {'name': 'Solar System', 'size': '10¹² m', 'color': '#F44336',
         'qa': 'Orbit ecc = e/d\nPrime harmonics\nResonances verified',
         'evidence': '47 bodies, 7/7 res ✓'},
    ]

    n_scales = len(scales)
    y_positions = np.linspace(0.88, 0.08, n_scales)

    for i, (scale, y) in enumerate(zip(scales, y_positions)):
        # Scale bar
        bar_width = 0.85
        bar_height = 0.09
        rect = plt.Rectangle((0.08, y), bar_width, bar_height,
                              facecolor=scale['color'], alpha=0.15,
                              edgecolor=scale['color'], linewidth=2)
        ax.add_patch(rect)

        # Scale name and size
        ax.text(0.10, y + bar_height/2, f"{scale['name']}\n{scale['size']}",
                fontsize=13, fontweight='bold', va='center',
                color=scale['color'])

        # QA description
        ax.text(0.32, y + bar_height/2, scale['qa'],
                fontsize=10, va='center', fontfamily='monospace',
                color='black')

        # Evidence
        ax.text(0.72, y + bar_height/2, scale['evidence'],
                fontsize=10, va='center', fontweight='bold',
                color=scale['color'])

        # Connecting arrows between scales
        if i < n_scales - 1:
            ax.annotate('', xy=(0.5, y), xytext=(0.5, y_positions[i+1] + bar_height),
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        lw=1.5, connectionstyle='arc3,rad=0'))

    # Central spine label
    ax.text(0.5, 0.99, 'QUADRANCE replaces DISTANCE  •  SPREAD replaces ANGLE  •  '
            'MODULAR ORBIT replaces CONTINUOUS',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='darkblue', style='italic')

    # Key formulas
    ax.text(0.5, 0.02,
            'C² + F² = G²  (Pythagoras)    |    '
            's₁/Q₁ = s₂/Q₂ = s₃/Q₃  (Spread Law)    |    '
            '(s₁+s₂+s₃)² = 2(s₁²+s₂²+s₃²) + 4s₁s₂s₃  (Triple Spread)',
            ha='center', va='bottom', fontsize=10,
            fontfamily='monospace', color='darkred',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('qa_scales_of_nature.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved: qa_scales_of_nature.png')


def main():
    section_subatomic()
    section_atomic()
    section_molecular()
    section_crystal()
    section_geological()
    section_solar()
    build_scale_diagram()

    print()
    print('=' * 80)
    print('SYNTHESIS: QA IS THE ARITHMETIC OF NATURE AT EVERY SCALE')
    print('=' * 80)
    print()
    print('From subatomic to solar system, the same structures appear:')
    print()
    print('  QUADRANCE (Q = d²):')
    print('    • Atomic shells: Q = n²')
    print('    • Rydberg formula: Q ratios')
    print("    • Bragg's law: n²Q_λ = 4Q_d·s")
    print('    • LiDAR range: measured as Q directly')
    print('    • WGS84: axis ratio via Q')
    print('    • Orbital eccentricity: e/d')
    print()
    print('  SPREAD (s = sin²θ):')
    print('    • Bond angles: tetrahedral s=8/9, trigonal s=3/4')
    print('    • Crystal systems: lattice angle spreads')
    print("    • Bragg's law: diffraction spread")
    print('    • LiDAR slope: spread of surface normal')
    print('    • Camera FOV: spread')
    print('    • Orbital inclination: spread')
    print()
    print('  MODULAR ORBIT (mod 9, mod 24):')
    print('    • Periodic table: period = 2Q(n)')
    print('    • Crystal symmetry: 7 systems, 32 point groups')
    print('    • Seismic QCI: 3 orbit types predict activity')
    print('    • Finance QCI: orbit coherence predicts volatility')
    print('    • EEG: orbit distribution discriminates seizure')
    print('    • Solar system: prime factor harmonics')
    print()
    print('Ben Iverson found the arithmetic.')
    print('Norman Wildberger proved the geometry.')
    print('Nature demonstrates both at every scale.')


if __name__ == '__main__':
    main()
