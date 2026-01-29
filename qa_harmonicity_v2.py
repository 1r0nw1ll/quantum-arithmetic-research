#!/usr/bin/env python3
"""
Harmonicity Index 2.0 for Quantum Arithmetic Systems

Three-component geometric metric based on hierarchical Pythagorean classification:
1. Angular Harmonicity: Pisano period alignment (mod-24 × mod-9)
2. Radial Harmonicity: Primitivity measure (1/gcd)
3. Family Harmonicity: Classical subfamily membership (Fermat/Pythagoras/Plato)

Reference: Enhanced Pythagorean Five Families Paper (2025)
Author: QA Research Team
Date: December 10, 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional
import math

# ============================================================================
# CORE QA TUPLE OPERATIONS
# ============================================================================

def qa_tuple(b: int, e: int, modulus: int = 24) -> Tuple[int, int, int, int]:
    """
    Generate QA tuple from (b, e) pair.

    Args:
        b: First parameter
        e: Second parameter
        modulus: Modular arithmetic base (default 24)

    Returns:
        (b, e, d, a) where d = (b+e) mod N, a = (b+2e) mod N
    """
    d = (b + e) % modulus
    a = (b + 2*e) % modulus
    return (b, e, d, a)


def pythagorean_triple(q: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """
    Generate Pythagorean triple (C, F, G) from QA tuple (b, e, d, a).

    Formula:
        C = 2de
        F = ab
        G = e² + d²

    Args:
        q: QA tuple (b, e, d, a)

    Returns:
        (C, F, G) Pythagorean triple
    """
    b, e, d, a = q
    C = 2 * d * e
    F = a * b
    G = e**2 + d**2
    return (C, F, G)


def digital_root(n: int) -> int:
    """
    Compute digital root (mod-9 with special handling of 0 → 9).

    Args:
        n: Positive integer

    Returns:
        Digital root in {1, 2, ..., 9}
    """
    if n == 0:
        return 9
    dr = n % 9
    return 9 if dr == 0 else dr


# ============================================================================
# COMPONENT 1: ANGULAR HARMONICITY (Pisano Period Alignment)
# ============================================================================

# Pisano period families (Five Families classification)
PISANO_FAMILIES = {
    'Fibonacci': {'period': 24, 'initial': (1, 1)},
    'Lucas': {'period': 24, 'initial': (2, 1)},
    'Phibonacci': {'period': 24, 'initial': (3, 1)},
    'Tribonacci': {'period': 8, 'initial': (3, 3)},
    'Ninbonacci': {'period': 1, 'initial': (9, 9)},
}

# Pre-computed Pisano cycle membership (mod-24 × mod-9 grid)
# Maps (dr(b), dr(e)) → family name
PISANO_FAMILY_MAP = {
    # Fibonacci family (24 unique digital root pairs)
    (1,1): 'Fibonacci', (1,2): 'Fibonacci', (1,5): 'Fibonacci', (1,8): 'Fibonacci', (1,9): 'Fibonacci',
    (2,2): 'Fibonacci', (2,6): 'Fibonacci', (2,8): 'Fibonacci',
    (4,1): 'Fibonacci', (4,3): 'Fibonacci', (7,1): 'Fibonacci', (7,7): 'Fibonacci',
    (8,1): 'Fibonacci', (8,4): 'Fibonacci', (8,5): 'Fibonacci', (8,8): 'Fibonacci', (8,9): 'Fibonacci',
    (9,2): 'Fibonacci', (9,8): 'Fibonacci',
    (3,5): 'Fibonacci', (5,6): 'Fibonacci', (6,8): 'Fibonacci',
    (3,7): 'Fibonacci', (5,8): 'Fibonacci',

    # Lucas family (24 unique digital root pairs)
    (2,1): 'Lucas', (2,3): 'Lucas', (2,5): 'Lucas', (2,7): 'Lucas', (2,9): 'Lucas',
    (1,3): 'Lucas', (3,4): 'Lucas', (4,6): 'Lucas', (5,3): 'Lucas', (6,1): 'Lucas',
    (7,2): 'Lucas', (7,3): 'Lucas', (7,6): 'Lucas', (7,8): 'Lucas', (7,9): 'Lucas',
    (9,3): 'Lucas', (9,7): 'Lucas',
    (3,8): 'Lucas', (4,7): 'Lucas', (5,2): 'Lucas', (6,7): 'Lucas', (8,3): 'Lucas',
    (8,6): 'Lucas', (8,2): 'Lucas',

    # Phibonacci family (unique digital root pairs - excludes Tribonacci overlaps)
    (3,1): 'Phibonacci', (3,2): 'Phibonacci', (1,4): 'Phibonacci', (1,6): 'Phibonacci',
    (2,4): 'Phibonacci', (4,2): 'Phibonacci', (4,4): 'Phibonacci', (4,5): 'Phibonacci',
    (5,1): 'Phibonacci', (5,4): 'Phibonacci', (5,5): 'Phibonacci', (5,7): 'Phibonacci', (5,9): 'Phibonacci',
    (7,4): 'Phibonacci', (7,5): 'Phibonacci',
    (8,7): 'Phibonacci', (9,1): 'Phibonacci', (9,4): 'Phibonacci', (9,5): 'Phibonacci',

    # Tribonacci family (8 unique digital root pairs - all have both components ≡ 0 mod 3)
    # Note: (9,9) is excluded - it belongs to Ninbonacci (fixed point)
    (3,3): 'Tribonacci', (3,6): 'Tribonacci', (3,9): 'Tribonacci',
    (6,3): 'Tribonacci', (6,6): 'Tribonacci', (6,9): 'Tribonacci',
    (9,3): 'Tribonacci', (9,6): 'Tribonacci',

    # Ninbonacci family (1 pair - the fixed point at digital root (9,9))
    (9,9): 'Ninbonacci',
}

def compute_angular_harmonicity(q: Tuple[int, int, int, int],
                                 modulus: int = 24) -> float:
    """
    Compute angular harmonicity component.

    Based on alignment with Pisano period structure (mod-24 × mod-9).

    Args:
        q: QA tuple (b, e, d, a)
        modulus: QA modulus (default 24)

    Returns:
        H_angular in [0, 1]
    """
    b, e, d, a = q

    # Get digital roots for mod-9 structure
    dr_b = digital_root(b)
    dr_e = digital_root(e)

    # Lookup family membership
    key = (dr_b, dr_e)
    family = PISANO_FAMILY_MAP.get(key, 'Unknown')

    # Mod-24 harmonic alignment (measures position within Pisano cycle)
    # Higher values for tuples near cycle start/end (strong periodicity)
    mod24_position = (b % modulus) / modulus
    mod24_harmonic = 1.0 - abs(0.5 - mod24_position)  # Peak at 0 and 24

    # Mod-9 harmonic alignment (measures digital root resonance)
    # Fibonacci/Lucas/Phibonacci have period 24 → high harmonicity
    # Tribonacci has period 8 → medium harmonicity
    # Ninbonacci has period 1 → low harmonicity (fixed point)
    if family in ['Fibonacci', 'Lucas', 'Phibonacci']:
        mod9_harmonic = 1.0  # 24-cycle families
    elif family == 'Tribonacci':
        mod9_harmonic = 0.6  # 8-cycle family
    elif family == 'Ninbonacci':
        mod9_harmonic = 0.3  # Fixed point (no dynamics)
    else:
        mod9_harmonic = 0.5  # Unknown (conservative estimate)

    # Combine mod-24 and mod-9 components (geometric mean)
    H_angular = math.sqrt(mod24_harmonic * mod9_harmonic)

    return H_angular


# ============================================================================
# COMPONENT 2: RADIAL HARMONICITY (Primitivity Measure)
# ============================================================================

def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)


def gcd_three(a: int, b: int, c: int) -> int:
    """Compute gcd of three numbers."""
    return gcd(gcd(a, b), c)


def compute_radial_harmonicity(q: Tuple[int, int, int, int]) -> float:
    """
    Compute radial harmonicity component.

    Based on primitivity measure H_radial = 1 / gcd(C, F, G).

    Interpretation:
    - Primitive tuples (gcd=1): H_radial = 1.0 → E8 root shell
    - Female tuples (gcd=2): H_radial = 0.5 → E8 first weight shell (√2× distance)
    - Composite tuples (gcd>2): H_radial < 0.5 → Higher E8 shells

    Args:
        q: QA tuple (b, e, d, a)

    Returns:
        H_radial in (0, 1]
    """
    C, F, G = pythagorean_triple(q)

    # Compute gcd of Pythagorean triple
    triple_gcd = gcd_three(C, F, G)

    # Radial harmonicity = 1 / gcd
    # Clamp to avoid division by zero (though gcd >= 1 always)
    H_radial = 1.0 / max(triple_gcd, 1)

    return H_radial


# ============================================================================
# COMPONENT 3: FAMILY HARMONICITY (Classical Subfamily Membership)
# ============================================================================

def is_fermat_family(C: int, F: int, G: int) -> bool:
    """
    Check if Pythagorean triple belongs to Fermat family.

    Criterion: |C - F| = 1 (consecutive legs)

    Examples: (3,4,5), (5,12,13), (7,24,25)
    """
    return abs(C - F) == 1


def is_pythagoras_family(q: Tuple[int, int, int, int]) -> bool:
    """
    Check if QA tuple belongs to Pythagoras family.

    Criterion: (d - e)² = 1 (1-step-off-diagonal in BEDA lattice)

    Examples from (b,e): (1,2) → (3,4,5), (3,1) → (8,15,17)
    """
    b, e, d, a = q
    return (d - e)**2 == 1


def is_plato_family(C: int, F: int, G: int) -> bool:
    """
    Check if Pythagorean triple belongs to Plato family.

    Criterion: |G - F| = 2 (hypotenuse 2 more than one leg)

    Examples: (15,112,113), (35,612,613)
    """
    return abs(G - F) == 2


def compute_family_harmonicity(q: Tuple[int, int, int, int]) -> float:
    """
    Compute family harmonicity component.

    Based on membership in classical Pythagorean subfamilies:
    - Fermat: |C - F| = 1
    - Pythagoras: (d - e)² = 1
    - Plato: |G - F| = 2

    H_family = (f_Fermat + f_Pythagoras + f_Plato) / 3

    Args:
        q: QA tuple (b, e, d, a)

    Returns:
        H_family in [0, 1]
    """
    C, F, G = pythagorean_triple(q)

    # Check membership in each subfamily
    f_fermat = 1.0 if is_fermat_family(C, F, G) else 0.0
    f_pythagoras = 1.0 if is_pythagoras_family(q) else 0.0
    f_plato = 1.0 if is_plato_family(C, F, G) else 0.0

    # Average membership across three families
    H_family = (f_fermat + f_pythagoras + f_plato) / 3.0

    return H_family


# ============================================================================
# HARMONICITY INDEX 2.0 (Full Three-Component Metric)
# ============================================================================

def compute_hi_2_0(q: Tuple[int, int, int, int],
                   w_ang: float = 0.4,
                   w_rad: float = 0.3,
                   w_fam: float = 0.3,
                   modulus: int = 24) -> Dict[str, float]:
    """
    Compute Harmonicity Index 2.0 with all three components.

    HI_2.0 = w_ang × H_angular + w_rad × H_radial + w_fam × H_family

    Args:
        q: QA tuple (b, e, d, a)
        w_ang: Weight for angular harmonicity (default 0.4)
        w_rad: Weight for radial harmonicity (default 0.3)
        w_fam: Weight for family harmonicity (default 0.3)
        modulus: QA modulus (default 24)

    Returns:
        Dictionary with:
            - 'HI_2.0': Overall harmonicity index [0, 1]
            - 'H_angular': Angular component [0, 1]
            - 'H_radial': Radial component (0, 1]
            - 'H_family': Family component [0, 1]
            - 'weights': (w_ang, w_rad, w_fam)
            - 'pythagorean_triple': (C, F, G)
            - 'gcd': gcd(C, F, G)
            - 'families': List of subfamily memberships
    """
    # Compute individual components
    H_angular = compute_angular_harmonicity(q, modulus)
    H_radial = compute_radial_harmonicity(q)
    H_family = compute_family_harmonicity(q)

    # Weighted combination
    HI_2_0 = w_ang * H_angular + w_rad * H_radial + w_fam * H_family

    # Get Pythagorean triple and gcd for diagnostics
    C, F, G = pythagorean_triple(q)
    triple_gcd = gcd_three(C, F, G)

    # Identify family memberships
    families = []
    if is_fermat_family(C, F, G):
        families.append('Fermat')
    if is_pythagoras_family(q):
        families.append('Pythagoras')
    if is_plato_family(C, F, G):
        families.append('Plato')

    return {
        'HI_2.0': HI_2_0,
        'H_angular': H_angular,
        'H_radial': H_radial,
        'H_family': H_family,
        'weights': (w_ang, w_rad, w_fam),
        'pythagorean_triple': (C, F, G),
        'gcd': triple_gcd,
        'families': families if families else ['None'],
    }


# ============================================================================
# BACKWARD COMPATIBILITY: HI 1.0 (E8-Only)
# ============================================================================

def compute_hi_1_0(q: Tuple[int, int, int, int],
                   modulus: int = 24) -> float:
    """
    Compute Harmonicity Index 1.0 (E8-only baseline).

    HI_1.0 ≈ H_angular (E8 alignment component)

    This is equivalent to HI_2.0 with weights (w_ang=1.0, w_rad=0, w_fam=0).

    Args:
        q: QA tuple (b, e, d, a)
        modulus: QA modulus (default 24)

    Returns:
        HI_1.0 in [0, 1]
    """
    return compute_angular_harmonicity(q, modulus)


# ============================================================================
# GENDER CLASSIFICATION (Primitive/Female/Composite)
# ============================================================================

def classify_gender(q: Tuple[int, int, int, int]) -> str:
    """
    Classify QA tuple into gender categories.

    Categories:
    - 'Male (Primitive)': gcd(C,F,G) = 1
    - 'Female': gcd(C,F,G) = 2 (octave harmonics)
    - 'Male (Composite)': gcd(C,F,G) > 2, not a power of 2

    Args:
        q: QA tuple (b, e, d, a)

    Returns:
        Gender classification string
    """
    C, F, G = pythagorean_triple(q)
    triple_gcd = gcd_three(C, F, G)

    if triple_gcd == 1:
        return 'Male (Primitive)'
    elif triple_gcd == 2:
        return 'Female'
    else:
        # Check if gcd is a power of 2
        if triple_gcd > 0 and (triple_gcd & (triple_gcd - 1)) == 0:
            return f'Female (2^{int(math.log2(triple_gcd))})'
        else:
            return 'Male (Composite)'


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("HARMONICITY INDEX 2.0 - Demonstration")
    print("="*80)
    print()

    # Test cases: Famous Pythagorean triples
    test_cases = [
        ((1, 1), "Fibonacci (3,4,5) - Classic primitive"),
        ((2, 1), "Lucas (6,8,10) - Female (gcd=2)"),
        ((3, 1), "Phibonacci (8,15,17) - Pythagoras family"),
        ((3, 3), "Tribonacci (36,27,45) - 8-cycle"),
        ((9, 9), "Ninbonacci (324,243,405) - Fixed point"),
        ((2, 3), "Fibonacci (30,16,34) - Female"),
        ((1, 2), "Fibonacci (12,5,13) - Pythagoras family"),
    ]

    print(f"{'(b,e)':<8} {'Triple':<18} {'HI 2.0':<8} {'H_ang':<8} {'H_rad':<8} {'H_fam':<8} {'gcd':<5} {'Gender':<20} {'Families'}")
    print("-" * 130)

    for (b, e), description in test_cases:
        q = qa_tuple(b, e, modulus=24)
        result = compute_hi_2_0(q)
        gender = classify_gender(q)

        C, F, G = result['pythagorean_triple']
        triple_str = f"({C},{F},{G})"
        families_str = ', '.join(result['families'])

        print(f"({b},{e})   {triple_str:<18} {result['HI_2.0']:.4f}   "
              f"{result['H_angular']:.4f}   {result['H_radial']:.4f}   "
              f"{result['H_family']:.4f}   {result['gcd']:<5} "
              f"{gender:<20} {families_str}")

    print()
    print("="*80)
    print("WEIGHT ABLATION STUDY")
    print("="*80)
    print()

    # Test HI 2.0 with different weight configurations
    q_test = qa_tuple(1, 1, modulus=24)  # Classic (3,4,5)

    weight_configs = [
        ('HI 1.0 (E8-only)', 1.0, 0.0, 0.0),
        ('Balanced (default)', 0.4, 0.3, 0.3),
        ('High Angular', 0.6, 0.2, 0.2),
        ('High Radial', 0.2, 0.6, 0.2),
        ('High Family', 0.2, 0.2, 0.6),
    ]

    print(f"Configuration for QA tuple (1,1) → Pythagorean triple (3,4,5):")
    print()
    print(f"{'Configuration':<20} {'w_ang':<8} {'w_rad':<8} {'w_fam':<8} {'HI 2.0':<8}")
    print("-" * 60)

    for name, w_ang, w_rad, w_fam in weight_configs:
        result = compute_hi_2_0(q_test, w_ang=w_ang, w_rad=w_rad, w_fam=w_fam)
        print(f"{name:<20} {w_ang:.2f}     {w_rad:.2f}     {w_fam:.2f}     {result['HI_2.0']:.4f}")

    print()
    print("Demonstration complete!")
    print()
