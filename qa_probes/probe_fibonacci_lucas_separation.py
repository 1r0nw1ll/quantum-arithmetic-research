#!/usr/bin/env python3
"""
Probe: Fibonacci ↔ Lucas separation within 24-cycle families

Test: Can generator compositions map Fibonacci family states to Lucas family
states (or vice versa), while staying within the 24-cycle regime?

This probe works at the digital root level (mod-9 with 0→9), making it
a finite 81-state problem that can be solved exactly.

Outcome:
- CONNECTED → Fibonacci and Lucas families are reachable from each other
- SEPARATED → structural boundary exists within 24-cycle families
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments')

from collections import defaultdict

# Import family map
from qa_harmonicity_v2 import PISANO_FAMILY_MAP


# === Digital root arithmetic ===

def dr(n: int) -> int:
    """Digital root: mod 9 with 0 → 9"""
    if n == 0:
        return 9
    r = n % 9
    return 9 if r == 0 else r


def dr_add(a: int, b: int) -> int:
    """Add two digital roots and return digital root of sum."""
    return dr(a + b)


# === Generators on digital root pairs ===

def sigma(dr_b: int, dr_e: int) -> tuple:
    """
    Fibonacci step on digital roots: (b, e) → (e, b+e)

    σ: (dr_b, dr_e) → (dr_e, dr(b+e))
    """
    return (dr_e, dr_add(dr_b, dr_e))


def mu(dr_b: int, dr_e: int) -> tuple:
    """
    Swap: (b, e) → (e, b)

    μ: (dr_b, dr_e) → (dr_e, dr_b)
    """
    return (dr_e, dr_b)


def lambda_2(dr_b: int, dr_e: int) -> tuple:
    """
    Scale by 2: (b, e) → (2b, 2e)

    λ₂: (dr_b, dr_e) → (dr(2b), dr(2e))
    """
    return (dr(2 * dr_b), dr(2 * dr_e))


def lambda_3(dr_b: int, dr_e: int) -> tuple:
    """
    Scale by 3: (b, e) → (3b, 3e)

    λ₃: This maps everything to (0,0) mod 3, i.e., Tribonacci regime.
    """
    return (dr(3 * dr_b), dr(3 * dr_e))


# Generator sets to test (σ-only is the core)
GENERATOR_SETS = {
    'sigma_only': {'σ': sigma},
    'sigma_mu': {'σ': sigma, 'μ': mu},
    'sigma_mu_lambda2': {'σ': sigma, 'μ': mu, 'λ₂': lambda_2},
}


# === Family classification ===

def get_family(dr_b: int, dr_e: int) -> str:
    """Get family name for a digital root pair."""
    return PISANO_FAMILY_MAP.get((dr_b, dr_e), 'Unknown')


def get_family_pairs(family: str) -> set:
    """Get all digital root pairs belonging to a family."""
    return {(dr_b, dr_e) for (dr_b, dr_e), fam in PISANO_FAMILY_MAP.items()
            if fam == family}


# === Reachability analysis ===

def compute_reachability(start_pairs: set, generators: dict, max_steps: int = 100) -> set:
    """
    Compute all digital root pairs reachable from start_pairs under generators.

    Uses BFS to find transitive closure.
    """
    reachable = set(start_pairs)
    frontier = list(start_pairs)

    for _ in range(max_steps):
        new_frontier = []
        for dr_b, dr_e in frontier:
            for gen_name, gen_func in generators.items():
                new_pair = gen_func(dr_b, dr_e)
                if new_pair not in reachable:
                    reachable.add(new_pair)
                    new_frontier.append(new_pair)

        if not new_frontier:
            break  # Fixed point reached
        frontier = new_frontier

    return reachable


def analyze_family_separation(family_a: str, family_b: str, generators: dict):
    """
    Analyze whether family_a and family_b are connected under generators.

    Returns:
        dict with analysis results
    """
    pairs_a = get_family_pairs(family_a)
    pairs_b = get_family_pairs(family_b)

    if not pairs_a or not pairs_b:
        return {'error': f'Empty family: {family_a}={len(pairs_a)}, {family_b}={len(pairs_b)}'}

    # Compute reachability from family A
    reachable_from_a = compute_reachability(pairs_a, generators)

    # Check if any of family B is reachable
    b_reachable = pairs_b & reachable_from_a
    b_unreachable = pairs_b - reachable_from_a

    # Compute reachability from family B
    reachable_from_b = compute_reachability(pairs_b, generators)

    # Check if any of family A is reachable from B
    a_reachable_from_b = pairs_a & reachable_from_b
    a_unreachable_from_b = pairs_a - reachable_from_b

    return {
        'family_a': family_a,
        'family_b': family_b,
        'pairs_a': pairs_a,
        'pairs_b': pairs_b,
        'reachable_from_a': reachable_from_a,
        'reachable_from_b': reachable_from_b,
        'b_reachable_from_a': b_reachable,
        'b_unreachable_from_a': b_unreachable,
        'a_reachable_from_b': a_reachable_from_b,
        'a_unreachable_from_b': a_unreachable_from_b,
        'a_to_b_connected': len(b_reachable) > 0,
        'b_to_a_connected': len(a_reachable_from_b) > 0,
        'bidirectional': len(b_reachable) > 0 and len(a_reachable_from_b) > 0,
    }


def probe():
    """Main probe: test Fibonacci ↔ Lucas separation under various generator sets."""

    print("=" * 70)
    print("PROBE: Fibonacci ↔ Lucas Separation (within 24-cycle)")
    print("=" * 70)
    print()

    # Show family sizes
    families_24 = ['Fibonacci', 'Lucas', 'Phibonacci']
    for fam in families_24:
        pairs = get_family_pairs(fam)
        print(f"{fam}: {len(pairs)} digital root pairs")
    print()

    # Test each generator set
    for gen_set_name, generators in GENERATOR_SETS.items():
        print("-" * 70)
        print(f"Generator set: {gen_set_name}")
        print(f"Generators: {list(generators.keys())}")
        print("-" * 70)
        print()

        # Test Fibonacci ↔ Lucas
        result = analyze_family_separation('Fibonacci', 'Lucas', generators)

        print(f"Fibonacci → Lucas:")
        print(f"  Fibonacci pairs: {len(result['pairs_a'])}")
        print(f"  Reachable from Fibonacci: {len(result['reachable_from_a'])} total pairs")
        print(f"  Lucas pairs reachable: {len(result['b_reachable_from_a'])} / {len(result['pairs_b'])}")

        if result['b_reachable_from_a']:
            print(f"  Examples: {list(result['b_reachable_from_a'])[:5]}")
        else:
            print(f"  NONE reachable!")

        print()
        print(f"Lucas → Fibonacci:")
        print(f"  Reachable from Lucas: {len(result['reachable_from_b'])} total pairs")
        print(f"  Fibonacci pairs reachable: {len(result['a_reachable_from_b'])} / {len(result['pairs_a'])}")

        if result['a_reachable_from_b']:
            print(f"  Examples: {list(result['a_reachable_from_b'])[:5]}")
        else:
            print(f"  NONE reachable!")

        print()

        # Summary
        if result['bidirectional']:
            print(f"RESULT: CONNECTED (bidirectional)")
        elif result['a_to_b_connected'] or result['b_to_a_connected']:
            direction = "Fib→Luc" if result['a_to_b_connected'] else "Luc→Fib"
            print(f"RESULT: PARTIALLY CONNECTED ({direction} only)")
        else:
            print(f"RESULT: SEPARATED (no connection)")

        print()

        # Also test Fibonacci ↔ Phibonacci
        result2 = analyze_family_separation('Fibonacci', 'Phibonacci', generators)
        print(f"Fibonacci ↔ Phibonacci: {'CONNECTED' if result2['bidirectional'] else 'SEPARATED'}")

        result3 = analyze_family_separation('Lucas', 'Phibonacci', generators)
        print(f"Lucas ↔ Phibonacci: {'CONNECTED' if result3['bidirectional'] else 'SEPARATED'}")

        print()


def detailed_orbit_analysis():
    """Show the full orbit structure under σ-only."""

    print("=" * 70)
    print("DETAILED: σ-only Orbit Structure")
    print("=" * 70)
    print()

    generators = GENERATOR_SETS['sigma_only']

    # Find all orbits under σ
    all_pairs = [(dr_b, dr_e) for dr_b in range(1, 10) for dr_e in range(1, 10)]
    visited = set()
    orbits = []

    for start in all_pairs:
        if start in visited:
            continue

        # Trace orbit
        orbit = []
        current = start
        while current not in visited:
            visited.add(current)
            orbit.append(current)
            current = sigma(*current)
            if current == start:
                break

        if orbit:
            # Classify by family
            families_in_orbit = defaultdict(list)
            for pair in orbit:
                fam = get_family(*pair)
                families_in_orbit[fam].append(pair)

            orbits.append({
                'start': start,
                'length': len(orbit),
                'pairs': orbit,
                'families': dict(families_in_orbit),
            })

    # Sort by length
    orbits.sort(key=lambda x: -x['length'])

    print(f"Found {len(orbits)} distinct σ-orbits in the 81-pair DR space:")
    print()

    for i, orb in enumerate(orbits[:10]):  # Show top 10
        fam_summary = ', '.join(f"{f}:{len(ps)}" for f, ps in orb['families'].items())
        print(f"Orbit {i+1}: length={orb['length']}, families: {fam_summary}")
        if orb['length'] <= 8:
            print(f"  Pairs: {orb['pairs']}")
        print()


if __name__ == "__main__":
    probe()
    print()
    detailed_orbit_analysis()
