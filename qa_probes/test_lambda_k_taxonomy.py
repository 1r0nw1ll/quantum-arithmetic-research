#!/usr/bin/env python3
"""
QA Certified Result: λₖ Connectivity Classification Theorem

Under generator set {σ, μ, λₖ}, reachability from the 24-cycle component
depends on k:

- k ∈ {3, 6}: Full collapse (81/81 reachable, all orbits connected)
- k = 9: Fixed-point-only collapse (73/81, reaches Ninbonacci but not Tribonacci)
- k ∈ {1, 2, 4, 5, 7, 8}: Separation preserved (72/81, original obstruction holds)

This demonstrates that component connectivity is a function of the allowed
scaling generators, with k=9 exhibiting unique "projection to fixed point"
behavior distinct from the k=3,6 full collapse.
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments')

import pytest
from qa_harmonicity_v2 import PISANO_FAMILY_MAP


# === Digital root arithmetic ===

def dr(n: int) -> int:
    """Digital root: mod 9 with 0 → 9"""
    if n == 0:
        return 9
    r = n % 9
    return 9 if r == 0 else r


# === Generators ===

def sigma(dr_b: int, dr_e: int) -> tuple:
    """Fibonacci step: (b, e) → (e, b+e)"""
    return (dr_e, dr(dr_b + dr_e))


def mu(dr_b: int, dr_e: int) -> tuple:
    """Swap: (b, e) → (e, b)"""
    return (dr_e, dr_b)


def lambda_k(k: int):
    """Create scaling generator for factor k."""
    def gen(dr_b: int, dr_e: int) -> tuple:
        return (dr(k * dr_b), dr(k * dr_e))
    return gen


# === Reachability ===

def compute_reachability(start_pairs: set, generators: dict, max_steps: int = 100) -> set:
    """Compute transitive closure under generators."""
    reachable = set(start_pairs)
    frontier = list(start_pairs)

    for _ in range(max_steps):
        new_frontier = []
        for pair in frontier:
            for gen_func in generators.values():
                new_pair = gen_func(*pair)
                if new_pair not in reachable:
                    reachable.add(new_pair)
                    new_frontier.append(new_pair)
        if not new_frontier:
            break
        frontier = new_frontier

    return reachable


# === Family helpers ===

def get_24_cycle_pairs() -> set:
    return {(dr_b, dr_e) for (dr_b, dr_e), fam in PISANO_FAMILY_MAP.items()
            if fam in ['Fibonacci', 'Lucas', 'Phibonacci']}


def get_tribonacci_pairs() -> set:
    return {(dr_b, dr_e) for (dr_b, dr_e), fam in PISANO_FAMILY_MAP.items()
            if fam == 'Tribonacci'}


def get_ninbonacci_pairs() -> set:
    return {(dr_b, dr_e) for (dr_b, dr_e), fam in PISANO_FAMILY_MAP.items()
            if fam == 'Ninbonacci'}


def get_reachability_stats(k: int) -> dict:
    """Compute reachability statistics for generator set {σ, μ, λₖ}."""
    gens = {'σ': sigma, 'μ': mu, f'λ_{k}': lambda_k(k)}
    start = get_24_cycle_pairs()
    trib = get_tribonacci_pairs()
    ninb = get_ninbonacci_pairs()

    reach = compute_reachability(start, gens)

    return {
        'reachable': len(reach),
        'tribonacci_reached': len(trib & reach),
        'ninbonacci_reached': len(ninb & reach),
    }


# === Tests ===

@pytest.mark.parametrize("k", [1, 2, 4, 5, 7, 8])
def test_separation_preserved(k):
    """k ∈ {1,2,4,5,7,8}: separation preserved, 72/81 reachable."""
    stats = get_reachability_stats(k)

    assert stats['reachable'] == 72, (
        f"k={k}: expected 72 reachable, got {stats['reachable']}"
    )
    assert stats['tribonacci_reached'] == 0, (
        f"k={k}: expected 0 Tribonacci reached, got {stats['tribonacci_reached']}"
    )
    assert stats['ninbonacci_reached'] == 0, (
        f"k={k}: expected 0 Ninbonacci reached, got {stats['ninbonacci_reached']}"
    )


@pytest.mark.parametrize("k", [3, 6])
def test_full_collapse(k):
    """k ∈ {3,6}: full collapse, 81/81 reachable."""
    stats = get_reachability_stats(k)

    assert stats['reachable'] == 81, (
        f"k={k}: expected 81 reachable (full collapse), got {stats['reachable']}"
    )
    assert stats['tribonacci_reached'] == 8, (
        f"k={k}: expected all 8 Tribonacci reached, got {stats['tribonacci_reached']}"
    )
    assert stats['ninbonacci_reached'] == 1, (
        f"k={k}: expected Ninbonacci reached, got {stats['ninbonacci_reached']}"
    )


def test_k9_fixed_point_only():
    """k=9: fixed-point-only collapse, 73/81 reachable."""
    stats = get_reachability_stats(9)

    assert stats['reachable'] == 73, (
        f"k=9: expected 73 reachable, got {stats['reachable']}"
    )
    assert stats['tribonacci_reached'] == 0, (
        f"k=9: expected 0 Tribonacci reached (not 8-cycle), got {stats['tribonacci_reached']}"
    )
    assert stats['ninbonacci_reached'] == 1, (
        f"k=9: expected Ninbonacci (9,9) reached, got {stats['ninbonacci_reached']}"
    )


def test_lambda_k_taxonomy():
    """
    Combined test: λₖ Connectivity Classification Theorem.

    Verifies the complete taxonomy of component behavior under scaling generators.
    """
    # Run all component tests
    for k in [1, 2, 4, 5, 7, 8]:
        test_separation_preserved(k)

    for k in [3, 6]:
        test_full_collapse(k)

    test_k9_fixed_point_only()

    print()
    print("=" * 70)
    print("CERTIFIED: λₖ Connectivity Classification Theorem")
    print("=" * 70)
    print()
    print("Under {σ, μ, λₖ}, reachability from S₂₄ (24-cycle) satisfies:")
    print()
    print("  k ∈ {3, 6}:       Full collapse (81/81), all orbits connected")
    print("  k = 9:            Fixed-point-only (73/81), reaches (9,9) not S₈")
    print("  k ∈ {1,2,4,5,7,8}: Separation preserved (72/81)")
    print()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
