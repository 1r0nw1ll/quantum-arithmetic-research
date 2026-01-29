#!/usr/bin/env python3
"""
QA Certified Result: Generator-Relative Obstruction

The 24-cycle ↛ 8-cycle separation is NOT absolute - it depends on which
generators are admitted:

- Under {σ} (Fibonacci step): obstruction holds
- Under {σ, μ} (+ swap): obstruction holds
- Under {σ, λ₃} (+ scale by 3): obstruction FAILS

This demonstrates that QA obstructions are generator-relative, not
absolute impossibilities.
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments')

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


def lambda_3(dr_b: int, dr_e: int) -> tuple:
    """Scale by 3: (b, e) → (3b, 3e)"""
    return (dr(3 * dr_b), dr(3 * dr_e))


# === Reachability computation ===

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
    """Get all digital root pairs from 24-cycle families."""
    return {(dr_b, dr_e) for (dr_b, dr_e), fam in PISANO_FAMILY_MAP.items()
            if fam in ['Fibonacci', 'Lucas', 'Phibonacci']}


def get_tribonacci_pairs() -> set:
    """Get all Tribonacci (8-cycle) digital root pairs."""
    return {(dr_b, dr_e) for (dr_b, dr_e), fam in PISANO_FAMILY_MAP.items()
            if fam == 'Tribonacci'}


# === Tests ===

def test_obstruction_holds_under_sigma():
    """Verify 24↛8 under σ-only."""
    gens = {'σ': sigma}
    start = get_24_cycle_pairs()
    trib = get_tribonacci_pairs()

    reachable = compute_reachability(start, gens)
    crossing = trib & reachable

    assert len(crossing) == 0, (
        f"24→8 crossing found under σ-only: {crossing}"
    )
    print(f"✓ σ-only: 24↛8 obstruction holds (reachable: {len(reachable)} pairs)")


def test_obstruction_holds_under_sigma_mu():
    """Verify 24↛8 under {σ, μ}."""
    gens = {'σ': sigma, 'μ': mu}
    start = get_24_cycle_pairs()
    trib = get_tribonacci_pairs()

    reachable = compute_reachability(start, gens)
    crossing = trib & reachable

    assert len(crossing) == 0, (
        f"24→8 crossing found under {{σ, μ}}: {crossing}"
    )
    print(f"✓ {{σ, μ}}: 24↛8 obstruction holds (reachable: {len(reachable)} pairs)")


def test_obstruction_fails_under_sigma_lambda3():
    """Verify 24→8 crossing exists under {σ, λ₃}."""
    gens = {'σ': sigma, 'λ₃': lambda_3}
    start = get_24_cycle_pairs()
    trib = get_tribonacci_pairs()

    reachable = compute_reachability(start, gens)
    crossing = trib & reachable

    assert len(crossing) == len(trib), (
        f"Expected all 8 Tribonacci pairs reachable, got {len(crossing)}"
    )
    print(f"✓ {{σ, λ₃}}: 24→8 crossing EXISTS (all {len(crossing)} Tribonacci pairs reachable)")


def test_lambda3_maps_to_mod3_zero():
    """Verify λ₃ always produces pairs with both components ≡ 0 (mod 3)."""
    for dr_b in range(1, 10):
        for dr_e in range(1, 10):
            new_b, new_e = lambda_3(dr_b, dr_e)
            assert new_b % 3 == 0, f"λ₃({dr_b}, {dr_e}) → ({new_b}, _) but {new_b} ≢ 0 (mod 3)"
            assert new_e % 3 == 0, f"λ₃({dr_b}, {dr_e}) → (_, {new_e}) but {new_e} ≢ 0 (mod 3)"

    print("✓ λ₃ always maps to (0, 0) mod 3 basin (Tribonacci regime)")


def test_generator_relative_obstruction():
    """
    Combined test: the obstruction is generator-relative.

    This is the main theorem being certified.
    """
    test_lambda3_maps_to_mod3_zero()
    test_obstruction_holds_under_sigma()
    test_obstruction_holds_under_sigma_mu()
    test_obstruction_fails_under_sigma_lambda3()

    print()
    print("=" * 70)
    print("CERTIFIED: Generator-Relative Obstruction Theorem")
    print("=" * 70)
    print()
    print("The 24-cycle ↛ 8-cycle separation is GENERATOR-RELATIVE:")
    print()
    print("  - Under {σ}:      obstruction HOLDS")
    print("  - Under {σ, μ}:   obstruction HOLDS")
    print("  - Under {σ, λ₃}:  obstruction FAILS")
    print()
    print("Mechanism: λ₃ (scale by 3) maps any digital root pair into the")
    print("(0, 0) mod 3 basin, bypassing the mod-3 fixed point isolation.")
    print()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
