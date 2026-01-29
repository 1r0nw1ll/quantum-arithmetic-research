#!/usr/bin/env python3
"""
QA Certified Result: Scale-Lift Theorem

The DR-lattice (81-state quotient) obstruction taxonomy correctly predicts
component connectivity in bounded QA state spaces Caps(N,N).

Predictions verified:
- {σ, μ}: Separation (24-cycle ↛ 8-cycle)
- {σ, μ, λ₃}: Full collapse (all components connected)
- {σ, μ, λ₆}: Full collapse
- {σ, μ, λ₉}: Fixed-point sink (reaches 1-cycle, not 8-cycle)

This validates QA as PREDICTIVE: the quotient structure captures real
behavior in concrete state spaces.
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments')

import pytest


# === Digital root ===

def dr(n: int) -> int:
    """Digital root: mod 9 with 0 → 9"""
    if n == 0:
        return 9
    r = n % 9
    return 9 if r == 0 else r


# === Family classification ===

TRIBONACCI_DR = {(3,3), (3,6), (3,9), (6,3), (6,6), (6,9), (9,3), (9,6)}
NINBONACCI_DR = {(9,9)}


def classify_state(b: int, e: int) -> str:
    """Classify a (b,e) state by its DR pair."""
    pair = (dr(b), dr(e))
    if pair in TRIBONACCI_DR:
        return '8-cycle'
    elif pair in NINBONACCI_DR:
        return '1-cycle'
    else:
        return '24-cycle'


# === Generators ===

def make_generators(cap: int, k_values: list = None):
    """Create generator functions for Caps(N,N)."""

    def sigma(b: int, e: int) -> tuple:
        new_e = (b + e) % cap
        if new_e == 0:
            new_e = cap
        return (e, new_e)

    def mu(b: int, e: int) -> tuple:
        return (e, b)

    def make_lambda_k(k: int):
        def gen(b: int, e: int) -> tuple:
            new_b = (k * b) % cap
            new_e = (k * e) % cap
            if new_b == 0:
                new_b = cap
            if new_e == 0:
                new_e = cap
            return (new_b, new_e)
        return gen

    gens = {'σ': sigma, 'μ': mu}
    if k_values:
        for k in k_values:
            gens[f'λ_{k}'] = make_lambda_k(k)
    return gens


# === Reachability ===

def compute_reachability(start_states: set, generators: dict,
                          cap: int, max_steps: int = 1000) -> set:
    """Compute reachability in Caps(N,N)."""
    reachable = set(start_states)
    frontier = list(start_states)

    for _ in range(max_steps):
        new_frontier = []
        for b, e in frontier:
            for gen_func in generators.values():
                new_b, new_e = gen_func(b, e)
                if 1 <= new_b <= cap and 1 <= new_e <= cap:
                    new_state = (new_b, new_e)
                    if new_state not in reachable:
                        reachable.add(new_state)
                        new_frontier.append(new_state)
        if not new_frontier:
            break
        frontier = new_frontier

    return reachable


def get_states_by_class(cap: int, class_name: str) -> set:
    """Get all states of a given class in Caps(N,N)."""
    return {(b, e) for b in range(1, cap + 1) for e in range(1, cap + 1)
            if classify_state(b, e) == class_name}


def check_lift_prediction(cap: int, k_values: list,
                           expect_24_to_8: bool, expect_24_to_1: bool) -> bool:
    """Check if a single lift prediction holds."""
    gens = make_generators(cap, k_values)
    s24 = get_states_by_class(cap, '24-cycle')
    s8 = get_states_by_class(cap, '8-cycle')
    s1 = get_states_by_class(cap, '1-cycle')

    reachable = compute_reachability(s24, gens, cap)

    has_24_to_8 = len(s8 & reachable) > 0
    has_24_to_1 = len(s1 & reachable) > 0

    return (has_24_to_8 == expect_24_to_8) and (has_24_to_1 == expect_24_to_1)


# === Tests ===

# Use Caps(27,27) as the canonical test case (small but complete)
TEST_CAP = 27


def test_separation_lifts():
    """Verify {σ, μ} separation lifts to Caps(N,N)."""
    assert check_lift_prediction(TEST_CAP, [], expect_24_to_8=False, expect_24_to_1=False), (
        "Separation prediction failed: expected no 24→8 or 24→1 under {σ, μ}"
    )
    print(f"✓ Caps({TEST_CAP},{TEST_CAP}): {{σ, μ}} separation confirmed")


def test_lambda3_collapse_lifts():
    """Verify {σ, μ, λ₃} full collapse lifts to Caps(N,N)."""
    assert check_lift_prediction(TEST_CAP, [3], expect_24_to_8=True, expect_24_to_1=True), (
        "λ₃ collapse prediction failed: expected 24→8 and 24→1 under {σ, μ, λ₃}"
    )
    print(f"✓ Caps({TEST_CAP},{TEST_CAP}): {{σ, μ, λ₃}} full collapse confirmed")


def test_lambda6_collapse_lifts():
    """Verify {σ, μ, λ₆} full collapse lifts to Caps(N,N)."""
    assert check_lift_prediction(TEST_CAP, [6], expect_24_to_8=True, expect_24_to_1=True), (
        "λ₆ collapse prediction failed: expected 24→8 and 24→1 under {σ, μ, λ₆}"
    )
    print(f"✓ Caps({TEST_CAP},{TEST_CAP}): {{σ, μ, λ₆}} full collapse confirmed")


def test_lambda9_fixed_point_sink_lifts():
    """Verify {σ, μ, λ₉} fixed-point sink lifts to Caps(N,N)."""
    assert check_lift_prediction(TEST_CAP, [9], expect_24_to_8=False, expect_24_to_1=True), (
        "λ₉ sink prediction failed: expected 24→1 but not 24→8 under {σ, μ, λ₉}"
    )
    print(f"✓ Caps({TEST_CAP},{TEST_CAP}): {{σ, μ, λ₉}} fixed-point sink confirmed")


@pytest.mark.parametrize("cap", [27, 36, 45])
def test_lift_across_caps(cap):
    """Verify lift theorem holds across multiple cap sizes."""
    # All four predictions must hold
    assert check_lift_prediction(cap, [], expect_24_to_8=False, expect_24_to_1=False)
    assert check_lift_prediction(cap, [3], expect_24_to_8=True, expect_24_to_1=True)
    assert check_lift_prediction(cap, [6], expect_24_to_8=True, expect_24_to_1=True)
    assert check_lift_prediction(cap, [9], expect_24_to_8=False, expect_24_to_1=True)
    print(f"✓ Caps({cap},{cap}): All predictions confirmed")


def test_scale_lift_theorem():
    """
    Combined test: Scale-Lift Theorem.

    The DR-lattice obstruction taxonomy correctly predicts component
    connectivity in bounded QA state spaces.
    """
    test_separation_lifts()
    test_lambda3_collapse_lifts()
    test_lambda6_collapse_lifts()
    test_lambda9_fixed_point_sink_lifts()

    print()
    print("=" * 70)
    print("CERTIFIED: Scale-Lift Theorem")
    print("=" * 70)
    print()
    print("The DR-lattice (81-state quotient) obstruction taxonomy correctly")
    print("predicts component connectivity in bounded QA state spaces.")
    print()
    print("This validates QA as PREDICTIVE: quotient structure captures")
    print("real behavior in concrete state spaces.")
    print()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
