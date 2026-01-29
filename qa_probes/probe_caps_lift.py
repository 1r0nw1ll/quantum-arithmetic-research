#!/usr/bin/env python3
"""
Probe: Lift Obstruction Atlas from DR-lattice to Caps(N,N)

Test whether the digital-root taxonomy predicts connectivity in bounded
(b,e) state spaces.

The DR-lattice (81 states) is a quotient. This probe verifies that the
same obstruction patterns manifest in actual integer state spaces under
bounded caps.

Predictions from DR taxonomy:
- {σ, μ}: Separation (24-cycle ↛ 8-cycle)
- {σ, μ, λ₃}: Full collapse
- {σ, μ, λ₆}: Full collapse
- {σ, μ, λ₉}: Fixed-point sink (reaches (9k,9k) but not general 8-cycle)
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments')

from collections import defaultdict


# === Digital root ===

def dr(n: int) -> int:
    """Digital root: mod 9 with 0 → 9"""
    if n == 0:
        return 9
    r = n % 9
    return 9 if r == 0 else r


# === Family classification (from DR) ===

# Tribonacci DR pairs (both components ≡ 0 mod 3, excluding (9,9))
TRIBONACCI_DR = {(3,3), (3,6), (3,9), (6,3), (6,6), (6,9), (9,3), (9,6)}

# Ninbonacci (fixed point)
NINBONACCI_DR = {(9,9)}

# 24-cycle families
FAMILIES_24_DR = set()
for dr_b in range(1, 10):
    for dr_e in range(1, 10):
        pair = (dr_b, dr_e)
        if pair not in TRIBONACCI_DR and pair not in NINBONACCI_DR:
            FAMILIES_24_DR.add(pair)


def classify_state(b: int, e: int) -> str:
    """Classify a (b,e) state by its DR pair."""
    pair = (dr(b), dr(e))
    if pair in TRIBONACCI_DR:
        return '8-cycle'
    elif pair in NINBONACCI_DR:
        return '1-cycle'
    else:
        return '24-cycle'


# === Generators on Caps(N,N) ===

def make_generators(cap: int, k_values: list = None):
    """Create generator functions for Caps(N,N) state space."""

    def sigma(b: int, e: int) -> tuple:
        """Fibonacci step: (b, e) → (e, b+e) mod cap"""
        new_e = (b + e) % cap
        if new_e == 0:
            new_e = cap  # Avoid 0
        return (e, new_e)

    def mu(b: int, e: int) -> tuple:
        """Swap: (b, e) → (e, b)"""
        return (e, b)

    def make_lambda_k(k: int):
        """Scale by k: (b, e) → (k*b, k*e) mod cap"""
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


# === Reachability in Caps(N,N) ===

def compute_reachability_caps(start_states: set, generators: dict,
                               cap: int, max_steps: int = 1000) -> set:
    """Compute reachability in Caps(N,N) state space."""
    reachable = set(start_states)
    frontier = list(start_states)

    for step in range(max_steps):
        new_frontier = []
        for b, e in frontier:
            for gen_func in generators.values():
                new_state = gen_func(b, e)
                # Validate state is in Caps(N,N)
                new_b, new_e = new_state
                if 1 <= new_b <= cap and 1 <= new_e <= cap:
                    if new_state not in reachable:
                        reachable.add(new_state)
                        new_frontier.append(new_state)

        if not new_frontier:
            break
        frontier = new_frontier

    return reachable


def get_24_cycle_states(cap: int) -> set:
    """Get all (b,e) states in Caps(N,N) classified as 24-cycle."""
    states = set()
    for b in range(1, cap + 1):
        for e in range(1, cap + 1):
            if classify_state(b, e) == '24-cycle':
                states.add((b, e))
    return states


def get_8_cycle_states(cap: int) -> set:
    """Get all (b,e) states in Caps(N,N) classified as 8-cycle (Tribonacci)."""
    states = set()
    for b in range(1, cap + 1):
        for e in range(1, cap + 1):
            if classify_state(b, e) == '8-cycle':
                states.add((b, e))
    return states


def get_1_cycle_states(cap: int) -> set:
    """Get all (b,e) states in Caps(N,N) classified as 1-cycle (Ninbonacci)."""
    states = set()
    for b in range(1, cap + 1):
        for e in range(1, cap + 1):
            if classify_state(b, e) == '1-cycle':
                states.add((b, e))
    return states


# === Main probe ===

def probe_caps(cap: int):
    """Run the lift probe for Caps(N,N)."""

    print(f"\n{'='*70}")
    print(f"CAPS({cap},{cap}) LIFT PROBE")
    print(f"{'='*70}")

    total_states = cap * cap
    s24 = get_24_cycle_states(cap)
    s8 = get_8_cycle_states(cap)
    s1 = get_1_cycle_states(cap)

    print(f"\nState space: {total_states} total")
    print(f"  24-cycle (S₂₄): {len(s24)} states")
    print(f"  8-cycle (S₈):   {len(s8)} states")
    print(f"  1-cycle (S₁):   {len(s1)} states")
    print()

    # Test configurations
    configs = [
        ('σ, μ', []),
        ('σ, μ, λ₃', [3]),
        ('σ, μ, λ₆', [6]),
        ('σ, μ, λ₉', [9]),
    ]

    # DR predictions
    predictions = {
        'σ, μ': ('Separation', False, False),
        'σ, μ, λ₃': ('Full collapse', True, True),
        'σ, μ, λ₆': ('Full collapse', True, True),
        'σ, μ, λ₉': ('Fixed-point sink', False, True),
    }

    print(f"{'Config':<15} | {'Reach':>8} | {'24→8':>8} | {'24→1':>8} | {'Predicted':>18} | Match?")
    print("-" * 80)

    results = []

    for name, k_values in configs:
        gens = make_generators(cap, k_values)

        # Start from 24-cycle states
        reachable = compute_reachability_caps(s24, gens, cap)

        # Check crossings
        s8_reached = len(s8 & reachable)
        s1_reached = len(s1 & reachable)

        has_24_to_8 = s8_reached > 0
        has_24_to_1 = s1_reached > 0

        # Compare with prediction
        pred_name, pred_8, pred_1 = predictions[name]
        match_8 = (has_24_to_8 == pred_8)
        match_1 = (has_24_to_1 == pred_1)
        match = match_8 and match_1

        results.append({
            'name': name,
            'reachable': len(reachable),
            's8_reached': s8_reached,
            's1_reached': s1_reached,
            'has_24_to_8': has_24_to_8,
            'has_24_to_1': has_24_to_1,
            'prediction': pred_name,
            'match': match,
        })

        reach_pct = 100 * len(reachable) / total_states
        s8_str = f"{s8_reached}/{len(s8)}" if s8 else "N/A"
        s1_str = f"{s1_reached}/{len(s1)}" if s1 else "N/A"
        match_str = "✓" if match else "✗"

        print(f"{name:<15} | {len(reachable):>8} | {s8_str:>8} | {s1_str:>8} | {pred_name:>18} | {match_str}")

    print()

    # Summary
    all_match = all(r['match'] for r in results)
    if all_match:
        print("✓ ALL PREDICTIONS CONFIRMED — DR taxonomy lifts to Caps!")
    else:
        mismatches = [r['name'] for r in results if not r['match']]
        print(f"✗ MISMATCHES: {mismatches}")

    return results


def main():
    print("=" * 70)
    print("OBSTRUCTION ATLAS LIFT PROBE")
    print("Testing whether DR-lattice taxonomy predicts Caps(N,N) behavior")
    print("=" * 70)

    # Test multiple cap sizes
    caps = [27, 36, 45, 54, 63]  # Multiples of 9 for clean DR alignment

    all_results = {}
    for cap in caps:
        all_results[cap] = probe_caps(cap)

    # Final summary
    print("\n" + "=" * 70)
    print("LIFT THEOREM SUMMARY")
    print("=" * 70)
    print()

    all_caps_match = True
    for cap, results in all_results.items():
        cap_match = all(r['match'] for r in results)
        status = "✓ CONFIRMED" if cap_match else "✗ MISMATCH"
        print(f"Caps({cap},{cap}): {status}")
        if not cap_match:
            all_caps_match = False

    print()
    if all_caps_match:
        print("=" * 70)
        print("SCALE-LIFT THEOREM CONFIRMED")
        print("=" * 70)
        print()
        print("The DR-lattice obstruction taxonomy correctly predicts")
        print("component connectivity in bounded QA state spaces.")
        print()
        print("This validates QA as PREDICTIVE, not just self-contained.")
    else:
        print("Some predictions failed — investigate mismatches.")


if __name__ == "__main__":
    main()
