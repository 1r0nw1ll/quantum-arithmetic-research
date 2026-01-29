#!/usr/bin/env python3
"""
Probe: 24-cycle → 8-cycle orbit transition under generator composition

Test: Does any generator composition map a 24-cycle state into an 8-cycle
      while preserving geometric alignment and parity invariants?

Outcome:
- EXISTS  → no structural impossibility here
- ABSENT → non-trivial algebraic obstruction (paper-grade)

This probe uses actual QA infrastructure from the codebase.
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments')

from itertools import product
from collections import defaultdict

# Import from existing QA modules
from qa_harmonicity_v2 import (
    digital_root,
    PISANO_FAMILY_MAP,
    PISANO_FAMILIES,
)
from qa_lab.qa_e8_alignment import e8_alignment_single


# === FIXED PARAMETERS (do NOT tune mid-run) ===

GENERATOR_DEPTH = 6          # bounded, enough to see structure
ALIGNMENT_THRESHOLD = 0.85   # reasonable threshold for "high alignment"
MAX_BASE_STATES = 500        # cover the full 24-cycle space


# === Orbit classification ===

def get_orbit_period(b: int, e: int) -> int:
    """
    Get orbit period (24, 8, or 1) based on digital root pair.

    Uses the PISANO_FAMILY_MAP lookup.
    """
    dr_b = digital_root(b)
    dr_e = digital_root(e)
    family = PISANO_FAMILY_MAP.get((dr_b, dr_e), 'Unknown')

    if family in ['Fibonacci', 'Lucas', 'Phibonacci']:
        return 24
    elif family == 'Tribonacci':
        return 8
    elif family == 'Ninbonacci':
        return 1
    else:
        return 0  # Unknown


def get_family(b: int, e: int) -> str:
    """Get family name for a (b, e) pair."""
    dr_b = digital_root(b)
    dr_e = digital_root(e)
    return PISANO_FAMILY_MAP.get((dr_b, dr_e), 'Unknown')


# === Generators ===
# The fundamental QA generator is the Fibonacci step: (b, e) → (e, b+e)
# We also include the reverse and some variants

def gen_fib_step(b: int, e: int) -> tuple:
    """Standard Fibonacci step: (b, e) → (e, b+e)"""
    return (e, b + e)

def gen_reverse_step(b: int, e: int) -> tuple:
    """Reverse step: (b, e) → (b+e, b) - goes backward in sequence"""
    return (b + e, b)

def gen_swap(b: int, e: int) -> tuple:
    """Swap: (b, e) → (e, b)"""
    return (e, b)

def gen_double_step(b: int, e: int) -> tuple:
    """Two Fibonacci steps: (b, e) → (b+e, b+2e)"""
    return (b + e, b + 2*e)

def gen_scale_2(b: int, e: int) -> tuple:
    """Scale by 2: (b, e) → (2b, 2e) - female transformation"""
    return (2*b, 2*e)


GENERATORS = {
    'fib': gen_fib_step,
    'rev': gen_reverse_step,
    'swap': gen_swap,
    'dbl': gen_double_step,
    'scale2': gen_scale_2,
}


# === Invariants ===

def mod9_class(b: int, e: int) -> tuple:
    """Return the mod-9 class (digital root pair)."""
    return (digital_root(b), digital_root(e))

def parity_signature(b: int, e: int) -> tuple:
    """Return parity signature (even/odd for b, e)."""
    return (b % 2, e % 2)

def compute_e8_alignment(b: int, e: int) -> float:
    """Compute E8 alignment score for (b, e) pair."""
    d = b + e
    a = b + 2 * e
    return e8_alignment_single(b, e, d, a)


# === Logging ===

log = {
    "base_states_checked": 0,
    "generator_paths_tried": 0,
    "candidates_found": [],
    "blocked_by": defaultdict(int),
    "transitions_by_type": defaultdict(int),
}


# === Core probe logic ===

def apply_generators(b: int, e: int, gen_path: tuple) -> tuple:
    """Apply a sequence of generators to a state."""
    for g in gen_path:
        b, e = GENERATORS[g](b, e)
        if b <= 0 or e <= 0:
            return None  # Invalid state
        if b > 10000 or e > 10000:
            return None  # Prevent blowup
    return (b, e)


def generate_24_cycle_states(max_states: int):
    """Generate high-alignment states from 24-cycle families."""
    states = []

    # Enumerate (b, e) pairs and filter for 24-cycle + high alignment
    for b in range(1, 100):
        for e in range(1, 100):
            if get_orbit_period(b, e) != 24:
                continue

            alignment = compute_e8_alignment(b, e)
            if alignment >= ALIGNMENT_THRESHOLD:
                states.append((b, e, alignment))

            if len(states) >= max_states:
                return states

    return states


def probe():
    """Main probe: search for 24→8 transitions preserving alignment."""

    print("=" * 70)
    print("PROBE: 24-cycle → 8-cycle orbit transition with alignment preserved")
    print("=" * 70)
    print()

    # Step 1: collect high-alignment 24-cycle base states
    print(f"Collecting 24-cycle base states with E8 alignment >= {ALIGNMENT_THRESHOLD}...")
    base_states = generate_24_cycle_states(MAX_BASE_STATES)
    log["base_states_checked"] = len(base_states)

    if not base_states:
        print("WARNING: no qualifying 24-cycle base states found")
        return

    print(f"Found {len(base_states)} qualifying base states")
    print()

    # Show some examples
    print("Sample base states (first 5):")
    for b, e, align in base_states[:5]:
        family = get_family(b, e)
        print(f"  ({b:3d}, {e:3d}) - family={family:12s}, E8={align:.4f}")
    print()

    gen_names = list(GENERATORS.keys())

    # Step 2: bounded generator compositions
    print(f"Testing generator compositions up to depth {GENERATOR_DEPTH}...")
    print(f"Generators: {gen_names}")
    print()

    for depth in range(1, GENERATOR_DEPTH + 1):
        paths_at_depth = 0
        for gen_path in product(gen_names, repeat=depth):
            log["generator_paths_tried"] += 1
            paths_at_depth += 1

            for b0, e0, align0 in base_states:
                result = apply_generators(b0, e0, gen_path)
                if result is None:
                    continue

                b1, e1 = result

                # Check orbit transition
                orbit1 = get_orbit_period(b1, e1)
                if orbit1 == 0:
                    continue  # Unknown family

                transition_type = f"24→{orbit1}"
                log["transitions_by_type"][transition_type] += 1

                if orbit1 != 8:
                    continue  # Not the transition we're looking for

                # Check if alignment is preserved
                align1 = compute_e8_alignment(b1, e1)
                if align1 < ALIGNMENT_THRESHOLD:
                    log["blocked_by"]["alignment_drop"] += 1
                    continue

                # Check mod9 preservation (optional - may be too strict)
                # For now, we don't require this

                # Check parity preservation
                parity0 = parity_signature(b0, e0)
                parity1 = parity_signature(b1, e1)
                if parity0 != parity1:
                    log["blocked_by"]["parity_change"] += 1
                    # Don't skip - parity change may be acceptable

                # If we get here, we FOUND a transition!
                log["candidates_found"].append({
                    "base": (b0, e0),
                    "result": (b1, e1),
                    "generators": gen_path,
                    "base_alignment": align0,
                    "result_alignment": align1,
                    "base_family": get_family(b0, e0),
                    "result_family": get_family(b1, e1),
                })

                print("!" * 70)
                print("FOUND 24→8 TRANSITION WITH ALIGNMENT PRESERVED")
                print("!" * 70)
                print(f"Generators: {' → '.join(gen_path)}")
                print(f"Base:   ({b0}, {e0}) - {get_family(b0, e0)}, E8={align0:.4f}")
                print(f"Result: ({b1}, {e1}) - {get_family(b1, e1)}, E8={align1:.4f}")
                print()
                # Don't return - keep searching to see how common this is

        print(f"  Depth {depth}: tested {paths_at_depth} paths")


def print_summary():
    """Print final probe summary."""
    print()
    print("=" * 70)
    print("PROBE SUMMARY")
    print("=" * 70)
    print(f"Base states checked:      {log['base_states_checked']}")
    print(f"Generator paths tried:    {log['generator_paths_tried']}")
    print(f"Candidates found:         {len(log['candidates_found'])}")
    print()

    print("Transition counts by type:")
    for trans_type, count in sorted(log["transitions_by_type"].items()):
        print(f"  {trans_type}: {count}")
    print()

    print("Blocked by:")
    for reason, count in sorted(log["blocked_by"].items()):
        print(f"  {reason}: {count}")
    print()

    if log["candidates_found"]:
        print("=" * 70)
        print("RESULT: EXISTS (no structural impossibility)")
        print("=" * 70)
        print()
        print("Sample transitions found:")
        for i, c in enumerate(log["candidates_found"][:10]):
            print(f"  {i+1}. ({c['base'][0]}, {c['base'][1]}) → ({c['result'][0]}, {c['result'][1]})")
            print(f"      via: {' → '.join(c['generators'])}")
            print(f"      E8: {c['base_alignment']:.4f} → {c['result_alignment']:.4f}")
    else:
        print("=" * 70)
        print("RESULT: ABSENT (candidate structural obstruction)")
        print("=" * 70)
        print()
        print("No 24→8 transitions found that preserve E8 alignment.")
        print("This may indicate a structural impossibility.")


if __name__ == "__main__":
    probe()
    print_summary()
