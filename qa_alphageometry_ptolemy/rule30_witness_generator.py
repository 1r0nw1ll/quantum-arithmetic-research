#!/usr/bin/env python3
"""
Rule 30 Center Column Non-Periodicity Witness Generator

Generates complete witness data proving bounded non-periodicity of Rule 30 center column.

Parameters:
- Rule: 30 (ECA - Elementary Cellular Automaton)
- Initial condition: single 1 at position 0
- Time horizon: T = 16384
- Period search: P_MAX = 1024
- Window: [-16384, +16384]
"""

import numpy as np
import json
import hashlib
from typing import List, Dict, Tuple

# ============================================================================
# RULE 30 TRUTH TABLE
# ============================================================================

def rule30_lookup():
    """
    Rule 30 explicit truth table.

    Neighborhood (L,C,R) -> new(C)
    111 -> 0
    110 -> 0
    101 -> 0
    100 -> 1
    011 -> 1
    010 -> 1
    001 -> 1
    000 -> 0
    """
    lookup = {
        (1, 1, 1): 0,
        (1, 1, 0): 0,
        (1, 0, 1): 0,
        (1, 0, 0): 1,
        (0, 1, 1): 1,
        (0, 1, 0): 1,
        (0, 0, 1): 1,
        (0, 0, 0): 0,
    }
    return lookup

def print_truth_table():
    """Print the truth table for verification."""
    print("=" * 60)
    print("RULE 30 TRUTH TABLE")
    print("=" * 60)
    lookup = rule30_lookup()
    print("Neighborhood (L,C,R) -> new(C)")
    for config in [(1,1,1), (1,1,0), (1,0,1), (1,0,0), (0,1,1), (0,1,0), (0,0,1), (0,0,0)]:
        result = lookup[config]
        print(f"{config[0]}{config[1]}{config[2]} -> {result}")
    print("=" * 60)
    print()

# ============================================================================
# RULE 30 EVOLUTION
# ============================================================================

def evolve_rule30(T_END: int) -> np.ndarray:
    """
    Evolve Rule 30 from t=0 to t=T_END and extract center column.

    Uses O(T) memory by maintaining only current and next row.
    Window: [-T_END, +T_END], so width = 2*T_END + 1.
    Center position is at index T_END (offset 0).

    Args:
        T_END: Maximum time step

    Returns:
        center: array of shape (T_END+1,) with center column values
    """
    lookup = rule30_lookup()

    # Window size: positions from -T_END to +T_END
    width = 2 * T_END + 1
    center_idx = T_END  # Position 0 maps to index T_END

    # Initialize: single 1 at position 0 (center)
    current = np.zeros(width, dtype=np.int8)
    current[center_idx] = 1

    # Store center column values
    center = np.zeros(T_END + 1, dtype=np.int8)
    center[0] = current[center_idx]

    # Evolve for T_END steps
    next_row = np.zeros(width, dtype=np.int8)

    for t in range(1, T_END + 1):
        # Update each cell using Rule 30
        for i in range(width):
            # Get neighbors (boundary is always 0)
            left = current[i-1] if i > 0 else 0
            center_cell = current[i]
            right = current[i+1] if i < width - 1 else 0

            # Apply Rule 30
            next_row[i] = lookup[(left, center_cell, right)]

        # Store center column value
        center[t] = next_row[center_idx]

        # Swap buffers
        current, next_row = next_row, current

        # Progress indicator
        if t % 1024 == 0:
            print(f"Evolution progress: t = {t} / {T_END}")

    return center

# ============================================================================
# WITNESS COMPUTATION
# ============================================================================

def find_period_witnesses(center: np.ndarray, P_MAX: int) -> Tuple[List[Dict], List[int]]:
    """
    For each period p in [1, P_MAX], find smallest t where center(t) ≠ center(t+p).

    Args:
        center: center column sequence of length T_END+1
        P_MAX: maximum period to check

    Returns:
        witnesses: list of dicts with keys {p, t, center_t, center_t_plus_p}
        failures: list of periods where no counterexample was found
    """
    T_END = len(center) - 1
    witnesses = []
    failures = []

    print(f"\nSearching for period counterexamples (P_MAX = {P_MAX})...")

    for p in range(1, P_MAX + 1):
        # Find smallest t in [0, T_END - p] where center[t] ≠ center[t+p]
        counterexample_found = False

        for t in range(T_END - p + 1):
            if center[t] != center[t + p]:
                # Found counterexample
                witnesses.append({
                    'p': int(p),
                    't': int(t),
                    'center_t': int(center[t]),
                    'center_t_plus_p': int(center[t + p])
                })
                counterexample_found = True
                break

        if not counterexample_found:
            failures.append(p)
            print(f"WARNING: No counterexample found for period p = {p}")

        # Progress indicator
        if p % 100 == 0:
            print(f"Period search progress: p = {p} / {P_MAX}")

    return witnesses, failures

# ============================================================================
# SANITY TESTS
# ============================================================================

def sanity_test():
    """Run sanity test: compute and print center(0..5)."""
    print("=" * 60)
    print("SANITY TEST: center(0..5)")
    print("=" * 60)

    T_TEST = 5
    center_test = evolve_rule30(T_TEST)

    print("Time | Center value")
    print("-" * 20)
    for t in range(T_TEST + 1):
        print(f"{t:4d} | {center_test[t]}")
    print("=" * 60)
    print()

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def write_csv(witnesses: List[Dict], filename: str):
    """Write witnesses to CSV file."""
    with open(filename, 'w') as f:
        f.write("p,t,center_t,center_t_plus_p\n")
        for w in witnesses:
            f.write(f"{w['p']},{w['t']},{w['center_t']},{w['center_t_plus_p']}\n")
    print(f"Written: {filename}")

def write_json(witnesses: List[Dict], P_MAX: int, T_END: int, failures: List[int], filename: str):
    """Write witnesses and summary to JSON file."""
    data = {
        'periods': witnesses,
        'summary': {
            'P_MAX': P_MAX,
            'T_END': T_END,
            'verified_count': len(witnesses),
            'failures': failures
        }
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Written: {filename}")

def write_center_sequence(center: np.ndarray, filename: str):
    """Write full center column sequence to text file."""
    with open(filename, 'w') as f:
        # Write as space-separated bits
        f.write(' '.join(str(int(bit)) for bit in center))
    print(f"Written: {filename}")

def compute_sha256(filename: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        sha256.update(f.read())
    return sha256.hexdigest()

def write_summary(csv_hash: str, json_hash: str, verified_count: int, P_MAX: int, failures: List[int], filename: str):
    """Write computation summary."""
    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RULE 30 CENTER COLUMN WITNESS COMPUTATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("PARAMETERS:\n")
        f.write(f"  Rule: 30\n")
        f.write(f"  Initial condition: single 1 at position 0\n")
        f.write(f"  Time horizon: T_END = 16384\n")
        f.write(f"  Period search: P_MAX = 1024\n")
        f.write(f"  Window: [-16384, +16384]\n\n")

        f.write("FILE HASHES:\n")
        f.write(f"  witness_rule30_center_P1024_T16384.csv: {csv_hash}\n")
        f.write(f"  witness_rule30_center_P1024_T16384.json: {json_hash}\n\n")

        f.write("RESULTS:\n")
        f.write(f"  Verified periods: {verified_count} / {P_MAX}\n")
        f.write(f"  Failures: {failures}\n\n")

        if len(failures) == 0:
            f.write("SUCCESS: All periods verified as non-periodic.\n")
        else:
            f.write(f"WARNING: {len(failures)} periods failed to find counterexample.\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Written: {filename}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configuration
    T_END = 16384
    P_MAX = 1024

    print("=" * 60)
    print("RULE 30 CENTER COLUMN NON-PERIODICITY WITNESS GENERATOR")
    print("=" * 60)
    print()

    # Print truth table
    print_truth_table()

    # Sanity test
    sanity_test()

    # Main computation
    print("=" * 60)
    print(f"MAIN COMPUTATION: T_END = {T_END}, P_MAX = {P_MAX}")
    print("=" * 60)
    print()

    print("Step 1: Evolving Rule 30 to extract center column...")
    center = evolve_rule30(T_END)
    print(f"Completed: center column extracted (length = {len(center)})\n")

    print("Step 2: Finding period witnesses...")
    witnesses, failures = find_period_witnesses(center, P_MAX)
    print(f"Completed: {len(witnesses)} witnesses found\n")

    # Check for failures
    if failures:
        print("=" * 60)
        print("CRITICAL WARNING: PERIODS WITHOUT COUNTEREXAMPLES FOUND")
        print("=" * 60)
        for p in failures:
            print(f"  Period p = {p}: NO counterexample found in [0, {T_END - p}]")
        print("=" * 60)
        print("\nSTOPPING: Investigation required.")
        return

    print("Step 3: Writing output files...")

    # Write CSV
    csv_filename = "witness_rule30_center_P1024_T16384.csv"
    write_csv(witnesses, csv_filename)

    # Write JSON
    json_filename = "witness_rule30_center_P1024_T16384.json"
    write_json(witnesses, P_MAX, T_END, failures, json_filename)

    # Write center sequence
    center_filename = "center_rule30_T16384.txt"
    write_center_sequence(center, center_filename)

    # Compute hashes
    print("\nStep 4: Computing SHA256 hashes...")
    csv_hash = compute_sha256(csv_filename)
    json_hash = compute_sha256(json_filename)

    print(f"  CSV SHA256:  {csv_hash}")
    print(f"  JSON SHA256: {json_hash}")

    # Write summary
    summary_filename = "computation_summary.txt"
    write_summary(csv_hash, json_hash, len(witnesses), P_MAX, failures, summary_filename)

    print("\n" + "=" * 60)
    print("COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"Verified periods: {len(witnesses)} / {P_MAX}")
    print(f"Failures: {failures}")
    print("=" * 60)

if __name__ == "__main__":
    main()
