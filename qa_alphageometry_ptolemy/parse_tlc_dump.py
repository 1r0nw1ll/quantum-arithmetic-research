#!/usr/bin/env python3
"""
Parse TLC state dump files and count failure types.
"""

import re
from collections import defaultdict

def parse_tlc_dump(filename):
    """Parse TLC dump file into list of state dicts."""
    states = []
    current_state = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('State '):
                # New state - save previous if exists
                if current_state:
                    states.append(current_state)
                current_state = {}

            elif line.startswith('/\\'):
                # Parse variable assignment: /\ varname = value
                match = re.match(r'/\\\s+(\w+)\s+=\s+(.+)', line)
                if match:
                    var_name = match.group(1)
                    var_value = match.group(2).strip('"')  # Remove quotes from strings
                    current_state[var_name] = var_value

        # Don't forget last state
        if current_state:
            states.append(current_state)

    return states

def analyze_states(states, label):
    """Analyze states and count by fail type and lastMove."""
    fail_counts = defaultdict(int)
    move_counts = defaultdict(int)
    fail_by_move = defaultdict(lambda: defaultdict(int))

    for state in states:
        fail = state.get('fail', 'UNKNOWN')
        move = state.get('lastMove', 'UNKNOWN')

        fail_counts[fail] += 1
        move_counts[move] += 1
        fail_by_move[fail][move] += 1

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Total states: {len(states)}")
    print()

    print("Failure Type Distribution:")
    print("-" * 70)
    for fail_type in sorted(fail_counts.keys()):
        count = fail_counts[fail_type]
        pct = 100.0 * count / len(states)
        print(f"  {fail_type:25s}: {count:5d} ({pct:5.1f}%)")
    print()

    print("Last Move Distribution:")
    print("-" * 70)
    for move in sorted(move_counts.keys()):
        count = move_counts[move]
        pct = 100.0 * count / len(states)
        print(f"  {move:25s}: {count:5d} ({pct:5.1f}%)")
    print()

    print("Failure × Move Crosstab:")
    print("-" * 70)
    for fail_type in sorted(fail_counts.keys()):
        print(f"{fail_type}:")
        for move in sorted(fail_by_move[fail_type].keys()):
            count = fail_by_move[fail_type][move]
            print(f"    {move:20s}: {count:5d}")

    return fail_counts, move_counts, fail_by_move

def main():
    print("="*70)
    print("TLC State Dump Analysis - QA/QARM Failure Counts")
    print("="*70)

    # Parse both dumps
    states_full = parse_tlc_dump('states_full.txt.dump')
    states_nomu = parse_tlc_dump('states_nomu.txt.dump')

    # Analyze
    fail_full, move_full, cross_full = analyze_states(states_full, "Run 1: Generator Set {σ, μ, λ}")
    fail_nomu, move_nomu, cross_nomu = analyze_states(states_nomu, "Run 2: Generator Set {σ, λ} - NO μ")

    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON: Failure Count Invariance Test")
    print(f"{'='*70}")
    print()

    all_fail_types = set(fail_full.keys()) | set(fail_nomu.keys())

    print(f"{'Failure Type':<25} {'Full {σ,μ,λ}':>12} {'No-μ {σ,λ}':>12} {'Δ':>8} {'Status':>10}")
    print("-" * 70)

    invariant = True
    for fail_type in sorted(all_fail_types):
        count_full = fail_full.get(fail_type, 0)
        count_nomu = fail_nomu.get(fail_type, 0)
        delta = count_full - count_nomu

        if fail_type == "OK":
            # OK states expected to change with generator set
            status = "(varies)"
        elif delta == 0:
            status = "✅ INVARIANT"
        else:
            status = "❌ CHANGED"
            if fail_type != "OK":
                invariant = False

        print(f"{fail_type:<25} {count_full:>12} {count_nomu:>12} {delta:>8} {status:>10}")

    print()
    print("="*70)
    if invariant:
        print("✅ FAILURE COUNTS ARE INVARIANT (excluding OK states)")
        print("   Hypothesis CONFIRMED by TLC model checking.")
    else:
        print("❌ FAILURE COUNTS CHANGED")
        print("   Hypothesis REJECTED by TLC model checking.")
    print("="*70)

if __name__ == "__main__":
    main()
