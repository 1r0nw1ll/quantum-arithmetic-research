"""
Five Families Complete Analysis
Rigorous verification of the complete partition of Pythagorean triples
into Fibonacci, Lucas, Tribonacci, Phibonacci, and Ninbonacci families
"""

import numpy as np
import pandas as pd
from collections import defaultdict

def digital_root(n):
    """Compute digital root (1-9, not 0-8)"""
    return 9 if n % 9 == 0 else n % 9

def generate_sequence(initial_terms, n_terms=50):
    """Generate generalized Fibonacci sequence"""
    seq = initial_terms.copy()
    while len(seq) < n_terms:
        seq.append(sum(seq[-len(initial_terms):]))
    return seq[:n_terms]

def analyze_family(sequence, name):
    """Complete analysis of a single family"""
    # Generate (b,e) pairs
    be_pairs = [(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)]

    # Digital roots
    dr_sequence = [digital_root(n) for n in sequence]
    dr_pairs = [(dr_sequence[i], dr_sequence[i+1]) for i in range(len(dr_sequence)-1)]

    # Find period
    unique_dr_pairs = []
    for pair in dr_pairs:
        if pair not in unique_dr_pairs:
            unique_dr_pairs.append(pair)
        else:
            # Found repeat - period detected
            period = len(unique_dr_pairs)
            break
    else:
        period = len(unique_dr_pairs)

    # Generate Pythagorean triples
    pythag_triples = []
    for b, e in be_pairs[:24]:  # First 24 for analysis
        d = b + e
        a = b + 2*e
        C = 2 * d * e
        F = a * b
        G = e**2 + d**2
        pythag_triples.append((C, F, G))

    return {
        'name': name,
        'initial': sequence[:3],
        'be_pairs': be_pairs[:24],
        'dr_sequence': dr_sequence[:25],
        'dr_pairs': dr_pairs[:24],
        'unique_dr_pairs': unique_dr_pairs,
        'period': period,
        'pythag_triples': pythag_triples[:10]
    }

# Generate all five families
print("="*80)
print("FIVE FAMILIES COMPLETE ANALYSIS")
print("="*80)

families = {
    'Fibonacci': analyze_family(generate_sequence([1, 1], 50), 'Fibonacci'),
    'Lucas': analyze_family(generate_sequence([2, 1], 50), 'Lucas'),
    'Tribonacci': analyze_family(generate_sequence([3, 3, 6], 50), 'Tribonacci'),
    'Phibonacci': analyze_family(generate_sequence([3, 1, 4], 50), 'Phibonacci'),
    'Ninbonacci': analyze_family(generate_sequence([9], 50), 'Ninbonacci')
}

# Print detailed analysis for each family
for name, fam in families.items():
    print(f"\n{'='*80}")
    print(f"{name.upper()} FAMILY")
    print(f"{'='*80}")
    print(f"Initial terms: {fam['initial']}")
    print(f"Digital root period: {fam['period']}")
    print(f"\nFirst 10 digital root (b,e) pairs:")
    for i, (drb, dre) in enumerate(fam['dr_pairs'][:10], 1):
        print(f"  {i:2d}. ({drb}, {dre})")

    print(f"\nUnique digital root pairs in orbit:")
    print(f"  {fam['unique_dr_pairs']}")

    print(f"\nFirst 5 Pythagorean triples (C, F, G):")
    for i, (C, F, G) in enumerate(fam['pythag_triples'][:5], 1):
        # Verify it's a valid triple
        is_valid = C**2 + F**2 == G**2
        check = "✓" if is_valid else "✗"
        print(f"  {i}. ({C:6d}, {F:6d}, {G:6d}) {check}")

# Verify complete partition
print(f"\n{'='*80}")
print("COMPLETE PARTITION VERIFICATION")
print(f"{'='*80}")

all_dr_pairs = set()
for name, fam in families.items():
    family_pairs = set(fam['unique_dr_pairs'])
    all_dr_pairs.update(family_pairs)
    print(f"{name:12s}: {len(family_pairs):2d} unique digital root pairs")

print(f"\nTotal unique digital root pairs across all families: {len(all_dr_pairs)}")
print(f"Theoretical maximum (1..9 × 1..9): 81")

# Check for overlaps
print(f"\n{'='*80}")
print("OVERLAP CHECK (should be NONE)")
print(f"{'='*80}")

overlap_found = False
for i, (name1, fam1) in enumerate(families.items()):
    for name2, fam2 in list(families.items())[i+1:]:
        overlap = set(fam1['unique_dr_pairs']) & set(fam2['unique_dr_pairs'])
        if overlap:
            print(f"OVERLAP between {name1} and {name2}: {overlap}")
            overlap_found = True

if not overlap_found:
    print("✓ NO OVERLAPS DETECTED - Each digital root pair belongs to exactly one family")

# Generate classification lookup table
print(f"\n{'='*80}")
print("CLASSIFICATION LOOKUP TABLE")
print(f"{'='*80}")

classification = {}
for name, fam in families.items():
    for pair in fam['unique_dr_pairs']:
        classification[pair] = name

print("\nComplete classification of all (dr(b), dr(e)) pairs:")
for drb in range(1, 10):
    row = []
    for dre in range(1, 10):
        pair = (drb, dre)
        if pair in classification:
            # First letter of family name
            abbrev = {'Fibonacci': 'F', 'Lucas': 'L', 'Tribonacci': 'T',
                     'Phibonacci': 'P', 'Ninbonacci': 'N'}
            row.append(abbrev[classification[pair]])
        else:
            row.append('.')
    print(f"  dr(b)={drb}: " + " ".join(row))

print("\nLegend: F=Fibonacci, L=Lucas, T=Tribonacci, P=Phibonacci, N=Ninbonacci, .=uncovered")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"24-cycle families: {sum(1 for f in families.values() if f['period'] == 24)}")
print(f"8-cycle families:  {sum(1 for f in families.values() if f['period'] == 8)}")
print(f"1-cycle families:  {sum(1 for f in families.values() if f['period'] == 1)}")
print(f"\nCoverage: {len(all_dr_pairs)}/81 possible (dr(b), dr(e)) pairs")
print(f"Uncovered pairs: {81 - len(all_dr_pairs)}")

# Export results
df_data = []
for name, fam in families.items():
    for pair in fam['unique_dr_pairs']:
        df_data.append({'family': name, 'dr_b': pair[0], 'dr_e': pair[1], 'period': fam['period']})

df = pd.DataFrame(df_data)
df.to_csv('/home/player2/signal_experiments/five_families_classification.csv', index=False)
print(f"\n✓ Classification table exported to: five_families_classification.csv")
