"""
Extract failure algebra counts for QA Time paper.
Mechanical extraction - raw counts only, no analysis.
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments/Formalizing tuple drift in quantum-native learning/files')

from qa_oracle import QAOracle, construct_qa_state, FailType
from collections import defaultdict

def extract_failure_counts():
    """Extract raw failure counts for Caps(30,30) under Σ₀"""

    oracle = QAOracle(30, q_def="none")

    # Σ₀ = {σ, μ, λ₂}
    generators = [
        ("sigma", 2),
        ("mu", 2),
        ("lambda2", 2)
    ]

    # Enumerate all Caps(30,30) states
    states = []
    for b in range(1, 31):
        for e in range(1, 31):
            states.append(construct_qa_state(b, e))

    # Count failures by type
    fail_counts = defaultdict(int)

    for state in states:
        for move, k_param in generators:
            if not oracle.is_legal(state, move, k_param):
                fail_type = oracle.get_fail_type(state, move, k_param)
                if fail_type:
                    fail_counts[fail_type.value] += 1

    return fail_counts

if __name__ == "__main__":
    print("="*60)
    print("FAILURE ALGEBRA EXTRACTION - QA TIME PAPER")
    print("="*60)
    print()
    print("Manifold: Caps(30,30)")
    print("Generators: Σ₀ = {σ, μ, λ₂}")
    print()
    print("="*60)
    print("RAW FAILURE COUNTS")
    print("="*60)

    counts = extract_failure_counts()

    # Map to paper terminology
    paper_names = {
        "OUT_OF_BOUNDS": "OUT_OF_BOUNDS",
        "PARITY": "PARITY",
        "PHASE_VIOLATION": "FIXED_Q",
        "INVARIANT": "INVARIANT_BREAK",
        "REDUCTION": "NON_REDUCTION"
    }

    # Print all 5 types (zeros if not present)
    for internal_name, paper_name in paper_names.items():
        count = counts.get(internal_name, 0)
        print(f"{paper_name:<20} {count:>8}")

    print("="*60)
    print(f"TOTAL FAILURES: {sum(counts.values())}")
    print("="*60)
