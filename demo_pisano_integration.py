#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
Demo: Pisano Period Analysis Integration with Signal Experiments

Shows how to add Pisano period classification to existing QA experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pisano_analysis import PisanoClassifier, add_pisano_analysis_to_results, visualize_pisano_distribution


# =============================================================================
# Mock QA System (simulates run_signal_experiments output)
# =============================================================================

class MockQASystem:
    """Simulates a QA system for demonstration."""

    def __init__(self, N: int = 24, modulus: int = 24):
        self.N = N
        self.modulus = modulus

        # Initialize with mixed Pisano families
        self.b = np.zeros(N)
        self.e = np.zeros(N)

        # Distribute nodes across different families
        # First 8 nodes: Fibonacci-like (1,1,2,3)
        for i in range(8):
            self.b[i] = 1
            self.e[i] = 1 + i

        # Next 8 nodes: Lucas-like (2,1,3,4)
        for i in range(8, 16):
            self.b[i] = 2
            self.e[i] = 1 + (i - 8)

        # Next 4 nodes: Phibonacci-like (3,1,4,5)
        for i in range(16, 20):
            self.b[i] = 3
            self.e[i] = 1 + (i - 16)

        # Last 4 nodes: Tribonacci-like (3,3,6,9)
        for i in range(20, 24):
            self.b[i] = 3
            self.e[i] = 3 + (i - 20)

        # Add some noise to make it realistic
        self.b += np.random.randn(N) * 0.1
        self.e += np.random.randn(N) * 0.1

        # Ensure positive values
        self.b = np.abs(self.b)
        self.e = np.abs(self.e)


# =============================================================================
# Demo Execution
# =============================================================================

def main():
    print("=" * 80)
    print("DEMO: PISANO PERIOD ANALYSIS INTEGRATION")
    print("=" * 80)
    print()

    # Create mock QA system
    print("[1] Creating mock QA system with mixed Pisano families...")
    system = MockQASystem(N=24, modulus=24)
    print(f"    ✓ System created: {system.N} nodes, modulus={system.modulus}")
    print()

    # Initialize classifier
    print("[2] Initializing Pisano classifier (mod 9)...")
    classifier = PisanoClassifier(modulus=9)
    print("    ✓ Classifier ready")
    print()

    # Analyze system
    print("[3] Analyzing QA system Pisano periods...")
    results = add_pisano_analysis_to_results(system, 'Demo Signal', classifier)
    print("    ✓ Analysis complete")
    print()

    # Display results
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print()

    print(f"Signal: {results['signal_name']}")
    print(f"Dominant Family: {results['dominant_family'].upper()}")
    print(f"Average Period: {results['avg_period']:.1f}")
    print()

    print("Family Distribution:")
    print("-" * 40)
    for family, count in sorted(results['family_distribution'].items(),
                                key=lambda x: x[1], reverse=True):
        period = PisanoClassifier.PERIOD_MAP.get(family, '?')
        pct = 100 * count / system.N
        print(f"  {family.capitalize():15s}: {count:2d} nodes ({pct:5.1f}%) [Period={period:2}]")

    print()
    print("Mod-9 Residue Distribution:")
    print("-" * 40)
    for residue, count in sorted(results['residue_histogram'].items()):
        print(f"  Residue {residue}: {count:3d} occurrences")

    print()

    # Sample detailed classifications
    print("Sample Node Classifications:")
    print("-" * 40)
    for cls in results['detailed_classifications'][:5]:
        node_id = cls['node_id']
        b_val = system.b[node_id]
        e_val = system.e[node_id]
        print(f"  Node {node_id:2d}: (b={b_val:5.2f}, e={e_val:5.2f}) → "
              f"{cls['family']:12s} (P={cls['period']:2}, conf={cls['confidence']:.1%})")

    print()

    # Visualize
    print("[4] Generating visualization...")
    fig = visualize_pisano_distribution(results,
                                       save_path='phase1_workspace/demo_pisano_analysis.png')
    print("    ✓ Saved to phase1_workspace/demo_pisano_analysis.png")
    print()

    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Integrate into run_signal_experiments_tight_bounds.py")
    print("  2. Add Pisano period to PAC bounds analysis")
    print("  3. Correlate period with generalization performance")
    print()


if __name__ == "__main__":
    main()
