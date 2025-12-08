#!/usr/bin/env python3
"""
Pisano Period Analysis for Quantum Arithmetic Systems

Analyzes mod-9 and mod-24 residue patterns to classify QA tuples
into periodic families (Fibonacci, Lucas, Phibonacci, Tribonacci, Ninbonacci).

Based on findings from: docs/ai_chats/QA system and Pisano periods.md
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from collections import Counter


# =============================================================================
# Pisano Period Classification
# =============================================================================

class PisanoClassifier:
    """
    Classifies QA tuples by their Pisano period under mod-9 and mod-24.

    Period Families (mod 9):
    - 24-period: Fibonacci (1,1,2,3), Lucas (2,1,3,4), Phibonacci (3,1,4,5)
    - 8-period: Tribonacci (3,3,6,9)
    - 1-period: Ninbonacci (9,9,9,9)
    """

    # Known seed tuples for each family
    SEEDS = {
        'fibonacci': (1, 1, 2, 3),
        'lucas': (2, 1, 3, 4),
        'phibonacci': (3, 1, 4, 5),
        'tribonacci': (3, 3, 6, 9),
        'ninbonacci': (9, 9, 9, 9)  # Degenerate case (fixed point mod 9)
    }

    PERIOD_MAP = {
        'fibonacci': 24,
        'lucas': 24,
        'phibonacci': 24,
        'tribonacci': 8,
        'ninbonacci': 1
    }

    def __init__(self, modulus: int = 9):
        """
        Args:
            modulus: Modulus for residue computation (9 or 24)
        """
        self.modulus = modulus
        self._generate_period_sequences()

    def _generate_period_sequences(self, max_length: int = 50):
        """
        Generate mod-9 sequences for each family to identify patterns.

        Args:
            max_length: Maximum sequence length to generate
        """
        self.sequences = {}

        for family, seed in self.SEEDS.items():
            seq = [tuple(x % self.modulus for x in seed)]

            # Generate sequence by evolving (b,e) → (b, e+1) or similar rule
            b, e, d, a = seed
            for _ in range(max_length - 1):
                # Simple evolution: increment e
                e_new = e + 1
                d_new = b + e_new
                a_new = b + 2 * e_new

                mod_tuple = (b % self.modulus,
                            e_new % self.modulus,
                            d_new % self.modulus,
                            a_new % self.modulus)

                seq.append(mod_tuple)

                # Check for period (return to start)
                if mod_tuple == seq[0] and len(seq) > 1:
                    break

                e = e_new

            self.sequences[family] = seq

    def mod_residues(self, b: float, e: float, d: float, a: float) -> Tuple[int, int, int, int]:
        """
        Compute modular residues of QA tuple.

        Args:
            b, e, d, a: QA tuple components

        Returns:
            Tuple of residues (b_mod, e_mod, d_mod, a_mod)
        """
        return (
            int(b) % self.modulus,
            int(e) % self.modulus,
            int(d) % self.modulus,
            int(a) % self.modulus
        )

    def classify_tuple(self, b: float, e: float, d: float, a: float) -> Dict:
        """
        Classify a QA tuple by Pisano period family.

        Args:
            b, e, d, a: QA tuple components

        Returns:
            Dictionary with classification results:
            - family: Detected family name or 'unknown'
            - period: Expected Pisano period
            - residues: Mod-9 residues
            - confidence: Match confidence (0-1)
        """
        residues = self.mod_residues(b, e, d, a)

        # Check against known seeds (initial match)
        best_match = 'unknown'
        best_confidence = 0.0

        for family, seed in self.SEEDS.items():
            seed_residues = tuple(x % self.modulus for x in seed)

            # Exact match
            if residues == seed_residues:
                return {
                    'family': family,
                    'period': self.PERIOD_MAP[family],
                    'residues': residues,
                    'confidence': 1.0
                }

            # Partial match (count matching components)
            matches = sum(r == s for r, s in zip(residues, seed_residues))
            confidence = matches / 4.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = family

        # If no good match, estimate period by checking sequence membership
        if best_confidence < 0.5:
            for family, seq in self.sequences.items():
                if residues in seq:
                    return {
                        'family': family,
                        'period': self.PERIOD_MAP[family],
                        'residues': residues,
                        'confidence': 0.8
                    }

        return {
            'family': best_match,
            'period': self.PERIOD_MAP.get(best_match, 'unknown'),
            'residues': residues,
            'confidence': best_confidence
        }

    def analyze_system(self, system) -> Dict:
        """
        Analyze an entire QA system (from run_signal_experiments).

        Args:
            system: QASystem object with attributes b, e, d, a (arrays)

        Returns:
            Dictionary with aggregate statistics:
            - family_distribution: Counter of detected families
            - avg_period: Average detected period
            - residue_histogram: Distribution of mod-9 residues
            - detailed_classifications: List of per-node classifications
        """
        N = len(system.b)
        classifications = []

        for i in range(N):
            # Compute d and a if not stored
            d = (system.b[i] + system.e[i]) % system.modulus
            a = (system.b[i] + 2 * system.e[i]) % system.modulus

            cls = self.classify_tuple(system.b[i], system.e[i], d, a)
            cls['node_id'] = i
            classifications.append(cls)

        # Aggregate statistics
        family_counts = Counter(cls['family'] for cls in classifications)
        periods = [cls['period'] for cls in classifications if isinstance(cls['period'], int)]
        avg_period = np.mean(periods) if periods else 0

        # Residue histogram (flattened across all components)
        all_residues = []
        for cls in classifications:
            all_residues.extend(cls['residues'])
        residue_hist = Counter(all_residues)

        return {
            'family_distribution': dict(family_counts),
            'avg_period': avg_period,
            'residue_histogram': dict(residue_hist),
            'detailed_classifications': classifications,
            'dominant_family': family_counts.most_common(1)[0][0] if family_counts else 'unknown'
        }


# =============================================================================
# Visualization
# =============================================================================

def visualize_pisano_distribution(analysis_results: Dict, save_path: str = None):
    """
    Visualize Pisano period classification results.

    Args:
        analysis_results: Output from PisanoClassifier.analyze_system()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Family distribution
    ax = axes[0, 0]
    families = list(analysis_results['family_distribution'].keys())
    counts = list(analysis_results['family_distribution'].values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    bars = ax.bar(families, counts, color=colors[:len(families)])
    ax.set_xlabel('Pisano Family', fontsize=12)
    ax.set_ylabel('Node Count', fontsize=12)
    ax.set_title('QA System: Pisano Period Family Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add period labels on bars
    for bar, family in zip(bars, families):
        period = PisanoClassifier.PERIOD_MAP.get(family, '?')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'P={period}', ha='center', fontsize=10, fontweight='bold')

    # 2. Residue histogram (mod 9)
    ax = axes[0, 1]
    residues = sorted(analysis_results['residue_histogram'].keys())
    res_counts = [analysis_results['residue_histogram'][r] for r in residues]

    ax.bar(residues, res_counts, color='#45B7D1', alpha=0.7)
    ax.set_xlabel('Residue (mod 9)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Mod-9 Residue Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(residues)
    ax.grid(axis='y', alpha=0.3)

    # 3. Confidence distribution
    ax = axes[1, 0]
    confidences = [cls['confidence'] for cls in analysis_results['detailed_classifications']]
    ax.hist(confidences, bins=20, color='#FFA07A', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(confidences):.3f}')
    ax.set_xlabel('Classification Confidence', fontsize=12)
    ax.set_ylabel('Node Count', fontsize=12)
    ax.set_title('Pisano Classification Confidence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    QA SYSTEM PISANO ANALYSIS SUMMARY
    ════════════════════════════════════

    Dominant Family: {analysis_results['dominant_family'].upper()}
    Average Period: {analysis_results['avg_period']:.1f}

    Family Breakdown:
    """

    for family, count in analysis_results['family_distribution'].items():
        period = PisanoClassifier.PERIOD_MAP.get(family, '?')
        pct = 100 * count / sum(analysis_results['family_distribution'].values())
        summary_text += f"    • {family.capitalize():12s}: {count:3d} nodes ({pct:5.1f}%) [P={period}]\n"

    summary_text += f"\n    Total Nodes: {sum(analysis_results['family_distribution'].values())}"

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Pisano analysis to {save_path}")

    return fig


# =============================================================================
# Integration with Existing Experiments
# =============================================================================

def add_pisano_analysis_to_results(system, signal_name: str,
                                   classifier: PisanoClassifier = None) -> Dict:
    """
    Add Pisano period analysis to existing experiment results.

    Args:
        system: QASystem from run_signal_experiments
        signal_name: Name of signal being analyzed
        classifier: Optional PisanoClassifier (creates new one if None)

    Returns:
        Dictionary with Pisano analysis results
    """
    if classifier is None:
        classifier = PisanoClassifier(modulus=9)

    analysis = classifier.analyze_system(system)
    analysis['signal_name'] = signal_name

    return analysis


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PISANO PERIOD ANALYSIS FOR QUANTUM ARITHMETIC SYSTEMS")
    print("=" * 80)
    print()

    # Initialize classifier
    classifier = PisanoClassifier(modulus=9)

    # Test on known seeds
    print("Testing known Pisano period seeds:")
    print("-" * 80)

    for family, seed in PisanoClassifier.SEEDS.items():
        b, e, d, a = seed
        result = classifier.classify_tuple(b, e, d, a)
        print(f"\n{family.upper():15s}: ({b},{e},{d},{a})")
        print(f"  Residues (mod 9): {result['residues']}")
        print(f"  Detected Family: {result['family']}")
        print(f"  Period: {result['period']}")
        print(f"  Confidence: {result['confidence']:.2%}")

    print("\n" + "=" * 80)
    print("MODULE READY FOR INTEGRATION")
    print("=" * 80)
    print()
    print("Usage:")
    print("  from pisano_analysis import PisanoClassifier, add_pisano_analysis_to_results")
    print("  classifier = PisanoClassifier(modulus=9)")
    print("  results = add_pisano_analysis_to_results(qa_system, 'Pure Tone')")
    print()
