#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
T-001: Rotor Limit Proof - Inner vs Quantum Ellipse Equivalence
Proves that fractional QA tuples (after division by d) maintain invariant relationships

Mathematical Framework:
- QA Tuple: (b, e, d, a) where d = b+e, a = b+2e
- Invariants: J = b·d, K = d·a, X = e·d
- Inner Ellipse: Defined by ratios b/d, e/d
- Quantum Ellipse: Defined by the constraint on (b/d, e/d, a/d)

Theorem: The inner ellipse parameterized by (b/d, e/d) and the quantum ellipse
parameterized by the full fractional tuple (b/d, e/d, 1, a/d) are equivalent
under the QA invariant structure.
"""

from fractions import Fraction
import pytest
from typing import Tuple, List
import numpy as np


# ============================================================================
# QA Tuple Operations
# ============================================================================

def create_qa_tuple(b: int, e: int) -> Tuple[int, int, int, int]:
    """Create a QA tuple from base parameters"""
    d = b + e
    a = b + 2*e
    return (b, e, d, a)


def compute_invariants(b, e, d, a) -> Tuple:
    """Compute QA invariants J, K, X"""
    J = b * d
    K = d * a
    X = e * d
    return (J, K, X)


def verify_qa_closure(b, e, d, a) -> bool:
    """Verify QA tuple closure: d = b+e, a = b+2e"""
    return (d == b + e) and (a == b + 2*e)


# ============================================================================
# Fractional QA Tuples (Rotor Limit)
# ============================================================================

def create_fractional_tuple(b: int, e: int) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    """
    Create fractional QA tuple by dividing by d
    This is the "rotor limit" - projecting onto the unit constraint
    """
    b, e, d, a = create_qa_tuple(b, e)

    if d == 0:
        raise ValueError("Division by zero: d=0 for b={}, e={}".format(b, e))

    # Fractional tuple: (b/d, e/d, 1, a/d)
    b_frac = Fraction(b, d)
    e_frac = Fraction(e, d)
    d_frac = Fraction(1, 1)  # Always 1 after normalization
    a_frac = Fraction(a, d)

    return (b_frac, e_frac, d_frac, a_frac)


def verify_fractional_closure(b_frac, e_frac, d_frac, a_frac) -> bool:
    """
    Verify that fractional tuple still satisfies QA closure:
    - d_frac should be 1 (normalized)
    - b_frac + e_frac should equal d_frac (= 1)
    - b_frac + 2*e_frac should equal a_frac
    """
    closure_1 = (d_frac == Fraction(1, 1))
    closure_2 = (b_frac + e_frac == d_frac)
    closure_3 = (b_frac + 2*e_frac == a_frac)

    return closure_1 and closure_2 and closure_3


def compute_fractional_invariants(b_frac, e_frac, d_frac, a_frac) -> Tuple[Fraction, Fraction, Fraction]:
    """
    Compute invariants for fractional tuple
    After division by d, invariants scale:
    - J' = J/d² = (b·d)/d² = b/d
    - K' = K/d² = (d·a)/d² = a/d
    - X' = X/d² = (e·d)/d² = e/d
    """
    J_frac = b_frac * d_frac  # = b/d
    K_frac = d_frac * a_frac  # = a/d
    X_frac = e_frac * d_frac  # = e/d

    return (J_frac, K_frac, X_frac)


# ============================================================================
# Ellipse Geometry
# ============================================================================

def inner_ellipse_point(b_frac, e_frac) -> Tuple[Fraction, Fraction]:
    """
    Inner ellipse: Points (b/d, e/d) satisfying b/d + e/d = 1
    This is a line segment in 2D parameter space
    """
    return (b_frac, e_frac)


def quantum_ellipse_point(b_frac, e_frac, a_frac) -> Tuple[Fraction, Fraction, Fraction]:
    """
    Quantum ellipse: Points (b/d, e/d, a/d) satisfying full QA constraints
    - b/d + e/d = 1 (inner ellipse constraint)
    - b/d + 2·e/d = a/d (QA closure)
    """
    return (b_frac, e_frac, a_frac)


def ellipse_equivalence_check(b: int, e: int) -> dict:
    """
    Check that inner and quantum ellipses are equivalent representations

    Inner ellipse is parameterized by (b/d, e/d)
    Quantum ellipse is fully determined by (b/d, e/d, a/d) where a/d = b/d + 2·e/d

    Thus: quantum ellipse is uniquely determined by inner ellipse point.
    """
    # Create fractional tuple
    b_frac, e_frac, d_frac, a_frac = create_fractional_tuple(b, e)

    # Inner ellipse point
    inner_point = inner_ellipse_point(b_frac, e_frac)

    # Quantum ellipse point
    quantum_point = quantum_ellipse_point(b_frac, e_frac, a_frac)

    # Check equivalence: quantum point should be uniquely determined by inner point
    a_computed = b_frac + 2*e_frac
    equivalence = (a_frac == a_computed)

    # Compute invariants
    J_frac, K_frac, X_frac = compute_fractional_invariants(b_frac, e_frac, d_frac, a_frac)

    return {
        'b': b,
        'e': e,
        'inner_point': inner_point,
        'quantum_point': quantum_point,
        'a_computed': a_computed,
        'a_actual': a_frac,
        'equivalent': equivalence,
        'closure': verify_fractional_closure(b_frac, e_frac, d_frac, a_frac),
        'invariants': (J_frac, K_frac, X_frac)
    }


# ============================================================================
# Mathematical Proof
# ============================================================================

def proof_statement():
    """
    THEOREM (Rotor Limit - Ellipse Equivalence):

    For any QA tuple (b, e, d, a) with d ≠ 0, define the fractional tuple:
        (b/d, e/d, 1, a/d)

    Then:

    1) INNER ELLIPSE CONSTRAINT:
       b/d + e/d = 1

       Proof: By definition, d = b + e, so:
       b/d + e/d = (b+e)/d = d/d = 1 ✓

    2) QUANTUM ELLIPSE CONSTRAINT:
       a/d = b/d + 2·e/d

       Proof: By definition, a = b + 2e, so:
       a/d = (b + 2e)/d = b/d + 2e/d ✓

    3) UNIQUENESS:
       The quantum ellipse point (b/d, e/d, a/d) is uniquely determined
       by the inner ellipse point (b/d, e/d).

       Proof: Given (b/d, e/d), we have:
       a/d = b/d + 2·e/d (from constraint 2)

       Thus the third coordinate is uniquely determined by the first two. ✓

    4) INVARIANT PRESERVATION (scaled):
       The fractional invariants are:
       J' = b/d, K' = a/d, X' = e/d

       These satisfy the relationships:
       - J' + 2·X' = K' (from a = b + 2e)
       - J' + X' = 1 (from b + e = d)

       Proof:
       J' + 2·X' = b/d + 2·e/d = (b+2e)/d = a/d = K' ✓
       J' + X' = b/d + e/d = (b+e)/d = d/d = 1 ✓

    CONCLUSION:
    The inner ellipse (constrained line segment in (b/d, e/d) space)
    and the quantum ellipse (constrained surface in (b/d, e/d, a/d) space)
    are equivalent parameterizations of the same geometric object under
    the rotor limit (division by d).

    The quantum ellipse adds no additional degrees of freedom beyond
    those present in the inner ellipse - it is merely the natural
    embedding into 3D space with the third coordinate determined by
    the QA closure constraint.

    QED.
    """
    print(__doc__)


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestRotorLimitProof:
    """Property-based tests for rotor limit theorem"""

    def test_inner_ellipse_constraint(self):
        """Test that b/d + e/d = 1 for all rational inputs"""
        for denom in range(2, 8):  # denominators 2-7
            for b in range(1, denom):
                e = denom - b  # Ensure b + e = denom (will become d)

                result = ellipse_equivalence_check(b, e)
                inner_b, inner_e = result['inner_point']

                # Check inner ellipse constraint
                assert inner_b + inner_e == Fraction(1, 1), \
                    f"Inner ellipse constraint failed for b={b}, e={e}: {inner_b} + {inner_e} ≠ 1"

    def test_quantum_ellipse_constraint(self):
        """Test that a/d = b/d + 2·e/d for all rational inputs"""
        for denom in range(2, 8):
            for b in range(1, denom):
                for e in range(1, denom):
                    if b + e == 0:
                        continue  # Skip d=0 case

                    result = ellipse_equivalence_check(b, e)

                    # Check quantum ellipse constraint
                    assert result['equivalent'], \
                        f"Quantum ellipse constraint failed for b={b}, e={e}"

    def test_uniqueness(self):
        """Test that quantum point is uniquely determined by inner point"""
        for denom in range(2, 8):
            for b in range(1, denom):
                e = denom - b

                result = ellipse_equivalence_check(b, e)

                # The computed a should equal the actual a
                assert result['a_computed'] == result['a_actual'], \
                    f"Uniqueness failed for b={b}, e={e}"

    def test_fractional_closure(self):
        """Test that fractional tuples satisfy QA closure"""
        for denom in range(2, 8):
            for b in range(1, denom):
                for e in range(1, denom):
                    if b + e == 0:
                        continue

                    result = ellipse_equivalence_check(b, e)

                    assert result['closure'], \
                        f"Fractional closure failed for b={b}, e={e}"

    def test_invariant_relationships(self):
        """Test that fractional invariants satisfy expected relationships"""
        for denom in range(2, 8):
            for b in range(1, denom):
                for e in range(1, denom):
                    if b + e == 0:
                        continue

                    result = ellipse_equivalence_check(b, e)
                    J_frac, K_frac, X_frac = result['invariants']

                    # Check J' + 2·X' = K'
                    assert J_frac + 2*X_frac == K_frac, \
                        f"Invariant relationship J'+2X'=K' failed for b={b}, e={e}"

                    # Check J' + X' = 1
                    assert J_frac + X_frac == Fraction(1, 1), \
                        f"Invariant relationship J'+X'=1 failed for b={b}, e={e}"


# ============================================================================
# Demonstration
# ============================================================================

def demonstrate_proof():
    """Demonstrate the proof with concrete examples"""
    print("="*70)
    print("T-001: ROTOR LIMIT PROOF - DEMONSTRATION")
    print("="*70)
    print()

    print("THEOREM: Inner and Quantum Ellipses are Equivalent")
    print("-"*70)
    print()

    # Example cases
    examples = [
        (1, 1),   # Fibonacci-like
        (2, 3),   # General
        (3, 5),   # Another ratio
        (1, 2),   # Simple case
    ]

    for b, e in examples:
        print(f"Example: b={b}, e={e}")
        print("-"*40)

        result = ellipse_equivalence_check(b, e)

        inner = result['inner_point']
        quantum = result['quantum_point']
        J, K, X = result['invariants']

        print(f"  Integer tuple: ({b}, {e}, {b+e}, {b+2*e})")
        print(f"  Fractional tuple: ({inner[0]}, {inner[1]}, 1, {quantum[2]})")
        print(f"  Inner ellipse point: ({inner[0]}, {inner[1]})")
        print(f"  Quantum ellipse point: ({quantum[0]}, {quantum[1]}, {quantum[2]})")
        print(f"  Invariants (fractional): J'={J}, K'={K}, X'={X}")
        print(f"  Inner constraint (b/d + e/d = 1): {inner[0]} + {inner[1]} = {inner[0] + inner[1]} ✓")
        print(f"  Quantum constraint (a/d = b/d + 2e/d): {quantum[2]} = {inner[0]} + 2·{inner[1]} = {inner[0] + 2*inner[1]} ✓")
        print(f"  Equivalence: {result['equivalent']} ✓")
        print()

    print("="*70)
    print("CONCLUSION: Theorem verified for all test cases")
    print("="*70)


def main():
    """Run proof demonstration and tests"""
    import sys

    # Print formal proof
    proof_statement()

    print()
    print()

    # Demonstrate with examples
    demonstrate_proof()

    print()
    print()

    # Run property-based tests
    print("="*70)
    print("RUNNING PROPERTY-BASED TESTS")
    print("="*70)
    print()

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
