#!/usr/bin/env python3
"""
QA Certified Obstruction: 24-cycle ↛ 8-cycle under Fibonacci generators

This test validates the mod-3 orbit separation theorem:
- Tribonacci (8-cycle) requires both digital roots ≡ 0 (mod 3)
- Under Fibonacci step (b,e)→(e,b+e), the state (0,0) mod 3 is isolated
- Therefore: 24-cycle families can NEVER transition to 8-cycle

This is a structural impossibility, not a search result.
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments')


def test_mod3_fixed_point_isolation():
    """
    Verify that (0,0) mod 3 is only reachable from itself under Fibonacci step.

    This is the algebraic foundation of the 24→8 orbit separation.
    """
    # Fibonacci step: (b, e) → (e, b+e) mod 3
    def fib_step_mod3(b, e):
        return (e, (b + e) % 3)

    # Build reachability from all non-(0,0) states
    states = [(b, e) for b in range(3) for e in range(3) if not (b == 0 and e == 0)]

    for start_b, start_e in states:
        b, e = start_b, start_e
        visited = set()

        # Run for enough steps to detect any cycle (max 9 states)
        for _ in range(20):
            if (b, e) in visited:
                break
            visited.add((b, e))

            # Check: did we reach (0, 0)?
            assert (b, e) != (0, 0), (
                f"OBSTRUCTION VIOLATED: ({start_b}, {start_e}) reached (0, 0) mod 3"
            )

            b, e = fib_step_mod3(b, e)

    print("✓ Mod-3 fixed point isolation verified: (0,0) unreachable from other states")


def test_tribonacci_requires_both_mod3_zero():
    """
    Verify that all Tribonacci (8-cycle) digital root pairs have both components ≡ 0 (mod 3).

    Also enforces:
    - Exactly 8 pairs (canonical definition)
    - (9,9) excluded (belongs to Ninbonacci fixed point)
    """
    from qa_harmonicity_v2 import PISANO_FAMILY_MAP

    tribonacci_pairs = [
        (dr_b, dr_e) for (dr_b, dr_e), family in PISANO_FAMILY_MAP.items()
        if family == 'Tribonacci'
    ]

    # Canonical count: exactly 8 pairs
    assert len(tribonacci_pairs) == 8, (
        f"Tribonacci should have exactly 8 pairs, found {len(tribonacci_pairs)}: {tribonacci_pairs}"
    )

    # (9,9) must NOT be in Tribonacci - it's the Ninbonacci fixed point
    assert (9, 9) not in tribonacci_pairs, (
        "(9,9) should be Ninbonacci (fixed point), not Tribonacci"
    )

    for dr_b, dr_e in tribonacci_pairs:
        assert dr_b % 3 == 0, f"Tribonacci pair ({dr_b}, {dr_e}) has dr_b ≢ 0 (mod 3)"
        assert dr_e % 3 == 0, f"Tribonacci pair ({dr_b}, {dr_e}) has dr_e ≢ 0 (mod 3)"

    print(f"✓ Tribonacci: exactly {len(tribonacci_pairs)} pairs, all ≡ 0 (mod 3), (9,9) excluded")


def test_24_cycle_families_have_nonzero_mod3():
    """
    Verify that 24-cycle families (Fibonacci/Lucas/Phibonacci) have at least one
    component ≢ 0 (mod 3) in their digital root pairs.
    """
    from qa_harmonicity_v2 import PISANO_FAMILY_MAP

    families_24 = ['Fibonacci', 'Lucas', 'Phibonacci']

    for family in families_24:
        pairs = [
            (dr_b, dr_e) for (dr_b, dr_e), fam in PISANO_FAMILY_MAP.items()
            if fam == family
        ]

        assert len(pairs) > 0, f"No pairs found for {family} family"

        for dr_b, dr_e in pairs:
            # At least one must be ≢ 0 (mod 3)
            both_zero = (dr_b % 3 == 0) and (dr_e % 3 == 0)
            assert not both_zero, (
                f"{family} pair ({dr_b}, {dr_e}) has BOTH components ≡ 0 (mod 3) - "
                f"this would allow 24→8 transition!"
            )

        print(f"✓ {family}: all {len(pairs)} pairs have at least one component ≢ 0 (mod 3)")


def test_ninbonacci_is_fixed_point():
    """
    Verify Ninbonacci is exactly the (9,9) fixed point.
    """
    from qa_harmonicity_v2 import PISANO_FAMILY_MAP

    ninbonacci_pairs = [
        (dr_b, dr_e) for (dr_b, dr_e), family in PISANO_FAMILY_MAP.items()
        if family == 'Ninbonacci'
    ]

    assert ninbonacci_pairs == [(9, 9)], (
        f"Ninbonacci should be exactly [(9,9)], found {ninbonacci_pairs}"
    )

    print("✓ Ninbonacci: exactly (9,9) - the fixed point")


def test_orbit_separation_theorem():
    """
    Combined test: the 24-cycle and 8-cycle families are algebraically disconnected.
    """
    # Run component tests
    test_mod3_fixed_point_isolation()
    test_tribonacci_requires_both_mod3_zero()
    test_ninbonacci_is_fixed_point()
    test_24_cycle_families_have_nonzero_mod3()

    print()
    print("=" * 70)
    print("CERTIFIED OBSTRUCTION: 24-cycle ↛ 8-cycle Orbit Separation")
    print("=" * 70)
    print()
    print("Theorem: Under Fibonacci-type generators (b,e)→(e,b+e), no sequence")
    print("of steps can map a 24-cycle family state to an 8-cycle family state.")
    print()
    print("Proof: The mod-3 residue class (0,0) is a fixed point, reachable only")
    print("from itself. Tribonacci requires both components ≡ 0 (mod 3), while")
    print("24-cycle families always have at least one component ≢ 0 (mod 3).")
    print()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
