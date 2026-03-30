#!/usr/bin/env python3
"""
V3 Validator Stress Test

Demonstrates that:
1. Current EEG demo FAILS v3 validation (cohens_d missing stats)
2. Fix Path A: Add required stats → PASSES
3. Fix Path B: Mark verifiable=false → PASSES with warnings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    UnderstandingCertificate,
    DerivationWitness,
    Strategy,
    validate_certificate_strict_v3,
    StrictValidationResult,
)


def create_eeg_style_cert_original():
    """Create certificate similar to EEG demo (will fail v3)."""
    return UnderstandingCertificate(
        target="EEG seizure detection",
        derived_invariants={
            "threshold_E": 186,
            "effect_size_E": 92,
        },
        derivation_witnesses=[
            DerivationWitness(
                invariant_name="threshold_E",
                derivation_operator="midpoint_threshold",
                input_data={
                    "baseline_mean": 84.12,
                    "seizure_mean": 288.43,
                },
                output_value=186,
                verifiable=True,
            ),
            DerivationWitness(
                invariant_name="effect_size_E",
                derivation_operator="cohens_d",
                input_data={
                    "n_baseline": 100,
                    "n_seizure": 100,
                    # MISSING: baseline_mean, seizure_mean, stds
                },
                output_value=92,
                verifiable=True,  # Claims verifiable but lacks data!
            ),
        ],
        strategy=Strategy(
            type="threshold_voting",
            key_insight="Classify using discriminative invariants",
            derivation_witness=DerivationWitness(
                invariant_name="strategy:threshold_voting",
                derivation_operator="discriminative_analysis",
                input_data={"method": "effect_size > 0.5"},
                output_value=1,
            ),
        ),
        strict_mode=True,
    )


def create_eeg_style_cert_fix_a():
    """Fix Path A: Add required stats for cohens_d."""
    return UnderstandingCertificate(
        target="EEG seizure detection (Fixed A)",
        derived_invariants={
            "threshold_E": 186,
            "effect_size_E": 92,
        },
        derivation_witnesses=[
            DerivationWitness(
                invariant_name="threshold_E",
                derivation_operator="midpoint_threshold",
                input_data={
                    "baseline_mean": 84.12,
                    "seizure_mean": 288.43,
                },
                output_value=186,
                verifiable=True,
            ),
            DerivationWitness(
                invariant_name="effect_size_E",
                derivation_operator="cohens_d",
                input_data={
                    "n_baseline": 100,
                    "n_seizure": 100,
                    "baseline_mean": 84.12,
                    "seizure_mean": 288.43,
                    "baseline_std": 45.2,  # ADDED
                    "seizure_std": 52.1,   # ADDED
                },
                output_value=92,
                verifiable=True,
            ),
        ],
        strategy=Strategy(
            type="threshold_voting",
            key_insight="Classify using discriminative invariants",
            derivation_witness=DerivationWitness(
                invariant_name="strategy:threshold_voting",
                derivation_operator="discriminative_analysis",
                input_data={"method": "effect_size > 0.5"},
                output_value=1,
            ),
        ),
        strict_mode=True,
    )


def create_eeg_style_cert_fix_b():
    """Fix Path B: Mark cohens_d as non-verifiable (downgraded claim)."""
    return UnderstandingCertificate(
        target="EEG seizure detection (Fixed B)",
        derived_invariants={
            "threshold_E": 186,
            "effect_size_E": 92,
        },
        derivation_witnesses=[
            DerivationWitness(
                invariant_name="threshold_E",
                derivation_operator="midpoint_threshold",
                input_data={
                    "baseline_mean": 84.12,
                    "seizure_mean": 288.43,
                },
                output_value=186,
                verifiable=True,
            ),
            DerivationWitness(
                invariant_name="effect_size_E",
                derivation_operator="cohens_d",
                input_data={
                    "n_baseline": 100,
                    "n_seizure": 100,
                },
                output_value=92,
                verifiable=False,  # DOWNGRADED - explicitly non-verifiable
            ),
        ],
        strategy=Strategy(
            type="threshold_voting",
            key_insight="Classify using discriminative invariants",
            derivation_witness=DerivationWitness(
                invariant_name="strategy:threshold_voting",
                derivation_operator="discriminative_analysis",
                input_data={"method": "effect_size > 0.5"},
                output_value=1,
            ),
        ),
        strict_mode=True,
    )


def main():
    print("\n" + "=" * 70)
    print("  V3 VALIDATOR STRESS TEST")
    print("  Demonstrating operator-specific verification rules")
    print("=" * 70)

    # Test 1: Original (should FAIL)
    print("\n" + "─" * 70)
    print("TEST 1: Original EEG-style certificate (cohens_d missing stats)")
    print("─" * 70)

    cert1 = create_eeg_style_cert_original()
    result1 = validate_certificate_strict_v3(cert1)

    print(f"\n{result1.summary()}")

    if not result1.valid:
        print("\n✓ EXPECTED: Certificate correctly flagged as INVALID")
        print("  → cohens_d claims verifiable but lacks std/pooled stats")
    else:
        print("\n✗ UNEXPECTED: Certificate passed when it should fail!")

    # Test 2: Fix A (add stats)
    print("\n" + "─" * 70)
    print("TEST 2: Fix Path A - Add required stats for cohens_d")
    print("─" * 70)

    cert2 = create_eeg_style_cert_fix_a()
    result2 = validate_certificate_strict_v3(cert2)

    print(f"\n{result2.summary()}")

    if result2.valid and len(result2.warnings) == 0:
        print("\n✓ EXPECTED: Certificate PASSES with full stats")
    else:
        print("\n✗ UNEXPECTED: Certificate failed or has warnings!")

    # Test 3: Fix B (mark non-verifiable)
    print("\n" + "─" * 70)
    print("TEST 3: Fix Path B - Mark cohens_d as verifiable=false")
    print("─" * 70)

    cert3 = create_eeg_style_cert_fix_b()
    result3 = validate_certificate_strict_v3(cert3)

    print(f"\n{result3.summary()}")

    if result3.valid and len(result3.warnings) == 1:
        print("\n✓ EXPECTED: Certificate PASSES with warning (downgraded claim)")
    else:
        print("\n✗ UNEXPECTED: Certificate failed or wrong warning count!")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
    The v3 strict validator enforces:

    1. cohens_d witnesses marked verifiable=true MUST include:
       - (baseline_std, seizure_std, baseline_mean, seizure_mean, n_*) OR
       - (pooled_std, baseline_mean, seizure_mean) OR
       - (pooled_variance, baseline_mean, seizure_mean)

    2. midpoint_threshold witnesses are RECOMPUTED and verified

    3. Strategy must have actual derivation_witness object (not just flag)

    Two valid remediation paths:
    A) Include full stats → verifiable=true is valid
    B) Mark verifiable=false → downgraded claim, passes with warning

    This prevents "gaming" the certificate by claiming verifiability
    without providing the data needed to verify.
    """)


if __name__ == "__main__":
    main()
