#!/usr/bin/env python3
"""
Understanding Certificate Examples

Concrete implementations of the three case studies from
"Beyond World Models: Rethinking Understanding in AI Models" (arXiv:2511.12239v1)
mapped to QA-RML certificate structures.
"""

import sys
sys.path.insert(0, '/home/player2/signal_experiments/qa_alphageometry_ptolemy')

from fractions import Fraction
from qa_certificate import (
    Generator, StateRef, MoveWitness, ObstructionEvidence, FailType,
    UnderstandingCertificate, KeyStep, Strategy, ProblemSituationCert,
    DerivationWitness, check_for_adhoc_injection
)
import json


def example_1_domino_primality():
    """
    Case Study 1: Hofstadter's Domino Computer

    The paper argues: tracking domino states doesn't explain primality.
    QA-RML approach: provide an invariant derivation + obstruction certificate.
    """

    # Example: n = 641 (prime)
    n = 641

    # The understanding certificate for "why does the prime stretch trigger?"
    cert = UnderstandingCertificate(
        target="PRIME_STRETCH_TRIGGERED",
        system_id="domino_computer_n641",

        # Layer 2: Reachability result
        reachable=True,  # Prime stretch IS reachable

        # Layer 3: Understanding
        derived_invariants={
            "Prime(n)": 1,  # 1 = True
            "divisor_count": 2,  # Only 1 and n
        },

        # Key insight: the invariant governs the outcome
        strategy=Strategy(
            type="invariant_dominance",
            key_insight="Prime(n) → no divisor test succeeds → prime stretch triggers",
            prerequisite_knowledge=["primality_definition", "divisibility_test_mechanics"]
        ),

        # Explanation path (the "understanding" not in world model)
        explanation_path=[
            "1. n = 641 is encoded as input stretch length",
            "2. Each divisor-test loop checks if d | n for d ∈ {2, ..., √n}",
            "3. Prime(641) ⇒ no d satisfies d | 641",
            "4. All divisor tests fail → no divisor stretch triggered",
            "5. Control reaches prime stretch → PRIME_STRETCH triggered",
        ],

        compression_ratio=47.0 / 5.0,  # Full trace (~47 domino states) / explanation (5 steps)

        # Derivation witnesses (ensures falsifiability)
        derivation_witnesses=[
            DerivationWitness(
                invariant_name="Prime(n)",
                derivation_operator="Miller-Rabin",
                input_data={"n": 641, "witnesses": [2, 3, 5, 7, 11]},
                output_value=1,  # True
                verifiable=True
            ),
            DerivationWitness(
                invariant_name="divisor_count",
                derivation_operator="trial_division",
                input_data={"n": 641, "range": "2..√641"},
                output_value=2,  # Only 1 and 641
                verifiable=True
            ),
        ],

        # Key steps with necessity witnesses
        key_steps=[
            KeyStep(
                index=3,
                description="Identify Prime(n) as governing invariant",
                necessity_witness=ObstructionEvidence(
                    fail_type=FailType.GENERATOR_INSUFFICIENT,
                    generator_set={Generator("PHYS:domino_propagation")},
                    max_depth_reached=1000,
                    states_explored=47,
                ),
                compression_contribution=0.7
            )
        ]
    )

    # Validate falsifiability
    violation = check_for_adhoc_injection(cert)
    assert violation is None, f"Certificate has ad-hoc injection: {violation}"

    return cert


def example_2_proof_understanding():
    """
    Case Study 2: Proof Understanding

    The paper argues: verifying proof steps ≠ understanding the proof.
    QA-RML approach: identify strategy, key steps, and compression.

    Example: Zagier's one-sentence proof that every prime p ≡ 1 (mod 4) is sum of two squares.
    """

    cert = UnderstandingCertificate(
        target="THEOREM: p ≡ 1 (mod 4) ⇒ p = a² + b²",
        system_id="zagier_two_squares",

        # Layer 2: The theorem is proven (reachable)
        reachable=True,

        # Layer 3: Understanding
        derived_invariants={
            "involution_fixed_points": 1,  # Odd count ⇒ at least one exists
            "target_set_cardinality_parity": 1,  # Odd
        },

        strategy=Strategy(
            type="involution_fixed_point_parity",
            key_insight="Define involution τ on set S. |S| odd ⇒ fixed point exists. Fixed point = solution.",
            prerequisite_knowledge=[
                "involution_definition",
                "fixed_point_parity_lemma",
                "set_S = {(x,y,z): x²+4yz=p, x,y,z>0}"
            ]
        ),

        explanation_path=[
            "1. Define S = {(x,y,z) ∈ ℤ³₊ : x² + 4yz = p}",
            "2. Show |S| is odd (partition argument)",
            "3. Define involution τ on S with 3 cases",
            "4. By parity lemma: odd |S| ⇒ ∃ fixed point",
            "5. Fixed point has x=y ⇒ p = x² + (2z)²",
        ],

        compression_ratio=1,  # Zagier's proof IS the compressed form

        derivation_witnesses=[
            DerivationWitness(
                invariant_name="involution_fixed_points",
                derivation_operator="parity_counting",
                input_data={"set": "S", "involution": "τ"},
                output_value=1,  # ≥1 guaranteed by parity
                verifiable=True
            ),
            DerivationWitness(
                invariant_name="target_set_cardinality_parity",
                derivation_operator="partition_counting",
                input_data={"p": "prime ≡ 1 (mod 4)"},
                output_value=1,  # Odd
                verifiable=True
            ),
        ],

        key_steps=[
            KeyStep(
                index=3,
                description="Define the involution τ with 3 cases",
                necessity_witness=ObstructionEvidence(
                    fail_type=FailType.GENERATOR_INSUFFICIENT,
                    generator_set={Generator("AG:standard_algebra")},
                    max_depth_reached=100,
                    states_explored=50,
                ),
                compression_contribution=0.8
            ),
            KeyStep(
                index=4,
                description="Apply fixed-point parity lemma",
                necessity_witness=ObstructionEvidence(
                    fail_type=FailType.INVARIANT_VIOLATION,
                    violated_invariants={"parity_preserved": 0},
                    blocked_move=MoveWitness(
                        gen=Generator("AG:counting"),
                        src=StateRef.from_coords_and_packet((0,), {"step": 3}),
                        dst=StateRef.from_coords_and_packet((0,), {"step": 5}),
                        packet_delta={"parity_preserved": 1},
                        legal=False
                    )
                ),
                compression_contribution=0.2
            )
        ]
    )

    violation = check_for_adhoc_injection(cert)
    assert violation is None

    return cert


def example_3_bohr_theory():
    """
    Case Study 3: Bohr Theory (Problem Situation)

    The paper (Popper): Understanding Bohr requires knowing the problem
    (discrete spectral lines), not just simulating electrons.

    QA-RML approach: obstruction under prior theory + generator extension.
    """

    cert = UnderstandingCertificate(
        target="DISCRETE_SPECTRAL_LINES",
        system_id="bohr_hydrogen_model",

        # Under classical theory: unreachable
        # Under Bohr theory: reachable
        reachable=True,

        derived_invariants={
            "quantization_postulate": 1,
            "energy_discretization": 1,
        },

        # The problem-situation certificate (Popper's insight)
        problem_gap=ProblemSituationCert(
            gap="Classical electrodynamics predicts continuous radiation from orbiting electrons",
            target_phenomenon="Observed: discrete spectral lines at specific frequencies",
            resolution="Quantized energy levels En = -13.6eV/n²; transitions emit ΔE = hν",
            necessity="Continuous model cannot produce discrete spectra; quantization is forced by observation",
            prior_generators={
                Generator("PHYS:classical_orbit"),
                Generator("PHYS:maxwell_radiation"),
            },
            new_generators={
                Generator("PHYS:quantized_energy_levels"),
                Generator("PHYS:discrete_transition"),
                Generator("PHYS:photon_emission"),
            }
        ),

        strategy=Strategy(
            type="postulate_extension",
            key_insight="Classical generators insufficient → add quantization postulates → discrete spectra derivable",
            prerequisite_knowledge=[
                "classical_electrodynamics",
                "spectroscopy_observations",
                "planck_constant"
            ]
        ),

        explanation_path=[
            "1. OBSERVATION: Hydrogen emits discrete spectral lines (Balmer series)",
            "2. OBSTRUCTION: Classical orbit + Maxwell ⇒ continuous spectrum",
            "3. GAP: No path from classical generators to discrete target",
            "4. EXTENSION: Add quantization postulates (En = -13.6/n², L = nℏ)",
            "5. REACHABLE: ΔE = Eₙ - Eₘ = hν ⇒ discrete frequencies",
        ],

        compression_ratio=5.0,  # Complex derivation compressed to 5 key steps

        derivation_witnesses=[
            DerivationWitness(
                invariant_name="quantization_postulate",
                derivation_operator="postulate_introduction",
                input_data={
                    "motivation": "discrete spectra cannot arise from continuous theory",
                    "form": "L = nℏ, n ∈ ℤ⁺"
                },
                output_value=1,
                verifiable=True
            ),
            DerivationWitness(
                invariant_name="energy_discretization",
                derivation_operator="algebraic_derivation",
                input_data={
                    "from": "L = nℏ + Coulomb force",
                    "to": "En = -13.6eV/n²"
                },
                output_value=1,
                verifiable=True
            ),
        ],

        # The obstruction under classical theory
        obstruction=ObstructionEvidence(
            fail_type=FailType.LAW_VIOLATION,
            law_name="Classical_Electrodynamics_Spectral_Prediction",
            measured_observables={
                "spectral_type": 0,  # 0 = discrete (observed)
                "expected": 1,  # 1 = continuous (classical prediction)
            },
            law_violation_delta=1,  # Categorical mismatch
            generator_set={
                Generator("PHYS:classical_orbit"),
                Generator("PHYS:maxwell_radiation"),
            },
            max_depth_reached=100,
        ),

        key_steps=[
            KeyStep(
                index=2,
                description="Identify classical obstruction",
                necessity_witness=ObstructionEvidence(
                    fail_type=FailType.LAW_VIOLATION,
                    law_name="Classical_Spectral_Continuity",
                    measured_observables={"spectrum": 0},
                    law_violation_delta=1,
                ),
                compression_contribution=0.3
            ),
            KeyStep(
                index=4,
                description="Introduce quantization postulates",
                necessity_witness=ObstructionEvidence(
                    fail_type=FailType.GENERATOR_INSUFFICIENT,
                    generator_set={Generator("PHYS:classical_orbit")},
                    max_depth_reached=1000,
                ),
                compression_contribution=0.5
            )
        ]
    )

    violation = check_for_adhoc_injection(cert)
    assert violation is None

    return cert


def main():
    """Generate and display all three example certificates."""

    print("=" * 70)
    print("UNDERSTANDING CERTIFICATES: Beyond World Models → QA-RML")
    print("Reference: Gupta & Pruthi (arXiv:2511.12239v1)")
    print("=" * 70)

    examples = [
        ("Case 1: Domino Primality", example_1_domino_primality),
        ("Case 2: Proof Understanding (Zagier)", example_2_proof_understanding),
        ("Case 3: Bohr Theory (Problem Situation)", example_3_bohr_theory),
    ]

    for name, func in examples:
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")

        cert = func()
        output = cert.to_json()

        print(f"\nTarget: {output['target']}")
        print(f"Reachable: {output.get('reachable', 'N/A')}")
        print(f"Falsifiability Valid: {output['falsifiability_valid']}")
        print(f"Compression Ratio: {output.get('compression_ratio', 'N/A')}")

        if 'strategy' in output:
            print(f"\nStrategy: {output['strategy']['type']}")
            print(f"Key Insight: {output['strategy']['key_insight']}")

        if 'problem_situation' in output:
            ps = output['problem_situation']
            print(f"\nProblem Situation:")
            print(f"  Gap: {ps['gap'][:60]}...")
            print(f"  Resolution: {ps['resolution'][:60]}...")

        print(f"\nExplanation Path:")
        for step in output.get('explanation_path', [])[:3]:
            print(f"  {step}")
        if len(output.get('explanation_path', [])) > 3:
            print(f"  ... ({len(output['explanation_path'])} total steps)")

        print(f"\nDerived Invariants: {output['derived_invariants']}")

        if output.get('key_steps'):
            print(f"\nKey Steps: {len(output['key_steps'])}")
            for ks in output['key_steps']:
                print(f"  [{ks['index']}] {ks['description']}")

    print("\n" + "=" * 70)
    print("All certificates validated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
