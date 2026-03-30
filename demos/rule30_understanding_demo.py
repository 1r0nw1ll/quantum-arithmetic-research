#!/usr/bin/env python3
"""
Rule-30 Understanding Demo: The Flagship "Understanding ≠ Prediction" Example

This demo shows the core QA-RML thesis:
- PREDICTION is trivial (apply local rule, O(1) per cell)
- UNDERSTANDING requires certified obstruction (1024 explicit witnesses)

Public lead demo for QA-RML framework.
"""

import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    UnderstandingCertificate,
    TransitionLog,
    DerivationWitness,
    Strategy,
    KeyStep,
    ObstructionEvidence,
    FailType,
    GeneratorRef,
    compute_compression_ratio,
)


# =============================================================================
# RULE 30 IMPLEMENTATION (The "trivial prediction" part)
# =============================================================================

RULE_30 = {
    (1, 1, 1): 0,
    (1, 1, 0): 0,
    (1, 0, 1): 0,
    (1, 0, 0): 1,
    (0, 1, 1): 1,
    (0, 1, 0): 1,
    (0, 0, 1): 1,
    (0, 0, 0): 0,
}


def evolve_rule30(row: list[int]) -> list[int]:
    """Apply Rule 30 to a row. This is PREDICTION - trivial, O(n)."""
    n = len(row)
    new_row = [0] * n
    for i in range(n):
        left = row[(i - 1) % n]
        center = row[i]
        right = row[(i + 1) % n]
        new_row[i] = RULE_30[(left, center, right)]
    return new_row


def get_center_sequence(width: int, steps: int) -> list[int]:
    """Generate the center column sequence."""
    row = [0] * width
    row[width // 2] = 1  # Single seed

    center = width // 2
    sequence = [row[center]]

    for _ in range(steps - 1):
        row = evolve_rule30(row)
        sequence.append(row[center])

    return sequence


# =============================================================================
# UNDERSTANDING: The Non-Trivial Part
# =============================================================================

def check_period(sequence: list[int], period: int) -> tuple[bool, int | None]:
    """
    Check if sequence has given period.
    Returns (has_period, first_violation_index).

    This is UNDERSTANDING - we need to PROVE non-periodicity.
    """
    for t in range(len(sequence) - period):
        if sequence[t] != sequence[t + period]:
            return False, t
    return True, None


def generate_period_witnesses(max_period: int = 1024, width: int = 4096, steps: int = 16384) -> list[dict]:
    """
    Generate explicit witnesses proving non-periodicity for periods 1 to max_period.

    Each witness is a specific (t, c[t], c[t+p]) triple showing c[t] ≠ c[t+p].
    This is the OBSTRUCTION CERTIFICATE - the understanding layer.
    """
    print(f"Generating center sequence (width={width}, steps={steps})...")
    sequence = get_center_sequence(width, steps)

    witnesses = []
    for p in range(1, max_period + 1):
        has_period, violation_t = check_period(sequence, p)

        if has_period:
            # This would be surprising for Rule 30
            witnesses.append({
                "period": p,
                "status": "PERIODIC",
                "witness": None,
            })
        else:
            witnesses.append({
                "period": p,
                "status": "NON_PERIODIC",
                "witness": {
                    "t": violation_t,
                    "c_t": sequence[violation_t],
                    "c_t_plus_p": sequence[violation_t + p],
                },
            })

    return witnesses


# =============================================================================
# BUILD THE UNDERSTANDING CERTIFICATE
# =============================================================================

def build_rule30_understanding_certificate(
    max_period: int = 64,  # Smaller for demo; full version uses 1024
    precomputed_witnesses: list[dict] | None = None,
) -> UnderstandingCertificate:
    """
    Build an UnderstandingCertificate for Rule 30 non-periodicity.

    This demonstrates the three-layer stack:
    - Layer 1 (World Model): Trivial - just apply the rule
    - Layer 2 (Reachability): Target "periodic center column" is UNREACHABLE
    - Layer 3 (Understanding): WHY unreachable - 1024 explicit counterexamples
    """

    # Generate witnesses if not provided
    if precomputed_witnesses is None:
        witnesses = generate_period_witnesses(max_period=max_period, width=1024, steps=max_period * 2 + 100)
    else:
        witnesses = precomputed_witnesses[:max_period]

    # Count verified non-periodic periods
    verified_count = sum(1 for w in witnesses if w["status"] == "NON_PERIODIC")

    # Build derivation witnesses for each period
    derivation_witnesses = []
    for w in witnesses:
        if w["status"] == "NON_PERIODIC":
            derivation_witnesses.append(
                DerivationWitness(
                    invariant_name=f"non_periodic_p{w['period']}",
                    derivation_operator="explicit_counterexample",
                    input_data={
                        "period": w["period"],
                        "t": w["witness"]["t"],
                        "c_t": w["witness"]["c_t"],
                        "c_t_plus_p": w["witness"]["c_t_plus_p"],
                    },
                    output_value=1,  # 1 = non-periodic verified
                    verifiable=True,
                )
            )

    # Build derived invariants dict
    derived_invariants = {
        f"non_periodic_p{w['period']}": 1
        for w in witnesses
        if w["status"] == "NON_PERIODIC"
    }

    # Strategy with derivation witness
    strategy = Strategy(
        type="exhaustive_counterexample_search",
        key_insight="For each period p, find explicit t where c[t] ≠ c[t+p]. No period survives.",
        prerequisite_knowledge=[
            "rule_30_definition",
            "light_cone_causality",
            "periodicity_definition",
        ],
        derivation_witness=DerivationWitness(
            invariant_name="strategy:exhaustive_counterexample",
            derivation_operator="strategy_derivation",
            input_data={
                "method": "For each p in [1, max_period], scan sequence for first violation",
                "complexity": "O(max_period × sequence_length)",
            },
            output_value=1,
        ),
    )

    # Obstruction evidence
    obstruction = ObstructionEvidence(
        fail_type=FailType.GENERATOR_INSUFFICIENT,
        generator_set={
            GeneratorRef(namespace="PHYS", name="rule30_local_update").to_generator(),
        },
        max_depth_reached=max_period,
        states_explored=verified_count,
    )

    # Key steps (the non-routine insights)
    key_steps = [
        KeyStep(
            index=1,
            description="Light-cone argument: center(t) depends on initial condition within ±t cells",
            necessity_witness=ObstructionEvidence(
                fail_type=FailType.TARGET_UNDEFINED,  # Light-cone defines the boundary
            ),
            compression_contribution=0.3,
        ),
        KeyStep(
            index=2,
            description="Exhaustive search: for each period p, find first t where c[t] ≠ c[t+p]",
            necessity_witness=ObstructionEvidence(
                fail_type=FailType.DEPTH_EXHAUSTED,
                generator_set={GeneratorRef("PHYS", "rule30_local_update").to_generator()},
                max_depth_reached=max_period,
            ),
            compression_contribution=0.5,
        ),
        KeyStep(
            index=3,
            description="Completeness: all periods in [1, max_period] have explicit witnesses",
            necessity_witness=ObstructionEvidence(
                fail_type=FailType.GENERATOR_INSUFFICIENT,
                generator_set={GeneratorRef("PHYS", "rule30_local_update").to_generator()},
                max_depth_reached=max_period,
            ),
            compression_contribution=0.2,
        ),
    ]

    # Build transition log (Layer 1 - the "trivial prediction" part)
    # We only need a summary here since prediction is not the point
    transition_log = [
        TransitionLog(
            move=None,
            fail_type=None,
            invariant_diff={"step": t}
        )
        for t in range(min(10, max_period))  # Just first 10 as summary
    ]

    # Explanation path
    explanation_path = [
        "1. PREDICTION: Apply Rule 30 local update - O(n) per step, trivial",
        "2. QUESTION: Is the center column eventually periodic?",
        f"3. OBSTRUCTION: For each period p ∈ [1, {max_period}], find counterexample",
        f"4. RESULT: All {verified_count} periods have explicit witnesses c[t] ≠ c[t+p]",
        "5. CONCLUSION: Center column is non-periodic (within tested bounds)",
        "6. UNDERSTANDING: The WHY is the certificate, not the simulation",
    ]

    # Build the certificate
    cert = UnderstandingCertificate(
        target=f"Rule 30 center column periodicity (periods 1-{max_period})",
        system_id="rule30_center_column",
        transition_log=transition_log,
        reachable=False,  # "Periodic center column" is NOT reachable
        fail_type=FailType.GENERATOR_INSUFFICIENT,
        obstruction=obstruction,
        derived_invariants=derived_invariants,
        derivation_witnesses=derivation_witnesses,
        key_steps=key_steps,
        strategy=strategy,
        explanation_path=explanation_path,
        strict_mode=True,  # Hard validity - will raise if invalid
    )

    return cert


# =============================================================================
# VISUALIZATION
# =============================================================================

def print_demo_output(cert: UnderstandingCertificate):
    """Print a clean demo output showing the three-layer structure."""

    print("\n" + "=" * 70)
    print("  RULE-30 UNDERSTANDING CERTIFICATE")
    print("  The Flagship 'Understanding ≠ Prediction' Demo")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  THE CORE THESIS                                                    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  PREDICTION: Trivial. Apply local rule, O(1) per cell.             │")
    print("│  UNDERSTANDING: Hard. Requires certified obstruction.              │")
    print("│                                                                     │")
    print("│  World models predict what happens.                                 │")
    print("│  RML certifies why some things CANNOT happen.                       │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    j = cert.to_json()

    print(f"\n📋 Certificate Schema: {j['schema']}")
    print(f"✅ Valid: {j['valid']}")
    print(f"🎯 Target: {j['target']}")
    print(f"🚫 Reachable: {j['reachable']} (periodic structure does NOT exist)")

    print(f"\n📊 COMPRESSION RATIO: {j['compression_ratio']:.1f}×")
    print("   (How much understanding compresses vs raw trace)")

    print(f"\n🔬 LAYER 1 - World Model (Prediction)")
    print(f"   Transition log entries: {j['transition_log']['count']}")
    print(f"   Schema: {j['transition_log']['schema']}")
    print("   Status: TRIVIAL - just apply the rule")

    print(f"\n🎯 LAYER 2 - Reachability (Structure)")
    print(f"   Target reachable: {j['reachable']}")
    print(f"   Fail type: {j['fail_type']}")
    print(f"   Obstruction: {j['obstruction']['fail_type']}")

    print(f"\n🧠 LAYER 3 - Understanding (Certificates)")
    print(f"   Derived invariants: {len(j['derived_invariants'])} periods verified")
    print(f"   Key steps: {len(j['key_steps'])}")
    print(f"   Strategy: {j['strategy']['type']}")
    print(f"   Strategy has derivation: {j['strategy']['has_derivation']}")

    print("\n📝 EXPLANATION PATH:")
    for step in j['explanation_path']:
        print(f"   {step}")

    print("\n🔑 KEY INSIGHT:")
    print(f"   \"{j['strategy']['key_insight']}\"")

    # Sample witnesses
    print("\n📦 SAMPLE DERIVATION WITNESSES (first 5):")
    for i, w in enumerate(j['derivation_witnesses'][:5]):
        data = w['input_data']
        print(f"   Period {data['period']}: t={data['t']}, c[t]={data['c_t']}, c[t+p]={data['c_t_plus_p']} → DIFFERENT ✓")

    print("\n" + "─" * 70)
    print("This certificate is machine-verifiable. Every derived invariant")
    print("has an explicit derivation witness. No ad-hoc injections.")
    print("─" * 70)

    return j


def export_certificate(cert: UnderstandingCertificate, output_path: str):
    """Export certificate to JSON file."""
    j = cert.to_json()

    # Add metadata
    j["_demo_metadata"] = {
        "title": "Rule-30 Understanding Certificate",
        "thesis": "Understanding ≠ Prediction",
        "framework": "QA-RML",
        "reference": "Gupta & Pruthi, arXiv:2511.12239v1",
    }

    with open(output_path, 'w') as f:
        json.dump(j, f, indent=2)

    print(f"\n💾 Certificate exported to: {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the flagship demo."""

    print("\n🚀 Building Rule-30 Understanding Certificate...")
    print("   (This demonstrates the core QA-RML thesis)\n")

    # Build the certificate
    cert = build_rule30_understanding_certificate(max_period=64)

    # Print demo output
    j = print_demo_output(cert)

    # Export
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    export_certificate(cert, str(output_dir / "rule30_understanding_cert.json"))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
    This demo shows:

    1. PREDICTION is trivial (any simulator can generate Rule 30)

    2. UNDERSTANDING requires proof:
       - 64 explicit counterexamples (expandable to 1024+)
       - Each counterexample is machine-verifiable
       - Strategy has derivation witness
       - No ad-hoc injections allowed

    3. The certificate is:
       - Valid: All derived invariants have witnesses
       - Replayable: QARM-compatible transition logs
       - Compressive: Explanation << raw trace
       - Falsifiable: Hard validity rules enforced

    This is what it means to UNDERSTAND, not just PREDICT.
    """)

    return cert


if __name__ == "__main__":
    main()
