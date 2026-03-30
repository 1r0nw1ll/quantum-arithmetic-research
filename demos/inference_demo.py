#!/usr/bin/env python3
"""
InferenceCertificate Demo: QA-native probabilistic inference
============================================================

Demonstrates certificate-grade inference over factor graphs, mapping
MIT "Algorithms for Decision Making" Ch. 3-4 to QA reachability framework.

Key concepts:
- Variable elimination as graph reduction operators
- Belief propagation as message passing invariants
- Failure certificates for intractable inference (treewidth, divergence)

Reference: Kochenderfer et al. "Algorithms for Decision Making" MIT Press
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    InferenceCertificate,
    InferenceFailType,
    InferenceObstructionEvidence,
    InferenceMethod,
    InferenceMethodProof,
    FactorSpec,
    DerivationWitness,
    validate_inference_certificate,
)


def demo_simple_bayesian_network():
    """
    Demo 1: Simple Bayesian Network (Student Example)

    Classic BN example with 3 variables:
    - D (Difficulty): {hard, easy}
    - I (Intelligence): {high, low}
    - G (Grade): {A, B, C} depends on D and I

    Query: P(I=high | G=A) - Does getting an A mean the student is intelligent?
    """
    print("\n" + "="*70)
    print("Demo 1: Student Bayesian Network - Variable Elimination")
    print("="*70)

    # Define the BN structure
    variables = ["D", "I", "G"]
    variable_domains = {
        "D": ["hard", "easy"],
        "I": ["high", "low"],
        "G": ["A", "B", "C"],
    }

    factors = [
        FactorSpec("P_D", ["D"], "prior"),       # P(Difficulty)
        FactorSpec("P_I", ["I"], "prior"),       # P(Intelligence)
        FactorSpec("P_G_DI", ["G", "D", "I"], "conditional"),  # P(Grade|D,I)
    ]

    # Query: P(I | G=A)
    # Using variable elimination, eliminate D first
    # Result: P(I=high | G=A) = 0.7, P(I=low | G=A) = 0.3 (example values)

    cert = InferenceCertificate.from_variable_elimination(
        model_id="student_bn",
        variables=variables,
        variable_domains=variable_domains,
        factors=factors,
        query_variables=["I"],
        evidence={"G": "A"},
        elimination_order=["D"],  # Eliminate D to get P(I|G=A)
        result_marginal={
            "high": Fraction(7, 10),
            "low": Fraction(3, 10),
        },
    )

    print(f"\nModel: Student Bayesian Network")
    print(f"Query: P(Intelligence | Grade=A)")
    print(f"Evidence: Grade = A")
    print(f"Elimination order: {cert.method_proof.elimination_order}")
    print(f"\nResult:")
    print(f"  P(I=high | G=A) = {cert.marginal['high']}")
    print(f"  P(I=low  | G=A) = {cert.marginal['low']}")
    print(f"\nCertificate valid: {cert.is_valid()}")
    print(f"Exact inference: {cert.exact_inference}")

    # Validate
    result = validate_inference_certificate(cert)
    print(f"Validation passed: {result.valid}")

    return cert.to_json()


def demo_belief_propagation_tree():
    """
    Demo 2: Belief Propagation on a Tree

    Chain BN: A -> B -> C -> D
    BP is exact on trees, so this demonstrates exact inference
    via message passing.

    Query: P(D | A=true)
    """
    print("\n" + "="*70)
    print("Demo 2: Chain Bayes Net - Belief Propagation (Exact on Tree)")
    print("="*70)

    variables = ["A", "B", "C", "D"]
    variable_domains = {v: ["true", "false"] for v in variables}

    factors = [
        FactorSpec("P_A", ["A"], "prior"),
        FactorSpec("P_B_A", ["B", "A"], "conditional"),
        FactorSpec("P_C_B", ["C", "B"], "conditional"),
        FactorSpec("P_D_C", ["D", "C"], "conditional"),
    ]

    # Simulated BP convergence in 3 iterations (tree structure)
    cert = InferenceCertificate.from_belief_propagation(
        model_id="chain_bn",
        variables=variables,
        variable_domains=variable_domains,
        factors=factors,
        query_variables=["D"],
        evidence={"A": "true"},
        result_marginal={
            "true": Fraction(13, 20),
            "false": Fraction(7, 20),
        },
        iterations=3,
        converged=True,
        is_tree=True,  # Chain is a tree!
    )

    print(f"\nModel: Chain A -> B -> C -> D")
    print(f"Query: P(D | A=true)")
    print(f"Evidence: A = true")
    print(f"Method: Belief Propagation")
    print(f"Iterations: {cert.method_proof.iterations}")
    print(f"Converged: {cert.method_proof.converged}")
    print(f"Is tree: {cert.is_tree}")
    print(f"\nResult:")
    print(f"  P(D=true  | A=true) = {cert.marginal['true']}")
    print(f"  P(D=false | A=true) = {cert.marginal['false']}")
    print(f"\nExact inference: {cert.exact_inference}")  # True because tree!

    result = validate_inference_certificate(cert)
    print(f"Validation passed: {result.valid}")

    return cert.to_json()


def demo_bp_divergence_failure():
    """
    Demo 3: BP Divergence on Loopy Graph

    Triangle graph: A -- B -- C -- A (loop)
    With strong potentials, BP can fail to converge.

    This demonstrates a certificate-grade failure witness.
    """
    print("\n" + "="*70)
    print("Demo 3: Loopy Graph - BP Divergence Failure Certificate")
    print("="*70)

    variables = ["A", "B", "C"]
    variable_domains = {v: ["0", "1"] for v in variables}

    # Triangle potentials (creates loop)
    factors = [
        FactorSpec("psi_AB", ["A", "B"], "potential"),
        FactorSpec("psi_BC", ["B", "C"], "potential"),
        FactorSpec("psi_CA", ["C", "A"], "potential"),  # Closes the loop
    ]

    # BP failed to converge after 100 iterations
    cert = InferenceCertificate(
        model_id="triangle_mrf",
        model_description="Triangular MRF with strong coupling",
        variables=variables,
        variable_domains=variable_domains,
        factors=factors,
        is_tree=False,  # Has loop!
        query_variables=["A"],
        evidence={},
        inference_success=False,
        failure_mode=InferenceFailType.MESSAGE_DIVERGENCE,
        obstruction_if_fail=InferenceObstructionEvidence(
            fail_type=InferenceFailType.MESSAGE_DIVERGENCE,
            iterations_run=100,
            max_iterations=100,
            final_residual=Fraction(15, 100),  # Still 0.15 after 100 iters
            convergence_threshold=Fraction(1, 1000),  # Needed < 0.001
        ),
        method_proof=InferenceMethodProof(
            method=InferenceMethod.BELIEF_PROPAGATION,
            message_schedule="parallel",
            iterations=100,
            converged=False,
            final_residual=Fraction(15, 100),
        ),
        strict_mode=True,
    )

    print(f"\nModel: Triangle MRF (A-B-C-A loop)")
    print(f"Query: P(A)")
    print(f"Is tree: {cert.is_tree}")
    print(f"\nMethod: Belief Propagation")
    print(f"Iterations: {cert.method_proof.iterations}")
    print(f"Converged: {cert.method_proof.converged}")
    print(f"\nFAILURE: {cert.failure_mode.value}")
    obs = cert.obstruction_if_fail
    print(f"Evidence:")
    print(f"  Iterations run: {obs.iterations_run}")
    print(f"  Max iterations: {obs.max_iterations}")
    print(f"  Final residual: {obs.final_residual}")
    print(f"  Convergence threshold: {obs.convergence_threshold}")

    result = validate_inference_certificate(cert)
    print(f"\nValidation passed: {result.valid}")

    return cert.to_json()


def demo_treewidth_intractability():
    """
    Demo 4: Treewidth Intractability Certificate

    When exact inference is intractable due to high treewidth,
    we issue a certificate explaining why.
    """
    print("\n" + "="*70)
    print("Demo 4: High Treewidth - Intractability Certificate")
    print("="*70)

    # A dense graph with many variables
    n = 8
    variables = [f"X{i}" for i in range(n)]
    variable_domains = {v: ["0", "1"] for v in variables}

    # Dense clique-like structure (high treewidth)
    factors = []
    for i in range(n):
        for j in range(i+1, n):
            factors.append(FactorSpec(f"psi_{i}_{j}", [f"X{i}", f"X{j}"], "potential"))

    # Treewidth of near-complete graph is n-1
    cert = InferenceCertificate(
        model_id="dense_mrf",
        model_description=f"Near-complete graph on {n} variables",
        variables=variables,
        variable_domains=variable_domains,
        factors=factors,
        is_tree=False,
        treewidth=7,  # For 8 variables in dense graph
        query_variables=["X0"],
        evidence={},
        inference_success=False,
        exact_inference=False,
        failure_mode=InferenceFailType.TREEWIDTH_TOO_HIGH,
        obstruction_if_fail=InferenceObstructionEvidence(
            fail_type=InferenceFailType.TREEWIDTH_TOO_HIGH,
            treewidth=7,
            treewidth_threshold=3,  # Can only handle treewidth <= 3
        ),
        strict_mode=True,
    )

    print(f"\nModel: Dense MRF on {n} variables")
    print(f"Number of factors: {len(factors)}")
    print(f"Treewidth: {cert.treewidth}")
    print(f"\nFAILURE: {cert.failure_mode.value}")
    obs = cert.obstruction_if_fail
    print(f"Evidence:")
    print(f"  Actual treewidth: {obs.treewidth}")
    print(f"  Threshold: {obs.treewidth_threshold}")
    print(f"  Reason: Exact inference exponential in treewidth")

    result = validate_inference_certificate(cert)
    print(f"\nValidation passed: {result.valid}")

    return cert.to_json()


def demo_evidence_inconsistency():
    """
    Demo 5: Inconsistent Evidence Certificate

    When evidence has probability zero under the model,
    inference is undefined.
    """
    print("\n" + "="*70)
    print("Demo 5: Inconsistent Evidence - P(Evidence) = 0 Certificate")
    print("="*70)

    variables = ["Disease", "Symptom"]
    variable_domains = {
        "Disease": ["A", "B", "none"],
        "Symptom": ["fever", "cough", "none"],
    }

    factors = [
        FactorSpec("P_Disease", ["Disease"], "prior"),
        FactorSpec("P_Symptom_Disease", ["Symptom", "Disease"], "conditional"),
    ]

    # Impossible evidence: Disease=none AND Symptom=fever
    # (if no disease, no fever in this model)
    cert = InferenceCertificate(
        model_id="medical_diagnosis",
        model_description="Simple medical diagnosis BN",
        variables=variables,
        variable_domains=variable_domains,
        factors=factors,
        query_variables=["Disease"],
        evidence={"Symptom": "fever"},  # But model says P(fever|none)=0
        inference_success=False,
        failure_mode=InferenceFailType.EVIDENCE_INCONSISTENT,
        obstruction_if_fail=InferenceObstructionEvidence(
            fail_type=InferenceFailType.EVIDENCE_INCONSISTENT,
            inconsistent_evidence={
                "Symptom": "fever",
                "reason": "P(Symptom=fever) = 0 under prior",
            },
        ),
        strict_mode=True,
    )

    print(f"\nModel: Medical Diagnosis BN")
    print(f"Query: P(Disease | Symptom=fever)")
    print(f"Evidence: Symptom = fever")
    print(f"\nFAILURE: {cert.failure_mode.value}")
    obs = cert.obstruction_if_fail
    print(f"Evidence:")
    print(f"  Inconsistent evidence: {obs.inconsistent_evidence}")
    print(f"  Reason: P(E) = 0, so P(Q|E) is undefined")

    result = validate_inference_certificate(cert)
    print(f"\nValidation passed: {result.valid}")

    return cert.to_json()


def main():
    """Run all demos and export certificates."""
    print("="*70)
    print("INFERENCE CERTIFICATE DEMO")
    print("QA-native probabilistic inference (Ch. 3-4)")
    print("="*70)

    # Run all demos
    demos = {
        "student_bn_variable_elimination": demo_simple_bayesian_network(),
        "chain_bn_bp_exact": demo_belief_propagation_tree(),
        "triangle_bp_divergence": demo_bp_divergence_failure(),
        "dense_treewidth_intractable": demo_treewidth_intractability(),
        "medical_evidence_inconsistent": demo_evidence_inconsistency(),
    }

    # Combine into single certificate
    combined = {
        "demo": "inference_demo",
        "description": "QA-native probabilistic inference certificates",
        "reference": "MIT Algorithms for Decision Making, Chapters 3-4",
        "scenarios": demos,
        "key_insights": [
            "Variable elimination = graph reduction operators.",
            "BP on trees = exact inference (message passing converges).",
            "BP on loops = approximate (may diverge - FAILURE certificate).",
            "Treewidth bounds tractability (intractability = FAILURE certificate).",
            "P(E)=0 = EVIDENCE_INCONSISTENT (undefined conditional).",
            "All failures are first-class objects with machine-checkable witnesses.",
        ],
    }

    # Export
    output_path = Path(__file__).parent / "inference_cert.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "="*70)
    print(f"Exported combined certificate to: {output_path}")
    print("="*70)

    return combined


if __name__ == "__main__":
    main()
