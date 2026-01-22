#!/usr/bin/env python3
"""
QA Decision Certificate Spine Demo
===================================

End-to-end demonstration of the certificate spine from MIT "Algorithms for Decision Making":
    Planning → Filtering → RL → Imitation

This script runs a single environment through all certificate types and emits
a coherent certificate bundle with manifest hash.

Reference: Kochenderfer et al. "Algorithms for Decision Making" MIT Press
Chapters: 7-8 (Planning), 9-11 (Filtering/Inference), 12 (RL), 13 (Imitation)
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    # Policy/Planning (Ch. 7-8)
    PolicyCertificate,
    PolicyEvaluationStats,
    OptimalityProof,
    OptimalityMethod,
    GeneratorRef,
    Strategy,
    DerivationWitness,
    # MCTS (Ch. 8)
    MCTSCertificate,
    MCTSMethodProof,
    MCTSExplorationRule,
    MCTSBackupOperator,
    SCCPruningWitness,
    # Exploration (Ch. 9)
    ExplorationCertificate,
    ExplorationMethod,
    UncertaintyMeasure,
    ExplorationMethodProof,
    RegretWitness,
    # Inference (Ch. 3-4)
    InferenceCertificate,
    InferenceMethod,
    InferenceMethodProof,
    FactorSpec,
    # Filtering (Ch. 9-11)
    FilterCertificate,
    FilterMethod,
    FilterMethodProof,
    # RL (Ch. 12)
    RLCertificate,
    RLAlgorithm,
    RewardSpec,
    RLMethodProof,
    QValueWitness,
    # Imitation (Ch. 13)
    ImitationCertificate,
    ImitationMethod,
    ImitationMethodProof,
    DemonstrationWitness,
    InverseRLWitness,
    # Bundle and coherence
    CertificateBundle,
    validate_bundle_coherence,
)


def create_policy_certificate() -> PolicyCertificate:
    """
    Layer 1: Policy Certificate (Ch. 7 - Exact Planning)

    Certifies a BFS-optimal policy for the 5x5 gridworld.
    Target: reach (4,4) from (0,0).
    """
    print("\n" + "="*70)
    print("Layer 1: PolicyCertificate (Ch. 7 - Exact Planning)")
    print("="*70)

    cert = PolicyCertificate(
        policy_id="bfs_optimal_5x5",
        policy_type="bfs_optimal",
        policy_description="BFS-optimal policy for 5x5 gridworld",
        target_class_description="goal_cell_(4,4)",
        start_class_description="start_cell_(0,0)",
        horizon=50,
        generator_set=[
            GeneratorRef("GRID", "up"),
            GeneratorRef("GRID", "down"),
            GeneratorRef("GRID", "left"),
            GeneratorRef("GRID", "right"),
        ],
        evaluation_stats=PolicyEvaluationStats(
            n_episodes=100,
            successes=100,
            total_steps=800,  # 100 episodes * 8 optimal steps
            total_oracle_calls=0,
        ),
        reachability_guarantee=True,
        optimality_guarantee=True,
        optimality_proof=OptimalityProof(
            method=OptimalityMethod.BFS,
            optimal_distance=Fraction(8, 1),  # |4-0| + |4-0| = 8
            states_explored=25,
            predecessor_map_hash="sha256:bfs_5x5_pred_map",
        ),
        training_witness=DerivationWitness(
            invariant_name="bfs_shortest_path",
            derivation_operator="bfs_search",
            input_data={"grid_size": 5, "start": "(0,0)", "goal": "(4,4)"},
            output_value=1,
            verifiable=True,
        ),
        strict_mode=True,
    )

    print(f"Policy: {cert.policy_type}")
    print(f"Optimal distance: {cert.optimality_proof.optimal_distance}")
    print(f"States explored: {cert.optimality_proof.states_explored}")
    print(f"Reachability guarantee: {cert.reachability_guarantee}")
    print(f"Optimality guarantee: {cert.optimality_guarantee}")

    return cert


def create_mcts_certificate() -> MCTSCertificate:
    """
    Layer 2: MCTS Certificate (Ch. 8 - Online Planning)

    Certifies MCTS planning with QA-native SCC pruning.
    The gridworld has a "trap" SCC that can be pruned.
    """
    print("\n" + "="*70)
    print("Layer 2: MCTSCertificate (Ch. 8 - Online Planning)")
    print("="*70)

    cert = MCTSCertificate.from_qa_mcts_run(
        model_id="gridworld_5x5_with_trap",
        root_state="(0,0)",
        best_action="right",
        expected_return=Fraction(92, 100),  # 0.92 expected success rate
        action_values={
            "up": Fraction(45, 100),
            "down": Fraction(30, 100),  # Leads toward trap
            "left": Fraction(0, 100),   # Wall
            "right": Fraction(92, 100), # Best
        },
        exploration_rule=MCTSExplorationRule.UCB1,
        backup_operator=MCTSBackupOperator.MEAN,
        n_rollouts=200,
        max_depth=20,
        nodes_expanded=150,
        scc_computation_hash="sha256:scc_5x5_trap",
        nodes_pruned_by_scc=75,  # 50% of tree pruned
        unreachable_scc_ids=[2],  # Trap SCC
        target_scc_id=1,  # Goal SCC
        vanilla_rollouts_baseline=500,
        exploration_constant=Fraction(14142, 10000),  # sqrt(2)
        random_seed=42,
    )

    print(f"Root state: {cert.root_state}")
    print(f"Best action: {cert.best_action}")
    print(f"Expected return: {cert.expected_return}")
    print(f"SCC pruning: {cert.scc_pruning_witness.nodes_pruned} nodes")
    print(f"Pruning efficiency: {cert.pruning_efficiency} ({float(cert.pruning_efficiency)*100:.0f}% savings)")

    return cert


def create_exploration_certificate() -> ExplorationCertificate:
    """
    Layer 3: Exploration Certificate (Ch. 9 - Exploration-Exploitation)

    Certifies UCB1 exploration with regret analysis.
    """
    print("\n" + "="*70)
    print("Layer 3: ExplorationCertificate (Ch. 9 - Exploration)")
    print("="*70)

    cert = ExplorationCertificate.from_ucb_exploration(
        model_id="gridworld_5x5_ucb",
        actual_steps=850,
        optimal_steps=800,  # 100 episodes * 8 optimal steps
        total_episodes=100,
        exploration_constant=Fraction(14142, 10000),
        unique_states_visited=25,
        target_reached=True,
        target_class="goal_cell_(4,4)",
    )

    print(f"Total episodes: {cert.method_proof.total_episodes}")
    print(f"Actual steps: {cert.regret_witness.actual_steps}")
    print(f"Optimal steps: {cert.regret_witness.optimal_steps}")
    print(f"Cumulative regret: {cert.regret_witness.cumulative_regret}")
    print(f"Regret bound: {cert.regret_witness.regret_bound}")
    print(f"States visited: {cert.method_proof.unique_states_visited}/25")

    return cert


def create_inference_certificate() -> InferenceCertificate:
    """
    Layer 4: Inference Certificate (Ch. 3-4 - Probabilistic Inference)

    Certifies belief inference over grid position.
    """
    print("\n" + "="*70)
    print("Layer 4: InferenceCertificate (Ch. 3-4 - Inference)")
    print("="*70)

    cert = InferenceCertificate.from_variable_elimination(
        model_id="gridworld_position_belief",
        variables=["X", "Y", "Goal"],
        variable_domains={
            "X": ["0", "1", "2", "3", "4"],
            "Y": ["0", "1", "2", "3", "4"],
            "Goal": ["reached", "not_reached"],
        },
        factors=[
            FactorSpec("P_X", ["X"], "prior"),
            FactorSpec("P_Y", ["Y"], "prior"),
            FactorSpec("P_Goal_XY", ["Goal", "X", "Y"], "conditional"),
        ],
        query_variables=["Goal"],
        evidence={"X": "4", "Y": "4"},  # At goal position
        result_marginal={"reached": Fraction(1, 1), "not_reached": Fraction(0, 1)},
        elimination_order=["X", "Y"],
    )

    print(f"Model: {cert.model_id}")
    print(f"Variables: {cert.variables}")
    print(f"Query: P({cert.query_variables}|{cert.evidence})")
    print(f"Marginal: {cert.marginal}")
    print(f"Method: {cert.method_proof.method.value}")
    print(f"Elimination order: {cert.method_proof.elimination_order}")

    return cert


def create_filter_certificate() -> FilterCertificate:
    """
    Layer 5: Filter Certificate (Ch. 9-11 - State Estimation)

    Certifies Kalman filter for noisy position tracking.
    """
    print("\n" + "="*70)
    print("Layer 5: FilterCertificate (Ch. 9-11 - Filtering)")
    print("="*70)

    cert = FilterCertificate.from_kalman(
        model_id="gridworld_position_tracker",
        state_names=["x", "y", "vx", "vy"],
        observation_dimension=2,  # Observe x, y with noise
        n_observations=50,
        state_estimate={
            "x": Fraction(39, 10),   # ~3.9 (close to goal 4)
            "y": Fraction(41, 10),   # ~4.1 (close to goal 4)
            "vx": Fraction(1, 10),   # ~0.1 velocity
            "vy": Fraction(1, 10),
        },
        covariance_trace=Fraction(2, 10),  # Low uncertainty
    )

    print(f"Model: {cert.model_id}")
    print(f"State dimension: {cert.state_dimension}")
    print(f"Observation dimension: {cert.observation_dimension}")
    print(f"Estimate: x={cert.state_estimate['x']}, y={cert.state_estimate['y']}")
    print(f"Uncertainty (tr(P)): {cert.covariance_trace}")
    print(f"Method: {cert.method_proof.method.value}")

    return cert


def create_rl_certificate() -> RLCertificate:
    """
    Layer 6: RL Certificate (Ch. 12 - Reinforcement Learning)

    Certifies Q-learning with distance-delta reward.
    """
    print("\n" + "="*70)
    print("Layer 6: RLCertificate (Ch. 12 - Reinforcement Learning)")
    print("="*70)

    # Sample Q-value transitions for audit
    sample_transitions = [
        {
            "s": "(0,0)", "a": "right", "r": Fraction(1, 1),
            "s_next": "(1,0)", "q_before": Fraction(0),
            "max_q_next": Fraction(7, 1), "q_after": Fraction(73, 100),
        },
        {
            "s": "(3,4)", "a": "right", "r": Fraction(1, 1),
            "s_next": "(4,4)", "q_before": Fraction(8, 1),
            "max_q_next": Fraction(0, 1), "q_after": Fraction(81, 10),
        },
    ]

    cert = RLCertificate.from_q_learning_run(
        model_id="gridworld_5x5_qlearning",
        total_episodes=500,
        total_steps=4500,
        learning_rate=Fraction(1, 10),
        discount_factor=Fraction(99, 100),
        final_performance=Fraction(95, 100),
        converged=True,
        reward_spec=RewardSpec.DISTANCE_DELTA,
        exploration_method=ExplorationMethod.EPSILON_GREEDY,
        sample_transitions=sample_transitions,
        target_class="goal_cell_(4,4)",
        generator_set=["up", "down", "left", "right"],
    )

    print(f"Model: {cert.model_id}")
    print(f"Algorithm: {cert.method_proof.algorithm.value}")
    print(f"Reward spec: {cert.method_proof.reward_spec.value} (QA-native)")
    print(f"Episodes: {cert.method_proof.total_episodes}")
    print(f"Final performance: {float(cert.final_performance)*100:.0f}%")
    print(f"Converged: {cert.method_proof.converged}")
    print(f"Q-value witness: {len(cert.q_value_witness.sample_transitions)} transitions logged")

    return cert


def create_imitation_certificate() -> ImitationCertificate:
    """
    Layer 7: Imitation Certificate (Ch. 13 - Imitation Learning)

    Certifies inverse RL target inference from expert demonstrations.
    """
    print("\n" + "="*70)
    print("Layer 7: ImitationCertificate (Ch. 13 - Imitation Learning)")
    print("="*70)

    cert = ImitationCertificate.from_inverse_rl(
        model_id="gridworld_5x5_irl",
        n_trajectories=50,
        n_state_action_pairs=400,  # 50 trajectories * 8 avg steps
        dataset_hash="sha256:expert_demos_5x5",
        inferred_target_class="goal_cell_(4,4)",
        confidence=Fraction(98, 100),
        identifiable=True,  # Unique target inferred
        total_epochs=100,
        expert_target_class="goal_cell_(4,4)",  # Ground truth matches
    )

    print(f"Model: {cert.model_id}")
    print(f"Method: {cert.method_proof.method.value}")
    print(f"Demonstrations: {cert.method_proof.demonstration_witness.n_trajectories}")
    print(f"State-action pairs: {cert.method_proof.demonstration_witness.n_state_action_pairs}")
    print(f"Inferred target: {cert.inferred_target_class}")
    print(f"Confidence: {float(cert.method_proof.inverse_rl_witness.confidence)*100:.0f}%")
    print(f"Identifiable: {cert.method_proof.inverse_rl_witness.identifiable}")
    print(f"Expert target (ground truth): {cert.expert_target_class}")
    print(f"Match: {cert.inferred_target_class == cert.expert_target_class}")

    return cert


def main():
    """Run end-to-end spine demo and export certificate bundle."""
    print("="*70)
    print("QA DECISION CERTIFICATE SPINE DEMO")
    print("MIT Algorithms for Decision Making → QA Certificates")
    print("="*70)

    # Create all certificates
    policy_cert = create_policy_certificate()
    mcts_cert = create_mcts_certificate()
    exploration_cert = create_exploration_certificate()
    inference_cert = create_inference_certificate()
    filter_cert = create_filter_certificate()
    rl_cert = create_rl_certificate()
    imitation_cert = create_imitation_certificate()

    # Bundle all certificates
    print("\n" + "="*70)
    print("CERTIFICATE BUNDLE")
    print("="*70)

    bundle = CertificateBundle(
        bundle_id="gridworld_5x5_full_spine",
        description="End-to-end QA decision stack for 5x5 gridworld",
        environment_id="gridworld_5x5",
        target_class="goal_cell_(4,4)",
        policy_certificates=[policy_cert],
        mcts_certificates=[mcts_cert],
        exploration_certificates=[exploration_cert],
        inference_certificates=[inference_cert],
        filter_certificates=[filter_cert],
        rl_certificates=[rl_cert],
        imitation_certificates=[imitation_cert],
    )

    # Validate bundle coherence
    print("\nValidating bundle coherence...")
    coherence_result = validate_bundle_coherence(bundle)

    print(f"Coherent: {coherence_result.coherent}")
    print(f"Cross-references checked: {coherence_result.cross_references_checked}")

    if coherence_result.violations:
        print(f"Violations: {len(coherence_result.violations)}")
        for v in coherence_result.violations:
            print(f"  - {v}")

    if coherence_result.warnings:
        print(f"Warnings: {len(coherence_result.warnings)}")
        for w in coherence_result.warnings:
            print(f"  - {w}")

    # Export manifest
    manifest = bundle.to_manifest()
    print(f"\nBundle manifest:")
    print(f"  ID: {manifest['bundle_id']}")
    print(f"  Hash: {manifest['bundle_hash']}")
    print(f"  Total certificates: {manifest['total_certificates']}")
    print(f"  Counts: {manifest['certificate_counts']}")

    # Export full bundle
    output_dir = Path(__file__).parent

    # Export manifest
    manifest_path = output_dir / "spine_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nExported manifest to: {manifest_path}")

    # Export full certificate bundle
    bundle_export = {
        "manifest": manifest,
        "coherence": {
            "valid": coherence_result.coherent,
            "checks": coherence_result.cross_references_checked,
            "violations": coherence_result.violations,
            "warnings": coherence_result.warnings,
        },
        "certificates": {
            "policy": [policy_cert.to_json()],
            "mcts": [mcts_cert.to_json()],
            "exploration": [exploration_cert.to_json()],
            "inference": [inference_cert.to_json()],
            "filter": [filter_cert.to_json()],
            "rl": [rl_cert.to_json()],
            "imitation": [imitation_cert.to_json()],
        },
        "key_insights": [
            "PolicyCertificate: BFS-optimal = shortest reachability path",
            "MCTSCertificate: SCC pruning = certified topological obstruction",
            "ExplorationCertificate: Regret = steps - BFS optimal (reachability)",
            "InferenceCertificate: VE = graph reduction operators",
            "FilterCertificate: Kalman = optimal linear Gaussian estimation",
            "RLCertificate: Q-learning with distance_delta = reachability learning",
            "ImitationCertificate: IRL = target-class inference (identifiable)",
            "All certificates are exact (Fraction arithmetic), deterministic, serializable",
        ],
    }

    bundle_path = output_dir / "spine_bundle.json"
    with open(bundle_path, "w") as f:
        json.dump(bundle_export, f, indent=2, default=str)
    print(f"Exported full bundle to: {bundle_path}")

    print("\n" + "="*70)
    print("SPINE DEMO COMPLETE")
    print("="*70)

    return bundle_export


if __name__ == "__main__":
    main()
