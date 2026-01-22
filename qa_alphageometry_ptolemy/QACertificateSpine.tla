--------------------------- MODULE QACertificateSpine ---------------------------
(****************************************************************************)
(* TLA+ Formal Specification of the QA Decision Certificate Spine          *)
(*                                                                          *)
(* This module specifies the certificate architecture for sequential        *)
(* decision making with machine-checkable witnesses and obstructions.       *)
(*                                                                          *)
(* Key insight: Failures are first-class objects with constructive proofs.  *)
(****************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    States,           \* Set of all possible states
    Actions,          \* Set of all possible actions
    TargetClasses,    \* Set of target classes (goal specifications)
    Variables,        \* Set of random variables (for inference)
    MaxHorizon        \* Maximum planning horizon

(****************************************************************************)
(* CERTIFICATE VALIDITY                                                     *)
(****************************************************************************)

\* A certificate is either valid with a success witness, or valid with an obstruction
CertificateStatus == {"SUCCESS", "OBSTRUCTION", "INVALID"}

\* Core certificate structure
Certificate == [
    status: CertificateStatus,
    witness: Seq(STRING),     \* Witness data (success proof or obstruction)
    verifiable: BOOLEAN       \* Can be machine-verified
]

(****************************************************************************)
(* POLICY CERTIFICATES (Chapter 7)                                          *)
(****************************************************************************)

PolicyType == {"bfs_optimal", "qawm_greedy", "value_iteration", "policy_iteration"}

PolicyFailType == {"NO_PATH_EXISTS", "HORIZON_EXCEEDED", "OBSTRUCTION_HIT"}

\* Policy certificate structure
PolicyCertificate == [
    policy_type: PolicyType,
    initial_state: States,
    target_class: TargetClasses,
    optimal_distance: Nat,
    reachable: BOOLEAN,
    fail_type: PolicyFailType \cup {NULL}
]

\* Policy certificate validator
ValidPolicyCert(cert) ==
    /\ cert.policy_type \in PolicyType
    /\ cert.initial_state \in States
    /\ cert.target_class \in TargetClasses
    /\ (cert.reachable = TRUE) => (cert.optimal_distance >= 0)
    /\ (cert.reachable = FALSE) => (cert.fail_type \in PolicyFailType)

(****************************************************************************)
(* MCTS CERTIFICATES (Chapter 8)                                            *)
(****************************************************************************)

MCTSExplorationRule == {"UCB1", "PUCT", "EPSILON_GREEDY"}

MCTSFailType == {"BUDGET_EXHAUSTED", "SCC_UNREACHABLE", "VALUE_DIVERGENCE"}

\* SCC Pruning Witness - the QA differentiator
SCCPruningWitness == [
    scc_hash: STRING,
    nodes_pruned: Nat,
    unreachable_sccs: SUBSET Nat
]

\* MCTS certificate structure
MCTSCertificate == [
    root_state: States,
    best_action: Actions,
    n_rollouts: Nat,
    exploration_rule: MCTSExplorationRule,
    scc_witness: SCCPruningWitness \cup {NULL},
    pruning_efficiency: 0..100,  \* Percentage
    fail_type: MCTSFailType \cup {NULL}
]

\* MCTS certificate validator
ValidMCTSCert(cert) ==
    /\ cert.root_state \in States
    /\ cert.n_rollouts > 0
    /\ cert.exploration_rule \in MCTSExplorationRule
    \* SCC pruning witness implies efficiency > 0
    /\ (cert.scc_witness /= NULL) => (cert.pruning_efficiency > 0)

(****************************************************************************)
(* EXPLORATION CERTIFICATES (Chapter 9)                                     *)
(****************************************************************************)

ExplorationMethod == {"UCB1", "THOMPSON_SAMPLING", "EPSILON_GREEDY"}

ExplorationFailType == {"HIGH_REGRET", "EXPLORATION_COLLAPSED", "BUDGET_EXHAUSTED"}

\* Regret witness - QA-native definition
RegretWitness == [
    actual_steps: Nat,
    optimal_steps: Nat,     \* BFS optimal
    cumulative_regret: Nat, \* actual - optimal
    regret_bound: STRING    \* e.g., "O(sqrt(T))"
]

\* Regret is well-formed
ValidRegretWitness(rw) ==
    /\ rw.actual_steps >= rw.optimal_steps
    /\ rw.cumulative_regret = rw.actual_steps - rw.optimal_steps

\* Exploration certificate structure
ExplorationCertificate == [
    method: ExplorationMethod,
    regret_witness: RegretWitness,
    states_visited: Nat,
    total_states: Nat,
    fail_type: ExplorationFailType \cup {NULL}
]

(****************************************************************************)
(* INFERENCE CERTIFICATES (Chapters 3-4)                                    *)
(****************************************************************************)

InferenceMethod == {"VARIABLE_ELIMINATION", "BELIEF_PROPAGATION", "EXACT", "APPROXIMATE"}

InferenceFailType == {"TREEWIDTH_TOO_HIGH", "MESSAGE_DIVERGENCE", "EVIDENCE_INCONSISTENT"}

\* Inference certificate structure
InferenceCertificate == [
    query_vars: SUBSET Variables,
    evidence: Variables -> STRING,  \* Variable -> value mapping
    method: InferenceMethod,
    is_tree: BOOLEAN,               \* Factor graph is tree
    exact: BOOLEAN,                 \* Result is exact (not approximate)
    elimination_order: Seq(Variables),
    fail_type: InferenceFailType \cup {NULL}
]

\* Strict validator: BP on non-tree claiming exact is a VIOLATION
ValidInferenceCert(cert) ==
    /\ cert.query_vars \subseteq Variables
    /\ cert.method \in InferenceMethod
    \* BP + non-tree + exact claim is invalid
    /\ ~(cert.method = "BELIEF_PROPAGATION" /\ ~cert.is_tree /\ cert.exact)

(****************************************************************************)
(* FILTER CERTIFICATES (Chapters 9-11)                                      *)
(****************************************************************************)

FilterMethod == {"KALMAN", "PARTICLE", "HISTOGRAM", "UNSCENTED"}

FilterFailType == {"PARTICLE_DEGENERACY", "STATE_UNOBSERVABLE",
                   "FILTER_DIVERGED", "COVARIANCE_SINGULAR"}

\* Filter certificate structure
FilterCertificate == [
    state_dimension: Nat,
    observation_dimension: Nat,
    method: FilterMethod,
    linear_system: BOOLEAN,
    gaussian_noise: BOOLEAN,
    n_timesteps: Nat,
    covariance_trace: STRING,  \* Exact rational as string
    fail_type: FilterFailType \cup {NULL}
]

\* Kalman requires linear + Gaussian
ValidFilterCert(cert) ==
    /\ cert.state_dimension > 0
    /\ cert.observation_dimension > 0
    /\ cert.method \in FilterMethod
    \* Kalman only valid for linear Gaussian systems
    /\ (cert.method = "KALMAN") => (cert.linear_system /\ cert.gaussian_noise)

(****************************************************************************)
(* RL CERTIFICATES (Chapter 12)                                             *)
(****************************************************************************)

RLAlgorithm == {"Q_LEARNING", "SARSA", "PPO", "DQN", "ACTOR_CRITIC"}

RewardSpec == {"DISTANCE_DELTA", "SPARSE", "SHAPED", "INTRINSIC"}

RLFailType == {"CONVERGENCE_TIMEOUT", "VALUE_DIVERGENCE", "EXPLORATION_FAILURE"}

\* Q-value witness for audit trail
QValueWitness == [
    state: States,
    action: Actions,
    q_before: STRING,   \* Exact rational
    q_after: STRING,
    reward: STRING,
    next_state: States
]

\* RL certificate structure
RLCertificate == [
    algorithm: RLAlgorithm,
    reward_spec: RewardSpec,
    learning_rate: STRING,      \* Exact rational
    discount_factor: STRING,
    n_episodes: Nat,
    final_performance: 0..100,  \* Percentage
    converged: BOOLEAN,
    q_witnesses: Seq(QValueWitness),
    fail_type: RLFailType \cup {NULL}
]

\* RL certificate validator
ValidRLCert(cert) ==
    /\ cert.algorithm \in RLAlgorithm
    /\ cert.reward_spec \in RewardSpec
    /\ cert.n_episodes > 0
    \* Q-learning requires learning rate and discount factor
    /\ (cert.algorithm = "Q_LEARNING") =>
       (cert.learning_rate /= "" /\ cert.discount_factor /= "")

(****************************************************************************)
(* IMITATION CERTIFICATES (Chapter 13)                                      *)
(****************************************************************************)

ImitationMethod == {"BEHAVIORAL_CLONING", "INVERSE_RL", "DAGGER"}

ImitationFailType == {"REWARD_NON_IDENTIFIABLE", "DISTRIBUTION_SHIFT",
                      "ORACLE_BUDGET_EXHAUSTED"}

\* Inverse RL witness - QA-native target inference
InverseRLWitness == [
    inferred_target: TargetClasses,
    confidence: 0..100,
    identifiable: BOOLEAN,
    alternative_targets: SUBSET TargetClasses  \* If non-identifiable
]

\* Imitation certificate structure
ImitationCertificate == [
    method: ImitationMethod,
    n_demonstrations: Nat,
    n_state_action_pairs: Nat,
    irl_witness: InverseRLWitness \cup {NULL},
    fail_type: ImitationFailType \cup {NULL}
]

\* Imitation certificate validator
ValidImitationCert(cert) ==
    /\ cert.method \in ImitationMethod
    /\ cert.n_demonstrations > 0
    \* IRL requires irl_witness
    /\ (cert.method = "INVERSE_RL") => (cert.irl_witness /= NULL)
    \* Non-identifiable means alternatives exist
    /\ (cert.irl_witness /= NULL /\ ~cert.irl_witness.identifiable) =>
       (Cardinality(cert.irl_witness.alternative_targets) > 0)

(****************************************************************************)
(* CERTIFICATE BUNDLE AND COHERENCE                                         *)
(****************************************************************************)

\* A bundle groups certificates from a single decision problem
CertificateBundle == [
    bundle_id: STRING,
    policy_certs: Seq(PolicyCertificate),
    mcts_certs: Seq(MCTSCertificate),
    exploration_certs: Seq(ExplorationCertificate),
    inference_certs: Seq(InferenceCertificate),
    filter_certs: Seq(FilterCertificate),
    rl_certs: Seq(RLCertificate),
    imitation_certs: Seq(ImitationCertificate)
]

(****************************************************************************)
(* COHERENCE RULES                                                          *)
(****************************************************************************)

\* RL success implies policy feasibility
CoherenceRL_Policy(bundle) ==
    \A rl \in bundle.rl_certs:
    \A pol \in bundle.policy_certs:
        (rl.converged /\ pol.target_class = rl.target_class) =>
        pol.reachable

\* Exploration coverage aligns with imitation demos
CoherenceExploration_Imitation(bundle) ==
    \A exp \in bundle.exploration_certs:
    \A imit \in bundle.imitation_certs:
        (imit.n_demonstrations > 0) =>
        (exp.states_visited >= imit.n_demonstrations)

\* Filter success relates to inference identifiability
CoherenceFilter_Inference(bundle) ==
    \A filt \in bundle.filter_certs:
    \A inf \in bundle.inference_certs:
        \* Observable filter + failed inference is suspicious
        (filt.fail_type = NULL /\ inf.fail_type \in InferenceFailType) =>
        (inf.fail_type = "TREEWIDTH_TOO_HIGH")  \* Only treewidth is acceptable

\* Bundle is coherent if all rules hold
BundleCoherent(bundle) ==
    /\ CoherenceRL_Policy(bundle)
    /\ CoherenceExploration_Imitation(bundle)
    /\ CoherenceFilter_Inference(bundle)

(****************************************************************************)
(* KEY INVARIANTS (Scale-Bearing Properties)                                *)
(****************************************************************************)

\* Invariant 1: Every certificate is either success or obstruction (never silent failure)
NoSilentFailures(cert) ==
    cert.status \in {"SUCCESS", "OBSTRUCTION"}

\* Invariant 2: Obstructions have witnesses
ObstructionHasWitness(cert) ==
    (cert.status = "OBSTRUCTION") => (Len(cert.witness) > 0)

\* Invariant 3: Regret is non-negative
RegretNonNegative(exp_cert) ==
    exp_cert.regret_witness.cumulative_regret >= 0

\* Invariant 4: SCC pruning efficiency bounded
PruningEfficiencyBounded(mcts_cert) ==
    mcts_cert.pruning_efficiency <= 100

(****************************************************************************)
(* THEOREM: Failure is First-Class                                          *)
(*                                                                          *)
(* Every decision layer admits a finite, machine-checkable witness or       *)
(* obstruction. Failures are not error codes - they are constructive        *)
(* proofs of impossibility.                                                 *)
(****************************************************************************)

THEOREM FailureFirstClass ==
    \A cert \in Certificate:
        NoSilentFailures(cert) /\ ObstructionHasWitness(cert)

=============================================================================
