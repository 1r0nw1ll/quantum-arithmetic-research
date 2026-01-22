# QA Decision Certificate Spine

**Certificate-grade coverage of MIT "Algorithms for Decision Making"**

This document specifies the complete certificate spine mapping the MIT book chapters to QA-native certificates with strict validators and recompute hooks.

---

## Overview

| Chapter | Certificate Type | QA-Native Feature | Recompute Hook |
|---------|-----------------|-------------------|----------------|
| Ch. 3-4 | `InferenceCertificate` | VE = graph reduction | `recompute_ve_marginal()` |
| Ch. 7 | `PolicyCertificate` | BFS = shortest reachability | - |
| Ch. 8 | `MCTSCertificate` | SCC pruning witness | - |
| Ch. 9 | `ExplorationCertificate` | Regret = steps - BFS optimal | - |
| Ch. 9-11 | `FilterCertificate` | Kalman/particle estimation | `recompute_kalman_update()` |
| Ch. 12 | `RLCertificate` | Reward = distance_delta | `recompute_q_learning_update()` |
| Ch. 13 | `ImitationCertificate` | IRL = target-class inference | - |

**Total: 295 tests passing**

---

## Certificate Types

### 1. InferenceCertificate (Ch. 3-4)

**Purpose**: Certify probabilistic inference over factor graphs.

**Key Fields**:
- `variables`, `variable_domains`, `factors`
- `query_variables`, `evidence`
- `marginal`: Dict[str, Fraction] (exact arithmetic)
- `method_proof`: VE elimination order or BP iterations

**Failure Modes** (`InferenceFailType`):
- `TREEWIDTH_TOO_HIGH`: Exact inference intractable
- `MESSAGE_DIVERGENCE`: BP didn't converge
- `EVIDENCE_INCONSISTENT`: P(evidence) = 0

**Strict Validator Rules**:
- BP + is_tree=False + exact=True → **violation** (not just warning)
- BP + is_tree=True + not exact → warning
- Success requires method_proof or inference_witness

**Recompute Hook**: `recompute_ve_marginal(cert, factor_tables)`
- Runs sum-product algorithm
- Verifies claimed marginal matches recomputation

---

### 2. PolicyCertificate (Ch. 7)

**Purpose**: Certify that a policy achieves a target under specified conditions.

**Key Fields**:
- `policy_type`: "bfs_optimal", "qawm_greedy", "rml"
- `generator_set`: List[GeneratorRef]
- `reachability_guarantee`, `optimality_guarantee`
- `optimality_proof`: OptimalityProof with `optimal_distance`

**Failure Modes** (`PolicyFailType`):
- `NO_PATH_EXISTS`: Target unreachable
- `HORIZON_EXCEEDED`: Didn't reach in time
- `OBSTRUCTION`: Hit obstruction class

**QA-Native Insight**: BFS-optimal = shortest reachability path in generator graph.

---

### 3. MCTSCertificate (Ch. 8)

**Purpose**: Certify Monte Carlo Tree Search with QA-native pruning.

**Key Fields**:
- `root_state`, `best_action`, `expected_return`, `action_values`
- `method_proof`: exploration rule (UCB1, PUCT), backup operator, n_rollouts
- `scc_pruning_witness`: **QA differentiator**
  - `scc_computation_hash`
  - `nodes_pruned`
  - `unreachable_scc_ids`
- `qawm_return_witness`: QAWM model prediction

**Failure Modes** (`MCTSFailType`):
- `BUDGET_EXHAUSTED`: Ran out of rollouts
- `SCC_UNREACHABLE`: Target SCC unreachable from root
- `VALUE_DIVERGENCE`: Value estimates unstable

**QA-Native Insight**: SCC pruning = certified topological obstruction inside rollout tree.

**Pruning Efficiency**: `1 - (qa_rollouts / vanilla_rollouts)`

---

### 4. ExplorationCertificate (Ch. 9)

**Purpose**: Certify exploration-exploitation strategy with regret analysis.

**Key Fields**:
- `method_proof`: ExplorationMethod (UCB1, Thompson, ε-greedy)
- `uncertainty_measure`: visit_count, posterior_variance, **packet_uncertainty**
- `regret_witness`:
  - `actual_steps`
  - `optimal_steps`
  - `cumulative_regret = actual - optimal`
  - `regret_bound`: "O(sqrt(T))" etc.

**Failure Modes** (`ExplorationFailType`):
- `HIGH_REGRET`: Regret exceeded threshold
- `EXPLORATION_COLLAPSED`: Premature convergence
- `BUDGET_EXHAUSTED`

**QA-Native Insight**: Regret = (steps to target) - (BFS optimal steps). Concrete reachability metric.

---

### 5. FilterCertificate (Ch. 9-11)

**Purpose**: Certify state estimation over dynamical systems.

**Key Fields**:
- `state_dimension`, `observation_dimension`, `state_names`
- `state_estimate`: Dict[str, Fraction]
- `covariance_trace` or `credible_interval_width`
- `method_proof`: FilterMethod (Kalman, Particle, Histogram)

**Failure Modes** (`FilterFailType`):
- `PARTICLE_DEGENERACY`: ESS too low
- `STATE_UNOBSERVABLE`: Observability rank deficient
- `FILTER_DIVERGED`: Estimate drifted from truth
- `COVARIANCE_SINGULAR`

**Recompute Hook**: `recompute_kalman_update(cert, A, H, Q, R, x0, P0, observations)`
- Runs Kalman filter equations (exact Fraction arithmetic)
- Verifies state estimate and covariance trace match

---

### 6. RLCertificate (Ch. 12)

**Purpose**: Certify reinforcement learning training run.

**Key Fields**:
- `method_proof`: RLAlgorithm (Q_LEARNING, SARSA, PPO, DQN)
- `reward_spec`: **distance_delta** (QA-native), obstruction_penalty, goal_reward
- `learning_rate`, `discount_factor`
- `q_value_witness`: sample transitions for audit
- `generator_set`: actions = generators

**Failure Modes** (`RLFailType`):
- `CONVERGENCE_TIMEOUT`
- `VALUE_DIVERGENCE`: Q-values exploded
- `EXPLORATION_FAILURE`: Never found target

**Strict Validator Rules**:
- Q-learning requires `discount_factor` and `learning_rate`
- Distance_delta reward noted as "QA-native"

**Recompute Hook**: `recompute_q_learning_update(cert, transitions)`
- Verifies: `Q_after = Q_before + α(r + γ·max_Q_next - Q_before)`
- Exact Fraction arithmetic

**QA-Native Insight**: Q-learning = generator-value learning. Reward = BFS distance delta.

---

### 7. ImitationCertificate (Ch. 13)

**Purpose**: Certify learning from demonstrations.

**Key Fields**:
- `method_proof`: ImitationMethod (BEHAVIORAL_CLONING, INVERSE_RL, DAGGER)
- `demonstration_witness`: n_trajectories, dataset_hash, coverage
- `inverse_rl_witness`: **QA differentiator**
  - `inferred_target_class`
  - `confidence`
  - `identifiable`: boolean
  - `alternative_targets`: if non-identifiable
- `dagger_witness`: oracle queries, uncertainty-triggered

**Failure Modes** (`ImitationFailType`):
- `REWARD_NON_IDENTIFIABLE`: IRL identifiability failure
- `DISTRIBUTION_SHIFT`: BC covariate shift
- `ORACLE_BUDGET_EXHAUSTED`: DAgger ran out of queries

**QA-Native Insight**: Inverse RL = target-class inference. Reuses identifiability machinery.

---

## Cross-Certificate Coherence

The `CertificateBundle` class groups certificates and validates coherence:

### Coherence Rules

1. **RL ↔ Policy**:
   - Generator sets must be compatible
   - Target classes must match
   - RL avg_steps should be ≥ policy optimal_distance

2. **Imitation ↔ Exploration**:
   - Demo coverage should align with exploration coverage
   - Sparse demo data relative to exploration steps triggers warning

3. **Filter ↔ Inference**:
   - State observability relates to inference identifiability
   - Filter success + inference failure is suspicious

4. **MCTS ↔ Exploration**:
   - Exploration methods should be consistent
   - UCB constants should match if both use UCB

### Bundle Validation

```python
bundle = CertificateBundle(
    policy_certificates=[...],
    rl_certificates=[...],
    ...
)
result = validate_bundle_coherence(bundle)
# result.coherent: bool
# result.cross_references_checked: int
# result.violations: List[str]
# result.warnings: List[str]
```

### Bundle Manifest

```python
manifest = bundle.to_manifest()
# {
#   "bundle_id": "...",
#   "bundle_hash": "sha256:...",
#   "total_certificates": 7,
#   "certificate_counts": {...}
# }
```

---

## Invariants (Scale-Bearing)

All certificates maintain these invariants:

1. **Exact arithmetic**: `Fraction` only, no floats
2. **Deterministic serialization**: Same inputs → same JSON
3. **Machine-checkable witnesses**: Every claim has auditable evidence
4. **Failures are first-class**: Obstruction evidence, not just "failed"

---

## Demo: End-to-End Spine

`demos/decision_spine_demo.py` runs a 5x5 gridworld through all layers:

```
Planning (BFS optimal, 8 steps)
    ↓
MCTS (UCB1, 60% pruning via SCC)
    ↓
Exploration (UCB1, regret=50)
    ↓
Inference (VE, P(Goal|position)=1)
    ↓
Filtering (Kalman, x≈4, y≈4)
    ↓
RL (Q-learning, distance_delta reward, 95% success)
    ↓
Imitation (IRL, target inferred with 98% confidence)
```

**Output**: `spine_bundle.json` with 7 certificates and coherence validation.

---

## Files

| File | Description |
|------|-------------|
| `qa_certificate.py` | All certificate dataclasses + validators |
| `test_understanding_certificate.py` | 295 tests |
| `demos/decision_spine_demo.py` | End-to-end demo |
| `demos/spine_bundle.json` | Exported bundle |
| `demos/inference_demo.py` | Inference scenarios |
| `demos/filter_demo.py` | Filter scenarios |

---

## Summary

The certificate spine provides **compiler-level rigor** for decision-making:

- Every chapter maps to a certificate type
- Every certificate has strict validators
- Failures have machine-checkable witnesses
- Cross-certificate coherence is enforced
- All arithmetic is exact (Fraction)
- Recompute hooks enable "auditable mode"

This is the QA-native decision stack: **reachability geometry as the unified substrate**.
