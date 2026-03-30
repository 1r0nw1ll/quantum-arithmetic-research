# QA Mapping: Algorithms for Decision Making (MIT Press)

## Reference

**Book**: Kochenderfer, Wheeler, & Wray (2022). *Algorithms for Decision Making*. MIT Press.
**URL**: https://algorithmsbook.com
**Mapping Date**: 2026-01-19

---

## Executive Summary

The MIT Decision Making book covers the complete stack from probabilistic reasoning through
sequential decisions, model uncertainty, and state uncertainty. This document maps each
major concept to its QA-native equivalent, showing how QA arithmetic provides a unified
substrate for decision-theoretic reasoning.

**Key Insight**: Where the book treats state spaces as given, QA generates state spaces
from tuple dynamics. Where the book uses continuous value functions, QA uses discrete
invariant packets. Where the book requires model parameters, QA derives structure from
reachability.

---

## Part I: Probabilistic Reasoning → QA Tuple Distributions

### Chapter 2: Probability Distributions

| Book Concept | QA Equivalent |
|-------------|---------------|
| Probability distribution P(X) | QA tuple frequency over orbits |
| Random variable X | Tuple coordinate (b, e, d, a) |
| Expectation E[X] | Invariant mean over state class |
| Variance Var(X) | Packet dispersion within orbit |

**QA Structure**: The mod-24 QA system partitions into orbits:
- 24-cycle Cosmos (72 starting pairs)
- 8-cycle Satellite (8 pairs)
- 1-cycle Singularity (fixed point)

Probability distributions emerge as *empirical frequencies* over these orbits under
generator application.

### Chapter 3: Bayesian Networks

| Book Concept | QA Equivalent |
|-------------|---------------|
| Directed acyclic graph | QA generator dependency graph |
| Conditional independence | Invariant separation |
| d-separation | Orbit isolation theorem |
| Bayes' rule update | Posterior tuple distribution |

**QA Implementation**:
```python
# Bayesian network → QA graph
class QABayesianNetwork:
    def __init__(self, qa_engine):
        self.engine = qa_engine
        self.dependency_graph = self.extract_generator_dag()

    def conditional_independence(self, A, B, given_C):
        """Check if invariant A ⊥ B | C via orbit structure."""
        orbit_A = self.engine.get_orbit_class(A)
        orbit_B = self.engine.get_orbit_class(B)
        orbit_C = self.engine.get_orbit_class(C)
        return not self.engine.orbits_connected(orbit_A, orbit_B, blocked=orbit_C)
```

### Chapter 4: Inference in Probabilistic Models

| Book Concept | QA Equivalent |
|-------------|---------------|
| Variable elimination | Invariant marginalization |
| Belief propagation | Tuple message passing |
| Exact inference | BFS reachability |
| Approximate inference | QAWM sampling |

---

## Part II: Sequential Problems → QA Reachability

### Chapter 5: Markov Decision Processes

| Book Concept | QA Equivalent |
|-------------|---------------|
| State space S | QA state lattice Z²₂₄ |
| Action space A | Generator set {σ, λ, μ, ν, ...} |
| Transition T(s'|s,a) | Deterministic: s' = g(s) |
| Reward R(s,a) | Invariant delta: Δpacket(s,a) |
| Discount γ | Depth horizon k |

**Critical Difference**: QA MDPs are *deterministic* and *structure-preserving*:
- Transitions are group operations, not probabilistic
- Value comes from reaching target states, not accumulated reward
- Policy optimality = shortest path in generator graph

**QA MDP Schema**:
```python
@dataclass
class QAMDP:
    """MDP over QA state space."""
    state_space: QALattice  # (b, e) pairs in Z²₂₄
    generators: Set[Generator]  # Action space
    target_class: Set[QAState]  # Goal states
    horizon: int  # Maximum depth k

    def transition(self, state: QAState, gen: Generator) -> QAState:
        """Deterministic transition."""
        return self.state_space.apply(gen, state)

    def is_legal(self, state: QAState, gen: Generator) -> bool:
        """Check invariant preservation."""
        next_state = self.transition(state, gen)
        return self.state_space.preserves_invariants(state, next_state)
```

### Chapter 6: Exact Solution Methods (Value Iteration, Policy Iteration)

| Book Concept | QA Equivalent |
|-------------|---------------|
| Value iteration | BFS distance propagation |
| Policy iteration | Generator ranking refinement |
| Bellman equation | Reachability recurrence |
| Optimal policy π* | Shortest-path generator sequence |

**QA Bellman Equation**:
```
V*(s) = min_g { 1 + V*(g(s)) }  if s ∉ Target
V*(s) = 0                        if s ∈ Target
```

This is *exactly* BFS shortest path, not approximate value iteration.

### Chapter 7: Approximate Value Functions

| Book Concept | QA Equivalent |
|-------------|---------------|
| Linear function approximation | Invariant linear combination |
| Neural network approximation | QAWM embedding |
| Basis functions | Packet components {B, E, D, A, X, C, F, H, φ₉, φ₂₄} |

**QA Approximation**:
```python
class QAValueApproximator:
    def __init__(self, packet_weights):
        self.weights = packet_weights  # Learned weights on invariants

    def value(self, state: QAState) -> float:
        packet = compute_qa_packet(state)
        return sum(w * packet[inv] for inv, w in self.weights.items())
```

### Chapter 8: Online Planning (MCTS, Forward Search)

| Book Concept | QA Equivalent |
|-------------|---------------|
| Monte Carlo Tree Search | QA-guided BFS with rollouts |
| UCB exploration | Orbit-aware exploration bonus |
| Rollout policy | Random-legal generator sampling |
| Value backup | Reachability probability update |

**QA MCTS Enhancement**: Use orbit structure to prune search:
- If state s is in unreachable SCC, abandon subtree
- Use QAWM to predict return-in-k without simulation

---

## Part III: Model Uncertainty → QA Structure Learning

### Chapter 9: Exploration and Exploitation

| Book Concept | QA Equivalent |
|-------------|---------------|
| ε-greedy | Random-legal with probability ε |
| UCB | Invariant uncertainty bonus |
| Thompson sampling | Posterior over QAWM parameters |
| Regret bounds | Steps to target vs optimal |

### Chapter 10: Belief-State Planning (POMDPs)

| Book Concept | QA Equivalent |
|-------------|---------------|
| Belief state b(s) | Distribution over QA state |
| Observation model | Packet measurement with noise |
| Belief update | Bayesian packet filter |
| Value over beliefs | Expected reachability |

**QA Belief State**:
```python
@dataclass
class QABeliefState:
    """Probability distribution over QA states."""
    particles: Dict[QAState, float]  # state → probability

    def update(self, observation: Dict[str, int]):
        """Bayesian update given observed packet values."""
        new_particles = {}
        for state, prob in self.particles.items():
            packet = compute_qa_packet(state)
            likelihood = self.observation_likelihood(packet, observation)
            new_particles[state] = prob * likelihood
        # Normalize
        total = sum(new_particles.values())
        return QABeliefState({s: p/total for s, p in new_particles.items()})
```

### Chapter 11: Model-Based Methods

| Book Concept | QA Equivalent |
|-------------|---------------|
| Model learning | QAWM training from transitions |
| Model-based planning | Planning on learned QAWM |
| Dyna architecture | Interleave QAWM training + planning |
| Model error | Reachability prediction error |

**This is exactly QAWM's role**: Learn QA transition structure from sparse oracle queries,
then plan using learned model.

---

## Part IV: State Uncertainty → QA Partial Observability

### Chapter 12: Reinforcement Learning

| Book Concept | QA Equivalent |
|-------------|---------------|
| Q-learning | Generator-value learning |
| SARSA | On-policy generator learning |
| Policy gradient | RML policy optimization |
| Actor-critic | QAWM (critic) + RML (actor) |

**QA-RL Distinction**: Standard RL assumes rewards are given. QA-RL derives "reward"
from reachability structure:
- Positive signal: moved closer to target (by BFS distance)
- Negative signal: moved away or hit obstruction

### Chapter 13: Imitation Learning

| Book Concept | QA Equivalent |
|-------------|---------------|
| Behavioral cloning | Copy oracle generator choices |
| Inverse RL | Infer target class from demonstrations |
| DAgger | Query oracle at QAWM uncertainty |

---

## QA Policy Certificate Schema

Building on the Understanding Certificate framework, we define policy certificates:

```python
@dataclass
class PolicyCertificate:
    """Certificate for a decision-making policy.

    Certifies that a policy achieves a target under specified conditions.
    """
    schema: str = "qa_policy_cert/v1"

    # Policy identification
    policy_id: str = ""
    policy_type: str = ""  # "random_legal", "qawm_greedy", "rml", "optimal"

    # Task specification
    target_class_description: str = ""
    horizon: int = 0
    generator_set: Set[GeneratorRef] = field(default_factory=set)

    # Performance certificate
    success_rate: Optional[Fraction] = None  # Empirical success rate
    expected_steps: Optional[Fraction] = None  # E[steps to target | success]
    oracle_calls_per_step: Optional[Fraction] = None

    # Reachability bounds
    reachability_guarantee: bool = False  # True if proven to reach target
    obstruction_if_fail: Optional[ObstructionEvidence] = None

    # Comparison baseline
    baseline_policy: Optional[str] = None
    improvement_over_baseline: Optional[Fraction] = None

    # Derivation (anti-ad-hoc)
    training_witness: Optional[DerivationWitness] = None
    evaluation_witness: Optional[DerivationWitness] = None
```

### Failure Algebra for Policies

Extend FailType enum:

```python
# Policy-specific failures
POLICY_DIVERGED = "policy_diverged"  # Policy loops forever
POLICY_STUCK = "policy_stuck"  # Policy chose illegal move, no fallback
HORIZON_EXCEEDED = "horizon_exceeded"  # Didn't reach target in k steps
EXPLORATION_EXHAUSTED = "exploration_exhausted"  # Ran out of exploration budget
```

---

## Concrete Demo: Gridworld MDP

Map classic gridworld to QA:

```
Standard Gridworld:           QA Mapping:
┌───┬───┬───┬───┐            ┌───────────┬───────────┬───────────┬───────────┐
│ S │   │   │   │            │(1,1)      │(1,2)      │(1,3)      │(1,4)      │
├───┼───┼───┼───┤            ├───────────┼───────────┼───────────┼───────────┤
│   │ X │   │   │    →       │(2,1)      │ BLOCKED   │(2,3)      │(2,4)      │
├───┼───┼───┼───┤            ├───────────┼───────────┼───────────┼───────────┤
│   │   │   │   │            │(3,1)      │(3,2)      │(3,3)      │(3,4)      │
├───┼───┼───┼───┤            ├───────────┼───────────┼───────────┼───────────┤
│   │   │   │ G │            │(4,1)      │(4,2)      │(4,3)      │ TARGET    │
└───┴───┴───┴───┘            └───────────┴───────────┴───────────┴───────────┘

Actions: {UP, DOWN, LEFT, RIGHT}  →  Generators: {σ:(+1,0), σ⁻¹:(-1,0), λ:(0,+1), λ⁻¹:(0,-1)}
```

**QA Certificate for Optimal Policy**:
```json
{
  "schema": "qa_policy_cert/v1",
  "policy_id": "gridworld_bfs_optimal",
  "policy_type": "optimal",
  "target_class_description": "State (4,4)",
  "horizon": 6,
  "generator_set": ["σ", "σ_inv", "λ", "λ_inv"],
  "success_rate": "1/1",
  "expected_steps": "6/1",
  "reachability_guarantee": true,
  "derivation_witness": {
    "invariant_name": "optimal_path_length",
    "derivation_operator": "bfs_shortest_path",
    "input_data": {"start": "(1,1)", "target": "(4,4)", "blocked": "(2,2)"},
    "output_value": 6,
    "verifiable": true
  }
}
```

---

## Integration with Existing QA Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Application Layer: Decision Making Algorithms                  │
│  - Value Iteration, Policy Iteration, MCTS, RL                 │
├─────────────────────────────────────────────────────────────────┤
│  Policy Certificates (qa_policy_cert/v1)                       │
│  - Success guarantees, obstruction evidence, comparison        │
├─────────────────────────────────────────────────────────────────┤
│  Understanding Certificates (qa_understanding_cert/v2)         │
│  - Key steps, strategy, derivation witnesses                   │
├─────────────────────────────────────────────────────────────────┤
│  QAWM (QA World Model)                                         │
│  - Learned transition structure, return-in-k predictions       │
├─────────────────────────────────────────────────────────────────┤
│  RML Policy Layer                                              │
│  - Meta-learning over QAWM hints, generator preferences        │
├─────────────────────────────────────────────────────────────────┤
│  QA Oracle                                                     │
│  - Ground-truth transitions, legality checks, BFS              │
├─────────────────────────────────────────────────────────────────┤
│  QA Engine                                                     │
│  - Tuple arithmetic, invariant packets, orbit structure        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Theorems Mapped

### Theorem: Policy Existence (Book Theorem 5.4.1)

**Book Statement**: For any finite MDP, there exists an optimal stationary policy.

**QA Translation**: For any finite QA lattice with reachable target class, there exists
an optimal generator sequence.

**QA Proof**:
1. QA lattice is finite (|Z²₂₄| = 576 states)
2. Generator set is finite
3. BFS from any state terminates
4. BFS path is optimal (shortest)
5. Generator sequence extracted from BFS path is optimal policy

### Theorem: Value Iteration Convergence (Book Theorem 6.2.2)

**Book Statement**: Value iteration converges to V* in finite MDPs.

**QA Translation**: BFS distance converges to true shortest path in QA lattice.

**QA Proof**: BFS is exact for unweighted graphs. QA lattice is an unweighted graph
where edges are generator applications. Therefore BFS computes exact shortest paths.

### Theorem: Exploration-Exploitation Tradeoff (Book Chapter 9)

**Book Statement**: Optimal exploration requires balancing known rewards vs uncertainty.

**QA Translation**: Optimal QAWM-guided policy balances:
- Known reachability (exploit QAWM predictions)
- Unknown structure (explore unvisited states/generators)

**QA Resolution**: RML policy (rml_policy.py) uses learned weights over QAWM hints,
adapting exploration based on trajectory success.

---

## Summary Table: Book → QA Dictionary

| Decision-Making Concept | QA Primitive |
|------------------------|--------------|
| State | QA tuple (b, e, d, a) |
| Action | Generator {σ, λ, μ, ν, ...} |
| Transition | Tuple update: s' = g(s) |
| Reward | Invariant delta or target indicator |
| Value function | BFS distance to target |
| Policy | Generator selection function |
| Belief | Distribution over QA states |
| Observation | Partial packet measurement |
| Model | QAWM learned structure |
| Exploration | Random-legal or UCB on generators |
| Exploitation | QAWM-greedy policy |
| Certificate | PolicyCertificate with derivation |

---

## Next Steps

1. **Implement PolicyCertificate** in qa_certificate.py
2. **Create gridworld demo** showing BFS-optimal policy certificate
3. **Extend RML policy** with policy certificate generation
4. **Add comparison experiments**: Random-legal vs QAWM-greedy vs RML vs BFS-optimal
