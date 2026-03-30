#!/usr/bin/env python3
"""
Gridworld Belief State Demo (POMDP Extension)

Extends the gridworld policy demo with partial observability:
- Agent has imperfect knowledge of its position
- Noisy sensor gives uncertain observations
- Belief state = probability distribution over possible states
- BELIEF_DEGENERACY failure when belief becomes unusable

Maps MIT "Algorithms for Decision Making" Chapter 10 (POMDPs) to QA:
- Belief state → Distribution over QA states
- Observation model → Noisy packet measurement
- Belief update → Bayesian filtering
- Value over beliefs → Expected reachability

Output: gridworld_belief_cert.json
"""

import sys
import json
import math
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from fractions import Fraction

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    PolicyCertificate,
    PolicyEvaluationStats,
    PolicyFailType,
    Generator,
    GeneratorRef,
    DerivationWitness,
    Strategy,
    ObstructionEvidence,
    FailType,
    validate_policy_certificate,
    to_scalar,
)


# ============================================================================
# GRIDWORLD WITH BELIEF STATE
# ============================================================================

@dataclass(frozen=True)
class GridState:
    """State in gridworld = (row, col) coordinate pair."""
    row: int
    col: int

    def __repr__(self):
        return f"({self.row},{self.col})"


class GridWorld:
    """Gridworld environment (same as base demo)."""

    def __init__(self, rows: int, cols: int, blocked: Set[GridState] = None):
        self.rows = rows
        self.cols = cols
        self.blocked = blocked or set()
        self.generators = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

    def is_valid(self, state: GridState) -> bool:
        if state.row < 0 or state.row >= self.rows:
            return False
        if state.col < 0 or state.col >= self.cols:
            return False
        if state in self.blocked:
            return False
        return True

    def is_legal(self, state: GridState, gen: str) -> bool:
        if gen not in self.generators:
            return False
        next_state = self.apply(state, gen)
        return self.is_valid(next_state)

    def apply(self, state: GridState, gen: str) -> GridState:
        dr, dc = self.generators[gen]
        return GridState(state.row + dr, state.col + dc)

    def all_states(self) -> List[GridState]:
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                s = GridState(r, c)
                if s not in self.blocked:
                    states.append(s)
        return states


# ============================================================================
# NOISY SENSOR (Observation Model)
# ============================================================================

@dataclass
class NoisySensor:
    """
    Imperfect position sensor for POMDP setting.

    Observation model:
    - With probability (1 - noise_level): observe true state
    - With probability noise_level: observe a random adjacent state or "LOST"

    This creates partial observability where the agent must maintain
    a belief state (distribution over possible positions).
    """
    world: GridWorld
    noise_level: float = 0.3  # P(wrong observation)

    def __post_init__(self):
        import random
        self.rng = random.Random(42)

    def observe(self, true_state: GridState) -> str:
        """
        Generate noisy observation of true state.

        Returns:
            State string if position observed, "LOST" if failed to observe
        """
        if self.rng.random() > self.noise_level:
            # Correct observation
            return str(true_state)

        # Noisy observation: adjacent state or "LOST"
        adjacent = []
        for gen, (dr, dc) in self.world.generators.items():
            adj = GridState(true_state.row + dr, true_state.col + dc)
            if self.world.is_valid(adj):
                adjacent.append(adj)

        if adjacent and self.rng.random() > 0.3:
            # Random adjacent
            return str(self.rng.choice(adjacent))
        else:
            # Complete loss
            return "LOST"

    def observation_likelihood(self, obs: str, true_state: GridState) -> float:
        """
        P(observation | true_state) for belief update.

        Observation model:
        - P(correct) = 1 - noise_level
        - P(adjacent) = noise_level * 0.7 / n_adjacent
        - P(LOST) = noise_level * 0.3
        """
        if obs == str(true_state):
            return 1.0 - self.noise_level

        if obs == "LOST":
            return self.noise_level * 0.3

        # Check if obs is adjacent to true_state
        try:
            # Parse observation
            obs_clean = obs.strip("()")
            r, c = map(int, obs_clean.split(","))
            obs_state = GridState(r, c)
        except:
            return 0.0  # Invalid observation

        # Check adjacency
        for gen, (dr, dc) in self.world.generators.items():
            adj = GridState(true_state.row + dr, true_state.col + dc)
            if obs_state == adj:
                n_adjacent = sum(1 for g in self.world.generators
                                 if self.world.is_valid(
                                     GridState(true_state.row + self.world.generators[g][0],
                                               true_state.col + self.world.generators[g][1])))
                return self.noise_level * 0.7 / max(1, n_adjacent)

        return 0.0  # Not adjacent


@dataclass
class AliasedSensor:
    """
    Sensor that produces identical observations for aliased regions.

    This creates NON_IDENTIFIABLE scenarios where multiple states
    are observationally indistinguishable - the agent cannot determine
    which state it's actually in.

    Example: If aliased_regions = {((0,0), (0,4)), ((1,0), (1,4))},
    then states (0,0) and (0,4) produce identical observations "REGION_A",
    and states (1,0) and (1,4) produce "REGION_B".
    """
    world: GridWorld
    aliased_regions: Dict[str, Set[GridState]] = field(default_factory=dict)
    noise_level: float = 0.1  # Small noise on top of aliasing

    def __post_init__(self):
        import random
        self.rng = random.Random(42)

        # Build reverse mapping: state -> region_id
        self.state_to_region = {}
        for region_id, states in self.aliased_regions.items():
            for state in states:
                self.state_to_region[state] = region_id

    def observe(self, true_state: GridState) -> str:
        """
        Generate observation - aliased states produce same region ID.
        """
        if true_state in self.state_to_region:
            # State is in aliased region - return region ID
            region_id = self.state_to_region[true_state]
            if self.rng.random() > self.noise_level:
                return f"REGION_{region_id}"
            else:
                return "LOST"
        else:
            # Normal state - return coordinate
            if self.rng.random() > self.noise_level:
                return str(true_state)
            else:
                return "LOST"

    def observation_likelihood(self, obs: str, hypothesized_state: GridState) -> float:
        """P(observation | hypothesized_state) for aliased sensor.

        Key insight: For belief update, we need P(obs | hypothesis).
        If hypothesis is in an aliased region and obs matches that region,
        the hypothesis is consistent but NOT uniquely identified.
        """
        if obs == "LOST":
            return self.noise_level

        if hypothesized_state in self.state_to_region:
            # Hypothesis is in aliased region
            region_id = self.state_to_region[hypothesized_state]
            expected_obs = f"REGION_{region_id}"
            if obs == expected_obs:
                # Observation is consistent with hypothesis
                # All states in this region are equally consistent
                return 1.0 - self.noise_level
            return 0.0
        else:
            # Hypothesis is in non-aliased region
            if obs == str(hypothesized_state):
                return 1.0 - self.noise_level
            # Check if observation is a region observation
            if obs.startswith("REGION_"):
                # This obs came from aliased region, hypothesis is not in aliased region
                return 0.0
            return 0.0

    def get_aliased_states(self, state: GridState) -> List[GridState]:
        """Return all states aliased with the given state."""
        if state in self.state_to_region:
            region_id = self.state_to_region[state]
            return list(self.aliased_regions[region_id])
        return [state]


@dataclass
class EnhancedSensor:
    """
    Sensor with additional observation channel that breaks aliasing.

    This demonstrates observer upgrade: by adding a "column indicator"
    observation, previously aliased states become distinguishable.

    The observation now has TWO components:
    1. Region/position (same as AliasedSensor)
    2. Column indicator (OBS:COL_x) - the extra channel

    This breaks NON_IDENTIFIABLE: even if states are in same "region",
    the column indicator distinguishes them.
    """
    world: GridWorld
    base_aliased_regions: Dict[str, Set[GridState]] = field(default_factory=dict)
    noise_level: float = 0.05

    def __post_init__(self):
        import random
        self.rng = random.Random(42)

        # Build reverse mapping
        self.state_to_region = {}
        for region_id, states in self.base_aliased_regions.items():
            for state in states:
                self.state_to_region[state] = region_id

    def observe(self, true_state: GridState) -> str:
        """
        Generate compound observation: region + column indicator.

        Format: "REGION_X|COL_y" for aliased states
                "(r,c)|COL_c" for non-aliased states
        """
        if self.rng.random() < self.noise_level:
            return "LOST"

        # Column indicator is the distinguishing observation
        col_obs = f"COL_{true_state.col}"

        if true_state in self.state_to_region:
            region_id = self.state_to_region[true_state]
            return f"REGION_{region_id}|{col_obs}"
        else:
            return f"{true_state}|{col_obs}"

    def observation_likelihood(self, obs: str, hypothesized_state: GridState) -> float:
        """P(observation | hypothesized_state) for enhanced sensor."""
        if obs == "LOST":
            return self.noise_level

        # Parse compound observation
        if "|" not in obs:
            return 0.0

        parts = obs.split("|")
        if len(parts) != 2:
            return 0.0

        region_obs, col_obs = parts

        # Check column indicator FIRST (the distinguishing channel)
        expected_col = f"COL_{hypothesized_state.col}"
        if col_obs != expected_col:
            return 0.0  # Column mismatch → probability zero

        # Column matches, now check region/position
        if hypothesized_state in self.state_to_region:
            region_id = self.state_to_region[hypothesized_state]
            expected_region = f"REGION_{region_id}"
            if region_obs == expected_region:
                return 1.0 - self.noise_level
            return 0.0
        else:
            if region_obs == str(hypothesized_state):
                return 1.0 - self.noise_level
            return 0.0

    def get_aliased_states(self, state: GridState) -> List[GridState]:
        """With enhanced sensor, no states are aliased (column distinguishes all)."""
        return [state]  # Each state is uniquely identifiable


# ============================================================================
# BELIEF STATE
# ============================================================================

@dataclass
class BeliefState:
    """
    Probability distribution over grid states.

    The belief state is the sufficient statistic for POMDPs:
    it captures all information needed for optimal decision making
    under partial observability.
    """
    probabilities: Dict[GridState, float] = field(default_factory=dict)

    @classmethod
    def uniform(cls, states: List[GridState]) -> "BeliefState":
        """Create uniform belief over given states."""
        p = 1.0 / len(states)
        return cls({s: p for s in states})

    @classmethod
    def point_mass(cls, state: GridState) -> "BeliefState":
        """Create belief concentrated on single state."""
        return cls({state: 1.0})

    def entropy(self) -> float:
        """Shannon entropy of belief distribution."""
        h = 0.0
        for p in self.probabilities.values():
            if p > 1e-10:
                h -= p * math.log2(p)
        return h

    def max_prob(self) -> Tuple[GridState, float]:
        """Return MAP estimate and its probability."""
        if not self.probabilities:
            return None, 0.0
        best_state = max(self.probabilities, key=self.probabilities.get)
        return best_state, self.probabilities[best_state]

    def normalize(self) -> "BeliefState":
        """Return normalized copy."""
        total = sum(self.probabilities.values())
        if total < 1e-10:
            return self  # Can't normalize
        return BeliefState({s: p/total for s, p in self.probabilities.items()})

    def bayesian_update(
        self,
        observation: str,
        sensor: NoisySensor,
    ) -> "BeliefState":
        """
        Bayesian update: P(s|o) ∝ P(o|s) * P(s)

        This is the core belief update in POMDPs.
        """
        new_probs = {}
        for state, prior in self.probabilities.items():
            likelihood = sensor.observation_likelihood(observation, state)
            new_probs[state] = prior * likelihood

        return BeliefState(new_probs).normalize()

    def transition_update(
        self,
        action: str,
        world: GridWorld,
    ) -> "BeliefState":
        """
        Predict belief after taking action (deterministic transitions).

        For deterministic transitions: P(s'|a) = sum_s P(s' = T(s,a)) * P(s)
        """
        new_probs = {}
        for state, prob in self.probabilities.items():
            if world.is_legal(state, action):
                next_state = world.apply(state, action)
            else:
                next_state = state  # Stay in place if illegal
            new_probs[next_state] = new_probs.get(next_state, 0.0) + prob

        return BeliefState(new_probs)

    def to_dict(self) -> Dict[str, float]:
        """Convert to serializable dict."""
        return {str(s): p for s, p in self.probabilities.items() if p > 1e-6}


# ============================================================================
# BELIEF-AWARE POLICIES
# ============================================================================

class BeliefPolicy:
    """Base class for belief-aware policies."""

    def select(self, belief: BeliefState, world: GridWorld, target: GridState) -> Optional[str]:
        """Select action given belief state."""
        raise NotImplementedError


class MAPGreedyPolicy(BeliefPolicy):
    """
    Greedy policy based on MAP (Maximum A Posteriori) estimate.

    Strategy: Treat MAP estimate as true state, then greedy Manhattan.

    Failure mode: If MAP is wrong, greedy toward wrong position.
    """

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.oracle_calls = 0

    def manhattan(self, s: GridState, target: GridState) -> int:
        return abs(s.row - target.row) + abs(s.col - target.col)

    def select(self, belief: BeliefState, world: GridWorld, target: GridState) -> Optional[str]:
        map_state, map_prob = belief.max_prob()

        if map_state is None:
            return None

        if map_state == target:
            return None  # Think we're at target

        # Greedy from MAP estimate
        best_gen = None
        best_dist = float('inf')
        legal_gens = []

        for gen in world.generators:
            self.oracle_calls += 1
            if world.is_legal(map_state, gen):
                next_state = world.apply(map_state, gen)
                dist = self.manhattan(next_state, target)
                legal_gens.append((gen, dist))
                if dist < best_dist:
                    best_dist = dist
                    best_gen = gen

        if not legal_gens:
            return None

        # Tie-breaking
        best_gens = [g for g, d in legal_gens if d == best_dist]
        return self.rng.choice(best_gens)

    def reset(self):
        self.oracle_calls = 0


class EntropyAwarePolicy(BeliefPolicy):
    """
    Policy that considers belief entropy.

    Strategy:
    - If entropy high → information-gathering actions
    - If entropy low → greedy toward target from MAP

    This models the explore/exploit tradeoff in POMDPs.
    """

    def __init__(self, entropy_threshold: float = 2.0, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.entropy_threshold = entropy_threshold
        self.oracle_calls = 0

    def manhattan(self, s: GridState, target: GridState) -> int:
        return abs(s.row - target.row) + abs(s.col - target.col)

    def select(self, belief: BeliefState, world: GridWorld, target: GridState) -> Optional[str]:
        entropy = belief.entropy()
        map_state, map_prob = belief.max_prob()

        if map_state is None:
            return None

        if map_state == target and map_prob > 0.8:
            return None  # Confident we're at target

        # Collect legal moves from MAP
        legal_gens = []
        for gen in world.generators:
            self.oracle_calls += 1
            if world.is_legal(map_state, gen):
                legal_gens.append(gen)

        if not legal_gens:
            return None

        if entropy > self.entropy_threshold:
            # High uncertainty → random exploration
            return self.rng.choice(legal_gens)
        else:
            # Low uncertainty → greedy toward target
            best_gen = None
            best_dist = float('inf')
            for gen in legal_gens:
                next_state = world.apply(map_state, gen)
                dist = self.manhattan(next_state, target)
                if dist < best_dist:
                    best_dist = dist
                    best_gen = gen
            return best_gen

    def reset(self):
        self.oracle_calls = 0


# ============================================================================
# BELIEF EPISODE RUNNER
# ============================================================================

def run_belief_episode(
    policy: BeliefPolicy,
    world: GridWorld,
    sensor: NoisySensor,
    true_start: GridState,
    target: GridState,
    horizon: int,
    initial_belief: BeliefState = None,
) -> Dict:
    """
    Run episode with belief tracking.

    The agent:
    1. Maintains belief state
    2. Selects action based on belief
    3. Takes action (in true state)
    4. Receives observation
    5. Updates belief

    Returns failure certificate data if belief degenerates.
    """
    # Initialize
    true_state = true_start
    belief = initial_belief or BeliefState.uniform(world.all_states())

    trajectory = []
    belief_trace = []
    observations = []

    policy.reset()

    for step in range(horizon):
        # Record pre-step state
        entropy = belief.entropy()
        map_state, map_prob = belief.max_prob()

        belief_trace.append({
            "step": step,
            "true_state": str(true_state),
            "map_state": str(map_state) if map_state else None,
            "map_prob": map_prob,
            "entropy": entropy,
            "belief_size": len([p for p in belief.probabilities.values() if p > 1e-6]),
        })

        # Check for success
        if true_state == target:
            return {
                "success": True,
                "steps": step,
                "oracle_calls": policy.oracle_calls,
                "trajectory": trajectory,
                "belief_trace": belief_trace,
                "observations": observations,
                "final_state": true_state,
                "final_belief": belief.to_dict(),
            }

        # Select action based on belief
        action = policy.select(belief, world, target)

        if action is None:
            # Policy can't decide - compute final belief stats
            final_entropy = belief.entropy()
            final_map, final_map_prob = belief.max_prob()
            return {
                "success": False,
                "failure_mode": "stuck",
                "steps": step,
                "oracle_calls": policy.oracle_calls,
                "trajectory": trajectory,
                "belief_trace": belief_trace,
                "observations": observations,
                "final_state": true_state,
                "final_belief": belief.to_dict(),
                "final_entropy": final_entropy,
                "final_map_prob": final_map_prob,
                "final_map_state": str(final_map) if final_map else None,
            }

        # Execute action in true state
        if world.is_legal(true_state, action):
            true_state = world.apply(true_state, action)

        trajectory.append((str(true_state), action))

        # Update belief for action
        belief = belief.transition_update(action, world)

        # Get observation
        obs = sensor.observe(true_state)
        observations.append(obs)

        # Bayesian update
        belief = belief.bayesian_update(obs, sensor)

        # Check for belief degeneracy
        post_entropy = belief.entropy()
        post_map, post_prob = belief.max_prob()

        if post_prob < 0.01:
            # Belief too diffuse - can't localize
            return {
                "success": False,
                "failure_mode": "belief_too_diffuse",
                "steps": step + 1,
                "oracle_calls": policy.oracle_calls,
                "trajectory": trajectory,
                "belief_trace": belief_trace,
                "observations": observations,
                "final_state": true_state,
                "final_belief": belief.to_dict(),
                "final_entropy": post_entropy,
                "final_map_prob": post_prob,
            }

        # Check for non-identifiable scenario (aliased sensor)
        if hasattr(sensor, 'get_aliased_states'):
            aliased = sensor.get_aliased_states(true_state)
            if len(aliased) >= 2:
                # Check if belief has converged but can't distinguish aliased states
                aliased_probs = [belief.probabilities.get(s, 0) for s in aliased]
                total_aliased_prob = sum(aliased_probs)
                if total_aliased_prob > 0.9 and min(aliased_probs) > 0.1:
                    # Belief has converged to aliased region but can't distinguish
                    return {
                        "success": False,
                        "failure_mode": "non_identifiable",
                        "steps": step + 1,
                        "oracle_calls": policy.oracle_calls,
                        "trajectory": trajectory,
                        "belief_trace": belief_trace,
                        "observations": observations,
                        "final_state": true_state,
                        "final_belief": belief.to_dict(),
                        "final_entropy": post_entropy,
                        "final_map_prob": post_prob,
                        "aliased_states": [str(s) for s in aliased],
                        "aliased_probs": aliased_probs,
                    }

    # Horizon exhausted
    map_state, map_prob = belief.max_prob()
    return {
        "success": False if true_state != target else True,
        "failure_mode": "horizon_exceeded" if true_state != target else None,
        "steps": horizon,
        "oracle_calls": policy.oracle_calls,
        "trajectory": trajectory,
        "belief_trace": belief_trace,
        "observations": observations,
        "final_state": true_state,
        "final_belief": belief.to_dict(),
        "final_entropy": belief.entropy(),
        "final_map_prob": map_prob,
        "map_correct": str(map_state) == str(true_state) if map_state else False,
    }


# ============================================================================
# CERTIFICATE GENERATION
# ============================================================================

def create_belief_failure_certificate(
    world: GridWorld,
    sensor: NoisySensor,
    start: GridState,
    target: GridState,
    horizon: int,
    result: Dict,
) -> PolicyCertificate:
    """
    Create certificate for belief-state failure.

    Failure modes:
    - BELIEF_TOO_DIFFUSE: Entropy exceeded threshold
    - BELIEF_COLLAPSE_WRONG: MAP converged to wrong state
    - HORIZON_EXCEEDED: Ran out of time
    """
    generators = [
        GeneratorRef("GRID", "UP"),
        GeneratorRef("GRID", "DOWN"),
        GeneratorRef("GRID", "LEFT"),
        GeneratorRef("GRID", "RIGHT"),
    ]

    failure_mode = result.get("failure_mode", "unknown")
    final_entropy = result.get("final_entropy", 0)
    final_map_prob = result.get("final_map_prob", 0)
    final_state = result.get("final_state")

    # Determine QA failure type
    if failure_mode == "belief_too_diffuse":
        fail_type = FailType.BELIEF_TOO_DIFFUSE
        policy_fail = PolicyFailType.POLICY_STUCK
        description = f"Belief entropy {final_entropy:.2f} exceeded usability threshold"

        obstruction = ObstructionEvidence(
            fail_type=fail_type,
            belief_entropy=to_scalar(Fraction(int(final_entropy * 100), 100)),
            entropy_threshold=to_scalar(Fraction(5, 1)),  # log2(32) ≈ 5 for 5x5 grid
            observations_received=len(result.get("observations", [])),
        )

    elif failure_mode == "non_identifiable":
        # States are observationally aliased - can't distinguish
        aliased_states = result.get("aliased_states", [])
        fail_type = FailType.NON_IDENTIFIABLE
        policy_fail = PolicyFailType.POLICY_STUCK
        description = f"Cannot distinguish between aliased states: {aliased_states[:3]}..."

        obstruction = ObstructionEvidence(
            fail_type=fail_type,
            aliased_states=aliased_states,
            belief_entropy=to_scalar(Fraction(int(final_entropy * 100), 100)),
            observations_received=len(result.get("observations", [])),
        )

    elif failure_mode == "stuck":
        # Check if belief collapsed to wrong state
        belief_trace = result.get("belief_trace", [])
        if belief_trace:
            last_trace = belief_trace[-1]
            if last_trace.get("map_prob", 0) > 0.8:
                # High confidence but wrong
                fail_type = FailType.BELIEF_COLLAPSE_WRONG
                policy_fail = PolicyFailType.POLICY_STUCK
                description = f"Belief collapsed to wrong state {last_trace.get('map_state')}"

                obstruction = ObstructionEvidence(
                    fail_type=fail_type,
                    belief_true_state=str(final_state),
                    belief_map_state=last_trace.get("map_state"),
                    belief_max_prob=to_scalar(Fraction(int(last_trace.get("map_prob", 0) * 100), 100)),
                    observations_received=len(result.get("observations", [])),
                )
            else:
                # Generic degeneracy
                fail_type = FailType.BELIEF_DEGENERACY
                policy_fail = PolicyFailType.POLICY_STUCK
                description = "Belief state became unusable"

                obstruction = ObstructionEvidence(
                    fail_type=fail_type,
                    belief_entropy=to_scalar(Fraction(int(final_entropy * 100), 100)),
                    observations_received=len(result.get("observations", [])),
                )
        else:
            fail_type = FailType.BELIEF_DEGENERACY
            policy_fail = PolicyFailType.POLICY_STUCK
            description = "Belief state became unusable"
            obstruction = ObstructionEvidence(
                fail_type=fail_type,
                belief_entropy=to_scalar(0),
                observations_received=0,
            )

    else:  # horizon_exceeded
        fail_type = FailType.DEPTH_EXHAUSTED
        policy_fail = PolicyFailType.HORIZON_EXCEEDED
        description = f"Horizon {horizon} exceeded without reaching target"

        grid_gens = {
            Generator("PHYS:GRID_UP"),
            Generator("PHYS:GRID_DOWN"),
            Generator("PHYS:GRID_LEFT"),
            Generator("PHYS:GRID_RIGHT"),
        }

        obstruction = ObstructionEvidence(
            fail_type=fail_type,
            generator_set=grid_gens,
            max_depth_reached=result.get("steps", horizon),
            states_explored=len(result.get("trajectory", [])),
        )

    return PolicyCertificate(
        policy_id="belief_policy_failure",
        policy_type="belief_map_greedy",
        policy_description=description,
        target_class_description=str(target),
        start_class_description=str(start),
        horizon=horizon,
        generator_set=generators,
        reachability_guarantee=False,
        optimality_guarantee=False,
        failure_mode=policy_fail,
        obstruction_if_fail=obstruction,
        training_witness=DerivationWitness(
            invariant_name="belief_failure_trace",
            derivation_operator="belief_episode_execution",
            input_data={
                "start": str(start),
                "target": str(target),
                "horizon": horizon,
                "noise_level": sensor.noise_level,
                "steps_taken": result.get("steps", 0),
                "final_entropy": final_entropy,
                "final_map_prob": final_map_prob,
                "observations_count": len(result.get("observations", [])),
            },
            output_value=0,  # 0 = failure
            verifiable=True,
        ),
        strategy=Strategy(
            type="belief_map_greedy",
            key_insight="Act greedily from MAP estimate, ignoring uncertainty",
            derivation_witness=DerivationWitness(
                invariant_name="strategy:belief_greedy",
                derivation_operator="pomdp_approximation",
                input_data={"method": "map_greedy"},
                output_value=1,
            ),
        ),
        strict_mode=True,
    )


def create_belief_success_certificate(
    world: GridWorld,
    sensor,  # Can be NoisySensor, AliasedSensor, or EnhancedSensor
    start: GridState,
    target: GridState,
    horizon: int,
    result: Dict,
    distinguishing_observation: Optional[str] = None,
    resolved_aliased_states: Optional[List[str]] = None,
) -> PolicyCertificate:
    """Create certificate for successful belief-based navigation.

    Args:
        distinguishing_observation: If this success resolved a NON_IDENTIFIABLE
            scenario, this is the observation channel that broke the aliasing.
        resolved_aliased_states: The states that were formerly aliased but are
            now distinguishable (empty list = no aliasing to resolve).
    """
    generators = [
        GeneratorRef("GRID", "UP"),
        GeneratorRef("GRID", "DOWN"),
        GeneratorRef("GRID", "LEFT"),
        GeneratorRef("GRID", "RIGHT"),
    ]

    # Build input data for training witness
    input_data = {
        "start": str(start),
        "target": str(target),
        "horizon": horizon,
        "noise_level": getattr(sensor, 'noise_level', 0),
        "steps_taken": result["steps"],
    }

    # Add observer upgrade proof if applicable
    if distinguishing_observation is not None:
        input_data["distinguishing_observation"] = distinguishing_observation
        input_data["aliased_states_resolved"] = resolved_aliased_states or []
        input_data["observer_upgrade_applied"] = True
    else:
        input_data["observer_upgrade_applied"] = False

    # Determine policy description
    if distinguishing_observation:
        description = (
            f"Reached target in {result['steps']} steps. "
            f"Observer upgrade ({distinguishing_observation}) resolved aliasing."
        )
    else:
        description = f"Reached target in {result['steps']} steps under partial observability"

    return PolicyCertificate(
        policy_id="belief_policy_success",
        policy_type="belief_map_greedy",
        policy_description=description,
        target_class_description=str(target),
        start_class_description=str(start),
        horizon=horizon,
        generator_set=generators,
        evaluation_stats=PolicyEvaluationStats(
            n_episodes=1,
            successes=1,
            total_steps=result["steps"],
            total_oracle_calls=result["oracle_calls"],
        ),
        reachability_guarantee=False,  # Empirical success, not proven
        optimality_guarantee=False,
        training_witness=DerivationWitness(
            invariant_name="belief_success_trace",
            derivation_operator="belief_episode_execution",
            input_data=input_data,
            output_value=1,  # 1 = success
            verifiable=True,
        ),
        strict_mode=True,
    )


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("=" * 70)
    print("  GRIDWORLD BELIEF STATE DEMO (POMDP)")
    print("  Partial Observability with Noisy Sensor")
    print("=" * 70)

    # Create 5x5 gridworld with obstacles
    #   0 1 2 3 4
    # 0 S . . . .
    # 1 . X . . .
    # 2 . X . . .
    # 3 . . . . .
    # 4 . . . . G
    world = GridWorld(
        rows=5,
        cols=5,
        blocked={GridState(1, 1), GridState(2, 1)},
    )
    start = GridState(0, 0)
    target = GridState(4, 4)
    horizon = 30

    print(f"\nGridworld: {world.rows}x{world.cols}")
    print("Layout:")
    for r in range(world.rows):
        row_str = f"{r} "
        for c in range(world.cols):
            s = GridState(r, c)
            if s == start:
                row_str += "S "
            elif s == target:
                row_str += "G "
            elif s in world.blocked:
                row_str += "X "
            else:
                row_str += ". "
        print(row_str)

    print(f"\nStart: {start}")
    print(f"Target: {target}")
    print(f"Horizon: {horizon}")

    # -------------------------------------------------------------------------
    # SCENARIO 1: Low Noise (Should Succeed)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 1: LOW NOISE (noise_level=0.1)")
    print("-" * 70)

    low_noise_sensor = NoisySensor(world, noise_level=0.1)
    map_policy = MAPGreedyPolicy(seed=42)

    result_low = run_belief_episode(
        map_policy, world, low_noise_sensor,
        start, target, horizon,
        initial_belief=BeliefState.point_mass(start),  # Start with known position
    )

    print(f"Success: {result_low['success']}")
    print(f"Steps: {result_low['steps']}")
    print(f"Oracle calls: {result_low['oracle_calls']}")

    if result_low["belief_trace"]:
        print("\nBelief evolution (first 5 steps):")
        for bt in result_low["belief_trace"][:5]:
            print(f"  Step {bt['step']}: true={bt['true_state']}, MAP={bt['map_state']} "
                  f"(p={bt['map_prob']:.2f}), H={bt['entropy']:.2f}")

    # -------------------------------------------------------------------------
    # SCENARIO 2: High Noise (May Fail)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 2: HIGH NOISE (noise_level=0.5)")
    print("-" * 70)

    high_noise_sensor = NoisySensor(world, noise_level=0.5)
    map_policy_high = MAPGreedyPolicy(seed=42)

    result_high = run_belief_episode(
        map_policy_high, world, high_noise_sensor,
        start, target, horizon,
        initial_belief=BeliefState.uniform(world.all_states()),  # Start uncertain!
    )

    print(f"Success: {result_high['success']}")
    print(f"Steps: {result_high['steps']}")
    if not result_high["success"]:
        print(f"Failure mode: {result_high.get('failure_mode', 'unknown')}")
    print(f"Oracle calls: {result_high['oracle_calls']}")

    if result_high["belief_trace"]:
        print("\nBelief evolution (first 5 steps):")
        for bt in result_high["belief_trace"][:5]:
            print(f"  Step {bt['step']}: true={bt['true_state']}, MAP={bt['map_state']} "
                  f"(p={bt['map_prob']:.2f}), H={bt['entropy']:.2f}")

    # -------------------------------------------------------------------------
    # SCENARIO 3: Entropy-Aware Policy
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 3: ENTROPY-AWARE POLICY (High Noise)")
    print("-" * 70)

    entropy_policy = EntropyAwarePolicy(entropy_threshold=2.5, seed=42)

    result_entropy = run_belief_episode(
        entropy_policy, world, high_noise_sensor,
        start, target, horizon,
        initial_belief=BeliefState.uniform(world.all_states()),
    )

    print(f"Success: {result_entropy['success']}")
    print(f"Steps: {result_entropy['steps']}")
    if not result_entropy["success"]:
        print(f"Failure mode: {result_entropy.get('failure_mode', 'unknown')}")
    print(f"Oracle calls: {result_entropy['oracle_calls']}")

    # -------------------------------------------------------------------------
    # SCENARIO 4: Extreme Noise (Designed to Fail)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 4: EXTREME NOISE (noise_level=0.8, shorter horizon)")
    print("-" * 70)

    extreme_sensor = NoisySensor(world, noise_level=0.8)
    extreme_sensor.rng.seed(123)  # Different seed for different observation sequence
    map_policy_extreme = MAPGreedyPolicy(seed=456)

    # Use short horizon and start with uniform belief to make failure likely
    result_extreme = run_belief_episode(
        map_policy_extreme, world, extreme_sensor,
        start, target, horizon=15,  # Shorter horizon
        initial_belief=BeliefState.uniform(world.all_states()),
    )

    print(f"Success: {result_extreme['success']}")
    print(f"Steps: {result_extreme['steps']}")
    if not result_extreme["success"]:
        print(f"Failure mode: {result_extreme.get('failure_mode', 'unknown')}")
        print(f"Final entropy: {result_extreme.get('final_entropy', 'N/A')}")
        print(f"Final MAP prob: {result_extreme.get('final_map_prob', 'N/A')}")
    print(f"Oracle calls: {result_extreme['oracle_calls']}")

    if result_extreme["belief_trace"]:
        print("\nBelief evolution (first 5 steps):")
        for bt in result_extreme["belief_trace"][:5]:
            print(f"  Step {bt['step']}: true={bt['true_state']}, MAP={bt['map_state']} "
                  f"(p={bt['map_prob']:.2f}), H={bt['entropy']:.2f}")

    # -------------------------------------------------------------------------
    # SCENARIO 5: Aliased Regions (NON_IDENTIFIABLE)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 5: ALIASED REGIONS (Observationally Indistinguishable)")
    print("-" * 70)

    # Create aliased sensor with bottom row aliased
    # (4,0), (4,1), (4,2), (4,3), (4,4) all report "REGION_BOTTOM"
    # This means agent cannot tell when it has reached target (4,4)!
    aliased_sensor = AliasedSensor(
        world=world,
        aliased_regions={
            "BOTTOM": {GridState(4, 0), GridState(4, 1), GridState(4, 2),
                       GridState(4, 3), GridState(4, 4)},
        },
        noise_level=0.05,
    )

    print("Aliased regions:")
    print("  REGION_BOTTOM: (4,0) ≡ (4,1) ≡ (4,2) ≡ (4,3) ≡ (4,4)")
    print("  [Entire bottom row indistinguishable - includes TARGET!]")

    # Start from WITHIN the aliased region with belief spread across it
    # Agent is at (4,0) but belief is uniform over bottom row
    aliased_start = GridState(4, 0)
    map_policy_aliased = MAPGreedyPolicy(seed=42)

    # Initial belief: uniform over aliased bottom row only
    aliased_initial_belief = BeliefState({
        GridState(4, 0): 0.2,
        GridState(4, 1): 0.2,
        GridState(4, 2): 0.2,
        GridState(4, 3): 0.2,
        GridState(4, 4): 0.2,  # Target
    })

    result_aliased = run_belief_episode(
        map_policy_aliased, world, aliased_sensor,
        aliased_start, target, horizon=20,
        initial_belief=aliased_initial_belief,
    )

    print(f"\nStart: {aliased_start} (TRUE position)")
    print(f"Initial belief: uniform over {list(aliased_initial_belief.probabilities.keys())}")
    print(f"Target: {target}")
    print(f"Success: {result_aliased['success']}")
    print(f"Steps: {result_aliased['steps']}")
    if not result_aliased["success"]:
        print(f"Failure mode: {result_aliased.get('failure_mode', 'unknown')}")
        if result_aliased.get("aliased_states"):
            print(f"Aliased states: {result_aliased['aliased_states']}")
    print(f"Oracle calls: {result_aliased['oracle_calls']}")

    if result_aliased["belief_trace"]:
        print("\nBelief evolution (first 5 steps):")
        for bt in result_aliased["belief_trace"][:5]:
            print(f"  Step {bt['step']}: true={bt['true_state']}, MAP={bt['map_state']} "
                  f"(p={bt['map_prob']:.2f}), H={bt['entropy']:.2f}")

    # -------------------------------------------------------------------------
    # SCENARIO 6: Observer Upgrade (Restore Identifiability)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 6: OBSERVER UPGRADE (Column Indicator Restores Identifiability)")
    print("-" * 70)

    # Same world, same aliased regions, but ENHANCED sensor with column indicator
    enhanced_sensor = EnhancedSensor(
        world=world,
        base_aliased_regions={
            "BOTTOM": {GridState(4, 0), GridState(4, 1), GridState(4, 2),
                       GridState(4, 3), GridState(4, 4)},
        },
        noise_level=0.05,
    )

    print("Same aliased regions as Scenario 5, BUT with column indicator:")
    print("  Observation format: 'REGION_BOTTOM|COL_x'")
    print("  COL_0, COL_1, COL_2, COL_3, COL_4 distinguish the states!")
    print("  [This is the 'distinguishing observation' that breaks aliasing]")

    # Same start position and initial belief as scenario 5
    map_policy_enhanced = MAPGreedyPolicy(seed=42)

    result_enhanced = run_belief_episode(
        map_policy_enhanced, world, enhanced_sensor,
        aliased_start, target, horizon=20,
        initial_belief=aliased_initial_belief,
    )

    print(f"\nStart: {aliased_start} (same as Scenario 5)")
    print(f"Initial belief: same as Scenario 5 (uniform over bottom row)")
    print(f"Target: {target}")
    print(f"Success: {result_enhanced['success']}")
    print(f"Steps: {result_enhanced['steps']}")
    if not result_enhanced["success"]:
        print(f"Failure mode: {result_enhanced.get('failure_mode', 'unknown')}")
    print(f"Oracle calls: {result_enhanced['oracle_calls']}")

    if result_enhanced["belief_trace"]:
        print("\nBelief evolution (first 5 steps):")
        for bt in result_enhanced["belief_trace"][:5]:
            print(f"  Step {bt['step']}: true={bt['true_state']}, MAP={bt['map_state']} "
                  f"(p={bt['map_prob']:.2f}), H={bt['entropy']:.2f}")

    # The key comparison
    print("\n" + "=" * 50)
    print("BEFORE/AFTER COMPARISON:")
    print("=" * 50)
    print(f"  Scenario 5 (aliased):  Success={result_aliased['success']}, Steps={result_aliased['steps']}")
    print(f"  Scenario 6 (enhanced): Success={result_enhanced['success']}, Steps={result_enhanced['steps']}")
    if result_aliased['success'] != result_enhanced['success']:
        print("  → Observer upgrade RESOLVED the identifiability problem!")

    # -------------------------------------------------------------------------
    # CERTIFICATE GENERATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("GENERATING CERTIFICATES")
    print("-" * 70)

    certificates = {}

    # Low noise (success)
    if result_low["success"]:
        cert_low = create_belief_success_certificate(
            world, low_noise_sensor, start, target, horizon, result_low
        )
        print(f"Low noise: SUCCESS certificate")
    else:
        cert_low = create_belief_failure_certificate(
            world, low_noise_sensor, start, target, horizon, result_low
        )
        print(f"Low noise: FAILURE certificate ({result_low.get('failure_mode')})")
    certificates["low_noise"] = cert_low

    # High noise (likely failure)
    if result_high["success"]:
        cert_high = create_belief_success_certificate(
            world, high_noise_sensor, start, target, horizon, result_high
        )
        print(f"High noise: SUCCESS certificate")
    else:
        cert_high = create_belief_failure_certificate(
            world, high_noise_sensor, start, target, horizon, result_high
        )
        print(f"High noise: FAILURE certificate ({result_high.get('failure_mode')})")
    certificates["high_noise"] = cert_high

    # Entropy-aware
    if result_entropy["success"]:
        cert_entropy = create_belief_success_certificate(
            world, high_noise_sensor, start, target, horizon, result_entropy
        )
        print(f"Entropy-aware: SUCCESS certificate")
    else:
        cert_entropy = create_belief_failure_certificate(
            world, high_noise_sensor, start, target, horizon, result_entropy
        )
        print(f"Entropy-aware: FAILURE certificate ({result_entropy.get('failure_mode')})")
    certificates["entropy_aware"] = cert_entropy

    # Extreme noise
    if result_extreme["success"]:
        cert_extreme = create_belief_success_certificate(
            world, extreme_sensor, start, target, 15, result_extreme
        )
        print(f"Extreme noise: SUCCESS certificate")
    else:
        cert_extreme = create_belief_failure_certificate(
            world, extreme_sensor, start, target, 15, result_extreme
        )
        print(f"Extreme noise: FAILURE certificate ({result_extreme.get('failure_mode')})")
    certificates["extreme_noise"] = cert_extreme

    # Aliased regions
    if result_aliased["success"]:
        cert_aliased = create_belief_success_certificate(
            world, aliased_sensor, aliased_start, target, 20, result_aliased
        )
        print(f"Aliased regions: SUCCESS certificate")
    else:
        cert_aliased = create_belief_failure_certificate(
            world, aliased_sensor, aliased_start, target, 20, result_aliased
        )
        print(f"Aliased regions: FAILURE certificate ({result_aliased.get('failure_mode')})")
    certificates["aliased_regions"] = cert_aliased

    # Observer upgrade (enhanced sensor)
    # This is the key certificate: it documents HOW the aliasing was resolved
    aliased_states_list = ["(4,0)", "(4,1)", "(4,2)", "(4,3)", "(4,4)"]
    if result_enhanced["success"]:
        cert_enhanced = create_belief_success_certificate(
            world, enhanced_sensor, aliased_start, target, 20, result_enhanced,
            distinguishing_observation="COL_x (column indicator)",
            resolved_aliased_states=aliased_states_list,
        )
        print(f"Observer upgrade: SUCCESS certificate (with distinguishing_observation proof)")
    else:
        cert_enhanced = create_belief_failure_certificate(
            world, enhanced_sensor, aliased_start, target, 20, result_enhanced
        )
        print(f"Observer upgrade: FAILURE certificate ({result_enhanced.get('failure_mode')})")
    certificates["observer_upgrade"] = cert_enhanced

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CERTIFICATE VALIDATION")
    print("-" * 70)

    for name, cert in certificates.items():
        result = validate_policy_certificate(cert)
        status = "VALID" if result.valid else "INVALID"
        print(f"{name}: {status}")
        if result.violations:
            for v in result.violations:
                print(f"  Violation: {v}")
        if result.warnings:
            for w in result.warnings:
                print(f"  Warning: {w}")

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXPORTING CERTIFICATES")
    print("-" * 70)

    output = {
        "demo": "gridworld_belief_demo",
        "description": "POMDP demo with belief states and noisy observations",
        "reference": "MIT Algorithms for Decision Making, Chapter 10",
        "gridworld": {
            "rows": world.rows,
            "cols": world.cols,
            "blocked": [str(b) for b in world.blocked],
            "start": str(start),
            "target": str(target),
            "horizon": horizon,
        },
        "scenarios": {
            # NOTE: Scenario summaries only contain configuration parameters.
            # All outcome data (success, steps, entropy, etc.) is derived from the certificate.
            # This ensures single source of truth - the certificate is canonical.
            "low_noise": {
                "config": {
                    "noise_level": 0.1,
                    "initial_belief": "point_mass(start)",
                    "horizon": horizon,
                },
                "certificate": certificates["low_noise"].to_json(),
            },
            "high_noise": {
                "config": {
                    "noise_level": 0.5,
                    "initial_belief": "uniform",
                    "horizon": horizon,
                },
                "certificate": certificates["high_noise"].to_json(),
            },
            "entropy_aware": {
                "config": {
                    "noise_level": 0.5,
                    "initial_belief": "uniform",
                    "horizon": horizon,
                    "policy_type": "entropy_aware",
                    "entropy_threshold": 2.5,
                },
                "certificate": certificates["entropy_aware"].to_json(),
            },
            "extreme_noise": {
                "config": {
                    "noise_level": 0.8,
                    "initial_belief": "uniform",
                    "horizon": 15,
                },
                "certificate": certificates["extreme_noise"].to_json(),
            },
            "aliased_regions": {
                "config": {
                    "sensor_type": "aliased",
                    "aliased_regions": {
                        "BOTTOM": ["(4,0)", "(4,1)", "(4,2)", "(4,3)", "(4,4)"],
                    },
                    "note": "Entire bottom row is aliased - includes target!",
                    "start": str(aliased_start),
                    "initial_belief": "uniform_over_bottom_row",
                    "horizon": 20,
                },
                "certificate": certificates["aliased_regions"].to_json(),
            },
            "observer_upgrade": {
                "config": {
                    "sensor_type": "enhanced",
                    "base_aliased_regions": {
                        "BOTTOM": ["(4,0)", "(4,1)", "(4,2)", "(4,3)", "(4,4)"],
                    },
                    "distinguishing_observation": "column_indicator (COL_x)",
                    "observation_format": "REGION_X|COL_y",
                    "note": "Same aliased regions, but column indicator breaks aliasing!",
                    "start": str(aliased_start),
                    "initial_belief": "uniform_over_bottom_row",
                    "horizon": 20,
                },
                "certificate": certificates["observer_upgrade"].to_json(),
            },
        },
        "observer_upgrade_comparison": {
            "description": "Before/after comparison showing observer upgrade resolves identifiability",
            "before": {
                "scenario": "aliased_regions",
                "observer": "AliasedSensor (REGION_BOTTOM for all bottom row states)",
                "outcome": "FAILED" if not result_aliased["success"] else "SUCCESS",
            },
            "after": {
                "scenario": "observer_upgrade",
                "observer": "EnhancedSensor (REGION_BOTTOM|COL_x)",
                "distinguishing_observation": "COL_x (column indicator)",
                "outcome": "SUCCESS" if result_enhanced["success"] else "FAILED",
            },
            "conclusion": (
                "Observer upgrade RESOLVED identifiability problem"
                if result_enhanced["success"] and not result_aliased["success"]
                else "Same outcome (no resolution needed or upgrade insufficient)"
            ),
        },
        "belief_failure_modes": {
            "BELIEF_DEGENERACY": "Belief became unusable (generic)",
            "BELIEF_COLLAPSE_WRONG": "Belief collapsed to incorrect state with high confidence",
            "BELIEF_TOO_DIFFUSE": "Belief entropy exceeded threshold - can't localize",
            "NON_IDENTIFIABLE": "States are observationally aliased - cannot distinguish",
        },
        "key_insights": [
            "Partial observability creates belief state (distribution over positions).",
            "Noisy sensors cause belief entropy to grow without informative observations.",
            "MAP-greedy policy fails when belief collapses to wrong state.",
            "Entropy-aware policy trades off exploration (reduce uncertainty) vs exploitation.",
            "Belief failure modes are certificate-grade: BELIEF_DEGENERACY, BELIEF_COLLAPSE_WRONG, etc.",
            "NON_IDENTIFIABLE captures observational aliasing: target reachable but states indistinguishable.",
            "Observer upgrade (adding distinguishing observation) can resolve NON_IDENTIFIABLE failures.",
        ],
    }

    output_path = Path(__file__).parent / "gridworld_belief_cert.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported to: {output_path}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY: QA-Native POMDP Mapping")
    print("=" * 70)
    print("""
    MIT Book Chapter 10 (POMDPs) → QA Mapping:

    Book Concept              QA Equivalent
    ------------              -------------
    Belief state b(s)         Distribution over QA states
    Observation model         Noisy packet measurement
    Belief update             Bayesian filtering
    Value over beliefs        Expected reachability
    POMDP policy              Belief → Generator mapping

    New Failure Modes (extending FailType enum):
    - BELIEF_DEGENERACY: Generic belief unusability
    - BELIEF_COLLAPSE_WRONG: MAP converged to wrong state
    - BELIEF_TOO_DIFFUSE: Entropy too high to act confidently
    - NON_IDENTIFIABLE: States observationally aliased (can't distinguish)

    Certificate Structure:
    - All belief failures are first-class obstruction types
    - Witness includes entropy, MAP probability, observation count
    - Structural consistency: belief failure ⇒ specific obstruction type
    """)


if __name__ == "__main__":
    main()
