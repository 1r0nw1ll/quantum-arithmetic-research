#!/usr/bin/env python3
"""
Gridworld Policy Certificate Demo

Demonstrates QA-native decision making by implementing a classic gridworld MDP
and generating policy certificates.

Maps MIT "Algorithms for Decision Making" concepts to QA:
- State space → QA coordinate pairs
- Actions → Generators
- Transitions → Deterministic tuple updates
- Value function → BFS distance
- Optimal policy → Shortest path generator sequence

Output: gridworld_policy_cert.json
"""

import sys
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass
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
)


# ============================================================================
# GRIDWORLD AS QA LATTICE
# ============================================================================

@dataclass(frozen=True)
class GridState:
    """State in gridworld = (row, col) coordinate pair."""
    row: int
    col: int

    def __repr__(self):
        return f"({self.row},{self.col})"


class GridWorld:
    """
    Gridworld environment mapped to QA lattice.

    Actions are generators that modify coordinates:
    - UP: row -= 1
    - DOWN: row += 1
    - LEFT: col -= 1
    - RIGHT: col += 1
    """

    def __init__(self, rows: int, cols: int, blocked: Set[GridState] = None):
        self.rows = rows
        self.cols = cols
        self.blocked = blocked or set()

        # Generators (actions)
        self.generators = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

    def is_valid(self, state: GridState) -> bool:
        """Check if state is within bounds and not blocked."""
        if state.row < 0 or state.row >= self.rows:
            return False
        if state.col < 0 or state.col >= self.cols:
            return False
        if state in self.blocked:
            return False
        return True

    def is_legal(self, state: GridState, gen: str) -> bool:
        """Check if generator application is legal."""
        if gen not in self.generators:
            return False
        next_state = self.apply(state, gen)
        return self.is_valid(next_state)

    def apply(self, state: GridState, gen: str) -> GridState:
        """Apply generator to state (deterministic transition)."""
        dr, dc = self.generators[gen]
        return GridState(state.row + dr, state.col + dc)

    def all_states(self) -> List[GridState]:
        """Return all valid states."""
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                s = GridState(r, c)
                if s not in self.blocked:
                    states.append(s)
        return states


# ============================================================================
# POLICIES
# ============================================================================

class RandomLegalPolicy:
    """Baseline: choose uniformly among legal generators."""

    def __init__(self, world: GridWorld, seed: int = 42):
        self.world = world
        import random
        self.rng = random.Random(seed)
        self.oracle_calls = 0

    def select(self, state: GridState) -> Optional[str]:
        legal = []
        for gen in self.world.generators:
            self.oracle_calls += 1
            if self.world.is_legal(state, gen):
                legal.append(gen)
        if not legal:
            return None
        return self.rng.choice(legal)

    def reset(self):
        self.oracle_calls = 0


class BFSOptimalPolicy:
    """
    Optimal policy via BFS shortest path.

    This is the QA-native equivalent of value iteration:
    - Compute shortest paths from all states to target
    - Policy at state s = first generator on shortest path
    """

    def __init__(self, world: GridWorld, target: GridState):
        self.world = world
        self.target = target
        self.oracle_calls = 0

        # Compute optimal policy via BFS
        self.distance, self.policy, self.states_explored = self._compute_bfs()

    def _compute_bfs(self) -> Tuple[Dict[GridState, int], Dict[GridState, str], int]:
        """BFS from target to compute shortest paths."""
        distance = {self.target: 0}
        policy = {}
        explored = 0

        # Reverse BFS: go backward from target
        queue = deque([self.target])

        while queue:
            current = queue.popleft()
            explored += 1

            # Try all generators (in reverse)
            for gen, (dr, dc) in self.world.generators.items():
                # Predecessor state (where we came from)
                pred = GridState(current.row - dr, current.col - dc)

                if not self.world.is_valid(pred):
                    continue
                if pred in distance:
                    continue

                distance[pred] = distance[current] + 1
                policy[pred] = gen  # From pred, go gen to get closer to target
                queue.append(pred)

        return distance, policy, explored

    def select(self, state: GridState) -> Optional[str]:
        """Select optimal generator (no oracle calls needed - precomputed)."""
        if state == self.target:
            return None  # Already at target
        return self.policy.get(state)

    def reset(self):
        self.oracle_calls = 0


class GreedyManhattanPolicy:
    """
    Heuristic policy: greedily minimize Manhattan distance.

    This is a "world model" approach - uses heuristic distance instead of
    true reachability. May fail if obstacles block the greedy path.
    """

    def __init__(self, world: GridWorld, target: GridState, seed: int = 42):
        self.world = world
        self.target = target
        import random
        self.rng = random.Random(seed)
        self.oracle_calls = 0

    def manhattan_distance(self, s: GridState) -> int:
        return abs(s.row - self.target.row) + abs(s.col - self.target.col)

    def select(self, state: GridState) -> Optional[str]:
        """Select generator that minimizes Manhattan distance."""
        if state == self.target:
            return None

        best_gen = None
        best_dist = float('inf')
        legal_gens = []

        for gen in self.world.generators:
            self.oracle_calls += 1
            if self.world.is_legal(state, gen):
                next_state = self.world.apply(state, gen)
                dist = self.manhattan_distance(next_state)
                legal_gens.append((gen, dist))
                if dist < best_dist:
                    best_dist = dist
                    best_gen = gen

        if best_gen is None:
            return None

        # Tie-breaking: random among best
        best_gens = [g for g, d in legal_gens if d == best_dist]
        return self.rng.choice(best_gens)

    def reset(self):
        self.oracle_calls = 0


# ============================================================================
# EPISODE RUNNER
# ============================================================================

def run_episode(
    policy,
    world: GridWorld,
    start: GridState,
    target: GridState,
    horizon: int,
) -> Dict:
    """Run single episode and return results."""
    state = start
    steps = 0
    trajectory = []

    policy.reset()

    for _ in range(horizon):
        if state == target:
            break

        gen = policy.select(state)
        if gen is None:
            # Stuck
            break

        trajectory.append((state, gen))
        state = world.apply(state, gen)
        steps += 1

    success = state == target
    oracle_calls = policy.oracle_calls

    return {
        "success": success,
        "steps": steps,
        "oracle_calls": oracle_calls,
        "trajectory": trajectory,
        "final_state": state,
    }


def evaluate_policy(
    policy,
    world: GridWorld,
    starts: List[GridState],
    target: GridState,
    horizon: int,
) -> PolicyEvaluationStats:
    """Evaluate policy over multiple start states."""
    successes = 0
    total_steps = 0
    total_oracle_calls = 0

    for start in starts:
        result = run_episode(policy, world, start, target, horizon)
        if result["success"]:
            successes += 1
            total_steps += result["steps"]
        total_oracle_calls += result["oracle_calls"]

    return PolicyEvaluationStats(
        n_episodes=len(starts),
        successes=successes,
        total_steps=total_steps,
        total_oracle_calls=total_oracle_calls,
    )


# ============================================================================
# CERTIFICATE GENERATION
# ============================================================================

def create_optimal_policy_certificate(
    world: GridWorld,
    start: GridState,
    target: GridState,
    horizon: int,
) -> PolicyCertificate:
    """Create certificate for BFS-optimal policy."""
    policy = BFSOptimalPolicy(world, target)

    generators = [
        GeneratorRef("GRID", "UP"),
        GeneratorRef("GRID", "DOWN"),
        GeneratorRef("GRID", "LEFT"),
        GeneratorRef("GRID", "RIGHT"),
    ]

    # Get optimal path length
    optimal_length = policy.distance.get(start, -1)

    if optimal_length == -1:
        # Target unreachable
        return PolicyCertificate(
            policy_id="gridworld_bfs_optimal_unreachable",
            policy_type="bfs_optimal",
            policy_description="BFS found no path to target",
            target_class_description=str(target),
            start_class_description=str(start),
            horizon=horizon,
            generator_set=generators,
            reachability_guarantee=False,
            optimality_guarantee=False,
            failure_mode=PolicyFailType.TARGET_UNREACHABLE,
            obstruction_if_fail=ObstructionEvidence(
                fail_type=FailType.SCC_UNREACHABLE,
                scc_id_reached="start_component",
                goal_state_id=str(target),
                states_explored=policy.states_explored,
                reachable_frontier_hash="n/a",
            ),
            training_witness=DerivationWitness(
                invariant_name="unreachable",
                derivation_operator="bfs_exhaustive",
                input_data={
                    "start": str(start),
                    "target": str(target),
                    "states_explored": policy.states_explored,
                },
                output_value=-1,
                verifiable=True,
            ),
            strict_mode=True,
        )

    return PolicyCertificate.from_bfs_optimal(
        policy_id="gridworld_bfs_optimal",
        target_class=str(target),
        start_class=str(start),
        horizon=horizon,
        generators=generators,
        optimal_path_length=optimal_length,
        states_explored=policy.states_explored,
    )


def create_evaluated_policy_certificate(
    policy,
    policy_id: str,
    policy_type: str,
    world: GridWorld,
    starts: List[GridState],
    target: GridState,
    horizon: int,
    baseline_id: Optional[str] = None,
    baseline_success_rate: Optional[Fraction] = None,
) -> PolicyCertificate:
    """Create certificate from empirical evaluation."""
    stats = evaluate_policy(policy, world, starts, target, horizon)

    generators = [
        GeneratorRef("GRID", "UP"),
        GeneratorRef("GRID", "DOWN"),
        GeneratorRef("GRID", "LEFT"),
        GeneratorRef("GRID", "RIGHT"),
    ]

    return PolicyCertificate.from_evaluation(
        policy_id=policy_id,
        policy_type=policy_type,
        target_class=str(target),
        start_class=f"Random from {len(starts)} states",
        horizon=horizon,
        generators=generators,
        n_episodes=stats.n_episodes,
        successes=stats.successes,
        total_steps=stats.total_steps,
        total_oracle_calls=stats.total_oracle_calls,
        baseline_policy_id=baseline_id,
        baseline_success_rate=baseline_success_rate,
    )


# ============================================================================
# MAIN DEMO
# ============================================================================

def extract_cycle_witness(trajectory: List[Tuple[GridState, str]], target: GridState) -> Dict:
    """
    Extract explicit cycle witness from trajectory.

    Returns dict with:
    - cycle_detected: bool
    - cycle_start_index: int (where cycle begins)
    - cycle_length: int
    - cycle_state: str (state where cycle was detected)
    - cycle_segment: List[(state, action)] - the repeating segment
    - manhattan_on_cycle: List[int] - d_M values showing non-progress
    """
    visited_states = [s for s, _ in trajectory]
    state_first_seen = {}

    cycle_detected = False
    cycle_start_index = None
    cycle_state = None
    cycle_segment = []
    manhattan_on_cycle = []

    for i, s in enumerate(visited_states):
        if s in state_first_seen:
            cycle_detected = True
            cycle_start_index = state_first_seen[s]
            cycle_state = s

            # Extract the repeating segment
            cycle_segment = [(str(visited_states[j]), trajectory[j][1])
                             for j in range(cycle_start_index, i)]

            # Compute Manhattan distance on cycle to show non-progress
            manhattan_on_cycle = [
                abs(visited_states[j].row - target.row) + abs(visited_states[j].col - target.col)
                for j in range(cycle_start_index, i)
            ]
            break
        state_first_seen[s] = i

    return {
        "cycle_detected": cycle_detected,
        "cycle_start_index": cycle_start_index,
        "cycle_length": len(cycle_segment) if cycle_detected else 0,
        "cycle_state": str(cycle_state) if cycle_state else None,
        "cycle_segment": cycle_segment,
        "manhattan_on_cycle": manhattan_on_cycle,
        "manhattan_non_decreasing": (
            len(set(manhattan_on_cycle)) <= 2 if manhattan_on_cycle else False
        ),  # True if progress measure oscillates between at most 2 values
    }


def create_greedy_trap_certificate(
    world: GridWorld,
    start: GridState,
    target: GridState,
    horizon: int,
) -> Tuple[PolicyCertificate, Dict]:
    """
    Create certificate demonstrating greedy failure on trap layout.

    Returns certificate and detailed failure trace.
    """
    policy = GreedyManhattanPolicy(world, target, seed=42)

    # Run single episode to capture failure
    result = run_episode(policy, world, start, target, horizon)

    generators = [
        GeneratorRef("GRID", "UP"),
        GeneratorRef("GRID", "DOWN"),
        GeneratorRef("GRID", "LEFT"),
        GeneratorRef("GRID", "RIGHT"),
    ]

    # GRID generators for obstruction (not the policy!)
    grid_generator_set = {
        Generator("PHYS:GRID_UP"),
        Generator("PHYS:GRID_DOWN"),
        Generator("PHYS:GRID_LEFT"),
        Generator("PHYS:GRID_RIGHT"),
    }

    if result["success"]:
        # Greedy succeeded - not a trap
        return create_evaluated_policy_certificate(
            policy, "greedy_not_trapped", "heuristic_greedy",
            world, [start], target, horizon
        ), result

    # Greedy failed - create failure certificate
    trajectory = result["trajectory"]
    final_state = result["final_state"]

    # Extract explicit cycle witness
    cycle_witness = extract_cycle_witness(trajectory, target)

    # BFS to check if target is actually reachable
    bfs_policy = BFSOptimalPolicy(world, target)
    target_reachable = start in bfs_policy.distance
    optimal_length = bfs_policy.distance.get(start, -1)

    # Determine failure mode and create appropriate obstruction
    if cycle_witness["cycle_detected"]:
        fail_mode = PolicyFailType.POLICY_DIVERGED
        fail_description = f"Greedy policy entered cycle at state {cycle_witness['cycle_state']}"

        obstruction = ObstructionEvidence(
            fail_type=FailType.CYCLE_DETECTED,
            generator_set=grid_generator_set,
            max_depth_reached=result["steps"],
            states_explored=len(set(s for s, _ in trajectory)),
            cycle_start_index=cycle_witness["cycle_start_index"],
            cycle_length=cycle_witness["cycle_length"],
            cycle_state=cycle_witness["cycle_state"],
            cycle_segment=cycle_witness["cycle_segment"],
        )
    elif result["steps"] >= horizon:
        fail_mode = PolicyFailType.HORIZON_EXCEEDED
        fail_description = f"Greedy policy exhausted horizon ({horizon} steps) without reaching target"

        obstruction = ObstructionEvidence(
            fail_type=FailType.DEPTH_EXHAUSTED,
            generator_set=grid_generator_set,
            max_depth_reached=result["steps"],
            states_explored=len(set(s for s, _ in trajectory)),
        )
    else:
        fail_mode = PolicyFailType.POLICY_STUCK
        fail_description = f"Greedy policy stuck at state {final_state}"

        obstruction = ObstructionEvidence(
            fail_type=FailType.TARGET_UNDEFINED,  # Generic "can't proceed"
            goal_state_id=str(target),
        )

    cert = PolicyCertificate(
        policy_id="greedy_manhattan_trapped",
        policy_type="heuristic_greedy",
        policy_description=fail_description,
        target_class_description=str(target),
        start_class_description=str(start),
        horizon=horizon,
        generator_set=generators,
        reachability_guarantee=False,
        optimality_guarantee=False,
        failure_mode=fail_mode,
        obstruction_if_fail=obstruction,
        training_witness=DerivationWitness(
            invariant_name="greedy_failure_trace",
            derivation_operator="episode_execution",
            input_data={
                "start": str(start),
                "target": str(target),
                "horizon": horizon,
                "cycle_detected": cycle_witness["cycle_detected"],
                "cycle_start_index": cycle_witness["cycle_start_index"],
                "cycle_length": cycle_witness["cycle_length"],
                "cycle_state": cycle_witness["cycle_state"],
                "final_state": str(final_state),
                "steps_taken": result["steps"],
                "target_reachable_by_bfs": target_reachable,
                "optimal_path_length": optimal_length,
                "manhattan_on_cycle": cycle_witness["manhattan_on_cycle"],
                "manhattan_non_decreasing": cycle_witness["manhattan_non_decreasing"],
            },
            output_value=0,  # 0 = failure
            verifiable=True,
        ),
        strategy=Strategy(
            type="greedy_manhattan_heuristic",
            key_insight="Minimize Manhattan distance at each step (local optimization)",
            derivation_witness=DerivationWitness(
                invariant_name="strategy:greedy",
                derivation_operator="heuristic_design",
                input_data={"metric": "manhattan_distance"},
                output_value=1,
            ),
        ),
        strict_mode=True,
    )

    return cert, {
        "trajectory": [(str(s), g) for s, g in trajectory],
        "cycle_witness": cycle_witness,
        "final_state": str(final_state),
        "target_reachable_by_bfs": target_reachable,
        "optimal_path_length": optimal_length,
    }


def main():
    print("=" * 70)
    print("  GRIDWORLD POLICY CERTIFICATE DEMO")
    print("  QA-Native Decision Making (MIT Book Mapping)")
    print("=" * 70)

    # Create 4x4 gridworld with obstacle
    #   0 1 2 3
    # 0 S . . .
    # 1 . X . .
    # 2 . . . .
    # 3 . . . G
    world = GridWorld(
        rows=4,
        cols=4,
        blocked={GridState(1, 1)},
    )
    start = GridState(0, 0)
    target = GridState(3, 3)
    horizon = 10

    print(f"\nGridworld: {world.rows}x{world.cols}")
    print(f"Blocked: {world.blocked}")
    print(f"Start: {start}")
    print(f"Target: {target}")
    print(f"Horizon: {horizon}")

    # -------------------------------------------------------------------------
    # Policy 1: BFS Optimal
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("POLICY 1: BFS OPTIMAL")
    print("-" * 70)

    bfs_policy = BFSOptimalPolicy(world, target)
    bfs_cert = create_optimal_policy_certificate(world, start, target, horizon)

    print(f"Optimal path length: {bfs_policy.distance.get(start, 'UNREACHABLE')}")
    print(f"States explored: {bfs_policy.states_explored}")
    print(f"Reachability guarantee: {bfs_cert.reachability_guarantee}")
    print(f"Optimality guarantee: {bfs_cert.optimality_guarantee}")

    # Trace optimal path
    path = []
    state = start
    while state != target and state in bfs_policy.policy:
        gen = bfs_policy.policy[state]
        path.append(f"{state} --{gen}--> ")
        state = world.apply(state, gen)
    path.append(str(state))
    print(f"Optimal path: {''.join(path)}")

    # -------------------------------------------------------------------------
    # Policy 2: Random Legal (Baseline)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("POLICY 2: RANDOM LEGAL (BASELINE)")
    print("-" * 70)

    # Use all non-blocked states as potential starts
    all_starts = [s for s in world.all_states() if s != target]
    random_policy = RandomLegalPolicy(world)
    random_cert = create_evaluated_policy_certificate(
        random_policy,
        "gridworld_random_legal",
        "random_legal",
        world,
        all_starts,
        target,
        horizon,
    )

    print(f"Episodes: {random_cert.evaluation_stats.n_episodes}")
    print(f"Successes: {random_cert.evaluation_stats.successes}")
    print(f"Success rate: {random_cert.evaluation_stats.success_rate}")
    print(f"Avg steps (on success): {random_cert.evaluation_stats.avg_steps}")

    # -------------------------------------------------------------------------
    # Policy 3: Greedy Manhattan (Heuristic)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("POLICY 3: GREEDY MANHATTAN (HEURISTIC)")
    print("-" * 70)

    greedy_policy = GreedyManhattanPolicy(world, target)
    greedy_cert = create_evaluated_policy_certificate(
        greedy_policy,
        "gridworld_greedy_manhattan",
        "heuristic_greedy",
        world,
        all_starts,
        target,
        horizon,
        baseline_id="gridworld_random_legal",
        baseline_success_rate=random_cert.evaluation_stats.success_rate,
    )

    print(f"Episodes: {greedy_cert.evaluation_stats.n_episodes}")
    print(f"Successes: {greedy_cert.evaluation_stats.successes}")
    print(f"Success rate: {greedy_cert.evaluation_stats.success_rate}")
    print(f"Avg steps (on success): {greedy_cert.evaluation_stats.avg_steps}")
    print(f"Improvement over baseline: {greedy_cert.improvement_over_baseline}")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CERTIFICATE VALIDATION")
    print("-" * 70)

    for cert, name in [
        (bfs_cert, "BFS Optimal"),
        (random_cert, "Random Legal"),
        (greedy_cert, "Greedy Manhattan"),
    ]:
        result = validate_policy_certificate(cert)
        status = "VALID" if result.valid else "INVALID"
        print(f"{name}: {status}")
        if result.warnings:
            for w in result.warnings:
                print(f"  Warning: {w}")

    # -------------------------------------------------------------------------
    # TRAP WORLD: Greedy Failure Demo
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAP WORLD: GREEDY HEURISTIC FAILURE")
    print("=" * 70)

    # Create trap world - pocket/corridor with tie-break oscillation
    #   0 1 2 3 4
    # 0 S . . . .
    # 1 . X X X X
    # 2 . X G . .
    # 3 . X . . .
    # 4 . . . . .
    #
    # Greedy from (0,0) to (2,2):
    # - Goes down to (1,0), (2,0)
    # - From (2,0), wants right but (2,1) blocked
    # - All legal moves have equal Manhattan distance → tie-break oscillation
    # - This is the classic "local minimum trap" for greedy heuristics
    trap_world = GridWorld(
        rows=5,
        cols=5,
        blocked={
            GridState(1, 1), GridState(1, 2), GridState(1, 3), GridState(1, 4),
            GridState(2, 1), GridState(3, 1),
        },
    )
    trap_start = GridState(0, 0)
    trap_target = GridState(2, 2)
    trap_horizon = 20

    print(f"\nTrap World: {trap_world.rows}x{trap_world.cols}")
    print("Layout:")
    print("  0 1 2 3 4")
    for r in range(trap_world.rows):
        row_str = f"{r} "
        for c in range(trap_world.cols):
            s = GridState(r, c)
            if s == trap_start:
                row_str += "S "
            elif s == trap_target:
                row_str += "G "
            elif s in trap_world.blocked:
                row_str += "X "
            else:
                row_str += ". "
        print(row_str)

    print(f"\nStart: {trap_start}")
    print(f"Target: {trap_target}")
    print(f"Horizon: {trap_horizon}")

    # Check BFS can solve it
    trap_bfs = BFSOptimalPolicy(trap_world, trap_target)
    if trap_start in trap_bfs.distance:
        print(f"BFS optimal path: {trap_bfs.distance[trap_start]} steps")
    else:
        print("BFS: Target unreachable!")

    # Run greedy and capture failure
    trap_cert, trap_trace = create_greedy_trap_certificate(
        trap_world, trap_start, trap_target, trap_horizon
    )

    print(f"\nGreedy result: {'SUCCESS' if trap_cert.failure_mode is None else 'FAILED'}")
    if trap_cert.failure_mode:
        print(f"Failure mode: {trap_cert.failure_mode.value}")
        print(f"Description: {trap_cert.policy_description}")

        cycle_w = trap_trace.get("cycle_witness", {})
        if cycle_w.get("cycle_detected"):
            print(f"\nCYCLE WITNESS:")
            print(f"  Cycle start index: {cycle_w['cycle_start_index']}")
            print(f"  Cycle length: {cycle_w['cycle_length']}")
            print(f"  Cycle state: {cycle_w['cycle_state']}")
            print(f"  Cycle segment: {cycle_w['cycle_segment'][:5]}...")
            print(f"  Manhattan on cycle: {cycle_w['manhattan_on_cycle']}")
            print(f"  Progress non-decreasing: {cycle_w['manhattan_non_decreasing']}")

        print(f"\nFinal state: {trap_trace['final_state']}")
        print(f"Target reachable by BFS: {trap_trace['target_reachable_by_bfs']}")
        print(f"Optimal path length: {trap_trace['optimal_path_length']}")

    # Validate trap certificate
    trap_result = validate_policy_certificate(trap_cert)
    print(f"\nTrap certificate valid: {trap_result.valid}")

    # -------------------------------------------------------------------------
    # Export to JSON
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXPORTING CERTIFICATES")
    print("-" * 70)

    output = {
        "demo": "gridworld_policy_demo",
        "description": "QA-native decision making demo mapping MIT Algorithms for Decision Making",
        "gridworld": {
            "rows": world.rows,
            "cols": world.cols,
            "blocked": [str(b) for b in world.blocked],
            "start": str(start),
            "target": str(target),
            "horizon": horizon,
        },
        "certificates": {
            "bfs_optimal": bfs_cert.to_json(),
            "random_legal": random_cert.to_json(),
            "greedy_manhattan": greedy_cert.to_json(),
        },
        "trap_world": {
            "description": "Pocket/corridor with tie-break oscillation trap for Manhattan-greedy",
            "rows": trap_world.rows,
            "cols": trap_world.cols,
            "blocked": [str(b) for b in trap_world.blocked],
            "start": str(trap_start),
            "target": str(trap_target),
            "horizon": trap_horizon,
            "bfs_optimal_length": trap_bfs.distance.get(trap_start, -1),
            "greedy_failure_certificate": trap_cert.to_json(),
            "failure_trace": trap_trace,
        },
        "comparison": {
            "policies": ["bfs_optimal", "random_legal", "greedy_manhattan"],
            "success_rates": {
                "bfs_optimal": "1 (guaranteed)",
                "random_legal": str(random_cert.evaluation_stats.success_rate),
                "greedy_manhattan": str(greedy_cert.evaluation_stats.success_rate),
            },
            "oracle_efficiency": {
                "bfs_optimal": "0 per step (precomputed)",
                "random_legal": "4 per step (check all generators)",
                "greedy_manhattan": "4 per step (check all generators)",
            },
        },
        "key_insights": [
            "BFS-optimal policy has 100% success guarantee with 0 runtime oracle calls.",
            "Greedy heuristics can fail on trap layouts even when target is reachable.",
            "The trap certificate proves 'heuristics are not guarantees' at the certificate level.",
            "QA-native advantage: structure is precomputed, not discovered at runtime.",
        ],
    }

    output_path = Path(__file__).parent / "gridworld_policy_cert.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported to: {output_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY: QA vs Traditional Decision Making")
    print("=" * 70)
    print("""
    Traditional MDP:
    - State space given, need to learn transitions
    - Value iteration converges asymptotically
    - Exploration-exploitation tradeoff required
    - No guarantees without sufficient exploration

    QA-Native Decision Making:
    - State space = QA lattice (known structure)
    - Transitions = deterministic generators
    - BFS = exact shortest path (no approximation)
    - Reachability guaranteed by certificate

    The gridworld demo shows:
    1. BFS-optimal policy has provable guarantees (certificate)
    2. Heuristic policies FAIL on trap layouts (certificate-grade proof)
    3. Random baseline provides comparison benchmark
    4. Failure modes are objects, not bugs - captured in certificates
    """)


if __name__ == "__main__":
    main()
