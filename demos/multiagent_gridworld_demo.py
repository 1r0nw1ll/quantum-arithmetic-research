#!/usr/bin/env python3
"""
multiagent_gridworld_demo.py

QA-native multiagent sequential decision making demo mapping MIT
"Algorithms for Decision Making" Ch. 25 (Markov Games) to QA coupled reachability.

Demonstrates:
1. Joint state space = product of individual agent states
2. Coupled generators with collision constraints
3. Certificate-grade failure modes (COORDINATION_DEADLOCK, MISCOORDINATION_CYCLE)
4. Success case: both agents reach goals without collision

Key insight: Multiagent coordination = reachability on product graph.
Deadlock = trap SCC in joint space. Cycle = limit cycle in joint dynamics.
"""

import sys
import json
import hashlib
from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    JointPolicyFailType,
    JointObstructionEvidence,
    CoordinationStats,
    JointPolicyCertificate,
    DerivationWitness,
    validate_joint_policy_certificate,
)


# ============================================================================
# MULTIAGENT GRIDWORLD
# ============================================================================

@dataclass(frozen=True)
class GridState:
    """A single cell in the grid."""
    row: int
    col: int

    def __str__(self) -> str:
        return f"({self.row},{self.col})"


@dataclass(frozen=True)
class JointState:
    """Joint state = tuple of agent positions."""
    positions: Tuple[GridState, ...]

    def __str__(self) -> str:
        return "(" + ", ".join(str(p) for p in self.positions) + ")"

    def has_collision(self) -> bool:
        """Check if any two agents occupy the same cell."""
        return len(set(self.positions)) < len(self.positions)


class Action(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"


@dataclass
class MultiagentGridWorld:
    """A gridworld with multiple agents and collision constraints."""
    rows: int
    cols: int
    blocked: Set[GridState]
    n_agents: int = 2

    def is_valid(self, state: GridState) -> bool:
        """Check if a cell is within bounds and not blocked."""
        return (0 <= state.row < self.rows and
                0 <= state.col < self.cols and
                state not in self.blocked)

    def apply_action(self, state: GridState, action: Action) -> GridState:
        """Apply action to a single agent, returning new position."""
        if action == Action.UP:
            new_state = GridState(state.row - 1, state.col)
        elif action == Action.DOWN:
            new_state = GridState(state.row + 1, state.col)
        elif action == Action.LEFT:
            new_state = GridState(state.row, state.col - 1)
        elif action == Action.RIGHT:
            new_state = GridState(state.row, state.col + 1)
        else:  # STAY
            return state

        return new_state if self.is_valid(new_state) else state

    def apply_joint_action(
        self,
        joint_state: JointState,
        joint_action: Tuple[Action, ...],
        collision_check: bool = True,
    ) -> Tuple[JointState, bool]:
        """
        Apply joint action, returning new joint state and collision flag.

        If collision_check=True and agents would collide, they stay in place.
        """
        new_positions = []
        for i, (pos, action) in enumerate(zip(joint_state.positions, joint_action)):
            new_positions.append(self.apply_action(pos, action))

        new_joint = JointState(tuple(new_positions))

        if collision_check and new_joint.has_collision():
            # Collision! Agents stay in original positions
            return joint_state, True

        return new_joint, False

    def get_legal_actions(self, state: GridState) -> List[Action]:
        """Get all legal single-agent actions from a state."""
        legal = [Action.STAY]  # STAY is always legal
        for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            new_state = self.apply_action(state, action)
            if new_state != state:  # Action had effect (wasn't blocked)
                legal.append(action)
        return legal

    def compute_hash(self) -> str:
        """Compute deterministic hash of the world."""
        canonical = {
            "rows": self.rows,
            "cols": self.cols,
            "blocked": sorted([str(b) for b in self.blocked]),
            "n_agents": self.n_agents,
        }
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================================
# POLICIES
# ============================================================================

def manhattan_distance(a: GridState, b: GridState) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def greedy_action(world: MultiagentGridWorld, pos: GridState, goal: GridState) -> Action:
    """Choose action that minimizes Manhattan distance to goal."""
    best_action = Action.STAY
    best_dist = manhattan_distance(pos, goal)

    for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
        new_pos = world.apply_action(pos, action)
        dist = manhattan_distance(new_pos, goal)
        if dist < best_dist:
            best_dist = dist
            best_action = action

    return best_action


def priority_greedy_policy(
    world: MultiagentGridWorld,
    joint_state: JointState,
    goals: Tuple[GridState, ...],
    priority: List[int],  # Agent indices in priority order
) -> Tuple[Action, ...]:
    """
    Priority-based greedy policy: higher-priority agents move first conceptually.

    In simultaneous execution, lower-priority agent yields if collision would occur.
    """
    n = len(joint_state.positions)
    actions = [Action.STAY] * n

    # Compute greedy action for each agent
    for i in range(n):
        if joint_state.positions[i] != goals[i]:
            actions[i] = greedy_action(world, joint_state.positions[i], goals[i])

    # Check for collision and have lower-priority agent yield
    new_positions = [world.apply_action(p, a) for p, a in zip(joint_state.positions, actions)]

    # If collision, lower-priority agent stays
    if len(set(new_positions)) < n:
        # Find colliding positions
        for i in range(n):
            for j in range(i + 1, n):
                if new_positions[i] == new_positions[j]:
                    # Lower priority (higher index in priority list) yields
                    if priority.index(i) > priority.index(j):
                        actions[i] = Action.STAY
                    else:
                        actions[j] = Action.STAY

    return tuple(actions)


def naive_greedy_policy(
    world: MultiagentGridWorld,
    joint_state: JointState,
    goals: Tuple[GridState, ...],
) -> Tuple[Action, ...]:
    """
    Naive greedy: each agent independently minimizes distance to goal.
    No coordination - can cause deadlock/collision.
    """
    actions = []
    for i, (pos, goal) in enumerate(zip(joint_state.positions, goals)):
        if pos == goal:
            actions.append(Action.STAY)
        else:
            actions.append(greedy_action(world, pos, goal))
    return tuple(actions)


# ============================================================================
# EXECUTION AND CYCLE DETECTION
# ============================================================================

def execute_joint_policy(
    world: MultiagentGridWorld,
    start: JointState,
    goals: Tuple[GridState, ...],
    policy_fn,
    horizon: int,
    collision_check: bool = True,
) -> Dict:
    """
    Execute joint policy and detect success, deadlock, or cycle.

    Returns dict with:
    - success: bool
    - trajectory: list of (joint_state, joint_action)
    - failure_type: None or JointPolicyFailType
    - failure_evidence: dict with failure details
    """
    trajectory = []
    visited = {}  # joint_state -> step number
    current = start
    collisions = 0
    stuck_count = 0

    for step in range(horizon):
        # Check if all agents at goals
        if all(p == g for p, g in zip(current.positions, goals)):
            return {
                "success": True,
                "trajectory": trajectory,
                "steps": step,
                "collisions": collisions,
                "failure_type": None,
                "failure_evidence": None,
            }

        # Check for cycle (revisiting a previously seen joint state)
        state_key = str(current)
        if state_key in visited:
            cycle_start = visited[state_key]
            cycle_length = step - cycle_start
            cycle_states = [str(t[0]) for t in trajectory[cycle_start:]]

            # Cycle of length 1 = stuck in same state = deadlock
            if cycle_length == 1:
                waiting = [i for i, (p, g) in enumerate(zip(current.positions, goals)) if p != g]
                return {
                    "success": False,
                    "trajectory": trajectory,
                    "steps": step,
                    "collisions": collisions,
                    "failure_type": JointPolicyFailType.COORDINATION_DEADLOCK,
                    "failure_evidence": {
                        "deadlock_joint_state": str(current),
                        "waiting_agents": waiting,
                        "stuck_steps": 1,
                    },
                }

            # True oscillation cycle (length >= 2)
            return {
                "success": False,
                "trajectory": trajectory,
                "steps": step,
                "collisions": collisions,
                "failure_type": JointPolicyFailType.MISCOORDINATION_CYCLE,
                "failure_evidence": {
                    "cycle_joint_states": cycle_states,
                    "cycle_length": cycle_length,
                    "cycle_start_step": cycle_start,
                },
            }

        visited[state_key] = step

        # Get joint action
        joint_action = policy_fn(world, current, goals)

        # Apply action
        next_state, collision = world.apply_joint_action(current, joint_action, collision_check)

        if collision:
            collisions += 1

        # Check for deadlock (no progress when not at goal)
        if next_state == current:
            stuck_count += 1
            if stuck_count >= 3:  # Stuck for 3 consecutive steps = deadlock
                # Identify which agents are waiting
                waiting = [i for i, (p, g) in enumerate(zip(current.positions, goals)) if p != g]
                return {
                    "success": False,
                    "trajectory": trajectory,
                    "steps": step,
                    "collisions": collisions,
                    "failure_type": JointPolicyFailType.COORDINATION_DEADLOCK,
                    "failure_evidence": {
                        "deadlock_joint_state": str(current),
                        "waiting_agents": waiting,
                        "stuck_steps": stuck_count,
                    },
                }
        else:
            stuck_count = 0

        trajectory.append((str(current), tuple(a.value for a in joint_action)))
        current = next_state

    # Horizon exceeded
    return {
        "success": False,
        "trajectory": trajectory,
        "steps": horizon,
        "collisions": collisions,
        "failure_type": JointPolicyFailType.HORIZON_EXCEEDED,
        "failure_evidence": {
            "final_joint_state": str(current),
            "agents_at_goal": [i for i, (p, g) in enumerate(zip(current.positions, goals)) if p == g],
        },
    }


# ============================================================================
# CERTIFICATE GENERATION
# ============================================================================

def create_joint_success_certificate(
    world: MultiagentGridWorld,
    start: JointState,
    goals: Tuple[GridState, ...],
    policy_name: str,
    result: Dict,
) -> JointPolicyCertificate:
    """Create certificate for successful joint coordination."""
    return JointPolicyCertificate(
        env_id=f"multiagent_gridworld_{world.rows}x{world.cols}",
        env_description=f"{world.n_agents}-agent gridworld with collision constraints",
        n_agents=world.n_agents,
        joint_state_space_description=f"Product of {world.rows}x{world.cols} grids minus blocked cells",
        joint_state_space_hash=world.compute_hash(),
        agent_goals={i: str(g) for i, g in enumerate(goals)},
        agent_policies={i: policy_name for i in range(world.n_agents)},
        collision_constraint=True,
        turn_based=False,
        joint_generators=["JOINT:UP", "JOINT:DOWN", "JOINT:LEFT", "JOINT:RIGHT", "JOINT:STAY"],
        horizon=len(result["trajectory"]) + 10,
        joint_success=True,
        coordination_stats=CoordinationStats(
            n_episodes=1,
            joint_successes=1,
            collisions=result["collisions"],
            deadlocks=0,
            cycles_detected=0,
            total_joint_steps=result["steps"],
        ),
        failure_mode=None,
        obstruction_if_fail=None,
        coordination_witness=DerivationWitness(
            invariant_name="joint_reachability_success",
            derivation_operator="joint_policy_execution",
            input_data={
                "start": str(start),
                "goals": [str(g) for g in goals],
                "steps": result["steps"],
                "collisions": result["collisions"],
                "policy": policy_name,
            },
            output_value=1,
            verifiable=True,
        ),
        joint_trajectory=[{"state": s, "action": a} for s, a in result["trajectory"]],
        strict_mode=True,
    )


def create_joint_failure_certificate(
    world: MultiagentGridWorld,
    start: JointState,
    goals: Tuple[GridState, ...],
    policy_name: str,
    result: Dict,
) -> JointPolicyCertificate:
    """Create certificate for joint policy failure."""
    fail_type = result["failure_type"]
    evidence = result["failure_evidence"]

    if fail_type == JointPolicyFailType.COORDINATION_DEADLOCK:
        obstruction = JointObstructionEvidence(
            fail_type=fail_type,
            deadlock_joint_state=evidence["deadlock_joint_state"],
            waiting_agents=evidence["waiting_agents"],
        )
    elif fail_type == JointPolicyFailType.MISCOORDINATION_CYCLE:
        obstruction = JointObstructionEvidence(
            fail_type=fail_type,
            cycle_joint_states=evidence["cycle_joint_states"],
            cycle_length=evidence["cycle_length"],
            cycle_start_step=evidence.get("cycle_start_step"),
        )
    else:
        obstruction = None

    return JointPolicyCertificate(
        env_id=f"multiagent_gridworld_{world.rows}x{world.cols}",
        env_description=f"{world.n_agents}-agent gridworld with collision constraints",
        n_agents=world.n_agents,
        joint_state_space_description=f"Product of {world.rows}x{world.cols} grids minus blocked cells",
        joint_state_space_hash=world.compute_hash(),
        agent_goals={i: str(g) for i, g in enumerate(goals)},
        agent_policies={i: policy_name for i in range(world.n_agents)},
        collision_constraint=True,
        turn_based=False,
        joint_generators=["JOINT:UP", "JOINT:DOWN", "JOINT:LEFT", "JOINT:RIGHT", "JOINT:STAY"],
        horizon=len(result["trajectory"]) + 10,
        joint_success=False,
        coordination_stats=CoordinationStats(
            n_episodes=1,
            joint_successes=0,
            collisions=result["collisions"],
            deadlocks=1 if fail_type == JointPolicyFailType.COORDINATION_DEADLOCK else 0,
            cycles_detected=1 if fail_type == JointPolicyFailType.MISCOORDINATION_CYCLE else 0,
            total_joint_steps=result["steps"],
        ),
        failure_mode=fail_type,
        obstruction_if_fail=obstruction,
        coordination_witness=DerivationWitness(
            invariant_name=f"joint_failure_{fail_type.value}",
            derivation_operator="joint_policy_execution",
            input_data={
                "start": str(start),
                "goals": [str(g) for g in goals],
                "steps": result["steps"],
                "collisions": result["collisions"],
                "policy": policy_name,
                "failure_evidence": evidence,
            },
            output_value=0,
            verifiable=True,
        ),
        joint_trajectory=[{"state": s, "action": a} for s, a in result["trajectory"]],
        strict_mode=True,
    )


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def run_demo():
    """Run the multiagent gridworld coordination demo."""
    print("=" * 70)
    print("  QA-Native Multiagent Coordination Demo: Joint Policy Certificates")
    print("  Ref: MIT 'Algorithms for Decision Making' Chapter 25")
    print("=" * 70)

    certificates = {}

    # -------------------------------------------------------------------------
    # Scenario 1: Simple coordination - both agents reach goals
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 1: Simple Coordination (Success)")
    print("-" * 70)

    # 4x4 grid, no obstacles
    world1 = MultiagentGridWorld(rows=4, cols=4, blocked=set(), n_agents=2)
    start1 = JointState((GridState(0, 0), GridState(3, 3)))
    goals1 = (GridState(3, 3), GridState(0, 0))

    print(f"World: {world1.rows}x{world1.cols} grid, no obstacles")
    print(f"Agent 0: {start1.positions[0]} -> {goals1[0]}")
    print(f"Agent 1: {start1.positions[1]} -> {goals1[1]}")
    print("Policy: Priority greedy (Agent 0 has priority)")

    policy1 = lambda w, s, g: priority_greedy_policy(w, s, g, [0, 1])
    result1 = execute_joint_policy(world1, start1, goals1, policy1, horizon=20)

    print(f"\nResult: {'SUCCESS' if result1['success'] else 'FAILED'}")
    print(f"Steps: {result1['steps']}, Collisions avoided: {result1['collisions']}")

    if result1["success"]:
        cert1 = create_joint_success_certificate(world1, start1, goals1, "priority_greedy", result1)
        certificates["simple_coordination_success"] = cert1

    # -------------------------------------------------------------------------
    # Scenario 2: Narrow corridor - COORDINATION_DEADLOCK
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 2: Narrow Corridor (Deadlock)")
    print("-" * 70)

    # 1x5 corridor - agents must pass each other but can't
    world2 = MultiagentGridWorld(rows=1, cols=5, blocked=set(), n_agents=2)
    start2 = JointState((GridState(0, 0), GridState(0, 4)))
    goals2 = (GridState(0, 4), GridState(0, 0))

    print("World: 1x5 corridor (single row)")
    print(f"Agent 0: {start2.positions[0]} -> {goals2[0]} (must go right)")
    print(f"Agent 1: {start2.positions[1]} -> {goals2[1]} (must go left)")
    print("Policy: Naive greedy (no coordination)")
    print("Problem: Agents meet in middle, can't pass (collision constraint)")

    result2 = execute_joint_policy(world2, start2, goals2, naive_greedy_policy, horizon=20)

    print(f"\nResult: {'SUCCESS' if result2['success'] else 'FAILED'}")
    if result2["failure_type"]:
        print(f"Failure type: {result2['failure_type'].value}")
        print(f"Evidence: {result2['failure_evidence']}")

    cert2 = create_joint_failure_certificate(world2, start2, goals2, "naive_greedy", result2)
    certificates["corridor_deadlock"] = cert2

    # -------------------------------------------------------------------------
    # Scenario 3: Chase cycle - MISCOORDINATION_CYCLE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 3: Chase Cycle (True Oscillation)")
    print("-" * 70)

    # 2x2 grid where agents chase each other in a circle
    world3 = MultiagentGridWorld(rows=2, cols=2, blocked=set(), n_agents=2)
    start3 = JointState((GridState(0, 0), GridState(0, 1)))
    # Agent 0 wants to catch agent 1, agent 1 wants to catch agent 0
    # But they use a "chase" policy that causes rotation

    print("World: 2x2 grid (no obstacles)")
    print(f"Agent 0 starts at: {start3.positions[0]}")
    print(f"Agent 1 starts at: {start3.positions[1]}")
    print("Policy: Each agent chases the other -> eternal rotation")

    def chase_policy(world, joint_state, goals):
        """
        Each agent moves toward where the OTHER agent currently is.
        This creates a rotation pattern: both move clockwise/counterclockwise.
        """
        p0, p1 = joint_state.positions

        # Agent 0 chases agent 1's current position
        # Agent 1 chases agent 0's current position
        def move_toward(current, target):
            if current.row < target.row:
                return Action.DOWN
            elif current.row > target.row:
                return Action.UP
            elif current.col < target.col:
                return Action.RIGHT
            elif current.col > target.col:
                return Action.LEFT
            return Action.STAY

        # Agent 0 moves toward p1, Agent 1 moves toward p0
        a0 = move_toward(p0, p1)
        a1 = move_toward(p1, p0)

        return (a0, a1)

    # For chase, goals don't matter - agents chase each other
    goals3_chase = (GridState(1, 1), GridState(0, 0))  # Dummy goals
    result3 = execute_joint_policy(world3, start3, goals3_chase, chase_policy, horizon=20, collision_check=False)

    print(f"\nResult: {'SUCCESS' if result3['success'] else 'FAILED'}")
    if result3["failure_type"]:
        print(f"Failure type: {result3['failure_type'].value}")
        if result3["failure_evidence"].get("cycle_joint_states"):
            print(f"Cycle states: {result3['failure_evidence']['cycle_joint_states'][:4]}...")
            print(f"Cycle length: {result3['failure_evidence']['cycle_length']}")

    if not result3["success"]:
        cert3 = create_joint_failure_certificate(world3, start3, goals3_chase, "chase_policy", result3)
        certificates["chase_cycle"] = cert3
    else:
        print("  (Chase unexpectedly succeeded - skipping certificate)")

    # -------------------------------------------------------------------------
    # Scenario 4: Wide corridor with priority - SUCCESS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 4: Wide Corridor with Priority (Success)")
    print("-" * 70)

    # 2x5 corridor - agents CAN pass each other with coordination
    world4 = MultiagentGridWorld(rows=2, cols=5, blocked=set(), n_agents=2)
    start4 = JointState((GridState(0, 0), GridState(0, 4)))
    goals4 = (GridState(0, 4), GridState(0, 0))

    print("World: 2x5 wide corridor (agents can pass via row 1)")
    print(f"Agent 0: {start4.positions[0]} -> {goals4[0]}")
    print(f"Agent 1: {start4.positions[1]} -> {goals4[1]}")
    print("Policy: Priority greedy - Agent 0 has priority, Agent 1 yields/detours")

    policy4 = lambda w, s, g: priority_greedy_policy(w, s, g, [0, 1])
    result4 = execute_joint_policy(world4, start4, goals4, policy4, horizon=20)

    print(f"\nResult: {'SUCCESS' if result4['success'] else 'FAILED'}")
    if result4["success"]:
        print(f"Steps: {result4['steps']}")

    if result4["success"]:
        cert4 = create_joint_success_certificate(world4, start4, goals4, "priority_greedy", result4)
        certificates["wide_corridor_priority_success"] = cert4
    else:
        cert4 = create_joint_failure_certificate(world4, start4, goals4, "priority_greedy", result4)
        certificates["wide_corridor_priority_fail"] = cert4

    # -------------------------------------------------------------------------
    # Scenario 5: T-junction deadlock
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SCENARIO 5: T-Junction Deadlock")
    print("-" * 70)

    # T-junction: horizontal corridor with one vertical branch
    #   . . .
    #   X . X
    #   X . X
    blocked5 = {
        GridState(1, 0), GridState(2, 0),  # Left blocked
        GridState(1, 2), GridState(2, 2),  # Right blocked
    }
    world5 = MultiagentGridWorld(rows=3, cols=3, blocked=blocked5, n_agents=2)
    start5 = JointState((GridState(0, 0), GridState(0, 2)))
    goals5 = (GridState(2, 1), GridState(2, 1))  # Both want to reach same goal!

    print("World: T-junction (vertical stem with horizontal top)")
    print("       [.][.][.]")
    print("       [X][.][X]")
    print("       [X][.][X]")
    print(f"Agent 0: {start5.positions[0]} -> {goals5[0]}")
    print(f"Agent 1: {start5.positions[1]} -> {goals5[1]}")
    print("Problem: Both agents want same goal cell!")

    result5 = execute_joint_policy(world5, start5, goals5, naive_greedy_policy, horizon=20)

    print(f"\nResult: {'SUCCESS' if result5['success'] else 'FAILED'}")
    if result5["failure_type"]:
        print(f"Failure type: {result5['failure_type'].value}")

    cert5 = create_joint_failure_certificate(world5, start5, goals5, "naive_greedy", result5)
    certificates["tjunction_deadlock"] = cert5

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CERTIFICATE VALIDATION")
    print("-" * 70)

    for name, cert in certificates.items():
        result = validate_joint_policy_certificate(cert)
        status = "VALID" if result.valid else f"INVALID: {result.violations}"
        print(f"{name}: {status}")

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXPORTING CERTIFICATES")
    print("-" * 70)

    output = {
        "demo": "multiagent_gridworld_demo",
        "description": "QA-native multiagent coordination: joint policies + failure certificates",
        "reference": "MIT Algorithms for Decision Making, Chapter 25",
        "scenarios": {
            name: cert.to_json()
            for name, cert in certificates.items()
        },
        "key_insights": [
            "Joint state = product of individual agent states.",
            "Coordination = reachability on product graph with collision constraints.",
            "COORDINATION_DEADLOCK = trap SCC in joint space (mutual waiting).",
            "MISCOORDINATION_CYCLE = limit cycle in joint dynamics (oscillation).",
            "Priority policies can resolve some deadlocks (asymmetric yielding).",
            "Same world + different policy = different certificate outcome.",
        ],
    }

    output_path = Path(__file__).parent / "multiagent_gridworld_cert.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported to: {output_path}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY: QA-Native Multiagent Coordination Mapping")
    print("=" * 70)
    print("""
    MIT Book Chapter 25 (Markov Games) -> QA Mapping:

    Book Concept              QA Equivalent
    ------------              -------------
    Joint state space         Product lattice S1 x S2 x ...
    Joint action              Tuple of generator applications
    Collision constraint      Coupled generator semantics
    Joint policy              Generator selection over joint states
    Coordination success      Reachability certificate on product graph
    Deadlock                  Trap SCC in joint space (COORDINATION_DEADLOCK)
    Oscillation               Limit cycle in joint dynamics (MISCOORDINATION_CYCLE)

    New Certificate Type: JointPolicyCertificate
    - n_agents, joint_state_space_hash
    - agent_goals, agent_policies
    - collision_constraint, turn_based
    - coordination_stats (collisions, deadlocks, cycles)
    - JointObstructionEvidence for failures

    Key Theorem Pattern:
    "Under coupled generators on joint lattice, a joint policy either
     reaches the joint goal set within horizon (certificate + witness),
     or admits a finite obstruction certificate (DEADLOCK, CYCLE, ...)."
""")


if __name__ == "__main__":
    run_demo()
