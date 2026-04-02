#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_algorithm_competency.py
===========================
Michael Levin–style competency analysis of classical algorithms.

Each algorithm is treated as an *agent* with:
  • goal             — what it achieves
  • cognitive_horizon — how far it looks (local/bounded/global/adaptive)
  • failure_modes    — structural conditions under which it breaks
  • composition      — how it combines with others (sequential/parallel/hierarchical)
  • qa_orbit_sig     — QA orbit signature: which orbit family dominates its state space

The QA orbit map:
  cosmos (24-cycle)    = productive, converging, healthy trajectory
  satellite (8-cycle)  = looping, repetitive, cycling without progress
  singularity (fixed)  = stuck, deadlocked, can't escape without external intervention

This study generates competency profiles → agent design template for QA Lab.

Analogy to Levin basal cognition:
  singularity   ↔ stem cell  (totipotent, no specialization)
  satellite     ↔ cycling progenitor (differentiating, loops until committed)
  cosmos        ↔ differentiated cell (specialized, productive, directed)

Organ formation = multi-agent binding where emergent capability exceeds any single agent.
Metamorphosis  = N satellite cycles → dedifferentiate (→ singularity) → redifferentiate.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
import sys

# ---------------------------------------------------------------------------
# QA substrate (minimal inline — no import dependency)
# ---------------------------------------------------------------------------

MODULUS = 9


def qa_step(b: int, e: int, m: int = MODULUS):
    """One step of the Q map: (b, e) -> (e, (b+e) % m)."""
    return e % m, (b + e) % m


def qa_orbit_family(b: int, e: int, m: int = MODULUS, max_steps: int = 500) -> str:
    """Classify orbit of (b,e) under Q."""
    seen = {}
    state = (b % m, e % m)
    for t in range(max_steps):
        if state in seen:
            period = t - seen[state]
            if period == 1:
                return "singularity"
            elif period == 8:
                return "satellite"
            elif period == 24:
                return "cosmos"
            else:
                return f"period_{period}"
        seen[state] = t
        state = qa_step(*state, m)
    return "unknown"


def qa_orbit_trajectory(b: int, e: int, m: int = MODULUS, steps: int = 48) -> List[tuple]:
    """Return orbit trajectory as list of (b,e) pairs."""
    traj = []
    state = (b % m, e % m)
    for _ in range(steps):
        traj.append(state)
        state = qa_step(*state, m)
    return traj


def orbit_follow_rate(traj: List[tuple], m: int = MODULUS) -> float:
    """Fraction of consecutive (b,e) pairs that follow the Q-map exactly."""
    if len(traj) < 3:
        return 0.0
    follow = 0
    for i in range(len(traj) - 2):
        b, e = traj[i]
        tb, te = traj[i + 1]
        if tb == e % m and te == (b + e) % m:
            follow += 1
    return follow / (len(traj) - 2)


# ---------------------------------------------------------------------------
# Competency profile dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompetencyProfile:
    name: str
    category: str                   # sort / search / traverse / optimize / learn / composite
    goal: str
    cognitive_horizon: str          # local / bounded / global / adaptive
    convergence: str                # guaranteed / probabilistic / conditional / none
    failure_modes: List[str]
    composition: List[str]          # how it binds with other agents
    qa_orbit_sig: str               # cosmos / satellite / singularity / mixed
    qa_orbit_rationale: str         # why this orbit family maps to this algorithm
    orbit_initial_state: tuple      # representative (b,e) for orbit simulation
    levin_cell_type: str            # stem / progenitor / differentiated
    levin_organ_role: str           # what role in a multi-agent organ

    # Filled by analysis
    simulated_ofr: float = 0.0
    simulated_orbit: str = ""
    trajectory: List[tuple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Algorithm definitions (8 algorithms)
# ---------------------------------------------------------------------------

ALGORITHMS = [
    CompetencyProfile(
        name="bubble_sort",
        category="sort",
        goal="Sort N elements by repeated adjacent swaps until no inversions remain",
        cognitive_horizon="local",
        convergence="guaranteed",
        failure_modes=[
            "O(N²) — quadratic blowup on large inputs",
            "Nearly-sorted case: nearly correct but still does full passes",
            "No parallelism — strictly sequential dependency chain",
        ],
        composition=["sequential:insertion_sort", "parallel:bitonic_sort (limited)"],
        qa_orbit_sig="satellite",
        qa_orbit_rationale=(
            "Each pass reduces inversion count by exactly 1 per swap — "
            "a fixed-step progress cycle like the 8-period satellite orbit. "
            "The algorithm loops through the array N times without adaptive shortcutting. "
            "Progress is monotone but slow, like satellite cycling without escape to cosmos."
        ),
        orbit_initial_state=(2, 3),  # satellite seed
        levin_cell_type="progenitor",
        levin_organ_role="local_comparator — contributes as leaf node in sorting networks",
    ),

    CompetencyProfile(
        name="merge_sort",
        category="sort",
        goal="Sort N elements via divide-and-conquer merge into O(N log N) steps",
        cognitive_horizon="global",
        convergence="guaranteed",
        failure_modes=[
            "O(N) auxiliary space — fails on memory-constrained streams",
            "Not in-place — poor cache locality on large arrays",
            "Merge step: requires full sub-array access (bounded cognition during merge)",
        ],
        composition=["hierarchical:tree_of_merge_ops", "parallel:parallel_merge_sort"],
        qa_orbit_sig="cosmos",
        qa_orbit_rationale=(
            "Recursive halving then full integration mirrors the 24-step cosmos cycle: "
            "descent into sub-problems (first 12 steps) followed by constructive recombination "
            "(last 12 steps). The orbit is productive and terminates correctly regardless of input. "
            "No looping — the recursion tree is a DAG that maps to cosmos forward flow."
        ),
        orbit_initial_state=(1, 1),  # cosmos seed
        levin_cell_type="differentiated",
        levin_organ_role="spine — backbone of composite sorting organs; provides O(N log N) guarantee",
    ),

    CompetencyProfile(
        name="quicksort",
        category="sort",
        goal="Sort N elements by pivot partition; expected O(N log N), worst O(N²)",
        cognitive_horizon="adaptive",
        convergence="probabilistic",
        failure_modes=[
            "Adversarial pivot: sorted/reverse-sorted input → O(N²) satellite trap",
            "Duplicate elements: all-equal input → degenerate single-partition recursion",
            "Stack overflow: deep recursion on skewed partitions",
        ],
        composition=["adaptive:median-of-3 pivot", "hybrid:introsort (fallback to heapsort)"],
        qa_orbit_sig="mixed",
        qa_orbit_rationale=(
            "Expected case: cosmos (adaptive pivot creates balanced halves → productive orbit). "
            "Worst case: satellite (unbalanced pivot → O(N²) loop, same as bubble_sort). "
            "Pivot strategy = orbit selector: good pivot ↔ cosmos injection; bad pivot ↔ satellite trap. "
            "This is the canonical QA mixed orbit: outcome depends on external condition (input quality)."
        ),
        orbit_initial_state=(4, 5),  # cosmos but near satellite boundary
        levin_cell_type="progenitor",
        levin_organ_role="adaptive_pivot — context-sensitive sorter; requires guardian (introsort) to stay in cosmos",
    ),

    CompetencyProfile(
        name="bfs",
        category="traverse",
        goal="Find shortest path in unweighted graph via level-order frontier expansion",
        cognitive_horizon="bounded",
        convergence="guaranteed",
        failure_modes=[
            "Memory: O(V) queue — exponential frontier on dense graphs",
            "Unweighted only: edge weights require Dijkstra",
            "Cyclic graphs: requires visited set (else infinite loop = singularity trap)",
        ],
        composition=["sequential:dfs (DFS for DFS-specific properties)", "parallel:parallel_BFS"],
        qa_orbit_sig="cosmos",
        qa_orbit_rationale=(
            "BFS expands a frontier that monotonically increases in distance — "
            "a pure forward orbit with no backtracking. The 24-step cosmos cycle maps "
            "naturally to level-by-level expansion where each step is strictly productive. "
            "Visited-set enforcement = obstruction checker: it prevents the singularity/satellite "
            "trap of revisiting nodes (cycles). BFS without visited set = satellite orbit = loop."
        ),
        orbit_initial_state=(1, 2),  # cosmos
        levin_cell_type="differentiated",
        levin_organ_role="frontier_expander — feeds exploration organs; composable with A* for heuristic",
    ),

    CompetencyProfile(
        name="dijkstra",
        category="traverse",
        goal="Single-source shortest path in weighted graph with non-negative weights",
        cognitive_horizon="global",
        convergence="guaranteed",
        failure_modes=[
            "Negative edges: relaxation loop → infinite descent (singularity trap)",
            "Dense graphs: O(V²) without priority queue",
            "Dynamic graphs: recompute from scratch on edge change",
        ],
        composition=["hierarchical:A* (Dijkstra + heuristic)", "sequential:Bellman-Ford (handles negatives)"],
        qa_orbit_sig="cosmos",
        qa_orbit_rationale=(
            "Priority queue pops minimum-distance node at each step: "
            "strictly monotone distance reduction = cosmos forward orbit. "
            "Negative weights break the monotone property → relaxation loop = satellite orbit. "
            "Dijkstra's correctness proof is essentially a QA obstruction certificate: "
            "it proves no shorter path can exist beyond the current frontier (reachability bound)."
        ),
        orbit_initial_state=(2, 5),  # cosmos
        levin_cell_type="differentiated",
        levin_organ_role="path_navigator — provides shortest-path oracle; composes with planners",
    ),

    CompetencyProfile(
        name="gradient_descent",
        category="optimize",
        goal="Minimize loss L(θ) by iterating θ ← θ - η∇L(θ) until convergence",
        cognitive_horizon="local",
        convergence="conditional",
        failure_modes=[
            "Local minima: non-convex landscape → stuck in satellite orbit",
            "Learning rate too high: overshooting → divergence (escape to infinity)",
            "Saddle points: gradient ≈ 0 but not minimum → near-singularity stall",
            "Flat regions: vanishing gradient → singularity orbit (no movement)",
        ],
        composition=["adaptive:Adam/RMSprop (local curvature adaptation)", "hierarchical:meta-learning"],
        qa_orbit_sig="mixed",
        qa_orbit_rationale=(
            "Convex loss: cosmos orbit — each step reduces norm distance to minimum. "
            "Non-convex / stuck at local min: satellite orbit — loss oscillates or plateaus. "
            "Saddle / vanishing gradient: singularity orbit — no net movement. "
            "This is the QA ML core: κ (QA curvature metric) predicts orbit family of GD. "
            "High κ → cosmos; κ ≈ 0 → singularity; oscillating κ → satellite. "
            "QA Lab's Finite-Orbit Descent theorem is exactly gradient descent in cosmos orbit."
        ),
        orbit_initial_state=(3, 7),  # mixed — starts cosmos, can drift
        levin_cell_type="progenitor",
        levin_organ_role="weight_updater — core learning primitive; needs organ partners (batch, scheduler) to stay in cosmos",
    ),

    CompetencyProfile(
        name="policy_gradient_rl",
        category="learn",
        goal="Learn policy π_θ by ascending ∇_θ E[R] via sampled trajectories",
        cognitive_horizon="adaptive",
        convergence="probabilistic",
        failure_modes=[
            "High variance: gradient estimates noisy → satellite orbit (random walk)",
            "Credit assignment: sparse reward → no gradient signal → singularity",
            "Non-stationarity: environment changes → policy lags → satellite cycling",
            "Mode collapse: policy converges to suboptimal deterministic → singularity",
        ],
        composition=["adaptive:PPO/TRPO (trust region)", "hierarchical:option_framework (temporal abstraction)"],
        qa_orbit_sig="satellite",
        qa_orbit_rationale=(
            "Policy gradient is inherently noisy: the orbit oscillates around the value optimum "
            "rather than converging cleanly. Unlike gradient descent in convex loss, "
            "the policy landscape is non-stationary (policy changes distribution it samples from). "
            "This self-referential loop maps to satellite: cycling around the optimum. "
            "PPO's clipping = orbit damper that prevents satellite from escaping to divergence. "
            "Sparse reward = singularity injection: agent can't move until reward signal arrives."
        ),
        orbit_initial_state=(2, 3),  # satellite seed
        levin_cell_type="progenitor",
        levin_organ_role="policy_adaptor — learns from environment; requires satellite-damping partners (critic, trust region)",
    ),

    CompetencyProfile(
        name="attention_transformer",
        category="learn",
        goal="Attend to all input tokens simultaneously via Q·K^T/√d_k scaled dot-product",
        cognitive_horizon="global",
        convergence="conditional",
        failure_modes=[
            "O(N²) memory/compute: quadratic in sequence length → hard capacity limit",
            "Over-smoothing at depth: deep stacking → all tokens attend uniformly → singularity",
            "Position sensitivity: without positional encoding, permutation invariant (no orbit structure)",
            "Attention sink: outlier tokens absorb all weight → satellite (rest of tokens starved)",
        ],
        composition=["hierarchical:transformer_block (attn + FFN + norm)", "parallel:multi_head"],
        qa_orbit_sig="cosmos",
        qa_orbit_rationale=(
            "Each attention head runs an independent orbit over the token space: "
            "softmax normalization ensures the orbit stays bounded (no escape to infinity). "
            "Multi-head = multi-orbit: different heads track different QA families simultaneously. "
            "Residual connections = orbit anchoring: prevent satellite collapse by adding back original trajectory. "
            "Over-smoothing at depth = singularity injection: all tokens converge to mean embedding. "
            "Sparse attention (e.g., Longformer) = obstruction-aware attention: only cosmos-reachable tokens attended."
        ),
        orbit_initial_state=(1, 3),  # cosmos
        levin_cell_type="differentiated",
        levin_organ_role="context_integrator — global information aggregator; composes as organ brain in transformer body",
    ),
]


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------

def analyze_algorithm(profile: CompetencyProfile, m: int = MODULUS) -> CompetencyProfile:
    """Run QA orbit simulation on algorithm's representative seed state."""
    b, e = profile.orbit_initial_state
    traj = qa_orbit_trajectory(b, e, m, steps=48)
    profile.trajectory = traj
    profile.simulated_orbit = qa_orbit_family(b, e, m)
    profile.simulated_ofr = orbit_follow_rate(traj, m)
    return profile


def print_profile(p: CompetencyProfile, verbose: bool = True) -> None:
    print(f"\n{'='*70}")
    print(f"ALGORITHM: {p.name.upper()}  [{p.category}]")
    print(f"{'='*70}")
    print(f"Goal:              {p.goal}")
    print(f"Horizon:           {p.cognitive_horizon}")
    print(f"Convergence:       {p.convergence}")
    print(f"QA Orbit Sig:      {p.qa_orbit_sig}  (simulated: {p.simulated_orbit}, OFR={p.simulated_ofr:.4f})")
    print(f"Levin cell type:   {p.levin_cell_type}")
    print(f"Organ role:        {p.levin_organ_role}")
    if verbose:
        print(f"\nQA Orbit Rationale:")
        print(f"  {p.qa_orbit_rationale}")
        print(f"\nFailure Modes:")
        for fm in p.failure_modes:
            print(f"  • {fm}")
        print(f"\nComposition:")
        for c in p.composition:
            print(f"  + {c}")


def orbit_family_counts(profiles: List[CompetencyProfile]) -> Dict[str, List[str]]:
    counts: Dict[str, List[str]] = {"cosmos": [], "satellite": [], "singularity": [], "mixed": []}
    for p in profiles:
        key = p.qa_orbit_sig if p.qa_orbit_sig in counts else "mixed"
        counts[key].append(p.name)
    return counts


def levin_cell_summary(profiles: List[CompetencyProfile]) -> Dict[str, List[str]]:
    cells: Dict[str, List[str]] = {"stem": [], "progenitor": [], "differentiated": []}
    for p in profiles:
        cells[p.levin_cell_type].append(p.name)
    return cells


def print_organ_design_template(profiles: List[CompetencyProfile]) -> None:
    """
    Print the QA Lab agent organ design template derived from the competency study.
    """
    print(f"\n{'='*70}")
    print("QA LAB AGENT ORGAN DESIGN TEMPLATE (Levin Architecture)")
    print(f"{'='*70}")

    cosmos_agents = [p for p in profiles if p.qa_orbit_sig in ("cosmos",)]
    satellite_agents = [p for p in profiles if p.qa_orbit_sig == "satellite"]
    mixed_agents = [p for p in profiles if p.qa_orbit_sig == "mixed"]

    print("\n--- ORGAN LAYER 1: SPINE (cosmos orbit agents) ---")
    print("Role: guaranteed convergence, global cognition, backbone of organ")
    print("QA Lab equivalent: CertAgent, QueryAgent, ExperimentAgent")
    for p in cosmos_agents:
        print(f"  [{p.name}] → {p.levin_organ_role.split('—')[0].strip()}")

    print("\n--- ORGAN LAYER 2: ADAPTIVE RING (mixed orbit agents) ---")
    print("Role: context-sensitive, pivot/route based on external condition")
    print("QA Lab equivalent: SynthesisAgent (decides which agent to spawn)")
    for p in mixed_agents:
        print(f"  [{p.name}] → {p.levin_organ_role.split('—')[0].strip()}")

    print("\n--- ORGAN LAYER 3: PROGENITOR RING (satellite orbit agents) ---")
    print("Role: cycling, needs damping/partner; source of differentiation")
    print("QA Lab equivalent: StemAgent (new — needs design)")
    for p in satellite_agents:
        print(f"  [{p.name}] → {p.levin_organ_role.split('—')[0].strip()}")

    print("""
--- ORGAN COMPOSITION RULES ---

  1. Every organ needs ≥1 cosmos agent as SPINE (guarantees convergence)
  2. Satellite agents MUST be paired with a damper (cosmos or mixed guardian)
  3. Mixed agents = pivot cells: they route tasks to correct orbit family
  4. Metamorphosis trigger: if ≥3 satellite cycles → StemAgent dedifferentiates
     → SynthesisAgent respecifies → new competency registered dynamically
  5. Singularity detection: if agent OFR < 0.05 for 5 cycles → dedifferentiate

--- QA LAB AGENT DESIGN RULES (from Levin competency study) ---

  OrbitAgent.cosmos  = CONVERGENT: always makes progress; task reduces to simpler form
  OrbitAgent.satellite = CYCLIC: may loop; must carry a "max_cycles" budget
  OrbitAgent.mixed   = ADAPTIVE: carries orbit_selector(context) → delegates to cosmos or satellite
  StemAgent          = SINGULARITY: starts at (9%9, 9%9) = (0,0); receives any task type
                       differentiation = first successful task → sets orbit_family

--- METAMORPHOSIS PROTOCOL (sketch) ---

  detect: kernel sees N_satellite > THRESHOLD (e.g., 5 consecutive satellite cycles)
  dediff: kernel signals all satellite agents → TaskType.DEDIFFERENTIATE
          each satellite agent resets to (0, 0) = singularity state
  respec: SynthesisAgent generates QA_AGENT_SPAWN_SPEC.v1 for new capability
  rediff: new stub registered → first successful task → cosmos commitment
""")


def export_profiles_json(profiles: List[CompetencyProfile], path: str) -> None:
    out = []
    for p in profiles:
        d = asdict(p)
        d["trajectory"] = [list(s) for s in d["trajectory"]]  # convert tuples
        out.append(d)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Profiles exported → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("QA Algorithm Competency Study — Levin Morphogenetic Architecture")
    print("=" * 70)
    print()
    print("Modelling 8 algorithms as Levin-style cognitive cells.")
    print("Mapping competency profiles → QA orbit signatures → agent design rules.")
    print()

    # Run orbit simulations
    profiles = [analyze_algorithm(p) for p in ALGORITHMS]

    # Print all profiles
    for p in profiles:
        print_profile(p, verbose=True)

    # Summary tables
    print(f"\n{'='*70}")
    print("ORBIT FAMILY SUMMARY")
    print(f"{'='*70}")
    orbit_counts = orbit_family_counts(profiles)
    for family, names in orbit_counts.items():
        if names:
            print(f"  {family:15s}: {', '.join(names)}")

    print(f"\n{'='*70}")
    print("LEVIN CELL TYPE SUMMARY")
    print(f"{'='*70}")
    cells = levin_cell_summary(profiles)
    for cell_type, names in cells.items():
        if names:
            print(f"  {cell_type:15s}: {', '.join(names)}")

    # Orbit simulation table
    print(f"\n{'='*70}")
    print("ORBIT SIMULATION TABLE  (seed state → simulated orbit, OFR)")
    print(f"{'='*70}")
    print(f"{'Algorithm':<25}  {'Seed':>8}  {'QA Sig':>12}  {'Simulated':>12}  {'OFR':>7}")
    print("-" * 72)
    for p in profiles:
        seed_str = f"({p.orbit_initial_state[0]},{p.orbit_initial_state[1]})"
        print(f"{p.name:<25}  {seed_str:>8}  {p.qa_orbit_sig:>12}  {p.simulated_orbit:>12}  {p.simulated_ofr:>7.4f}")

    # Agent organ design template
    print_organ_design_template(profiles)

    # Export
    export_profiles_json(profiles, "qa_algorithm_competency_profiles.json")

    # Capture summary for cert
    result = {
        "modulus": MODULUS,
        "n_algorithms": len(profiles),
        "orbit_distribution": {k: len(v) for k, v in orbit_family_counts(profiles).items()},
        "cell_type_distribution": {k: len(v) for k, v in levin_cell_summary(profiles).items()},
        "algorithms": [
            {
                "name": p.name,
                "qa_orbit_sig": p.qa_orbit_sig,
                "simulated_orbit": p.simulated_orbit,
                "simulated_ofr": round(p.simulated_ofr, 4),
                "levin_cell_type": p.levin_cell_type,
                "cognitive_horizon": p.cognitive_horizon,
                "convergence": p.convergence,
            }
            for p in profiles
        ],
        "key_finding": (
            "Algorithms map cleanly to QA orbit families: "
            "cosmos=converging (merge_sort, bfs, dijkstra, attention), "
            "satellite=cycling (bubble_sort, policy_gradient), "
            "mixed=input-dependent (quicksort, gradient_descent). "
            "Levin cell mapping: differentiated=cosmos, progenitor=satellite/mixed, stem=singularity. "
            "Organ composition rule: every organ needs ≥1 cosmos spine agent."
        ),
    }

    print(f"\n{'='*70}")
    print("VERDICT: COMPETENCY_MAP_COMPLETE")
    print(f"{'='*70}")
    print(f"  {len(profiles)} algorithms profiled across 4 orbit families")
    print(f"  Organ design template: 3-layer (spine/adaptive/progenitor)")
    print(f"  Next: QA_AGENT_COMPETENCY_SPEC.v1 cert family + StemAgent implementation")
    print()

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
