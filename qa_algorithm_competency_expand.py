#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_algorithm_competency_expand.py
===================================
Expanded QA Algorithm Competency Registry — 22 algorithms, 6 families.

Extends the 8-algorithm study in qa_algorithm_competency.py with:
  • source_corpus_refs     — which extracted corpus files informed this entry
  • confidence             — high/medium/low based on corpus evidence
  • needs_ocr_backfill     — True if QA-3/Pyth-1/Pyth-2 OCR may add vocabulary
  • organ_roles            — roles this algorithm plays in multi-agent organs
  • differentiation_profile — dediff conditions, recommitment, drift thresholds

Corpus status at time of writing (2026-03-26):
  EXTRACTED: QA-1, QA-2, QA-4, Quadrature, Pyth-3 (Enneagram)
  OCR RUNNING: QA-3 (background, PID 4148)
  OCR PENDING: Pyth-1, Pyth-2, QA-Workbook

Output: qa_algorithm_competency_registry.json

Usage:
  python qa_algorithm_competency_expand.py
  python qa_algorithm_competency_expand.py --summary
  python qa_algorithm_competency_expand.py --family optimize
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

# ── QA substrate (inline, no import dependency) ───────────────────────────────

MODULUS = 9

def qa_step(b: int, e: int, m: int = MODULUS):
    return e % m, (b + e) % m

def qa_orbit_family(b: int, e: int, m: int = MODULUS, max_steps: int = 500) -> str:
    seen = {}
    state = (b % m, e % m)
    for t in range(max_steps):
        if state in seen:
            period = t - seen[state]
            if period == 1:   return "singularity"
            elif period == 8:  return "satellite"
            elif period == 24: return "cosmos"
            else:              return f"period_{period}"
        seen[state] = t
        state = qa_step(*state, m)
    return "unknown"

def orbit_follow_rate(b: int, e: int, m: int = MODULUS, steps: int = 48) -> float:
    traj = []
    state = (b % m, e % m)
    for _ in range(steps):
        traj.append(state)
        state = qa_step(*state, m)
    if len(traj) < 3:
        return 0.0
    follow = sum(
        1 for i in range(len(traj) - 2)
        if traj[i+1][0] == traj[i][1] % m and traj[i+1][1] == (traj[i][0] + traj[i][1]) % m
    )
    return follow / (len(traj) - 2)

# ── Registry schema ───────────────────────────────────────────────────────────

@dataclass
class DifferentiationProfile:
    """Conditions governing state transitions for this algorithm-as-agent."""
    dediff_conditions: List[str]      # when to dedifferentiate (reset to stem)
    recommit_conditions: List[str]    # when a progenitor commits to cosmos
    max_satellite_cycles: int         # before forced dediff / metamorphosis trigger
    drift_threshold: float            # OFR below this = orbit drift warning
    partial_fail_threshold: int       # consecutive failures before PARTIAL_FAIL

@dataclass
class AlgorithmEntry:
    # Identity
    name: str
    family: str                       # sort | search | graph | optimize | learn | control | distributed
    goal: str

    # Cognitive structure
    cognitive_horizon: str            # local | bounded | global | adaptive
    convergence: str                  # guaranteed | probabilistic | conditional | none
    time_complexity: str              # big-O
    space_complexity: str             # big-O

    # QA orbit mapping
    orbit_signature: str              # cosmos | satellite | singularity | mixed
    orbit_rationale: str
    orbit_seed: tuple                 # representative (b,e) for QA simulation

    # Levin architecture
    levin_cell_type: str              # stem | progenitor | differentiated
    organ_roles: List[str]            # roles in multi-agent organs

    # Failure modes
    failure_modes: List[str]
    composition_rules: List[str]      # how it binds with others

    # Differentiation protocol
    differentiation_profile: DifferentiationProfile

    # Corpus provenance
    source_corpus_refs: List[str]     # corpus file slugs from qa_corpus_text/
    corpus_concepts: List[str]        # QA vocabulary terms from corpus that map here
    needs_ocr_backfill: bool          # True = QA-3/Pyth-1/Pyth-2 may add vocabulary
    confidence: str                   # high | medium | low

    # Computed (filled at runtime)
    simulated_orbit: str = ""
    simulated_ofr: float = 0.0

    def simulate(self) -> "AlgorithmEntry":
        b, e = self.orbit_seed
        self.simulated_orbit = qa_orbit_family(b, e)
        self.simulated_ofr   = round(orbit_follow_rate(b, e), 4)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orbit_seed"] = list(d["orbit_seed"])
        return d

# ── Corpus reference slugs ────────────────────────────────────────────────────

QA1  = "qa-1__qa_1_all_pages__docx.md"     # Parity laws, roots, Fibonacci structure
QA2  = "qa-2__001_qa_2_all_pages__docx.md" # Natural arithmetic, prime factorization
QA4  = "qa-4__00_qa_books_3_&_4_all_pages__pdf.md"  # Wave theory, harmonics, Quantize Code
QUAD = "quadrature__00_quadratureprint__pdf.md"      # Area quantization
P3   = "pyth-3__pythagoras_vol3_enneagram__docx.md"  # Enneagram, cycle structure
# QA3 not yet available (OCR running) — flagged with needs_ocr_backfill=True

# ── Algorithm definitions ─────────────────────────────────────────────────────

ALGORITHMS: List[AlgorithmEntry] = [

    # ── FAMILY: sort ─────────────────────────────────────────────────────────

    AlgorithmEntry(
        name="bubble_sort", family="sort",
        goal="Sort N elements by repeated adjacent swaps until no inversions remain",
        cognitive_horizon="local", convergence="guaranteed",
        time_complexity="O(N²)", space_complexity="O(1)",
        orbit_signature="satellite",
        orbit_rationale=(
            "Each pass reduces inversion count by one swap per cycle — fixed-step "
            "progress matching the 8-period satellite orbit. No adaptive shortcutting; "
            "loops N times regardless of partial order. QA-1 parity laws: adjacent swap "
            "preserves parity of inversion count mod 2 — the satellite orbit's 8-cycle "
            "exhibits the same mod-8 closure. Progress is monotone but slow; the algorithm "
            "cannot escape to cosmos without external intervention (e.g., early-exit flag)."
        ),
        orbit_seed=(2, 3),
        levin_cell_type="progenitor",
        organ_roles=["local_comparator", "leaf_node_in_sorting_network"],
        failure_modes=[
            "O(N²) blowup on large inputs — no divide strategy",
            "Nearly-sorted: does full passes even when 1 inversion remains",
            "No parallelism — strictly sequential dependency chain",
        ],
        composition_rules=[
            "sequential:insertion_sort (same horizon, compatible pipeline)",
            "parallel:bitonic_sort (limited — requires power-of-2 length)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Input size > 10^4 (O(N²) becomes satellite trap)",
                "OFR < 0.1 for 5 consecutive tasks",
            ],
            recommit_conditions=[
                "Input confirmed small (N < 100) and nearly-sorted",
            ],
            max_satellite_cycles=5,
            drift_threshold=0.10,
            partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA1, QA2],
        corpus_concepts=["parity_law", "inversion_count", "mod_2_closure", "sequential_orbit"],
        needs_ocr_backfill=True,  # QA-3 may have sorting/ordering examples
        confidence="high",
    ),

    AlgorithmEntry(
        name="insertion_sort", family="sort",
        goal="Build sorted array one element at a time by inserting into correct position",
        cognitive_horizon="local", convergence="guaranteed",
        time_complexity="O(N²) worst, O(N) best", space_complexity="O(1)",
        orbit_signature="satellite",
        orbit_rationale=(
            "Like bubble_sort, insertion_sort cycles through the unsorted region element "
            "by element — a bounded repeating pattern matching satellite orbit. However, "
            "on nearly-sorted input it escapes the satellite toward cosmos: O(N) comparisons "
            "when already sorted. This adaptive edge maps to the satellite↔cosmos boundary "
            "in QA. The 'insertion index' search within each step is a bounded local scan "
            "matching the 8-cycle period of satellite orbits."
        ),
        orbit_seed=(2, 4),
        levin_cell_type="progenitor",
        organ_roles=["local_inserter", "online_sorter — handles streaming input"],
        failure_modes=[
            "O(N²) worst case on reverse-sorted input",
            "Shift operations expensive on arrays (linked list preferred)",
            "Cannot parallelize the inner insertion scan",
        ],
        composition_rules=[
            "hybrid:timsort (insertion_sort as base case for small runs)",
            "sequential:binary_search (for insertion index lookup)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["Input reverse-sorted with N > 10^3"],
            recommit_conditions=["Input nearly-sorted (< 5% inversions)"],
            max_satellite_cycles=5, drift_threshold=0.10, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA1, QA2],
        corpus_concepts=["parity_law", "insertion_identity", "ordered_sequence"],
        needs_ocr_backfill=True,
        confidence="medium",
    ),

    AlgorithmEntry(
        name="merge_sort", family="sort",
        goal="Sort N elements via divide-and-conquer merge into guaranteed O(N log N)",
        cognitive_horizon="global", convergence="guaranteed",
        time_complexity="O(N log N)", space_complexity="O(N)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "Recursive halving then full integration mirrors the 24-step cosmos cycle: "
            "descent into sub-problems (first half) then constructive recombination (second half). "
            "The orbit is strictly productive; no looping. QA-1 Fibonacci-structure: the merge "
            "tree has Fibonacci-like branching where sub-problem sizes follow the (b,e,d,a) "
            "recurrence. The O(N log N) barrier is the QA orbit-length theorem applied to "
            "comparison-based sorting."
        ),
        orbit_seed=(1, 1),
        levin_cell_type="differentiated",
        organ_roles=["spine", "backbone_of_sorting_organ", "O(N_log_N)_guarantor"],
        failure_modes=[
            "O(N) auxiliary space — fails on memory-constrained streams",
            "Not in-place — poor cache locality on large arrays",
            "Merge step requires full sub-array access simultaneously",
        ],
        composition_rules=[
            "hierarchical:tree_of_merge_ops",
            "parallel:parallel_merge_sort (split at top level)",
            "hybrid:timsort (merge_sort + insertion_sort for small N)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["Memory constraint prevents O(N) auxiliary allocation"],
            recommit_conditions=["Sufficient memory available, N > 64"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=3,
        ),
        source_corpus_refs=[QA1, QA2],
        corpus_concepts=["fibonacci_roots", "divide_recombine", "cosmos_forward_orbit", "b_e_d_a_recurrence"],
        needs_ocr_backfill=False,  # Core merge structure well-covered in QA-1/QA-2
        confidence="high",
    ),

    AlgorithmEntry(
        name="heap_sort", family="sort",
        goal="Sort N elements in-place via max-heap construction then extraction",
        cognitive_horizon="global", convergence="guaranteed",
        time_complexity="O(N log N)", space_complexity="O(1)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "Heapify builds a complete binary tree satisfying the heap property — "
            "a structured, bounded orbit. Each extract-max step reduces heap size by 1 "
            "and sifts down in O(log N): a guaranteed convergent operation. The heap "
            "invariant maps to the QA 'every quantum number contains ordered roots' law (QA-1 Law 7): "
            "parent ≥ children is the sorted root ordering principle. Unlike merge_sort, "
            "heap_sort does not need auxiliary space — the in-place orbit is still cosmos "
            "because each sift-down is a bounded descent toward the leaf level."
        ),
        orbit_seed=(3, 5),
        levin_cell_type="differentiated",
        organ_roles=["in_place_sorter", "priority_oracle", "spine_alternative_to_merge_sort"],
        failure_modes=[
            "Poor cache performance — non-sequential memory access pattern",
            "Not stable — equal elements may reorder",
            "Heapify direction matters — min-heap vs max-heap confusion",
        ],
        composition_rules=[
            "sequential:priority_queue (heap_sort is implicit priority queue drain)",
            "hybrid:introsort (heapsort as fallback when quicksort depth exceeded)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["Cache-sensitive workload where sequential access required"],
            recommit_conditions=["Memory-constrained environment, N > 64"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=3,
        ),
        source_corpus_refs=[QA1],
        corpus_concepts=["ordered_roots", "parent_child_relation", "heap_property", "law_7_roots"],
        needs_ocr_backfill=True,
        confidence="medium",
    ),

    AlgorithmEntry(
        name="quicksort", family="sort",
        goal="Sort N elements by pivot partition; expected O(N log N), worst O(N²)",
        cognitive_horizon="adaptive", convergence="probabilistic",
        time_complexity="O(N log N) expected, O(N²) worst", space_complexity="O(log N) stack",
        orbit_signature="mixed",
        orbit_rationale=(
            "Expected case: cosmos — adaptive pivot creates balanced halves → productive orbit. "
            "Worst case: satellite — unbalanced pivot → O(N²) loop identical to bubble_sort. "
            "Pivot strategy = orbit selector: good pivot ↔ cosmos injection; bad pivot ↔ satellite trap. "
            "QA-2 prime factorization analogy: the pivot is the 'quantum number' that partitions "
            "the set into roots. A well-chosen pivot (median) is a true QA root; a degenerate pivot "
            "(min or max) is a non-quantum number that breaks the partition."
        ),
        orbit_seed=(4, 5),
        levin_cell_type="progenitor",
        organ_roles=["adaptive_pivot", "context_sensitive_sorter"],
        failure_modes=[
            "Adversarial pivot on sorted/reverse-sorted input → O(N²) satellite trap",
            "All-equal elements → degenerate single-partition recursion",
            "Stack overflow on deep recursion with skewed partitions",
        ],
        composition_rules=[
            "adaptive:median_of_3_pivot (reduces satellite probability)",
            "hybrid:introsort (quicksort + heapsort fallback + insertion_sort base)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Recursion depth exceeds 2*log2(N) (introsort switch criterion)",
                "Detected adversarial input (sorted/reverse-sorted)",
            ],
            recommit_conditions=["Random pivot selected, input not adversarial"],
            max_satellite_cycles=3, drift_threshold=0.15, partial_fail_threshold=3,
        ),
        source_corpus_refs=[QA1, QA2],
        corpus_concepts=["pivot_as_quantum_number", "partition_into_roots", "mixed_orbit"],
        needs_ocr_backfill=True,
        confidence="high",
    ),

    # ── FAMILY: search ────────────────────────────────────────────────────────

    AlgorithmEntry(
        name="binary_search", family="search",
        goal="Find target in sorted array by halving search interval each step",
        cognitive_horizon="bounded", convergence="guaranteed",
        time_complexity="O(log N)", space_complexity="O(1)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "Binary search halves the search space each step — a strictly convergent orbit "
            "that reaches the target (or proves absence) in ceil(log2 N) steps. "
            "QA-1 Law 5: divisibility test maps to binary search's halving: checking midpoint "
            "is equivalent to checking divisibility by 2 at each level. The bounded "
            "cognitive horizon (always looks at midpoint of current interval) maps to "
            "the QA 'bounded root' concept — the search space is always a valid quantum interval."
        ),
        orbit_seed=(1, 2),
        levin_cell_type="differentiated",
        organ_roles=["lookup_oracle", "index_spine", "sorted_structure_navigator"],
        failure_modes=[
            "Requires sorted input — fails silently on unsorted array",
            "Integer overflow in midpoint calculation: (lo+hi)//2 vs lo+(hi-lo)//2",
            "Infinite loop if invariant broken (lo > hi not caught)",
        ],
        composition_rules=[
            "sequential:insertion_sort (binary search for insertion index)",
            "hierarchical:B-tree (binary search at each node)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["Input not sorted (precondition violation)"],
            recommit_conditions=["Sorted array confirmed"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=2,
        ),
        source_corpus_refs=[QA1, QA2],
        corpus_concepts=["divisibility_law_5", "halving_interval", "quantum_interval", "bounded_root"],
        needs_ocr_backfill=False,
        confidence="high",
    ),

    AlgorithmEntry(
        name="linear_search", family="search",
        goal="Find target by scanning all elements sequentially",
        cognitive_horizon="local", convergence="guaranteed",
        time_complexity="O(N)", space_complexity="O(1)",
        orbit_signature="satellite",
        orbit_rationale=(
            "Linear search advances one position per step with no adaptive shortcutting — "
            "a fixed-step satellite orbit. It cycles through all N elements without "
            "exploiting structure. Unlike binary_search (cosmos), linear_search has no "
            "orbit-collapsing step. QA-1: this is the 'unquantized' search — it does not "
            "use the quantum number structure of the array. Useful only as a baseline "
            "or when the array is provably unstructured (white noise signal)."
        ),
        orbit_seed=(2, 6),
        levin_cell_type="progenitor",
        organ_roles=["baseline_scanner", "unstructured_fallback"],
        failure_modes=[
            "O(N) per query — degrades on large N",
            "No benefit from sorted order",
            "Cannot parallelize without losing early-exit guarantee",
        ],
        composition_rules=[
            "sequential:binary_search (upgrade when sorted)",
            "parallel:SIMD_scan (vectorized linear search for small N)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["N > 10^4 and array is sortable"],
            recommit_conditions=["Array known to be unsorted and small N < 100"],
            max_satellite_cycles=8, drift_threshold=0.05, partial_fail_threshold=10,
        ),
        source_corpus_refs=[QA1],
        corpus_concepts=["unquantized_scan", "sequential_orbit", "no_structure_exploitation"],
        needs_ocr_backfill=False,
        confidence="high",
    ),

    AlgorithmEntry(
        name="a_star", family="search",
        goal="Find optimal path in weighted graph using heuristic h(n) to guide search",
        cognitive_horizon="adaptive", convergence="conditional",
        time_complexity="O(b^d) worst, O(d) with perfect heuristic", space_complexity="O(b^d)",
        orbit_signature="mixed",
        orbit_rationale=(
            "With admissible + consistent heuristic: cosmos orbit — monotone f(n)=g(n)+h(n) "
            "guarantees optimal path. With inadmissible heuristic: satellite orbit — may "
            "oscillate around optimal. With h=0 (pure Dijkstra): cosmos guaranteed but slow. "
            "With h=infinity (greedy): satellite/singularity — may miss optimal. "
            "The heuristic function h is the orbit selector: admissible h ↔ cosmos injection. "
            "QA-4 Quantize Code: the heuristic is analogous to the 'quantum distance estimate' "
            "that guides the orbit toward the target state."
        ),
        orbit_seed=(3, 7),
        levin_cell_type="progenitor",
        organ_roles=["heuristic_navigator", "adaptive_pathfinder", "planner_core"],
        failure_modes=[
            "Inadmissible heuristic: suboptimal path, no guarantee",
            "Memory blowup: O(b^d) open list on large state spaces",
            "Inconsistency: h violates triangle inequality → re-expansion needed",
        ],
        composition_rules=[
            "hierarchical:hierarchical_A_star (abstract + concrete level)",
            "adaptive:weighted_A_star (suboptimal for speed)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Heuristic determined inadmissible at runtime",
                "Memory limit reached before goal found",
            ],
            recommit_conditions=["Admissible heuristic verified, bounded state space"],
            max_satellite_cycles=3, drift_threshold=0.15, partial_fail_threshold=4,
        ),
        source_corpus_refs=[QA4, QA2],
        corpus_concepts=["quantum_distance_estimate", "heuristic_as_orbit_selector", "monotone_f"],
        needs_ocr_backfill=True,  # QA-3 may have heuristic search vocabulary
        confidence="medium",
    ),

    # ── FAMILY: graph ─────────────────────────────────────────────────────────

    AlgorithmEntry(
        name="bfs", family="graph",
        goal="Find shortest path in unweighted graph via level-order frontier expansion",
        cognitive_horizon="bounded", convergence="guaranteed",
        time_complexity="O(V+E)", space_complexity="O(V)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "BFS expands a frontier that monotonically increases in distance — "
            "a pure forward orbit with no backtracking. The 24-step cosmos cycle maps "
            "to level-by-level expansion. Visited-set enforcement = obstruction certificate: "
            "prevents singularity/satellite trap of revisiting nodes. BFS without visited set "
            "= satellite orbit = infinite loop on cyclic graphs. "
            "QA-4 wave propagation: BFS is literally a discrete wave front — each level "
            "is one wave period, matching the harmonic expansion of QA Wave Theory."
        ),
        orbit_seed=(1, 2),
        levin_cell_type="differentiated",
        organ_roles=["frontier_expander", "wave_propagator", "unweighted_shortest_path_oracle"],
        failure_modes=[
            "O(V) queue — exponential frontier on dense graphs",
            "Unweighted only: edge weights require Dijkstra",
            "No visited set: infinite satellite loop on cyclic graphs",
        ],
        composition_rules=[
            "sequential:dijkstra (upgrade for weighted edges)",
            "parallel:parallel_BFS (frontier split across workers)",
            "hierarchical:bidirectional_BFS (meet in middle)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["Graph has edge weights (switch to Dijkstra)"],
            recommit_conditions=["Unweighted graph confirmed"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=2,
        ),
        source_corpus_refs=[QA4, QA2],
        corpus_concepts=["wave_front", "harmonic_expansion", "level_order", "obstruction_certificate"],
        needs_ocr_backfill=True,  # QA-3 may have graph/network vocabulary
        confidence="high",
    ),

    AlgorithmEntry(
        name="dfs", family="graph",
        goal="Traverse all reachable nodes via depth-first recursion/stack",
        cognitive_horizon="adaptive", convergence="guaranteed",
        time_complexity="O(V+E)", space_complexity="O(V) stack",
        orbit_signature="cosmos",
        orbit_rationale=(
            "DFS follows one branch to its limit then backtracks — a cosmos orbit "
            "that exhausts the search space via deterministic recursion. Unlike BFS, "
            "the orbit goes deep before wide. The recursion stack is the orbit trajectory; "
            "backtracking is the return half of the cosmos 24-cycle. "
            "DFS without visited set: satellite on cyclic graphs (revisits nodes = 8-cycle loop). "
            "QA-1 root structure: DFS explores the prime factor tree of the graph, mirroring "
            "the hierarchical root decomposition in QA natural arithmetic."
        ),
        orbit_seed=(2, 5),
        levin_cell_type="differentiated",
        organ_roles=["deep_explorer", "cycle_detector", "topological_sorter"],
        failure_modes=[
            "Stack overflow on deep graphs (switch to iterative DFS)",
            "No shortest-path guarantee (BFS needed for that)",
            "Order-sensitive: different edge orderings give different DFS trees",
        ],
        composition_rules=[
            "sequential:topological_sort (DFS-based on DAGs)",
            "sequential:SCC_Tarjan (DFS + low-link values)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["Shortest path required (use BFS/Dijkstra)", "Stack depth exceeds system limit"],
            recommit_conditions=["Exhaustive traversal or cycle detection task"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=2,
        ),
        source_corpus_refs=[QA1, QA4],
        corpus_concepts=["root_tree_traversal", "prime_factor_tree", "backtrack_orbit"],
        needs_ocr_backfill=True,
        confidence="medium",
    ),

    AlgorithmEntry(
        name="dijkstra", family="graph",
        goal="Single-source shortest path in weighted graph with non-negative weights",
        cognitive_horizon="global", convergence="guaranteed",
        time_complexity="O((V+E) log V) with heap", space_complexity="O(V)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "Priority queue pops minimum-distance node at each step: "
            "strictly monotone distance reduction = cosmos forward orbit. "
            "Negative weights break monotonicity → relaxation loop = satellite orbit. "
            "Dijkstra's correctness proof is a QA obstruction certificate: no shorter path "
            "can exist beyond the current frontier. "
            "QA-4 Quantize Code: distance relaxation (if dist[v] > dist[u]+w: update) "
            "maps to the QA quantization step where sub-optimal states are replaced by "
            "their quantum (optimal) equivalent."
        ),
        orbit_seed=(2, 5),
        levin_cell_type="differentiated",
        organ_roles=["path_navigator", "shortest_path_oracle", "planner_backbone"],
        failure_modes=[
            "Negative edge weights: relaxation loop → satellite trap",
            "Dense graphs O(V²) without binary heap",
            "Dynamic graphs: full recompute needed on edge change",
        ],
        composition_rules=[
            "hierarchical:A_star (Dijkstra + admissible heuristic)",
            "sequential:bellman_ford (handles negative weights)",
            "parallel:delta_stepping (parallel Dijkstra approximation)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["Negative edge weights detected (switch to Bellman-Ford)"],
            recommit_conditions=["All edge weights confirmed non-negative"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=2,
        ),
        source_corpus_refs=[QA4, QA2],
        corpus_concepts=["quantize_step", "distance_relaxation", "obstruction_certificate", "monotone_orbit"],
        needs_ocr_backfill=False,
        confidence="high",
    ),

    AlgorithmEntry(
        name="bellman_ford", family="graph",
        goal="Single-source shortest path supporting negative weights; detects negative cycles",
        cognitive_horizon="global", convergence="guaranteed",
        time_complexity="O(VE)", space_complexity="O(V)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "Bellman-Ford relaxes all edges V-1 times — a fixed iteration count with "
            "guaranteed convergence for graphs without negative cycles. The V-1 bound "
            "matches the QA quantum number structure: the maximum orbit length before "
            "the state space closes. Negative cycle detection (V-th relaxation still changes) "
            "is the QA orbit singularity detector: if the orbit hasn't stabilized after "
            "V-1 steps, it's in a satellite/singularity trap."
        ),
        orbit_seed=(1, 3),
        levin_cell_type="differentiated",
        organ_roles=["negative_weight_handler", "cycle_detector_path", "robust_path_oracle"],
        failure_modes=[
            "O(VE) — much slower than Dijkstra for non-negative graphs",
            "Negative cycle: algorithm detects but cannot return valid path",
            "No priority queue optimization possible (must relax all edges each pass)",
        ],
        composition_rules=[
            "sequential:dijkstra (upgrade when no negative weights)",
            "sequential:SPFA (Bellman-Ford with queue optimization)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=["No negative weights (Dijkstra preferred for speed)"],
            recommit_conditions=["Negative weights present or negative cycle detection needed"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=2,
        ),
        source_corpus_refs=[QA2, QA4],
        corpus_concepts=["orbit_closure_V_minus_1", "singularity_detector", "fixed_iteration_count"],
        needs_ocr_backfill=True,
        confidence="medium",
    ),

    # ── FAMILY: optimize ──────────────────────────────────────────────────────

    AlgorithmEntry(
        name="gradient_descent", family="optimize",
        goal="Minimize loss L(θ) by iterating θ ← θ - η∇L(θ) until convergence",
        cognitive_horizon="local", convergence="conditional",
        time_complexity="O(D·T) where D=params, T=steps", space_complexity="O(D)",
        orbit_signature="mixed",
        orbit_rationale=(
            "Convex loss: cosmos orbit — each step reduces norm distance to minimum. "
            "Non-convex / local min: satellite orbit — loss oscillates or plateaus. "
            "Saddle / vanishing gradient: singularity orbit — no net movement. "
            "QA Lab's κ (curvature metric) predicts orbit family: high κ → cosmos; "
            "κ ≈ 0 → singularity; oscillating κ → satellite. "
            "The Finite-Orbit Descent theorem (L_{t+L} = ρ(O)·L_t) is gradient descent "
            "in cosmos orbit. QA-4 Quantize Code is a discrete version of gradient descent "
            "on the QA state space."
        ),
        orbit_seed=(3, 7),
        levin_cell_type="progenitor",
        organ_roles=["weight_updater", "loss_minimizer", "core_learning_primitive"],
        failure_modes=[
            "Local minima: non-convex → satellite orbit trap",
            "Learning rate too high: overshoot → divergence",
            "Saddle points: gradient ≈ 0 → near-singularity stall",
            "Flat regions: vanishing gradient → singularity (no movement)",
        ],
        composition_rules=[
            "adaptive:adam (adds momentum + adaptive lr)",
            "adaptive:rmsprop (per-parameter lr scaling)",
            "hierarchical:meta_learning (learns the learning algorithm itself)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Loss plateau for > 50 steps (satellite detected)",
                "Gradient norm < 1e-8 for 10 steps (singularity detected)",
                "Loss increasing for 5 consecutive steps (divergence)",
            ],
            recommit_conditions=["Convex loss confirmed, lr in stable range"],
            max_satellite_cycles=3, drift_threshold=0.20, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA4, QUAD],
        corpus_concepts=["quantize_code_iteration", "orbit_descent", "kappa_curvature", "finite_orbit_descent"],
        needs_ocr_backfill=True,  # QA-3 may have optimization examples
        confidence="high",
    ),

    AlgorithmEntry(
        name="adam", family="optimize",
        goal="Adaptive gradient optimizer: per-parameter lr via first+second moment estimates",
        cognitive_horizon="adaptive", convergence="conditional",
        time_complexity="O(D·T)", space_complexity="O(D) — stores 2 moment vectors",
        orbit_signature="mixed",
        orbit_rationale=(
            "Adam combines momentum (β1) and adaptive scaling (β2) — two orbit dampers "
            "that keep gradient descent from escaping to satellite or singularity. "
            "β1 (momentum) adds inertia: smooths satellite oscillations. "
            "β2 (RMSprop) scales by inverse sqrt of second moment: shrinks steps in "
            "high-curvature directions. Together they keep the orbit near cosmos in "
            "practice, but non-convex landscapes still allow satellite trapping. "
            "Bias correction (1-β^t denominator) is the QA orbit normalization: "
            "ensures the initial steps are correctly scaled."
        ),
        orbit_seed=(4, 6),
        levin_cell_type="progenitor",
        organ_roles=["adaptive_optimizer", "gradient_damper", "momentum_carrier"],
        failure_modes=[
            "Non-convex: still converges to local minimum (satellite orbit)",
            "High β2 → slow adaptation to sudden loss landscape changes",
            "Weight decay interaction: AdamW needed for L2 regularization",
            "Large batch: bias correction insufficient — gradient noise too low",
        ],
        composition_rules=[
            "sequential:gradient_descent (Adam is augmented GD)",
            "adaptive:lr_scheduler (cosine decay on top of Adam)",
            "hybrid:adamw (Adam + decoupled weight decay)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Loss plateau despite momentum (satellite with inertia)",
                "Gradient explosion: moment estimates blow up",
            ],
            recommit_conditions=["Stable loss decrease observed over 20 steps"],
            max_satellite_cycles=3, drift_threshold=0.20, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA4, QUAD],
        corpus_concepts=["orbit_damper", "moment_estimate", "adaptive_scaling", "bias_correction"],
        needs_ocr_backfill=True,
        confidence="medium",
    ),

    AlgorithmEntry(
        name="simulated_annealing", family="optimize",
        goal="Global optimization via probabilistic acceptance of worse solutions (temperature-driven)",
        cognitive_horizon="adaptive", convergence="probabilistic",
        time_complexity="O(T·N) where T=annealing steps", space_complexity="O(1)",
        orbit_signature="mixed",
        orbit_rationale=(
            "High temperature: cosmos orbit — accepts all moves, explores freely. "
            "Low temperature: satellite orbit — only accepts improvements, cycling near optimum. "
            "Temperature = orbit selector: cooling schedule is the cosmos→satellite transition. "
            "At T→0: singularity if stuck at local optimum. "
            "QA-4 Wave Theory: the cooling schedule maps to the damping of harmonics — "
            "high-frequency (noisy) moves are suppressed as temperature drops, "
            "leaving only the fundamental (global optimum) signal. "
            "Metropolis criterion = QA obstruction-aware acceptance: allows temporary "
            "orbit escape to avoid satellite trapping."
        ),
        orbit_seed=(5, 3),
        levin_cell_type="progenitor",
        organ_roles=["global_explorer", "escape_from_local_minima", "temperature_controller"],
        failure_modes=[
            "Cooling too fast: trapped in local optimum (premature singularity)",
            "Cooling too slow: never converges (stays in satellite exploration)",
            "Neighbor function too local: cannot escape basin of attraction",
        ],
        composition_rules=[
            "adaptive:parallel_tempering (multiple temperatures, swap states)",
            "hierarchical:simulated_annealing_in_A_star (SA for heuristic search)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Acceptance rate < 0.01 for 100 steps (frozen = singularity)",
                "Solution quality not improving over 500 steps at current T",
            ],
            recommit_conditions=["Temperature reset (restart), new neighborhood defined"],
            max_satellite_cycles=5, drift_threshold=0.15, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA4],
        corpus_concepts=["harmonic_damping", "temperature_as_orbit_selector", "metropolis_criterion"],
        needs_ocr_backfill=True,  # QA-3 may have thermodynamic analogies
        confidence="medium",
    ),

    # ── FAMILY: learn ─────────────────────────────────────────────────────────

    AlgorithmEntry(
        name="backpropagation", family="learn",
        goal="Compute gradients of loss w.r.t. all parameters via chain rule through layers",
        cognitive_horizon="global", convergence="conditional",
        time_complexity="O(D·L) where D=params per layer, L=layers", space_complexity="O(D·L) activations",
        orbit_signature="mixed",
        orbit_rationale=(
            "Forward pass: cosmos orbit — fixed deterministic computation, no branching. "
            "Backward pass: mixed — depends on loss landscape. Vanishing gradient (deep nets) "
            "= singularity orbit: gradients → 0 at early layers, no update. "
            "Exploding gradient = escape from orbit: divergence. "
            "Batch normalization / residual connections = orbit anchors that prevent "
            "gradient vanishing (singularity trap). "
            "QA-1 Law 7 (Fibonacci roots): the chain rule product across L layers mirrors "
            "the QA (b,e,d,a) recurrence — each layer's gradient is the 'next root' "
            "in the Fibonacci-like composition."
        ),
        orbit_seed=(3, 6),
        levin_cell_type="progenitor",
        organ_roles=["gradient_computer", "learning_backbone", "chain_rule_organ"],
        failure_modes=[
            "Vanishing gradient: deep layers get near-zero updates (singularity)",
            "Exploding gradient: norm blows up (escape from orbit)",
            "Numerical precision: float16 underflow in deep nets",
            "Incorrect implementation: missing activation derivative in chain rule",
        ],
        composition_rules=[
            "sequential:gradient_descent (backprop computes, GD applies)",
            "hierarchical:automatic_differentiation (modern backprop)",
            "adaptive:gradient_clipping (prevents exploding gradient)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Gradient norm < 1e-7 for 5 layers (vanishing detected)",
                "Gradient norm > 10^4 (exploding detected)",
            ],
            recommit_conditions=["Residual connections or batch norm applied"],
            max_satellite_cycles=3, drift_threshold=0.15, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA1, QA4],
        corpus_concepts=["fibonacci_root_recurrence", "chain_rule_as_b_e_d_a", "gradient_orbit"],
        needs_ocr_backfill=True,
        confidence="medium",
    ),

    AlgorithmEntry(
        name="k_means", family="learn",
        goal="Partition N points into K clusters by iterating assignment + centroid update",
        cognitive_horizon="bounded", convergence="guaranteed",
        time_complexity="O(N·K·T) where T=iterations", space_complexity="O(N+K)",
        orbit_signature="satellite",
        orbit_rationale=(
            "K-means alternates between two steps (E-step: assign, M-step: update centroid) — "
            "a two-phase cycle that maps to satellite orbit. Convergence to local optimum is "
            "guaranteed (monotone decrease in inertia) but the loop count is unbounded in theory. "
            "In practice, K-means cycles for tens of iterations — typical satellite period. "
            "QA-4 harmonic structure: K clusters = K harmonic modes. The centroid update "
            "is the 'quantum mean' of the cluster — the average quantum number. "
            "Multiple restarts (to escape bad local optima) = QA dedifferentiation: "
            "reset to singularity (random init) then redifferentiate."
        ),
        orbit_seed=(2, 3),
        levin_cell_type="progenitor",
        organ_roles=["cluster_partitioner", "harmonic_mode_finder", "unsupervised_organ"],
        failure_modes=[
            "Local optima: result depends on initialization (restart needed)",
            "K must be specified: wrong K → poor partition",
            "Non-convex clusters: centroids don't capture true cluster shape",
            "Empty cluster: centroid undefined if no points assigned",
        ],
        composition_rules=[
            "adaptive:k_means_plus_plus (smart initialization → better cosmos probability)",
            "hierarchical:hierarchical_clustering (tree of k_means results)",
            "sequential:PCA (dimensionality reduction before k_means)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Inertia not decreasing for 3 iterations (local optimum)",
                "Empty cluster detected",
            ],
            recommit_conditions=["Re-initialized with k-means++, inertia decreasing"],
            max_satellite_cycles=5, drift_threshold=0.12, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA4, P3],
        corpus_concepts=["harmonic_mode", "quantum_mean", "cluster_period", "satellite_convergence"],
        needs_ocr_backfill=True,  # QA-3 may have grouping/clustering vocabulary
        confidence="medium",
    ),

    AlgorithmEntry(
        name="policy_gradient_rl", family="learn",
        goal="Learn policy π_θ by ascending ∇_θ E[R] via sampled trajectories",
        cognitive_horizon="adaptive", convergence="probabilistic",
        time_complexity="O(T·H·D) T=episodes, H=horizon, D=params", space_complexity="O(D+H)",
        orbit_signature="satellite",
        orbit_rationale=(
            "Policy gradient is inherently noisy: the orbit oscillates around the value "
            "optimum rather than converging cleanly. Unlike GD in convex loss, the policy "
            "landscape is non-stationary (policy changes the distribution it samples from). "
            "This self-referential loop maps to satellite orbit. PPO's clipping = orbit damper. "
            "Sparse reward = singularity injection: agent cannot move until reward arrives. "
            "QA-4: the policy update is a discrete harmonic adjustment — each trajectory "
            "sample is one wave measurement used to estimate the gradient."
        ),
        orbit_seed=(2, 3),
        levin_cell_type="progenitor",
        organ_roles=["policy_adaptor", "environment_interface", "reward_follower"],
        failure_modes=[
            "High variance: noisy gradient estimates → random walk (satellite)",
            "Credit assignment: sparse reward → no signal → singularity",
            "Non-stationarity: environment changes → policy lags",
            "Mode collapse: convergence to suboptimal deterministic policy → singularity",
        ],
        composition_rules=[
            "adaptive:PPO (trust region clips satellite oscillation)",
            "hierarchical:option_framework (temporal abstraction)",
            "sequential:value_function (critic reduces variance)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Gradient variance > 10x gradient mean for 10 updates",
                "Reward not improving over 100 episodes",
            ],
            recommit_conditions=["Critic trained, variance reduced, reward increasing"],
            max_satellite_cycles=5, drift_threshold=0.15, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA4],
        corpus_concepts=["harmonic_trajectory_sample", "reward_as_quantum_signal", "orbit_damper_PPO"],
        needs_ocr_backfill=True,
        confidence="high",
    ),

    AlgorithmEntry(
        name="attention_transformer", family="learn",
        goal="Attend to all input tokens simultaneously via Q·K^T/√d_k scaled dot-product",
        cognitive_horizon="global", convergence="conditional",
        time_complexity="O(N²·D) N=seq_len, D=model_dim", space_complexity="O(N²)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "Each attention head runs an independent orbit over the token space: "
            "softmax normalization ensures the orbit stays bounded. Multi-head = multi-orbit: "
            "different heads track different QA families simultaneously. "
            "Residual connections = orbit anchoring: prevent satellite collapse. "
            "Over-smoothing at depth = singularity injection: all tokens converge to mean. "
            "QA-4 Synchronous Harmonics: multi-head attention is a synchronous harmonic decomposition — "
            "each head extracts a different frequency component of the input signal, "
            "exactly as in Ben Iverson's Wave Theory chapter."
        ),
        orbit_seed=(1, 3),
        levin_cell_type="differentiated",
        organ_roles=["context_integrator", "global_information_aggregator", "transformer_brain"],
        failure_modes=[
            "O(N²) memory: sequence length quadratic bottleneck",
            "Over-smoothing at depth: all tokens attend uniformly → singularity",
            "Attention sink: outlier tokens absorb all weight → satellite (rest starved)",
            "Without positional encoding: permutation invariant (no orbit structure)",
        ],
        composition_rules=[
            "hierarchical:transformer_block (attn + FFN + LayerNorm)",
            "parallel:multi_head (independent orbit per head)",
            "adaptive:sparse_attention (obstruction-aware: only reachable tokens attended)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Attention entropy < 0.1 (attention sink — satellite)",
                "All heads attending same tokens (multi-orbit collapsed to one)",
            ],
            recommit_conditions=["Attention patterns diversified, positional encoding active"],
            max_satellite_cycles=2, drift_threshold=0.10, partial_fail_threshold=3,
        ),
        source_corpus_refs=[QA4, QUAD],
        corpus_concepts=["synchronous_harmonics", "multi_frequency_decomposition", "orbit_per_head", "residual_anchoring"],
        needs_ocr_backfill=False,
        confidence="high",
    ),

    # ── FAMILY: control ───────────────────────────────────────────────────────

    AlgorithmEntry(
        name="pid_controller", family="control",
        goal="Minimize error e(t) via Proportional + Integral + Derivative feedback",
        cognitive_horizon="adaptive", convergence="conditional",
        time_complexity="O(1) per step", space_complexity="O(1)",
        orbit_signature="mixed",
        orbit_rationale=(
            "Well-tuned PID: cosmos orbit — error decays to zero with no overshoot. "
            "Under-damped (high Kp): satellite orbit — oscillates around setpoint. "
            "Integral windup: singularity — integral term dominates, controller frozen. "
            "The three terms map to QA orbit components: P = current deviation (b), "
            "I = accumulated history (e), D = rate of change (d). Together (b,e,d,a) "
            "is the full QA tuple controlling the system. "
            "QA-4 New Wave Theory: PID is a discrete harmonic controller — the setpoint "
            "is the target quantum number, and the controller drives the system toward it."
        ),
        orbit_seed=(4, 5),
        levin_cell_type="progenitor",
        organ_roles=["error_minimizer", "feedback_controller", "setpoint_tracker"],
        failure_modes=[
            "Integral windup: large integral term saturates actuator → singularity",
            "Derivative kick: large D on setpoint change → spike",
            "Wrong gains: oscillation (high Kp = satellite) or slow (low Kp)",
            "Nonlinear plant: linear PID fails on highly nonlinear systems",
        ],
        composition_rules=[
            "hierarchical:cascade_PID (inner/outer loop for complex systems)",
            "adaptive:auto_tuning_PID (Ziegler-Nichols or relay feedback)",
            "sequential:feedforward (add model-based term to reduce PID load)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Oscillation amplitude > 10% setpoint for 5 cycles (satellite)",
                "Integral term > 5x proportional term (windup → singularity)",
            ],
            recommit_conditions=["Gains tuned, step response shows < 5% overshoot"],
            max_satellite_cycles=5, drift_threshold=0.20, partial_fail_threshold=5,
        ),
        source_corpus_refs=[QA4],
        corpus_concepts=["b_e_d_a_as_pid_terms", "harmonic_controller", "setpoint_quantum_number", "wave_theory_control"],
        needs_ocr_backfill=True,  # QA-3 may have control/feedback vocabulary
        confidence="medium",
    ),

    AlgorithmEntry(
        name="kalman_filter", family="control",
        goal="Optimal linear state estimator combining noisy measurements with process model",
        cognitive_horizon="adaptive", convergence="guaranteed",
        time_complexity="O(D³) per step (matrix inversion)", space_complexity="O(D²)",
        orbit_signature="cosmos",
        orbit_rationale=(
            "Kalman filter converges to the MMSE estimate under linear Gaussian assumptions: "
            "guaranteed cosmos orbit for LTI systems. The predict-update cycle is a "
            "two-phase cosmos iteration: predict extends the orbit, update corrects it. "
            "The Kalman gain K = orbit weight: K→0 (trust model) or K→1 (trust sensor). "
            "QA-4 Synchronous Harmonics: Kalman filter is the harmonic estimator — "
            "it separates signal from noise exactly as QA frequency decomposition separates "
            "harmonic modes. The covariance matrix P is the 'orbit uncertainty tensor.'"
        ),
        orbit_seed=(1, 4),
        levin_cell_type="differentiated",
        organ_roles=["state_estimator", "noise_filter", "sensor_fusion_backbone"],
        failure_modes=[
            "Non-linear systems: EKF/UKF approximation may diverge",
            "Model mismatch: wrong process noise Q → filter too slow/fast",
            "Numerical: P matrix loses positive-definiteness (use square-root form)",
            "Observability: unobservable states → estimate doesn't converge",
        ],
        composition_rules=[
            "hierarchical:extended_kalman_filter (EKF for nonlinear systems)",
            "sequential:particle_filter (non-Gaussian alternative)",
            "parallel:information_filter (distributed Kalman)",
        ],
        differentiation_profile=DifferentiationProfile(
            dediff_conditions=[
                "Innovation sequence non-white (model mismatch detected)",
                "P matrix near-singular (numerical breakdown)",
            ],
            recommit_conditions=["Linear system confirmed, Q/R matrices calibrated"],
            max_satellite_cycles=0, drift_threshold=0.05, partial_fail_threshold=3,
        ),
        source_corpus_refs=[QA4],
        corpus_concepts=["harmonic_estimator", "signal_noise_separation", "orbit_uncertainty", "predict_update_cycle"],
        needs_ocr_backfill=True,
        confidence="medium",
    ),
]


# ── Analysis and export ───────────────────────────────────────────────────────

def run_simulations(algorithms: List[AlgorithmEntry]) -> List[AlgorithmEntry]:
    for alg in algorithms:
        alg.simulate()
    return algorithms


def build_registry(algorithms: List[AlgorithmEntry]) -> dict:
    corpus_files = [
        QA1, QA2, QA4, QUAD, P3,
        "qa_3__ocr__qa3.md (OCR running — backfill pending)",
    ]
    return {
        "schema_version": "1.0",
        "generated": "2026-03-26",
        "total_algorithms": len(algorithms),
        "corpus_status": {
            "extracted": [QA1, QA2, QA4, QUAD, P3],
            "ocr_running": ["qa_3__ocr__qa3.md"],
            "ocr_pending": [
                "pyth-1__ocr__pyth1.md",
                "pyth-2__ocr__pyth2.md",
                "qa_workbook__ocr__workbook.md",
            ],
            "total_extracted_chars": 2198298,
        },
        "families": sorted(set(a.family for a in algorithms)),
        "orbit_distribution": {
            fam: [a.name for a in algorithms if a.orbit_signature == fam]
            for fam in ("cosmos", "satellite", "mixed", "singularity")
        },
        "cell_type_distribution": {
            ct: [a.name for a in algorithms if a.levin_cell_type == ct]
            for ct in ("differentiated", "progenitor", "stem")
        },
        "needs_ocr_backfill": [a.name for a in algorithms if a.needs_ocr_backfill],
        "algorithms": [a.to_dict() for a in algorithms],
    }


def print_summary(registry: dict) -> None:
    print(f"\n{'='*72}")
    print(f"QA ALGORITHM COMPETENCY REGISTRY  v{registry['schema_version']}")
    print(f"{'='*72}")
    print(f"  Total algorithms: {registry['total_algorithms']}")
    print(f"  Families: {', '.join(registry['families'])}")

    print(f"\n  Orbit distribution:")
    for fam, names in registry["orbit_distribution"].items():
        if names:
            print(f"    {fam:12s}: {len(names):2d}  [{', '.join(names)}]")

    print(f"\n  Levin cell types:")
    for ct, names in registry["cell_type_distribution"].items():
        if names:
            print(f"    {ct:15s}: {len(names):2d}  [{', '.join(names)}]")

    print(f"\n  Corpus status:")
    cs = registry["corpus_status"]
    print(f"    Extracted: {len(cs['extracted'])} files, {cs['total_extracted_chars']:,} chars")
    print(f"    OCR running: {cs['ocr_running']}")
    print(f"    OCR pending: {cs['ocr_pending']}")

    print(f"\n  Needs OCR backfill ({len(registry['needs_ocr_backfill'])} algorithms):")
    print(f"    {', '.join(registry['needs_ocr_backfill'])}")

    print(f"\n  {'Algorithm':<25} {'Family':<12} {'Orbit':>10} {'Simulated':>10} {'OFR':>6} {'Cell':>15} {'Conf':>6}")
    print(f"  {'-'*88}")
    for a in ALGORITHMS:
        print(f"  {a.name:<25} {a.family:<12} {a.orbit_signature:>10} "
              f"{a.simulated_orbit:>10} {a.simulated_ofr:>6.3f} "
              f"{a.levin_cell_type:>15} {a.confidence:>6}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="QA Algorithm Competency Registry Builder")
    parser.add_argument("--summary", action="store_true", help="Print summary table only")
    parser.add_argument("--family", help="Show only algorithms from this family")
    parser.add_argument("--out", type=Path,
                        default=Path("qa_algorithm_competency_registry.json"))
    args = parser.parse_args(argv)

    algs = run_simulations(ALGORITHMS)

    if args.family:
        algs = [a for a in algs if a.family == args.family]

    registry = build_registry(algs if not args.family else algs)

    if not args.summary:
        args.out.write_text(json.dumps(build_registry(ALGORITHMS), indent=2), encoding="utf-8")
        print(f"Registry written → {args.out}  ({len(ALGORITHMS)} algorithms)")

    print_summary(registry)


if __name__ == "__main__":
    main()
# NOTE: second batch appended 2026-03-26 — run qa_algorithm_competency_registry_v2.py for full 36-alg registry
