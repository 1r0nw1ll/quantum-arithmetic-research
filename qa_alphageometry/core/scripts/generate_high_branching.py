#!/usr/bin/env python3
"""
High-branching synthetic problem generator using only valid fact types.

Uses ChatGPT's Injector A + B approach:
- Injector A: Parallel fan-out (transitivity chains)
- Injector B: Shared-perpendicular hubs (O(n²) branching)

No angles, no circles - just Parallel + Perpendicular.
"""

import json
import sys
from pathlib import Path

# Import branching score
sys.path.insert(0, str(Path(__file__).parent))
from branching_score import branching_score

def generate_s01_high_branching():
    """
    3x3 lattice with massive branching from injectors

    Target: branching score ≥ 100 (conservative proxy for 30+ successors)
    """
    next_id = 7  # Lines 1-6 for base lattice

    givens = [
        # Base lattice (h1-h3 = 1-3, v1-v3 = 4-6)
        {"Parallel": [1, 2]},  # h1 || h2
        {"Parallel": [2, 3]},  # h2 || h3
        {"Parallel": [4, 5]},  # v1 || v2
        {"Parallel": [5, 6]},  # v2 || v3
        {"Perpendicular": [1, 4]},  # h1 ⊥ v1
        {"Perpendicular": [2, 5]},  # h2 ⊥ v2
    ]

    # Injector A: Parallel fan-out (create long transitivity chains)
    # Add 8 diagonal lines d1-d8 with transitivity surface
    d1, d2, d3, d4, d5, d6, d7, d8 = range(next_id, next_id + 8)
    next_id += 8

    givens.extend([
        {"Parallel": [d1, d2]},
        {"Parallel": [d2, d3]},
        {"Parallel": [d3, d4]},
        {"Parallel": [d4, d5]},
        {"Parallel": [d5, d6]},
        {"Parallel": [d6, d7]},
        {"Parallel": [d7, d8]},
    ])

    # Injector B: Shared-perpendicular hubs (O(n²) explosion)
    # Hub 1: line 1 (h1) perpendicular to 6 spokes
    spokes1 = range(next_id, next_id + 6)
    next_id += 6
    for spoke in spokes1:
        givens.append({"Perpendicular": [1, spoke]})

    # Hub 2: line 4 (v1) perpendicular to 6 spokes
    spokes2 = range(next_id, next_id + 6)
    next_id += 6
    for spoke in spokes2:
        givens.append({"Perpendicular": [4, spoke]})

    # Hub 3: Create another hub with the diagonal fan-out
    # d1 perpendicular to 5 spokes (connects to parallel surface)
    spokes3 = range(next_id, next_id + 5)
    next_id += 5
    for spoke in spokes3:
        givens.append({"Perpendicular": [d1, spoke]})

    problem = {
        "id": "s01_lattice_3x3",
        "description": "Small perpendicular lattice with high branching",
        "difficulty": 4,
        "givens": givens,
        "goals": [{"Perpendicular": [1, 6]}]  # h1 ⊥ v3
    }

    return problem

def generate_s02_high_branching():
    """4x4 lattice with massive branching"""
    next_id = 9

    givens = [
        # Base lattice (h1-h4 = 1-4, v1-v4 = 5-8)
        {"Parallel": [1, 2]},  # h1 || h2
        {"Parallel": [2, 3]},  # h2 || h3
        {"Parallel": [3, 4]},  # h3 || h4
        {"Parallel": [5, 6]},  # v1 || v2
        {"Parallel": [6, 7]},  # v2 || v3
        {"Parallel": [7, 8]},  # v3 || v4
        {"Perpendicular": [1, 5]},  # h1 ⊥ v1
        {"Perpendicular": [2, 7]},  # h2 ⊥ v3
        {"Perpendicular": [3, 6]},  # h3 ⊥ v2
    ]

    # Injector A: Longer parallel chain
    d = list(range(next_id, next_id + 10))
    next_id += 10
    for i in range(len(d) - 1):
        givens.append({"Parallel": [d[i], d[i+1]]})

    # Injector B: Multiple shared-perp hubs
    for hub in [1, 2, 5, 6]:
        spokes = range(next_id, next_id + 7)
        next_id += 7
        for spoke in spokes:
            givens.append({"Perpendicular": [hub, spoke]})

    problem = {
        "id": "s02_lattice_4x4",
        "description": "Medium perpendicular lattice with high branching",
        "difficulty": 5,
        "givens": givens,
        "goals": [{"Perpendicular": [1, 8]}]  # h1 ⊥ v4
    }

    return problem

def generate_s03_high_branching():
    """5x5 lattice with massive branching"""
    next_id = 11

    givens = [
        # Base lattice (h1-h5 = 1-5, v1-v5 = 6-10)
        {"Parallel": [1, 2]},  # h1 || h2
        {"Parallel": [2, 3]},  # h2 || h3
        {"Parallel": [3, 4]},  # h3 || h4
        {"Parallel": [4, 5]},  # h4 || h5
        {"Parallel": [6, 7]},  # v1 || v2
        {"Parallel": [7, 8]},  # v2 || v3
        {"Parallel": [8, 9]},  # v3 || v4
        {"Parallel": [9, 10]}, # v4 || v5
        {"Perpendicular": [1, 6]},  # h1 ⊥ v1
        {"Perpendicular": [2, 8]},  # h2 ⊥ v3
        {"Perpendicular": [4, 7]},  # h4 ⊥ v2
    ]

    # Injector A: Very long parallel chain
    d = list(range(next_id, next_id + 12))
    next_id += 12
    for i in range(len(d) - 1):
        givens.append({"Parallel": [d[i], d[i+1]]})

    # Injector B: Many shared-perp hubs
    for hub in [1, 2, 3, 6, 7, 8]:
        spokes = range(next_id, next_id + 8)
        next_id += 8
        for spoke in spokes:
            givens.append({"Perpendicular": [hub, spoke]})

    problem = {
        "id": "s03_lattice_5x5",
        "description": "Large perpendicular lattice with high branching",
        "difficulty": 6,
        "givens": givens,
        "goals": [{"Perpendicular": [1, 10]}]  # h1 ⊥ v5
    }

    return problem

def main():
    problems = [
        ("s01_lattice_3x3.json", generate_s01_high_branching()),
        ("s02_lattice_4x4.json", generate_s02_high_branching()),
        ("s03_lattice_5x5.json", generate_s03_high_branching()),
    ]

    print("Generating high-branching problems with Injector A + B...\n")

    for filename, problem in problems:
        # Compute branching score
        score = branching_score(problem)

        # Save
        path = Path(f"tests/fixtures/problems/synthetic/{filename}")
        path.write_text(json.dumps(problem, indent=2))

        print(f"✅ {filename}")
        print(f"   Branching score: {score['total']}")
        print(f"   - Shared-perp pairs: {score['shared_perp_pairs']}")
        print(f"   - Parallel len-2 paths: {score['parallel_len2']}")
        print(f"   - Perp*Parallel prop: {score['perp_parallel_prop']}")
        print()

    print("✨ Generated 3 problems with high-branching injectors")
    print("📊 Next: Run structure probe to validate actual branching ≥ 30")

if __name__ == "__main__":
    main()
