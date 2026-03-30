#!/usr/bin/env python3
"""
Enhanced synthetic problem generator with staged branching injectors.

Implements ChatGPT's recommendations:
Stage 1: Parallel bundles with gaps
Stage 2: Angle equality webs (8-12 facts)
Stage 3: Circle distractor surfaces
Stage 4: Shared-perp hubs for route diversity

Target: max_successors_generated ≥ 30 per expansion
"""

import json
from typing import List, Dict, Any

def generate_s01_enhanced(stage: int = 4) -> Dict[str, Any]:
    """
    3x3 lattice with staged branching injectors

    Base structure:
    - Lines 1-3: h1, h2, h3 (horizontals)
    - Lines 4-6: v1, v2, v3 (verticals)

    Goal: h1 ⊥ v3 (i.e., line 1 ⊥ line 6)
    """
    next_id = 7  # Start allocating IDs after base lattice

    # Base lattice facts
    givens = [
        {"Parallel": [1, 2]},  # h1 || h2
        {"Parallel": [2, 3]},  # h2 || h3
        {"Parallel": [4, 5]},  # v1 || v2
        {"Parallel": [5, 6]},  # v2 || v3
        {"Perpendicular": [1, 4]},  # h1 ⊥ v1
        {"Perpendicular": [2, 5]}   # h2 ⊥ v2
    ]

    # Stage 1: Parallel bundles with gaps
    if stage >= 1:
        # Add diagonal lines d1, d2, d3, d4 with gaps
        d1, d2, d3, d4 = next_id, next_id+1, next_id+2, next_id+3
        next_id += 4

        givens.extend([
            {"Parallel": [d1, d2]},  # d1 || d2 (gap before d3)
            {"Parallel": [d3, d4]},  # d3 || d4 (gap after d2)
        ])

    # Stage 2: Angle equality web (8-12 facts)
    if stage >= 2:
        # Create angle equality web among existing lines
        # Use combinations of (h_i, v_j, d_k) to create branching
        givens.extend([
            {"EqualAngle": [1, 4, 2, 5]},   # ∠(h1,v1) = ∠(h2,v2)
            {"EqualAngle": [2, 5, 3, 6]},   # ∠(h2,v2) = ∠(h3,v3)
            {"EqualAngle": [1, 5, 2, 6]},   # ∠(h1,v2) = ∠(h2,v3)
            {"EqualAngle": [1, d1, 2, d2]}, # ∠(h1,d1) = ∠(h2,d2)
            {"EqualAngle": [2, d2, 3, d3]}, # ∠(h2,d2) = ∠(h3,d3)
            {"EqualAngle": [4, d1, 5, d2]}, # ∠(v1,d1) = ∠(v2,d2)
            {"EqualAngle": [5, d2, 6, d3]}, # ∠(v2,d2) = ∠(v3,d3)
            {"EqualAngle": [1, 6, d1, d3]}, # ∠(h1,v3) = ∠(d1,d3)
        ])

    # Stage 3: Circle distractor surface
    if stage >= 3:
        # Add 2 circles with points
        c1, c2 = next_id, next_id+1
        next_id += 2

        # Points on circle 1 (6 points)
        p1, p2, p3, p4, p5, p6 = next_id, next_id+1, next_id+2, next_id+3, next_id+4, next_id+5
        next_id += 6

        # Points on circle 2 (6 points)
        q1, q2, q3, q4, q5, q6 = next_id, next_id+1, next_id+2, next_id+3, next_id+4, next_id+5
        next_id += 6

        givens.extend([
            {"OnCircle": [p1, c1]},
            {"OnCircle": [p2, c1]},
            {"OnCircle": [p3, c1]},
            {"OnCircle": [p4, c1]},
            {"OnCircle": [p5, c1]},
            {"OnCircle": [p6, c1]},
            {"OnCircle": [q1, c2]},
            {"OnCircle": [q2, c2]},
            {"OnCircle": [q3, c2]},
            {"OnCircle": [q4, c2]},
            {"OnCircle": [q5, c2]},
            {"OnCircle": [q6, c2]},
        ])

    # Stage 4: Shared-perp hubs for route diversity
    if stage >= 4:
        # Add hub: both h2 and h3 perpendicular to v2
        # This creates parallel inference h2 || h3
        givens.append({"Perpendicular": [3, 5]})  # h3 ⊥ v2 (h2 ⊥ v2 already exists)

        # Add second hub: both v1 and v3 perpendicular to h2
        # This creates parallel inference v1 || v3
        givens.append({"Perpendicular": [2, 6]})  # h2 ⊥ v3 (h2 ⊥ v2 already exists)

    return {
        "id": f"s01_lattice_3x3_stage{stage}",
        "description": f"Small perpendicular lattice (stage {stage} branching)",
        "difficulty": 3 + stage,
        "givens": givens,
        "goals": [{"Perpendicular": [1, 6]}]  # h1 ⊥ v3
    }

def generate_s02_enhanced(stage: int = 4) -> Dict[str, Any]:
    """4x4 lattice with staged branching"""
    next_id = 9  # Lines 1-8 for base lattice

    givens = [
        {"Parallel": [1, 2]},  # h1 || h2
        {"Parallel": [2, 3]},  # h2 || h3
        {"Parallel": [3, 4]},  # h3 || h4
        {"Parallel": [5, 6]},  # v1 || v2
        {"Parallel": [6, 7]},  # v2 || v3
        {"Parallel": [7, 8]},  # v3 || v4
        {"Perpendicular": [1, 5]},  # h1 ⊥ v1
        {"Perpendicular": [2, 7]},  # h2 ⊥ v3
        {"Perpendicular": [3, 6]}   # h3 ⊥ v2
    ]

    # Stage 1: Parallel bundles with gaps
    if stage >= 1:
        d1, d2, d3, d4, d5 = next_id, next_id+1, next_id+2, next_id+3, next_id+4
        next_id += 5
        givens.extend([
            {"Parallel": [d1, d2]},
            {"Parallel": [d2, d3]},  # Gap before d4
            {"Parallel": [d4, d5]},
        ])

    # Stage 2: Angle equality web
    if stage >= 2:
        givens.extend([
            {"EqualAngle": [1, 5, 2, 6]},
            {"EqualAngle": [2, 6, 3, 7]},
            {"EqualAngle": [3, 7, 4, 8]},
            {"EqualAngle": [1, 6, 2, 7]},
            {"EqualAngle": [2, 7, 3, 8]},
            {"EqualAngle": [1, d1, 2, d2]},
            {"EqualAngle": [2, d2, 3, d3]},
            {"EqualAngle": [3, d3, 4, d4]},
            {"EqualAngle": [5, d1, 6, d2]},
            {"EqualAngle": [6, d2, 7, d3]},
        ])

    # Stage 3: Circle distractors
    if stage >= 3:
        c1, c2 = next_id, next_id+1
        next_id += 2
        p1, p2, p3, p4, p5, p6 = next_id, next_id+1, next_id+2, next_id+3, next_id+4, next_id+5
        next_id += 6
        q1, q2, q3, q4, q5, q6 = next_id, next_id+1, next_id+2, next_id+3, next_id+4, next_id+5
        next_id += 6

        givens.extend([
            {"OnCircle": [p1, c1]}, {"OnCircle": [p2, c1]}, {"OnCircle": [p3, c1]},
            {"OnCircle": [p4, c1]}, {"OnCircle": [p5, c1]}, {"OnCircle": [p6, c1]},
            {"OnCircle": [q1, c2]}, {"OnCircle": [q2, c2]}, {"OnCircle": [q3, c2]},
            {"OnCircle": [q4, c2]}, {"OnCircle": [q5, c2]}, {"OnCircle": [q6, c2]},
        ])

    # Stage 4: Shared-perp hubs
    if stage >= 4:
        givens.extend([
            {"Perpendicular": [4, 6]},  # h4 ⊥ v2 (creates h3 || h4 via shared v2)
            {"Perpendicular": [2, 8]},  # h2 ⊥ v4 (alternate route)
        ])

    return {
        "id": f"s02_lattice_4x4_stage{stage}",
        "description": f"Medium perpendicular lattice (stage {stage} branching)",
        "difficulty": 4 + stage,
        "givens": givens,
        "goals": [{"Perpendicular": [1, 8]}]  # h1 ⊥ v4
    }

def generate_s03_enhanced(stage: int = 4) -> Dict[str, Any]:
    """5x5 lattice with staged branching"""
    next_id = 11  # Lines 1-10 for base lattice

    givens = [
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
        {"Perpendicular": [4, 7]}   # h4 ⊥ v2
    ]

    # Stage 1: Parallel bundles with gaps
    if stage >= 1:
        d1, d2, d3, d4, d5, d6 = next_id, next_id+1, next_id+2, next_id+3, next_id+4, next_id+5
        next_id += 6
        givens.extend([
            {"Parallel": [d1, d2]},
            {"Parallel": [d2, d3]},  # Gap before d4
            {"Parallel": [d4, d5]},
            {"Parallel": [d5, d6]},
        ])

    # Stage 2: Angle equality web (larger for 5x5)
    if stage >= 2:
        givens.extend([
            {"EqualAngle": [1, 6, 2, 7]},
            {"EqualAngle": [2, 7, 3, 8]},
            {"EqualAngle": [3, 8, 4, 9]},
            {"EqualAngle": [4, 9, 5, 10]},
            {"EqualAngle": [1, 7, 2, 8]},
            {"EqualAngle": [2, 8, 3, 9]},
            {"EqualAngle": [3, 9, 4, 10]},
            {"EqualAngle": [1, d1, 2, d2]},
            {"EqualAngle": [2, d2, 3, d3]},
            {"EqualAngle": [3, d3, 4, d4]},
            {"EqualAngle": [6, d1, 7, d2]},
            {"EqualAngle": [7, d2, 8, d3]},
        ])

    # Stage 3: Circle distractors
    if stage >= 3:
        c1, c2 = next_id, next_id+1
        next_id += 2
        p1, p2, p3, p4, p5, p6, p7, p8 = [next_id+i for i in range(8)]
        next_id += 8
        q1, q2, q3, q4, q5, q6, q7, q8 = [next_id+i for i in range(8)]
        next_id += 8

        givens.extend([
            {"OnCircle": [p1, c1]}, {"OnCircle": [p2, c1]}, {"OnCircle": [p3, c1]}, {"OnCircle": [p4, c1]},
            {"OnCircle": [p5, c1]}, {"OnCircle": [p6, c1]}, {"OnCircle": [p7, c1]}, {"OnCircle": [p8, c1]},
            {"OnCircle": [q1, c2]}, {"OnCircle": [q2, c2]}, {"OnCircle": [q3, c2]}, {"OnCircle": [q4, c2]},
            {"OnCircle": [q5, c2]}, {"OnCircle": [q6, c2]}, {"OnCircle": [q7, c2]}, {"OnCircle": [q8, c2]},
        ])

    # Stage 4: Shared-perp hubs
    if stage >= 4:
        givens.extend([
            {"Perpendicular": [5, 7]},  # h5 ⊥ v2
            {"Perpendicular": [3, 10]}, # h3 ⊥ v5
            {"Perpendicular": [2, 10]}, # h2 ⊥ v5 (creates route diversity)
        ])

    return {
        "id": f"s03_lattice_5x5_stage{stage}",
        "description": f"Large perpendicular lattice (stage {stage} branching)",
        "difficulty": 5 + stage,
        "givens": givens,
        "goals": [{"Perpendicular": [1, 10]}]  # h1 ⊥ v5
    }

def main():
    """Generate enhanced problems at different stages"""
    import sys

    stage = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"Generating problems with stage {stage} branching injectors...")

    problems = [
        ("s01_lattice_3x3.json", generate_s01_enhanced(stage)),
        ("s02_lattice_4x4.json", generate_s02_enhanced(stage)),
        ("s03_lattice_5x5.json", generate_s03_enhanced(stage)),
    ]

    for filename, problem in problems:
        path = f"tests/fixtures/problems/synthetic/{filename}"
        with open(path, 'w') as f:
            json.dump(problem, f, indent=2)
        print(f"  ✅ Generated {filename} (stage {stage})")

    print(f"\n✨ Generated 3 problems with stage {stage} branching")
    print(f"📊 Next: Run structure probe to validate branching ≥ 30")

if __name__ == "__main__":
    main()
