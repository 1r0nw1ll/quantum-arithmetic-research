#!/usr/bin/env python3
"""
Rule-batch discriminative problem generator.

Creates problems that trigger multiple rule families (high rule surface)
with large fact batches (high fact volume), matching the actual beam search
architecture (1 successor per rule, not per fact).

Target criteria:
- rule_surface_score ≥ 4 (distinct rule families)
- fact_volume_score ≥ 25 (total new facts)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from branching_score import branching_score


def generate_t01_dual_route():
    """
    Dual-route problem with 8 rule families but low fact volume.
    Used as conceptual reference.
    """
    return {
        "id": "t01_dual_route_rulebatch",
        "description": "Multi-surface reference problem (8 rule families, low fact volume)",
        "difficulty": 4,
        "givens": [
            # Route A: PerpendicularToParallel hub
            {"Perpendicular": [1, 3]},
            {"Perpendicular": [1, 6]},
            # Route B: Parallel transitivity chain
            {"Parallel": [3, 4]},
            {"Parallel": [4, 5]},
            {"Parallel": [5, 6]},
            # Perp + Parallel propagation
            {"Perpendicular": [2, 4]},
            # Coincident lines chain
            {"CoincidentLines": [7, 8]},
            {"CoincidentLines": [8, 9]},
            # Concentric circles chain
            {"ConcentricCircles": [100, 101]},
            {"ConcentricCircles": [101, 102]},
            # Segment equality chain
            {"EqualLength": [400, 401]},
            {"EqualLength": [401, 402]},
            # OnCircle → Concyclic
            {"OnCircle": [200, 300]},
            {"OnCircle": [201, 300]},
            {"OnCircle": [202, 300]},
            {"OnCircle": [203, 300]},
            {"OnCircle": [204, 300]},
            # OnLine → Collinear
            {"OnLine": [2000, 3]},
            {"OnLine": [2001, 3]},
            {"OnLine": [2002, 3]},
        ],
        "goals": [{"Parallel": [3, 6]}]
    }


def generate_t02_scaled():
    """
    Scaled-up multi-surface problem hitting both targets.

    Strategy: Scale up T01's design with larger injectors.
    """
    next_id = 10

    givens = [
        # Larger perp hub for PerpendicularToParallel
        {"Perpendicular": [1, 3]},
        {"Perpendicular": [1, 4]},
        {"Perpendicular": [1, 5]},
        {"Perpendicular": [1, 6]},  # 4 spokes → C(4,2) = 6 parallel facts

        # Longer parallel chain for ParallelTransitivity
        {"Parallel": [3, 4]},
        {"Parallel": [4, 5]},
        {"Parallel": [5, 6]},
        {"Parallel": [6, 7]},
        {"Parallel": [7, 8]},  # 6 lines → more len-2 paths

        # Perp + Parallel propagation surface
        {"Perpendicular": [2, 4]},
        {"Perpendicular": [2, 5]},

        # Coincident lines: longer chain
        {"CoincidentLines": [next_id, next_id+1]},
        {"CoincidentLines": [next_id+1, next_id+2]},
        {"CoincidentLines": [next_id+2, next_id+3]},
    ]
    next_id += 4

    # Concentric circles: longer chain
    givens.extend([
        {"ConcentricCircles": [next_id, next_id+1]},
        {"ConcentricCircles": [next_id+1, next_id+2]},
        {"ConcentricCircles": [next_id+2, next_id+3]},
    ])
    next_id += 4

    # Segment equality: longer chain
    givens.extend([
        {"EqualLength": [next_id, next_id+1]},
        {"EqualLength": [next_id+1, next_id+2]},
        {"EqualLength": [next_id+2, next_id+3]},
    ])
    next_id += 4

    # OnCircle → Concyclic: 6 points for C(6,4) = 15 quartets
    circle_id = next_id
    next_id += 1
    for i in range(6):
        givens.append({"OnCircle": [next_id + i, circle_id]})
    next_id += 6

    # OnLine → Collinear: 4 points for C(4,3) = 4 triplets
    line_id = 3  # Reuse line from parallel chain
    for i in range(4):
        givens.append({"OnLine": [next_id + i, line_id]})
    next_id += 4

    return {
        "id": "t02_scaled_multisurface",
        "description": "Scaled multi-surface problem (8 families, high fact volume)",
        "difficulty": 5,
        "givens": givens,
        "goals": [{"Parallel": [3, 8]}]  # Parallel transitivity goal
    }


def generate_t03_mega():
    """
    Maximum discrimination problem with massive rule surface + fact volume.
    """
    next_id = 12

    givens = [
        # TWO large perp hubs
        {"Perpendicular": [1, 3]},
        {"Perpendicular": [1, 4]},
        {"Perpendicular": [1, 5]},
        {"Perpendicular": [1, 6]},
        {"Perpendicular": [1, 7]},  # Hub 1: 5 spokes → C(5,2) = 10

        {"Perpendicular": [2, 8]},
        {"Perpendicular": [2, 9]},
        {"Perpendicular": [2, 10]},
        {"Perpendicular": [2, 11]},  # Hub 2: 4 spokes → C(4,2) = 6

        # Very long parallel chain
        {"Parallel": [3, 4]},
        {"Parallel": [4, 5]},
        {"Parallel": [5, 6]},
        {"Parallel": [6, 7]},
        {"Parallel": [7, 8]},
        {"Parallel": [8, 9]},  # 7 lines → many len-2 paths

        # Multiple perp + parallel intersections
        {"Perpendicular": [2, 5]},
        {"Perpendicular": [2, 6]},
    ]

    # Long coincident chain
    for i in range(5):
        givens.append({"CoincidentLines": [next_id + i, next_id + i + 1]})
    next_id += 6

    # Long concentric chain
    for i in range(5):
        givens.append({"ConcentricCircles": [next_id + i, next_id + i + 1]})
    next_id += 6

    # Long equality chain
    for i in range(5):
        givens.append({"EqualLength": [next_id + i, next_id + i + 1]})
    next_id += 6

    # Large OnCircle surface: 7 points → C(7,4) = 35 quartets
    circle_id = next_id
    next_id += 1
    for i in range(7):
        givens.append({"OnCircle": [next_id + i, circle_id]})
    next_id += 7

    # Large OnLine surface: 5 points → C(5,3) = 10 triplets
    line_id = 5
    for i in range(5):
        givens.append({"OnLine": [next_id + i, line_id]})
    next_id += 5

    return {
        "id": "t03_mega_discrimination",
        "description": "Maximum discrimination (8 families, massive fact volume)",
        "difficulty": 7,
        "givens": givens,
        "goals": [{"Parallel": [3, 9]}]
    }


def main():
    problems = [
        ("t01_dual_route.json", generate_t01_dual_route()),
        ("t02_scaled_multisurface.json", generate_t02_scaled()),
        ("t03_mega_discrimination.json", generate_t03_mega()),
    ]

    print("Generating rule-batch discriminative problems...\n")

    for filename, problem in problems:
        score = branching_score(problem)

        # Save
        path = Path(f"tests/fixtures/problems/synthetic/{filename}")
        path.write_text(json.dumps(problem, indent=2))

        # Report
        is_discriminative = (score['rule_surface_score'] >= 4 and
                            score['fact_volume_score'] >= 25)
        status = "✅ PASS" if is_discriminative else "❌ FAIL"

        print(f"{status} {filename}")
        print(f"   Rule surface: {score['rule_surface_score']} (need ≥4)")
        print(f"   Fact volume:  {score['fact_volume_score']} (need ≥25)")
        print()

    print("✨ Generated 3 rule-batch discriminative problems")
    print("📊 Next: Run actual beam search to validate rules fired matches prediction")


if __name__ == "__main__":
    main()
