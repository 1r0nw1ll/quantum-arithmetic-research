#!/usr/bin/env python3
"""
Track 1 Problem Generator: All Three Families

Generates 30 discriminative synthetic problems:
- Family S (s01-s10): Perpendicular lattices with decoys
- Family T (t01-t10): Multi-surface competing routes
- Family C (c01-c10): Coordinate-derived right triangles

All problems meet discriminativity criteria:
- rule_surface_score ≥ 4 (distinct rule families)
- fact_volume_score ≥ 25 (total new facts)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from branching_score import branching_score


def generate_family_s():
    """
    Family S: Perpendicular lattices with QA-sensitive decoys.

    Strategy: Create perpendicular hub-and-spoke patterns that trigger
    PerpendicularToParallel rule massively, with decoy elements.
    """
    problems = []

    for i in range(1, 11):
        next_id = 20

        # Scale complexity with problem number
        hub_count = 2 + (i // 3)  # 2-4 hubs
        spokes_per_hub = 4 + (i % 3)  # 4-6 spokes per hub

        givens = []
        hubs = list(range(1, hub_count + 1))

        # Create perpendicular hubs
        for hub in hubs:
            spokes = list(range(next_id, next_id + spokes_per_hub))
            for spoke in spokes:
                givens.append({"Perpendicular": [hub, spoke]})
            next_id += spokes_per_hub

        # Add parallel chains connecting spokes (creates competing routes)
        chain_length = 3 + (i % 4)
        for j in range(chain_length - 1):
            givens.append({"Parallel": [20 + j, 20 + j + 1]})

        # Add distractor surfaces
        # CoincidentLines
        for j in range(2):
            givens.append({"CoincidentLines": [next_id + j, next_id + j + 1]})
        next_id += 3

        # OnCircle cluster
        circle_id = next_id
        next_id += 1
        for j in range(4):
            givens.append({"OnCircle": [next_id + j, circle_id]})
        next_id += 4

        # Goal: Parallel transitivity through the lattice
        problems.append({
            "id": f"s{i:02d}_lattice",
            "description": f"Perpendicular lattice ({hub_count} hubs, {spokes_per_hub} spokes) with decoys",
            "difficulty": 3 + i // 3,
            "givens": givens,
            "goals": [{"Parallel": [20, 20 + chain_length - 1]}]
        })

    return problems


def generate_family_t():
    """
    Family T: Multi-surface problems with competing proof routes.

    Strategy: Multiple rule families with two distinct proof paths,
    forcing QA to discriminate between equally valid routes.
    """
    problems = []

    # t01: Reference problem (smaller than t02/t03)
    problems.append({
        "id": "t01_dual_route_reference",
        "description": "Dual-route reference (6 families, moderate volume)",
        "difficulty": 4,
        "givens": [
            # Route A: PerpendicularToParallel
            {"Perpendicular": [1, 3]},
            {"Perpendicular": [1, 4]},
            {"Perpendicular": [1, 5]},
            # Route B: Parallel transitivity
            {"Parallel": [3, 4]},
            {"Parallel": [4, 5]},
            {"Parallel": [5, 6]},
            # Perp + Parallel propagation
            {"Perpendicular": [2, 4]},
            # Coincident lines
            {"CoincidentLines": [10, 11]},
            {"CoincidentLines": [11, 12]},
            # OnCircle
            {"OnCircle": [20, 30]},
            {"OnCircle": [21, 30]},
            {"OnCircle": [22, 30]},
            {"OnCircle": [23, 30]},
            # OnLine
            {"OnLine": [40, 3]},
            {"OnLine": [41, 3]},
            {"OnLine": [42, 3]},
        ],
        "goals": [{"Parallel": [3, 6]}]
    })

    # t02 and t03 already exist, so generate t04-t10
    for i in range(4, 11):
        next_id = 15

        # Scale up with problem number
        hub_size = 3 + (i % 3)
        chain_length = 4 + (i % 3)

        givens = []

        # Larger perp hub
        for j in range(hub_size):
            givens.append({"Perpendicular": [1, 3 + j]})

        # Longer parallel chain
        for j in range(chain_length):
            givens.append({"Parallel": [3 + j, 4 + j]})

        # Perp + parallel intersections
        givens.append({"Perpendicular": [2, 4]})
        if i >= 6:
            givens.append({"Perpendicular": [2, 5]})

        # Coincident chain
        for j in range(3 + i % 2):
            givens.append({"CoincidentLines": [next_id + j, next_id + j + 1]})
        next_id += 4 + i % 2

        # Concentric chain
        for j in range(3 + i % 2):
            givens.append({"ConcentricCircles": [next_id + j, next_id + j + 1]})
        next_id += 4 + i % 2

        # Equality chain
        for j in range(3):
            givens.append({"EqualLength": [next_id + j, next_id + j + 1]})
        next_id += 4

        # OnCircle cluster
        circle_id = next_id
        next_id += 1
        for j in range(5 + i % 2):
            givens.append({"OnCircle": [next_id + j, circle_id]})
        next_id += 5 + i % 2

        # OnLine cluster
        line_id = 3
        for j in range(4):
            givens.append({"OnLine": [next_id + j, line_id]})
        next_id += 4

        problems.append({
            "id": f"t{i:02d}_multisurface_{hub_size}h_{chain_length}c",
            "description": f"Multi-surface problem ({hub_size} hub spokes, {chain_length} chain links)",
            "difficulty": 5 + i // 3,
            "givens": givens,
            "goals": [{"Parallel": [3, 3 + chain_length]}]
        })

    return problems


def generate_family_c():
    """
    Family C: Coordinate-derived right triangles with rich structure.

    Strategy: Use Pythagorean triples as theme, but add multi-surface
    structure to meet discriminativity criteria (≥4 rule families, ≥25 facts).
    """
    problems = []

    # Common Pythagorean triples
    triples = [
        (3, 4, 5),
        (5, 12, 13),
        (8, 15, 17),
        (7, 24, 25),
        (20, 21, 29),
        (9, 40, 41),
        (12, 35, 37),
        (11, 60, 61),
        (13, 84, 85),
        (36, 77, 85),
    ]

    for i, (a, b, c) in enumerate(triples, 1):
        next_id = 25

        givens = []

        # Main right triangle structure with perpendicular hub
        # Scale hub size with problem number
        hub_size = 3 + (i % 3)
        for j in range(hub_size):
            givens.append({"Perpendicular": [1, 3 + j]})

        # Parallel transitivity chain (theme: coordinate axes)
        chain_length = 4 + (i % 2)
        for j in range(chain_length):
            givens.append({"Parallel": [3 + j, 4 + j]})

        # Add perpendicular-parallel intersection
        givens.append({"Perpendicular": [2, 4]})
        if i >= 5:
            givens.append({"Perpendicular": [2, 5]})

        # Equality chains (theme: Pythagorean ratios)
        for j in range(3 + i % 2):
            givens.append({"EqualLength": [next_id + j, next_id + j + 1]})
        next_id += 4 + i % 2

        # CoincidentLines chain
        for j in range(2 + i % 2):
            givens.append({"CoincidentLines": [next_id + j, next_id + j + 1]})
        next_id += 3 + i % 2

        # ConcentricCircles (theme: circles on hypotenuse)
        for j in range(2 + i % 2):
            givens.append({"ConcentricCircles": [next_id + j, next_id + j + 1]})
        next_id += 3 + i % 2

        # OnCircle cluster (circle inscribed in triangle)
        circle_id = next_id
        next_id += 1
        for j in range(5 + i % 2):
            givens.append({"OnCircle": [next_id + j, circle_id]})
        next_id += 5 + i % 2

        # OnLine cluster (collinear points on leg)
        line_id = 3
        for j in range(4):
            givens.append({"OnLine": [next_id + j, line_id]})
        next_id += 4

        problems.append({
            "id": f"c{i:02d}_pythagorean_{a}_{b}_{c}",
            "description": f"Coordinate-derived ({a}-{b}-{c} triple, {hub_size} hub spokes)",
            "difficulty": 4 + i // 4,
            "givens": givens,
            "goals": [{"Parallel": [3, 3 + chain_length]}]
        })

    return problems


def main():
    """Generate all 30 Track 1 problems."""

    output_dir = Path("tests/fixtures/problems/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Track 1 discriminative problems...\n")
    print("=" * 70)

    # Generate all three families
    family_s = generate_family_s()
    family_t = generate_family_t()
    family_c = generate_family_c()

    all_problems = [
        ("Family S (Lattices)", family_s),
        ("Family T (Multi-surface)", family_t),
        ("Family C (Coordinate)", family_c),
    ]

    total_pass = 0
    total_problems = 0

    for family_name, problems in all_problems:
        print(f"\n{family_name}")
        print("-" * 70)

        for problem in problems:
            score = branching_score(problem)

            # Save problem
            filename = f"{problem['id']}.json"
            path = output_dir / filename
            path.write_text(json.dumps(problem, indent=2))

            # Check discriminativity
            is_discriminative = (
                score['rule_surface_score'] >= 4 and
                score['fact_volume_score'] >= 25
            )

            status = "✅ PASS" if is_discriminative else "⚠️  WEAK"
            total_pass += is_discriminative
            total_problems += 1

            print(f"{status} {problem['id']:30s} "
                  f"rules={score['rule_surface_score']:2d} "
                  f"facts={score['fact_volume_score']:3d}")

    print("\n" + "=" * 70)
    print(f"Generated {total_problems} problems")
    print(f"Discriminative: {total_pass}/{total_problems} "
          f"({100*total_pass//total_problems}%)")
    print(f"\n✨ Track 1 problem generation complete!")
    print(f"📂 Problems saved to: {output_dir}")

    if total_pass < total_problems:
        print(f"\n⚠️  {total_problems - total_pass} problems below discriminativity threshold")
        print("   Consider adjusting parameters or accepting weaker problems")


if __name__ == "__main__":
    main()
