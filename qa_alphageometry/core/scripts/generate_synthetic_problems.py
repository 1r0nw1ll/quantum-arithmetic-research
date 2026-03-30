#!/usr/bin/env python3
"""
Generate discriminative synthetic problems for QA-AlphaGeometry

Family S: Perpendicular Lattices with Decoys
- High branching factor from parallel/perpendicular combinations
- Multiple proof paths via transitivity chains
- QA-sensitive structure (perpendicular-heavy)
"""

import json
import sys
from pathlib import Path


def generate_s01_lattice_3x3():
    """
    3×3 perpendicular lattice

    Structure:
    - 3 horizontal lines (h1, h2, h3) all parallel
    - 3 vertical lines (v1, v2, v3) all parallel
    - Some h ⊥ v facts given
    - Goal: Prove h1 ⊥ v3 (requires transitivity)

    Branching factor: ~20-30 from perpendicular transitivity applications
    Multiple paths: Can go h1→v1→h2→v2→h3→v3 or direct routes
    """
    problem = {
        "id": "s01_lattice_3x3",
        "description": "Small perpendicular lattice with parallel families",
        "difficulty": 3,
        "symbols": {
            "lines": ["h1", "h2", "h3", "v1", "v2", "v3"]
        },
        "facts": [
            # Horizontal lines are parallel to each other
            {"Parallel": ["h1", "h2"]},
            {"Parallel": ["h2", "h3"]},

            # Vertical lines are parallel to each other
            {"Parallel": ["v1", "v2"]},
            {"Parallel": ["v2", "v3"]},

            # Some perpendicular facts (not all - force derivation)
            {"Perpendicular": ["h1", "v1"]},
            {"Perpendicular": ["h2", "v2"]},
        ],
        "goal": [
            {"Perpendicular": ["h1", "v3"]}
        ]
    }
    return problem


def generate_s02_lattice_4x4():
    """4×4 lattice - more branching"""
    problem = {
        "id": "s02_lattice_4x4",
        "description": "Medium perpendicular lattice",
        "difficulty": 4,
        "symbols": {
            "lines": ["h1", "h2", "h3", "h4", "v1", "v2", "v3", "v4"]
        },
        "facts": [
            # Horizontal parallels
            {"Parallel": ["h1", "h2"]},
            {"Parallel": ["h2", "h3"]},
            {"Parallel": ["h3", "h4"]},

            # Vertical parallels
            {"Parallel": ["v1", "v2"]},
            {"Parallel": ["v2", "v3"]},
            {"Parallel": ["v3", "v4"]},

            # Seed perpendiculars
            {"Perpendicular": ["h1", "v1"]},
            {"Perpendicular": ["h2", "v3"]},
            {"Perpendicular": ["h3", "v2"]},
        ],
        "goal": [
            {"Perpendicular": ["h1", "v4"]}
        ]
    }
    return problem


def generate_s03_lattice_5x5():
    """5×5 lattice - high branching"""
    problem = {
        "id": "s03_lattice_5x5",
        "description": "Large perpendicular lattice with high branching",
        "difficulty": 5,
        "symbols": {
            "lines": ["h1", "h2", "h3", "h4", "h5", "v1", "v2", "v3", "v4", "v5"]
        },
        "facts": [
            # Horizontal parallels
            {"Parallel": ["h1", "h2"]},
            {"Parallel": ["h2", "h3"]},
            {"Parallel": ["h3", "h4"]},
            {"Parallel": ["h4", "h5"]},

            # Vertical parallels
            {"Parallel": ["v1", "v2"]},
            {"Parallel": ["v2", "v3"]},
            {"Parallel": ["v3", "v4"]},
            {"Parallel": ["v4", "v5"]},

            # Seed perpendiculars (sparse coverage)
            {"Perpendicular": ["h1", "v1"]},
            {"Perpendicular": ["h3", "v3"]},
            {"Perpendicular": ["h5", "v5"]},
        ],
        "goal": [
            {"Perpendicular": ["h1", "v5"]}
        ]
    }
    return problem


def generate_s04_lattice_with_parallels():
    """
    3×3 lattice + many parallel distractors
    Tests if QA ignores irrelevant parallel structure
    """
    problem = {
        "id": "s04_lattice_with_parallels",
        "description": "Lattice with heavy parallel distractors",
        "difficulty": 3,
        "symbols": {
            "lines": ["h1", "h2", "h3", "v1", "v2", "v3",
                     "d1", "d2", "d3", "d4", "d5"]  # Distractor lines
        },
        "facts": [
            # Core lattice
            {"Parallel": ["h1", "h2"]},
            {"Parallel": ["h2", "h3"]},
            {"Parallel": ["v1", "v2"]},
            {"Parallel": ["v2", "v3"]},
            {"Perpendicular": ["h1", "v1"]},
            {"Perpendicular": ["h2", "v2"]},

            # DISTRACTORS: Many parallel lines unrelated to goal
            {"Parallel": ["d1", "d2"]},
            {"Parallel": ["d2", "d3"]},
            {"Parallel": ["d3", "h1"]},  # Link to core
            {"Parallel": ["d4", "d5"]},
            {"Parallel": ["d4", "v1"]},  # Link to core
        ],
        "goal": [
            {"Perpendicular": ["h1", "v3"]}
        ]
    }
    return problem


def generate_s05_lattice_with_equalities():
    """
    3×3 lattice + equality distractors
    Tests if QA ignores irrelevant equality structure
    """
    problem = {
        "id": "s05_lattice_with_equalities",
        "description": "Lattice with heavy equality distractors",
        "difficulty": 3,
        "symbols": {
            "lines": ["h1", "h2", "h3", "v1", "v2", "v3"],
            "segments": ["s1", "s2", "s3", "s4", "s5", "s6"]
        },
        "facts": [
            # Core lattice
            {"Parallel": ["h1", "h2"]},
            {"Parallel": ["h2", "h3"]},
            {"Parallel": ["v1", "v2"]},
            {"Parallel": ["v2", "v3"]},
            {"Perpendicular": ["h1", "v1"]},
            {"Perpendicular": ["h2", "v2"]},

            # DISTRACTORS: Segment equalities
            {"EqualLength": ["s1", "s2"]},
            {"EqualLength": ["s2", "s3"]},
            {"EqualLength": ["s3", "s4"]},
            {"EqualLength": ["s4", "s5"]},
            {"EqualLength": ["s5", "s6"]},
        ],
        "goal": [
            {"Perpendicular": ["h1", "v3"]}
        ]
    }
    return problem


def save_problem(problem, output_dir):
    """Save problem to JSON file"""
    output_path = output_dir / f"{problem['id']}.json"
    with open(output_path, 'w') as f:
        json.dump(problem, f, indent=2)
    print(f"✅ Generated: {output_path.name}")


def main():
    # Output directory
    output_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "problems" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔬 Generating Family S: Perpendicular Lattices\n")

    # Generate first 5 problems
    problems = [
        generate_s01_lattice_3x3(),
        generate_s02_lattice_4x4(),
        generate_s03_lattice_5x5(),
        generate_s04_lattice_with_parallels(),
        generate_s05_lattice_with_equalities(),
    ]

    for problem in problems:
        save_problem(problem, output_dir)

    print(f"\n📁 Saved to: {output_dir}")
    print(f"✅ Generated {len(problems)} problems")
    print("\n📝 Next: Create problems s06-s10 for complete Family S")


if __name__ == "__main__":
    main()
