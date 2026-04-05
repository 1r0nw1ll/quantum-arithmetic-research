#!/usr/bin/env python3
"""
Gate-cost proxy comparison between:
- a balanced-ternary full-adder slice
- a binary full-adder slice

Explicit assumptions:
1. Input state predicates are already available.
2. Boolean output rails are synthesized as minimized multi-valued PLA covers.
3. Primitive gates are 2-input AND and 2-input OR only; larger fan-in is built as trees.
4. Output encoding is signed-rail:
   - binary slice: sum_1, carry_1
   - ternary slice: sum_pos, sum_neg, carry_pos, carry_neg

This is a combinational logic cost proxy, not a transistor-level or timing model.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path


BIN_DOMAIN = (0, 1)
TER_DOMAIN = (-1, 0, 1)


def full_adder_ternary(x: int, y: int, carry_in: int) -> tuple[int, int]:
    total = x + y + carry_in
    if total > 1:
        return total - 3, 1
    if total < -1:
        return total + 3, -1
    return total, 0


def full_adder_binary(x: int, y: int, carry_in: int) -> tuple[int, int]:
    total = x + y + carry_in
    return total & 1, 1 if total >= 2 else 0


def nonempty_subsets(domain: tuple[int, ...]) -> list[frozenset[int]]:
    subsets: list[frozenset[int]] = []
    for r in range(1, len(domain) + 1):
        for combo in combinations(domain, r):
            subsets.append(frozenset(combo))
    return subsets


def cube_covers(cube: tuple[frozenset[int], ...], point: tuple[int, ...]) -> bool:
    return all(value in allowed for value, allowed in zip(point, cube))


def cube_literal_cost(cube: tuple[frozenset[int], ...], domain: tuple[int, ...]) -> int:
    factor_sizes = [len(allowed) for allowed in cube if len(allowed) < len(domain)]
    if not factor_sizes:
        return 0
    subset_or_cost = sum(size - 1 for size in factor_sizes)
    and_cost = len(factor_sizes) - 1
    return subset_or_cost + and_cost


def cube_to_text(cube: tuple[frozenset[int], ...], labels: tuple[str, ...]) -> str:
    parts: list[str] = []
    for label, allowed in zip(labels, cube):
        values = sorted(allowed)
        parts.append(f"{label}∈{{{','.join(str(v) for v in values)}}}")
    return " & ".join(parts)


def generate_candidate_cubes(
    domain: tuple[int, ...],
    labels: tuple[str, ...],
    on_set: set[tuple[int, ...]],
) -> list[dict[str, object]]:
    all_points = list(product(domain, repeat=len(labels)))
    subset_choices = nonempty_subsets(domain)
    coverage_to_best: dict[frozenset[tuple[int, ...]], dict[str, object]] = {}

    for cube in product(subset_choices, repeat=len(labels)):
        covered = frozenset(point for point in all_points if cube_covers(cube, point))
        if not covered:
            continue
        if not covered.issubset(on_set):
            continue
        cost = cube_literal_cost(cube, domain)
        candidate = {
            "cube": cube,
            "coverage": covered,
            "term_cost": cost,
            "text": cube_to_text(cube, labels),
        }
        existing = coverage_to_best.get(covered)
        if existing is None or (cost, candidate["text"]) < (existing["term_cost"], existing["text"]):
            coverage_to_best[covered] = candidate

    candidates = list(coverage_to_best.values())
    reduced: list[dict[str, object]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if candidate is other:
                continue
            if candidate["coverage"].issubset(other["coverage"]) and other["term_cost"] <= candidate["term_cost"]:
                if candidate["coverage"] != other["coverage"] or other["term_cost"] < candidate["term_cost"]:
                    dominated = True
                    break
        if not dominated:
            reduced.append(candidate)
    return reduced


def minimize_cover(
    domain: tuple[int, ...],
    labels: tuple[str, ...],
    on_set: set[tuple[int, ...]],
) -> dict[str, object]:
    if not on_set:
        return {"terms": [], "term_count": 0, "total_cost": 0}

    candidates = generate_candidate_cubes(domain, labels, on_set)
    point_to_candidates: dict[tuple[int, ...], list[int]] = {point: [] for point in on_set}
    for idx, candidate in enumerate(candidates):
        for point in candidate["coverage"]:
            point_to_candidates[point].append(idx)

    for point in point_to_candidates:
        point_to_candidates[point].sort(
            key=lambda idx: (
                -(len(candidates[idx]["coverage"])),
                candidates[idx]["term_cost"],
                candidates[idx]["text"],
            )
        )

    best_cost = math.inf
    best_choice: list[int] = []

    @lru_cache(maxsize=None)
    def optimistic_bound(uncovered: frozenset[tuple[int, ...]]) -> int:
        if not uncovered:
            return 0
        min_term_plus_or = min(int(c["term_cost"]) + 1 for c in candidates)
        max_cover = max(len(c["coverage"] & uncovered) for c in candidates)
        needed = math.ceil(len(uncovered) / max_cover)
        return needed * min_term_plus_or

    def search(uncovered: frozenset[tuple[int, ...]], chosen: tuple[int, ...], transformed_cost: int) -> None:
        nonlocal best_cost, best_choice
        if not uncovered:
            final_cost = transformed_cost - 1
            if final_cost < best_cost:
                best_cost = final_cost
                best_choice = list(chosen)
            return

        if transformed_cost + optimistic_bound(uncovered) - 1 >= best_cost:
            return

        pivot = min(uncovered, key=lambda point: len(point_to_candidates[point]))
        options = point_to_candidates[pivot]
        options = sorted(
            options,
            key=lambda idx: (
                -len(candidates[idx]["coverage"] & uncovered),
                candidates[idx]["term_cost"] + 1,
                candidates[idx]["text"],
            ),
        )

        for idx in options:
            candidate = candidates[idx]
            new_uncovered = frozenset(uncovered - candidate["coverage"])
            add_cost = int(candidate["term_cost"]) + 1
            if transformed_cost + add_cost - 1 >= best_cost:
                continue
            search(new_uncovered, chosen + (idx,), transformed_cost + add_cost)

    search(frozenset(on_set), tuple(), 0)

    chosen_terms = [candidates[idx] for idx in best_choice]
    return {
        "terms": [
            {
                "text": term["text"],
                "covered_points": sorted(list(term["coverage"])),
                "term_cost": term["term_cost"],
            }
            for term in chosen_terms
        ],
        "term_count": len(chosen_terms),
        "total_cost": 0 if not chosen_terms else sum(int(term["term_cost"]) for term in chosen_terms) + (len(chosen_terms) - 1),
    }


def binary_functions() -> dict[str, set[tuple[int, ...]]]:
    on_sets = {"sum_1": set(), "carry_1": set()}
    for point in product(BIN_DOMAIN, repeat=3):
        s, c = full_adder_binary(*point)
        if s == 1:
            on_sets["sum_1"].add(point)
        if c == 1:
            on_sets["carry_1"].add(point)
    return on_sets


def ternary_functions() -> dict[str, set[tuple[int, ...]]]:
    on_sets = {"sum_pos": set(), "sum_neg": set(), "carry_pos": set(), "carry_neg": set()}
    for point in product(TER_DOMAIN, repeat=3):
        s, c = full_adder_ternary(*point)
        if s == 1:
            on_sets["sum_pos"].add(point)
        if s == -1:
            on_sets["sum_neg"].add(point)
        if c == 1:
            on_sets["carry_pos"].add(point)
        if c == -1:
            on_sets["carry_neg"].add(point)
    return on_sets


def equivalent_signed_binary_bits(max_abs: int) -> int:
    bits = 1
    while (2 ** (bits - 1)) - 1 < max_abs:
        bits += 1
    return bits


def run_experiment(width_trits: int) -> dict[str, object]:
    labels = ("x", "y", "cin")

    binary_on_sets = binary_functions()
    ternary_on_sets = ternary_functions()

    binary_minimized = {
        name: minimize_cover(BIN_DOMAIN, labels, on_set) for name, on_set in binary_on_sets.items()
    }
    ternary_minimized = {
        name: minimize_cover(TER_DOMAIN, labels, on_set) for name, on_set in ternary_on_sets.items()
    }

    binary_slice_cost = sum(int(info["total_cost"]) for info in binary_minimized.values())
    ternary_slice_cost = sum(int(info["total_cost"]) for info in ternary_minimized.values())

    max_abs = (3**width_trits - 1) // 2
    binary_bits = equivalent_signed_binary_bits(max_abs)
    ternary_total_cost = ternary_slice_cost * width_trits
    binary_total_cost = binary_slice_cost * binary_bits

    verdict = "PASS" if binary_slice_cost > 0 and ternary_slice_cost > 0 else "FAIL"

    return {
        "experiment_id": "arto_ternary_gate_cost_experiment_2026-03-30",
        "hypothesis": (
            "Under a decoded-state multi-valued PLA cost model, balanced-ternary addition may or may not "
            "retain an implementation advantage once explicit combinational logic cost is counted."
        ),
        "success_criteria": (
            "Produce an explicit gate-cost comparison under stated assumptions for a balanced-ternary "
            "adder slice versus a binary full-adder slice, and propagate that comparison to equal-range "
            "ripple adders."
        ),
        "result": verdict,
        "assumptions": [
            "Input state predicates are already available.",
            "Primitive gates are 2-input AND and 2-input OR.",
            "Large fan-in is implemented as trees of 2-input gates.",
            "Adder outputs use signed-rail Boolean functions.",
            "Costs below are combinational logic proxy costs, not transistor counts or timing measurements.",
        ],
        "binary_slice": {
            "input_state_count": len(BIN_DOMAIN) ** 3,
            "functions": binary_minimized,
            "total_slice_cost": binary_slice_cost,
            "cost_per_input_state": binary_slice_cost / (len(BIN_DOMAIN) ** 3),
            "cost_per_input_bit": binary_slice_cost / math.log2(len(BIN_DOMAIN) ** 3),
        },
        "ternary_slice": {
            "input_state_count": len(TER_DOMAIN) ** 3,
            "functions": ternary_minimized,
            "total_slice_cost": ternary_slice_cost,
            "cost_per_input_state": ternary_slice_cost / (len(TER_DOMAIN) ** 3),
            "cost_per_input_bit": ternary_slice_cost / math.log2(len(TER_DOMAIN) ** 3),
        },
        "equal_range_comparison": {
            "balanced_ternary_trits": width_trits,
            "balanced_ternary_range": [-max_abs, max_abs],
            "minimum_signed_binary_bits": binary_bits,
            "binary_range": [-(2 ** (binary_bits - 1)), (2 ** (binary_bits - 1)) - 1],
            "estimated_ternary_ripple_cost": ternary_total_cost,
            "estimated_binary_ripple_cost": binary_total_cost,
            "ternary_over_binary_cost_ratio": ternary_total_cost / binary_total_cost,
        },
        "summary": {
            "note": (
                "This experiment compares minimized combinational covers under one explicit cost model. "
                "Different assumptions about primitive gates, available complements, native ternary devices, "
                "or transistor-level implementations could change the ranking."
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ternary vs binary adder gate-cost proxies.")
    parser.add_argument(
        "--width-trits",
        type=int,
        default=4,
        help="Balanced-ternary ripple width for equal-range comparison.",
    )
    parser.add_argument(
        "--out",
        default="results/arto_ternary_gate_cost_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    result = run_experiment(width_trits=args.width_trits)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
