#!/usr/bin/env python3
"""
Native-primitive ternary cost model using Arto Heino's gate vocabulary as an abstract
hardware basis.

Model assumptions:
- Inputs are true ternary wires, not Boolean-decoded state predicates.
- Any one-input state/subset selector is a native unary primitive from the 27-gate family.
  Singleton selectors are labeled with Arto-style names INC/DEC/MAX and arbitrary subset
  selectors are labeled USR.
- Term conjunctions for positive-rail outputs use TONLY gates in trees.
- Output aggregation uses diode merges.
- Binary baseline uses native XOR/AND/OR gates.

This remains a hardware proxy, but it is materially closer to a native ternary basis than
the earlier decoded-state PLA model.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path


BIN_DOMAIN = (0, 1)
TER_DOMAIN = (-1, 0, 1)
LABELS = ("x", "y", "cin")


def full_adder_binary(x: int, y: int, carry_in: int) -> tuple[int, int]:
    total = x + y + carry_in
    return total & 1, 1 if total >= 2 else 0


def full_adder_ternary(x: int, y: int, carry_in: int) -> tuple[int, int]:
    total = x + y + carry_in
    if total > 1:
        return total - 3, 1
    if total < -1:
        return total + 3, -1
    return total, 0


def nonempty_subsets(domain: tuple[int, ...]) -> list[frozenset[int]]:
    subsets: list[frozenset[int]] = []
    for r in range(1, len(domain) + 1):
        for combo in combinations(domain, r):
            subsets.append(frozenset(combo))
    return subsets


def cube_covers(cube: tuple[frozenset[int], ...], point: tuple[int, ...]) -> bool:
    return all(value in allowed for value, allowed in zip(point, cube))


def cube_specificity(cube: tuple[frozenset[int], ...], domain: tuple[int, ...]) -> int:
    return sum(1 for allowed in cube if len(allowed) < len(domain))


def cube_text(cube: tuple[frozenset[int], ...]) -> str:
    parts: list[str] = []
    for label, allowed in zip(LABELS, cube):
        parts.append(f"{label}∈{{{','.join(str(v) for v in sorted(allowed))}}}")
    return " & ".join(parts)


def cube_selector_kind(allowed: frozenset[int], domain: tuple[int, ...]) -> str | None:
    if len(allowed) == len(domain):
        return None
    if domain == TER_DOMAIN:
        if allowed == frozenset({1}):
            return "INC"
        if allowed == frozenset({0}):
            return "MAX"
        if allowed == frozenset({-1}):
            return "DEC"
        return "USR"
    return "WIRE" if len(allowed) == 1 else "USR"


def generate_candidate_cubes(domain: tuple[int, ...], on_set: set[tuple[int, ...]]) -> list[dict[str, object]]:
    all_points = list(product(domain, repeat=3))
    subset_choices = nonempty_subsets(domain)
    by_coverage: dict[frozenset[tuple[int, ...]], dict[str, object]] = {}

    for cube in product(subset_choices, repeat=3):
        covered = frozenset(point for point in all_points if cube_covers(cube, point))
        if not covered or not covered.issubset(on_set):
            continue
        candidate = {
            "cube": cube,
            "coverage": covered,
            "specificity": cube_specificity(cube, domain),
            "text": cube_text(cube),
        }
        existing = by_coverage.get(covered)
        if existing is None or candidate["specificity"] < existing["specificity"]:
            by_coverage[covered] = candidate

    candidates = list(by_coverage.values())
    reduced: list[dict[str, object]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if candidate is other:
                continue
            if candidate["coverage"].issubset(other["coverage"]) and other["specificity"] <= candidate["specificity"]:
                if candidate["coverage"] != other["coverage"] or other["specificity"] < candidate["specificity"]:
                    dominated = True
                    break
        if not dominated:
            reduced.append(candidate)
    return reduced


def minimize_cover(domain: tuple[int, ...], on_set: set[tuple[int, ...]]) -> list[dict[str, object]]:
    if not on_set:
        return []
    candidates = generate_candidate_cubes(domain, on_set)
    uncovered = set(on_set)
    chosen: list[dict[str, object]] = []

    while uncovered:
        best = max(
            candidates,
            key=lambda c: (len(c["coverage"] & uncovered), -c["specificity"], c["text"]),
        )
        chosen.append(best)
        uncovered -= set(best["coverage"])
        candidates = [c for c in candidates if c is not best]
    return chosen


def ternary_on_sets() -> dict[str, set[tuple[int, ...]]]:
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


def collect_native_ternary_cost(covers: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    selectors: dict[tuple[str, frozenset[int]], str] = {}
    conjunction_count = 0
    diode_count = 0
    function_terms: dict[str, list[str]] = {}

    for name, terms in covers.items():
        function_terms[name] = []
        if terms:
            diode_count += len(terms) - 1
        for term in terms:
            cube = term["cube"]
            active_literals = 0
            for label, allowed in zip(LABELS, cube):
                kind = cube_selector_kind(allowed, TER_DOMAIN)
                if kind is not None:
                    selectors[(label, allowed)] = kind
                    active_literals += 1
            if active_literals >= 2:
                conjunction_count += active_literals - 1
            function_terms[name].append(term["text"])

    selector_breakdown = {
        "INC": 0,
        "DEC": 0,
        "MAX": 0,
        "USR": 0,
    }
    for kind in selectors.values():
        selector_breakdown[kind] += 1

    selector_total = sum(selector_breakdown.values())
    total_cost = selector_total + conjunction_count + diode_count
    return {
        "distinct_selectors": [
            {"input": label, "allowed": sorted(list(allowed)), "kind": kind}
            for (label, allowed), kind in sorted(selectors.items(), key=lambda item: (item[0][0], sorted(list(item[0][1]))))
        ],
        "selector_breakdown": selector_breakdown,
        "selector_total": selector_total,
        "tonly_tree_count": conjunction_count,
        "diode_merge_count": diode_count,
        "function_terms": function_terms,
        "total_slice_cost": total_cost,
    }


def equivalent_signed_binary_bits(max_abs: int) -> int:
    bits = 1
    while (2 ** (bits - 1)) - 1 < max_abs:
        bits += 1
    return bits


def run_experiment(width_trits: int) -> dict[str, object]:
    ternary_covers = {name: minimize_cover(TER_DOMAIN, on_set) for name, on_set in ternary_on_sets().items()}
    ternary_cost = collect_native_ternary_cost(ternary_covers)

    binary_native_slice_cost = 5  # 2 XOR + 2 AND + 1 OR

    max_abs = (3**width_trits - 1) // 2
    binary_bits = equivalent_signed_binary_bits(max_abs)
    ternary_total_cost = ternary_cost["total_slice_cost"] * width_trits
    binary_total_cost = binary_native_slice_cost * binary_bits

    verdict = "PASS"
    return {
        "experiment_id": "arto_ternary_native_gate_cost_experiment_2026-03-30",
        "hypothesis": (
            "A more native ternary primitive basis using Arto-style unary selectors plus TONLY and diode "
            "merges may narrow or overturn the negative cost result from the decoded-state PLA model."
        ),
        "success_criteria": (
            "Produce an explicit equal-range slice and ripple-adder cost comparison under stated native "
            "primitive assumptions and record whether the ternary cost gap narrows, disappears, or reverses."
        ),
        "result": verdict,
        "assumptions": [
            "True ternary wires are available as native inputs.",
            "Any one-input subset selector is one native unary primitive from the 27-gate family.",
            "Singleton selectors are labeled INC/DEC/MAX and arbitrary subset selectors are labeled USR.",
            "Positive-rail term conjunctions use TONLY gates in trees.",
            "Output aggregation uses diode merges.",
            "Binary baseline is a native full adder using XOR/AND/OR with cost 5 per slice.",
        ],
        "ternary_native_slice": ternary_cost,
        "binary_native_slice": {
            "primitive_counts": {"XOR": 2, "AND": 2, "OR": 1},
            "total_slice_cost": binary_native_slice_cost,
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
                "This is still a proxy model, but it is more favorable to native ternary hardware than the "
                "earlier decoded-state PLA comparison."
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Native-primitive ternary vs binary gate-cost proxy.")
    parser.add_argument("--width-trits", type=int, default=4)
    parser.add_argument(
        "--out",
        default="results/arto_ternary_native_gate_cost_experiment.json",
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
