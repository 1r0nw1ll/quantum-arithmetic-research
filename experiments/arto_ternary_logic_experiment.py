#!/usr/bin/env python3
"""
Software-side evaluation of concrete ternary-logic claims attributed to Arto Heino.

This experiment does not attempt to validate hardware performance or physics claims.
It only checks exact discrete-logic facts that are computable from first principles:

1. The number of 2-input ternary gates is 3^(3^2) = 19,683.
2. For equal wire count n, ternary state-space exceeds binary by (3/2)^n.
3. For equal state capacity, ternary needs fewer symbols than binary.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path


def count_two_input_ternary_gates() -> int:
    """Brute-force count all 2-input ternary truth tables."""
    count = 0
    for _table in product(range(3), repeat=9):
        count += 1
    return count


def equivalent_trits_for_bits(bits: int) -> int:
    """Smallest ternary symbol count with at least the same state capacity."""
    states = 2**bits
    trits = 0
    capacity = 1
    while capacity < states:
        trits += 1
        capacity *= 3
    return trits


def equivalent_bits_for_trits(trits: int) -> int:
    """Smallest binary symbol count with at least the same state capacity."""
    states = 3**trits
    bits = 0
    capacity = 1
    while capacity < states:
        bits += 1
        capacity *= 2
    return bits


def build_state_growth_table(max_wires: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wires in range(1, max_wires + 1):
        binary_states = 2**wires
        ternary_states = 3**wires
        rows.append(
            {
                "wires": wires,
                "binary_states": binary_states,
                "ternary_states": ternary_states,
                "ternary_over_binary_ratio": ternary_states / binary_states,
            }
        )
    return rows


def build_capacity_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bits in (4, 8, 16, 32, 64):
        trits = equivalent_trits_for_bits(bits)
        rows.append(
            {
                "binary_bits": bits,
                "binary_states": 2**bits,
                "equivalent_ternary_trits": trits,
                "ternary_states_at_equivalent_trits": 3**trits,
            }
        )
    for trits in (3, 6, 9, 12):
        bits = equivalent_bits_for_trits(trits)
        rows.append(
            {
                "ternary_trits": trits,
                "ternary_states": 3**trits,
                "equivalent_binary_bits": bits,
                "binary_states_at_equivalent_bits": 2**bits,
            }
        )
    return rows


def run_experiment(max_wires: int) -> dict[str, object]:
    brute_force_gate_count = count_two_input_ternary_gates()
    formula_gate_count = 3 ** (3**2)
    gate_count_ok = brute_force_gate_count == formula_gate_count == 19683

    state_growth = build_state_growth_table(max_wires=max_wires)
    capacity_equivalence = build_capacity_table()

    max_ratio_row = max(state_growth, key=lambda row: float(row["ternary_over_binary_ratio"]))
    bits_64_row = next(
        row for row in capacity_equivalence if row.get("binary_bits") == 64
    )

    verdict = "PASS" if gate_count_ok and float(max_ratio_row["ternary_over_binary_ratio"]) > 1.0 else "FAIL"

    return {
        "experiment_id": "arto_ternary_logic_experiment_2026-03-29",
        "hypothesis": (
            "Arto Heino's discrete ternary-logic claims contain a solid engineering core: "
            "the 2-input ternary gate count should equal 19,683 exactly, and ternary should "
            "show a real state-space density advantage over binary for equal wire count."
        ),
        "success_criteria": (
            "PASS if brute-force enumeration matches 19,683 and if ternary state-space "
            "strictly exceeds binary state-space for equal wire count."
        ),
        "result": verdict,
        "gate_count_check": {
            "formula_gate_count": formula_gate_count,
            "brute_force_gate_count": brute_force_gate_count,
            "matches_expected_19683": gate_count_ok,
        },
        "state_growth_equal_wires": state_growth,
        "capacity_equivalence_examples": capacity_equivalence,
        "summary": {
            "max_equal_wire_advantage_row": max_ratio_row,
            "equivalent_trits_for_64_bits": bits_64_row["equivalent_ternary_trits"],
            "note": (
                "The ternary advantage shown here is purely combinatorial state density. "
                "It does not by itself prove better real-world hardware, speed, power, or noise behavior."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate exact ternary-logic counting claims.")
    parser.add_argument(
        "--max-wires",
        type=int,
        default=12,
        help="Highest equal-wire comparison row to emit.",
    )
    parser.add_argument(
        "--out",
        default="results/arto_ternary_logic_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    result = run_experiment(max_wires=args.max_wires)
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
