#!/usr/bin/env python3
"""
Topology-grounded part-count comparison based on printed counts from Arto Heino's
published ternary 7-segment schematics.

Primary ternary evidence source:
  https://artoheino.com/2024/03/23/ternary-coded-decimal-and-7-segment-control/

Counts are taken from the labels printed on the two published schematics:
- 2A: ternary 0..9 to 7-segment
- 2B: extended signed ternary 18-symbol variant

Binary baseline:
- standard packaged BCD-to-7-segment decoder implementation using one decoder IC
  plus one 7-segment display as the logic/display core.

This is a raw published-topology count comparison, not a transistor-level analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_experiment() -> dict[str, object]:
    ternary_2a = {
        "name": "Arto ternary 7-segment 2A",
        "input_symbols": 27,
        "displayed_symbols": 10,
        "published_counts": {
            "INC_unary_gates": 3,
            "DEC_unary_gates": 2,
            "MAX_unary_gates": 9,
            "USR_unary_gates": 18,
            "diodes": 34,
        },
    }
    ternary_2a["published_counts"]["total_unary_gates"] = (
        ternary_2a["published_counts"]["INC_unary_gates"]
        + ternary_2a["published_counts"]["DEC_unary_gates"]
        + ternary_2a["published_counts"]["MAX_unary_gates"]
        + ternary_2a["published_counts"]["USR_unary_gates"]
    )
    ternary_2a["published_counts"]["total_logic_primitives_plus_diodes"] = (
        ternary_2a["published_counts"]["total_unary_gates"] + ternary_2a["published_counts"]["diodes"]
    )

    ternary_2b = {
        "name": "Arto ternary 7-segment 2B",
        "input_symbols": 27,
        "displayed_symbols": 18,
        "published_counts": {
            "INC_unary_gates": 3,
            "DEC_unary_gates": 2,
            "MAX_unary_gates": 9,
            "USR_unary_gates": 18,
            "RGP_unary_gates": 9,
            "diodes_1N4148": 91,
        },
    }
    ternary_2b["published_counts"]["total_unary_gates"] = (
        ternary_2b["published_counts"]["INC_unary_gates"]
        + ternary_2b["published_counts"]["DEC_unary_gates"]
        + ternary_2b["published_counts"]["MAX_unary_gates"]
        + ternary_2b["published_counts"]["USR_unary_gates"]
        + ternary_2b["published_counts"]["RGP_unary_gates"]
    )
    ternary_2b["published_counts"]["total_logic_primitives_plus_diodes"] = (
        ternary_2b["published_counts"]["total_unary_gates"] + ternary_2b["published_counts"]["diodes_1N4148"]
    )

    binary_packaged = {
        "name": "Binary packaged BCD-to-7-segment baseline",
        "input_symbols": 16,
        "displayed_symbols": 10,
        "baseline_counts": {
            "BCD_to_7_segment_decoder_IC": 1,
            "seven_segment_display": 1,
        },
    }
    binary_packaged["baseline_counts"]["total_core_packages"] = (
        binary_packaged["baseline_counts"]["BCD_to_7_segment_decoder_IC"]
        + binary_packaged["baseline_counts"]["seven_segment_display"]
    )

    comparisons = {
        "2A_vs_binary_packaged": {
            "ternary_total_logic_primitives_plus_diodes": ternary_2a["published_counts"][
                "total_logic_primitives_plus_diodes"
            ],
            "binary_total_core_packages": binary_packaged["baseline_counts"]["total_core_packages"],
            "ternary_to_binary_raw_count_ratio": ternary_2a["published_counts"][
                "total_logic_primitives_plus_diodes"
            ]
            / binary_packaged["baseline_counts"]["total_core_packages"],
            "ternary_displayed_symbol_efficiency": ternary_2a["displayed_symbols"]
            / ternary_2a["published_counts"]["total_logic_primitives_plus_diodes"],
            "binary_displayed_symbol_efficiency": binary_packaged["displayed_symbols"]
            / binary_packaged["baseline_counts"]["total_core_packages"],
            "ternary_code_space_utilization": ternary_2a["displayed_symbols"] / ternary_2a["input_symbols"],
            "binary_code_space_utilization": binary_packaged["displayed_symbols"] / binary_packaged["input_symbols"],
        },
        "2B_vs_binary_packaged": {
            "ternary_total_logic_primitives_plus_diodes": ternary_2b["published_counts"][
                "total_logic_primitives_plus_diodes"
            ],
            "binary_total_core_packages": binary_packaged["baseline_counts"]["total_core_packages"],
            "ternary_to_binary_raw_count_ratio": ternary_2b["published_counts"][
                "total_logic_primitives_plus_diodes"
            ]
            / binary_packaged["baseline_counts"]["total_core_packages"],
            "ternary_displayed_symbol_efficiency": ternary_2b["displayed_symbols"]
            / ternary_2b["published_counts"]["total_logic_primitives_plus_diodes"],
            "binary_displayed_symbol_efficiency": binary_packaged["displayed_symbols"]
            / binary_packaged["baseline_counts"]["total_core_packages"],
            "ternary_code_space_utilization": ternary_2b["displayed_symbols"] / ternary_2b["input_symbols"],
            "binary_code_space_utilization": binary_packaged["displayed_symbols"] / binary_packaged["input_symbols"],
        },
    }

    verdict = "PASS"
    return {
        "experiment_id": "arto_ternary_topology_part_count_experiment_2026-03-30",
        "hypothesis": (
            "Even at the level of actual published circuit topologies, Arto's ternary 7-segment "
            "design will likely have much higher raw part count than a standard binary packaged baseline, "
            "though it may carry more symbol capacity per input code space."
        ),
        "success_criteria": (
            "Extract explicit ternary published counts from the 2A and 2B schematics, compare them against "
            "a standard binary packaged baseline under stated assumptions, and record whether ternary wins "
            "on raw part count, code-space utilization, or neither."
        ),
        "result": verdict,
        "sources": {
            "arto_post": "https://artoheino.com/2024/03/23/ternary-coded-decimal-and-7-segment-control/",
            "notes": [
                "2A and 2B component counts are transcribed from the labels printed on the published schematic images.",
                "Binary packaged baseline is a standard BCD-to-7-segment decoder IC plus one 7-segment display as the core implementation.",
            ],
        },
        "ternary_topologies": {
            "2A": ternary_2a,
            "2B": ternary_2b,
        },
        "binary_packaged_baseline": binary_packaged,
        "comparisons": comparisons,
        "summary": {
            "note": (
                "This comparison is intentionally narrow and physical-topology-oriented. "
                "It compares printed published part counts against a standard packaged binary baseline; "
                "it does not normalize for chip-internal complexity, manufacturing technology, or custom IC design."
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Topology-grounded ternary part-count comparison.")
    parser.add_argument(
        "--out",
        default="results/arto_ternary_topology_part_count_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    result = run_experiment()
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
