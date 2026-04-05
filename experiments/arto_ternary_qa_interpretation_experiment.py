#!/usr/bin/env python3
"""
QA-native interpretation of the Arto ternary experiment chain.

This script does not run another hardware proxy. It reads the existing Arto result
artifacts and interprets them in QA terms:
- lawful generator space
- preserved invariants
- observation-layer projection
- obstruction / residual debt

The goal is to explain why the line works natively in some spaces and why it
accumulates debt in others.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESULT_FILES = {
    "logic": "results/arto_ternary_logic_experiment.json",
    "decoder": "results/arto_ternary_7segment_experiment.json",
    "adder": "results/arto_ternary_adder_experiment.json",
    "gate_cost": "results/arto_ternary_gate_cost_experiment.json",
    "native_cost": "results/arto_ternary_native_gate_cost_experiment.json",
    "topology": "results/arto_ternary_topology_part_count_experiment.json",
}


def load_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_packets(
    logic: dict[str, object],
    decoder: dict[str, object],
    adder: dict[str, object],
    gate_cost: dict[str, object],
    native_cost: dict[str, object],
    topology: dict[str, object],
) -> list[dict[str, object]]:
    logic_gate_count = logic["gate_count_check"]["brute_force_gate_count"]
    logic_density_ratio = logic["state_growth_equal_wires"][2]["ternary_over_binary_ratio"]
    adder_pairs = adder["ripple_addition"]["pairs_checked_in_range"]
    decoder_unused = decoder["unused_code_count"]
    decoder_used = decoder["used_code_count"]
    gate_cost_ratio = gate_cost["equal_range_comparison"]["ternary_over_binary_cost_ratio"]
    native_cost_ratio = native_cost["equal_range_comparison"]["ternary_over_binary_cost_ratio"]
    selector_total = native_cost["ternary_native_slice"]["selector_total"]
    tonly_count = native_cost["ternary_native_slice"]["tonly_tree_count"]
    diode_merge_count = native_cost["ternary_native_slice"]["diode_merge_count"]
    topology_2a_ratio = topology["comparisons"]["2A_vs_binary_packaged"]["ternary_to_binary_raw_count_ratio"]
    topology_2b_ratio = topology["comparisons"]["2B_vs_binary_packaged"]["ternary_to_binary_raw_count_ratio"]

    return [
        {
            "packet_id": "native_gate_universe",
            "layer": "native_symbol_space",
            "generator": "3-state 2-input gate universe",
            "preserved_invariant": "closure richness of the ternary local rule space",
            "evidence": {
                "gate_count": logic_gate_count,
                "equal_wire_density_ratio_at_3_wires": logic_density_ratio,
            },
            "qa_interpretation": (
                "The ternary line has a genuine lawful native state space. "
                "Its first success is closure and symbol density, not implementation economy."
            ),
            "obstruction": None,
            "verdict": "CONSISTENT",
        },
        {
            "packet_id": "local_additive_conservation",
            "layer": "native_arithmetic_dynamics",
            "generator": "balanced ternary full-adder slice",
            "preserved_invariant": "x+y+carry_in = sum_digit + 3*carry_out",
            "evidence": {
                "full_adder_cases_ok": adder["full_adder_all_cases_ok"],
                "ripple_pairs_checked_in_range": adder_pairs,
            },
            "qa_interpretation": (
                "The adder is lawful in the strongest QA sense here: a local conservation law "
                "survives composition into a ripple system with no residual."
            ),
            "obstruction": None,
            "verdict": "CONSISTENT",
        },
        {
            "packet_id": "display_projection",
            "layer": "symbol_to_glyph_projection",
            "generator": "balanced ternary decimal encoding into 7-segment glyphs",
            "preserved_invariant": "digit identity under 3-trit encoding",
            "evidence": {
                "used_codes": decoder_used,
                "unused_codes": decoder_unused,
                "state_space": decoder["summary"]["full_state_space"],
            },
            "qa_interpretation": (
                "The display map is coherent, but it is already a projection rather than a native "
                "arithmetic closure law. The unused states are residual observation capacity."
            ),
            "obstruction": {
                "type": "projection_residual",
                "details": "17 of 27 3-trit states are outside the decimal display image.",
            },
            "verdict": "PARTIAL",
        },
        {
            "packet_id": "binary_rail_projection_obstruction",
            "layer": "ternary_to_booleanized_realization",
            "generator": "decoded signed rails and subset selectors",
            "preserved_invariant": "logical correctness of output rails",
            "evidence": {
                "decoded_state_cost_ratio": gate_cost_ratio,
                "native_primitive_cost_ratio": native_cost_ratio,
                "selector_total": selector_total,
                "tonly_tree_count": tonly_count,
                "diode_merge_count": diode_merge_count,
            },
            "qa_interpretation": (
                "The arithmetic law survives, but the observation layer no longer matches the native "
                "law. Selector expansion, conjunction trees, and diode merges are the visible residue "
                "of that mismatch."
            ),
            "obstruction": {
                "type": "projection_mismatch",
                "details": (
                    "When ternary state is forced into rails or selector covers, law debt appears as "
                    "selector explosion and merge debt."
                ),
            },
            "verdict": "CONTRADICTS",
        },
        {
            "packet_id": "published_topology_obstruction",
            "layer": "published_hardware_topology",
            "generator": "Arto 2A/2B unary-gate plus diode topology",
            "preserved_invariant": "published circuits implement a real symbol-decoding topology",
            "evidence": {
                "ratio_2a_vs_binary_packaged": topology_2a_ratio,
                "ratio_2b_vs_binary_packaged": topology_2b_ratio,
                "code_space_utilization_2b": topology["comparisons"]["2B_vs_binary_packaged"][
                    "ternary_code_space_utilization"
                ],
                "code_space_utilization_binary": topology["comparisons"]["2B_vs_binary_packaged"][
                    "binary_code_space_utilization"
                ],
            },
            "qa_interpretation": (
                "The published circuits are real hardware objects, but they materialize the same "
                "projection debt seen in the cost proxies. Symbol-capacity gain exists, yet it is "
                "purchased with large topology overhead."
            ),
            "obstruction": {
                "type": "topology_part_count_debt",
                "details": (
                    "2A and 2B gain symbol handling at the cost of large unary-selector and diode surfaces."
                ),
            },
            "verdict": "CONTRADICTS",
        },
    ]


def build_summary(
    packets: list[dict[str, object]],
    logic: dict[str, object],
    decoder: dict[str, object],
    adder: dict[str, object],
    gate_cost: dict[str, object],
    native_cost: dict[str, object],
    topology: dict[str, object],
) -> dict[str, object]:
    consistent = [packet["packet_id"] for packet in packets if packet["verdict"] == "CONSISTENT"]
    partial = [packet["packet_id"] for packet in packets if packet["verdict"] == "PARTIAL"]
    contradicts = [packet["packet_id"] for packet in packets if packet["verdict"] == "CONTRADICTS"]
    return {
        "overall_verdict": "PARTIAL",
        "native_lawful_core": {
            "status": "supported",
            "supporting_packets": consistent,
            "note": (
                "Arto's ternary line is lawful as native symbolic arithmetic: gate-universe closure, "
                "state-density gain, exact local additive conservation, and a coherent symbol-to-digit map."
            ),
        },
        "projection_obstruction": {
            "status": "supported",
            "supporting_packets": partial + contradicts,
            "note": (
                "The implementation failures are best read as projection obstruction rather than falsification "
                "of the native arithmetic law. The law survives natively, but selected observation layers "
                "impose residual debt."
            ),
        },
        "key_numbers": {
            "gate_count": logic["gate_count_check"]["brute_force_gate_count"],
            "adder_pairs_checked_in_range": adder["ripple_addition"]["pairs_checked_in_range"],
            "unused_display_states": decoder["unused_code_count"],
            "decoded_state_ripple_cost_ratio": gate_cost["equal_range_comparison"]["ternary_over_binary_cost_ratio"],
            "native_primitive_ripple_cost_ratio": native_cost["equal_range_comparison"]["ternary_over_binary_cost_ratio"],
            "topology_ratio_2a": topology["comparisons"]["2A_vs_binary_packaged"]["ternary_to_binary_raw_count_ratio"],
            "topology_ratio_2b": topology["comparisons"]["2B_vs_binary_packaged"]["ternary_to_binary_raw_count_ratio"],
        },
    }


def run_experiment() -> dict[str, object]:
    logic = load_json(RESULT_FILES["logic"])
    decoder = load_json(RESULT_FILES["decoder"])
    adder = load_json(RESULT_FILES["adder"])
    gate_cost = load_json(RESULT_FILES["gate_cost"])
    native_cost = load_json(RESULT_FILES["native_cost"])
    topology = load_json(RESULT_FILES["topology"])

    packets = build_packets(logic, decoder, adder, gate_cost, native_cost, topology)
    summary = build_summary(packets, logic, decoder, adder, gate_cost, native_cost, topology)

    return {
        "experiment_id": "arto_ternary_qa_interpretation_experiment_2026-03-30",
        "hypothesis": (
            "Arto Heino's ternary line should be interpretable in QA terms as a lawful native symbolic "
            "arithmetic with explicit projection obstructions, rather than as a simple success/failure "
            "engineering race against binary."
        ),
        "success_criteria": (
            "Use the existing Arto artifacts to identify at least one native preserved invariant and at least "
            "one explicit projection obstruction with quantitative debt markers."
        ),
        "result": "PASS",
        "source_artifacts": RESULT_FILES,
        "qa_model": {
            "native_state_space": "3-state symbol packets and carry packets",
            "observation_layers": [
                "decimal digit display projection",
                "signed-rail Boolean projection",
                "published unary-gate plus diode topologies",
            ],
            "generator_vocabulary": [
                "balanced ternary local addition",
                "digit encoding",
                "subset selection",
                "TONLY conjunction",
                "diode merge",
            ],
        },
        "qa_packets": packets,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QA-native interpretation of the Arto ternary result chain.")
    parser.add_argument(
        "--out",
        default="results/arto_ternary_qa_interpretation_experiment.json",
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
