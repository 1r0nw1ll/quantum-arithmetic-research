#!/usr/bin/env python3
"""Map the stable Sixto two-branch timing packet onto QA / contra-cyclic variables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
TWO_BRANCH_PATH = OUT_DIR / "sixto_graph_two_branch_law_packet.json"
CORRESPONDENCE_PATH = OUT_DIR / "arto_contra_cyclic_qa_correspondence.json"
MOMENT_ARM_PATH = OUT_DIR / "arto_contra_cyclic_moment_arm_packet.json"
ARCH_PACKET_PATH = OUT_DIR / "archimedean_twin_circle_qa_candidate.json"
WAVE_CHECK_PATH = OUT_DIR / "sixtwave2_notch_check.json"
OUTPUT_PATH = OUT_DIR / "sixto_graph_qa_variable_mapping.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload() -> dict[str, Any]:
    two_branch = read_json(TWO_BRANCH_PATH)
    correspondence = read_json(CORRESPONDENCE_PATH)
    moment_arm = read_json(MOMENT_ARM_PATH)
    arch_packet = read_json(ARCH_PACKET_PATH)
    wave_check = read_json(WAVE_CHECK_PATH)

    qa_relations = [item["equation"] for item in correspondence["qa_core"]["relations"]]
    packet_chain = moment_arm["packet_chain"]["relations"]
    arch_definitions = [item["equation"] for item in arch_packet["corpus_native_packet"]["definitions"]]

    negative_family = two_branch["shared_law_packet"]["negative_branch_family"]
    positive_family = two_branch["shared_law_packet"]["positive_branch_family"]
    cyan_lane = two_branch["cyan_anomaly_lane"]

    curve_roles = []
    for role in two_branch["curve_roles"]:
        curve_roles.append(
            {
                "curve_id": role["curve_id"],
                "phase_handoff_x": role["anchors"]["crossover_x"],
                "negative_branch_family": role["negative_branch_family"],
                "positive_branch_family": role["positive_branch_family"],
                "observed_amplitude_pair": {
                    "dip_y": role["anchors"]["dip_y"],
                    "peak_y": role["anchors"]["peak_y"],
                },
            }
        )

    return {
        "artifact_id": "sixto_graph_qa_variable_mapping",
        "source_packets": {
            "two_branch_packet": "pythagoras_quantum_world_rt/sixto_graph_two_branch_law_packet.json",
            "contra_cyclic_correspondence": "pythagoras_quantum_world_rt/arto_contra_cyclic_qa_correspondence.json",
            "moment_arm_packet": "pythagoras_quantum_world_rt/arto_contra_cyclic_moment_arm_packet.json",
            "archimedean_packet": "pythagoras_quantum_world_rt/archimedean_twin_circle_qa_candidate.json",
            "sixtwave2_check": "pythagoras_quantum_world_rt/sixtwave2_notch_check.json",
        },
        "qa_core": {
            "tuple": correspondence["qa_core"]["tuple"],
            "relations": qa_relations,
            "stage_drive_law": "v = F/C",
            "scheduler_rule": "Advance stage state on contra-cyclic residue windows; the timing graph is read as scheduler output, not as a direct geometry axis.",
        },
        "geometry_packet_layer": {
            "stable_definitions": arch_definitions,
            "moment_arm_packet_chain": packet_chain,
            "operational_reading": [
                "Use W/P for circle-branch carrier claims.",
                "Use D/X/J/K for ellipse / offset packet claims.",
                "Do not identify the timing-graph ordinate directly with D, X, J, K, W, or P.",
            ],
        },
        "timing_graph_mapping": {
            "x_axis": {
                "meaning": "Stage-local scheduler phase over one revolution / active window.",
                "left_branch_coordinate": "t_left = x / crossover_x",
                "right_branch_coordinate": "t_right = (x - crossover_x) / (1212 - crossover_x)",
            },
            "y_axis": {
                "meaning": "Signed normalized stage-response carrier on top of the QA drive law.",
                "shared_carrier_form": "response_stage(t) = A_stage * U_branch(t)",
                "qa_bridge_form": "output_stage(t) ∝ (F/C) * U_branch(t), with packet geometry entering through the chosen moment-arm branch rather than replacing U_branch(t).",
                "non_identity_rule": "The timing-graph ordinate is a scheduler/response layer fed by QA quantities; it is not itself a direct D/X/J/K/W/P coordinate.",
            },
            "shared_branch_families": {
                "negative_branch": {
                    "family_id": negative_family["family_id"],
                    "member_curve_ids": negative_family["member_curve_ids"],
                    "medoid_curve_id": negative_family["medoid_curve_id"],
                    "template_profile": negative_family["template_profile"],
                    "qa_reading": "Shared loading / ingress branch before phase handoff.",
                },
                "positive_branch": {
                    "family_id": positive_family["family_id"],
                    "member_curve_ids": positive_family["member_curve_ids"],
                    "medoid_curve_id": positive_family["medoid_curve_id"],
                    "template_profile": positive_family["template_profile"],
                    "qa_reading": "Shared release / return branch after phase handoff.",
                },
            },
            "curve_roles": curve_roles,
        },
        "cyan_anomaly_lane": {
            "status": "quarantined_witness_local_modulation",
            "base_family": cyan_lane["base_family"],
            "support_window": cyan_lane["support_window"],
            "extrema": cyan_lane["extrema"],
            "qa_reading": "Treat as a localized response modulation on top of the shared positive branch, not as a new universal packet law.",
            "cross_witness_check": {
                "sixtwave2_reproduces_notch": wave_check["verdict"]["localized_cyan_notch_present"],
                "sixtwave2_verdict": wave_check["verdict"]["honest_summary"],
            },
        },
        "stable_bridge_rules": [
            {
                "id": "sixto_qa_map_01",
                "rule": "Read the timing graph as scheduler output driven by v = F/C, not as a standalone geometric primitive.",
            },
            {
                "id": "sixto_qa_map_02",
                "rule": "Use D/X/J/K/W/P only in the geometry packet beneath the graph; keep the graph ordinates in branch-response form U_branch(t).",
            },
            {
                "id": "sixto_qa_map_03",
                "rule": "Promote the shared negative and positive branches as stable source-native timing families.",
            },
            {
                "id": "sixto_qa_map_04",
                "rule": "Keep the cyan positive-branch notch quarantined as witness-local modulation unless another witness reproduces it.",
            },
        ],
        "verdict": {
            "qa_mapping_ready": True,
            "honest_summary": "The stable Sixto timing packet now maps cleanly onto the QA / contra-cyclic layer: branch phase on the time axis, v = F/C as the drive carrier, D/X/J/K/W/P as the underlying geometry packet, and the cyan notch retained only as a local modulation rather than promoted law.",
        },
    }


def self_test() -> int:
    payload = build_payload()
    ok = True
    ok = ok and payload["qa_core"]["stage_drive_law"] == "v = F/C"
    ok = ok and payload["timing_graph_mapping"]["shared_branch_families"]["negative_branch"]["member_curve_ids"] == [
        "curve_graph_light_green",
        "curve_graph_green",
        "curve_graph_cyan",
        "curve_graph_blue",
    ]
    ok = ok and payload["timing_graph_mapping"]["shared_branch_families"]["positive_branch"]["member_curve_ids"] == [
        "curve_graph_light_green",
        "curve_graph_green",
        "curve_graph_blue",
    ]
    ok = ok and payload["cyan_anomaly_lane"]["cross_witness_check"]["sixtwave2_reproduces_notch"] is False
    ok = ok and payload["verdict"]["qa_mapping_ready"] is True
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    write_json(OUTPUT_PATH, payload)
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/sixto_graph_qa_variable_mapping.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
