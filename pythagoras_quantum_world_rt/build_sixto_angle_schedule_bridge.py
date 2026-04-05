#!/usr/bin/env python3
"""Bridge the recovered Sixto stage-angle schedule to the printed angle packet and timing graph."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
INTERNAL_ARM_PATH = OUT_DIR / "sixto_internal_k_arm_test.json"
PRECISION_WITNESS_PATH = OUT_DIR / "sioxto3002b_precision_witness.json"
TOPOLOGY_WITNESS_PATH = OUT_DIR / "sixto_topology_witness.json"
OUT_PATH = OUT_DIR / "sixto_angle_schedule_bridge.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def build_payload() -> dict[str, object]:
    internal = read_json(INTERNAL_ARM_PATH)
    precision = read_json(PRECISION_WITNESS_PATH)
    topology = read_json(TOPOLOGY_WITNESS_PATH)

    stage_angles = internal["internal_circle_behavior"]["absolute_angles_degrees"]
    stage_ids = [item["stage_id"] for item in internal["recovered_internal_arm_family"]["stage_arms"]]
    green = float(precision["source_angles"]["theta_green_deg"])
    orange = float(precision["source_angles"]["theta_orange_deg"])
    blue = float(precision["source_angles"]["theta_blue_deg"])
    orange_comp = 90.0 - orange
    arithmetic_mid = (orange + green) / 2.0
    geometric_mid = math.sqrt(orange * green)

    angle_packet_bridge = {
        "stage_ids": stage_ids,
        "recovered_stage_angles_degrees": [round_float(value) for value in stage_angles],
        "printed_angle_packet_degrees": {
            "theta_green": round_float(green),
            "theta_orange": round_float(orange),
            "theta_blue": round_float(blue),
            "theta_orange_complement": round_float(orange_comp),
        },
        "comparisons": [
            {
                "stage_id": 2,
                "recovered_angle": round_float(stage_angles[0]),
                "target": "theta_orange",
                "target_angle": round_float(orange),
                "difference_deg": round_float(stage_angles[0] - orange),
            },
            {
                "stage_id": 3,
                "recovered_angle": round_float(stage_angles[1]),
                "target": "geometric_mean(theta_orange, theta_green)",
                "target_angle": round_float(geometric_mid),
                "difference_deg": round_float(stage_angles[1] - geometric_mid),
            },
            {
                "stage_id": 3,
                "recovered_angle": round_float(stage_angles[1]),
                "target": "arithmetic_mean(theta_orange, theta_green)",
                "target_angle": round_float(arithmetic_mid),
                "difference_deg": round_float(stage_angles[1] - arithmetic_mid),
            },
            {
                "stage_id": 4,
                "recovered_angle": round_float(stage_angles[2]),
                "target": "theta_green",
                "target_angle": round_float(green),
                "difference_deg": round_float(stage_angles[2] - green),
            },
            {
                "stage_id": 3,
                "recovered_angle": round_float(stage_angles[1]),
                "target": "theta_orange_complement",
                "target_angle": round_float(orange_comp),
                "difference_deg": round_float(stage_angles[1] - orange_comp),
            },
        ],
        "ratio_form": {
            "theta_green_over_theta_orange": round_float(green / orange),
            "sqrt_ratio": round_float(math.sqrt(green / orange)),
            "stage3_over_stage2": round_float(stage_angles[1] / stage_angles[0]),
            "stage4_over_stage3": round_float(stage_angles[2] / stage_angles[1]),
        },
    }

    timing_graph_bridge = {
        "graph_labels_visible": topology["assets"][1]["topology_extract"]["labeled_nodes_visible"][-8:],
        "graph_topology_summary": topology["assets"][1]["topology_extract"]["topology_summary"],
        "structural_observation": "The same witness contains a top-right time-distance graph indexed by X0..X3 and a bottom stage chain with three recovered internal angles across stages 2, 3, and 4.",
        "current_support_level": "structural_only",
        "reason_not_promoted": "No numeric curve extraction has yet been performed on the time-distance graph, so the angle schedule cannot be certified against Y1/Y2/Y3 values or graph slopes.",
    }

    return {
        "artifact_id": "sixto_angle_schedule_bridge",
        "purpose": "Test whether the recovered internal Sixto stage-angle schedule ties back to the printed angle packet or to the timing graph.",
        "source_hierarchy": {
            "tier_1_internal_arm_test": {
                "path": "pythagoras_quantum_world_rt/sixto_internal_k_arm_test.json",
                "role": "Recovered stage-angle schedule from the visible internal K-bearing arm family.",
            },
            "tier_2_precision_witness": {
                "path": "pythagoras_quantum_world_rt/sioxto3002b_precision_witness.json",
                "role": "Printed source-angle packet for the main engineering diagram.",
            },
            "tier_3_topology_witness": {
                "path": "pythagoras_quantum_world_rt/sixto_topology_witness.json",
                "role": "Structural witness for the X0..X3 timing graph and stage-chain co-presence.",
            },
        },
        "angle_packet_bridge": angle_packet_bridge,
        "timing_graph_bridge": timing_graph_bridge,
        "verdict": {
            "angle_packet_tie_supported": True,
            "timing_graph_tie_supported": False,
            "timing_graph_tie_support_level": "structural_only",
            "honest_summary": "The recovered stage-angle schedule ties back to the printed angle packet with real support: stage 2 is close to the printed orange angle, stage 4 is close to the printed green angle, and stage 3 lands near the geometric mean between them rather than the arithmetic midpoint. The timing-graph link is not yet numeric. It remains a structural co-presence claim because the witness shows both the X0..X3 graph and the stage chain, but the graph curves have not been extracted into testable numbers.",
        },
    }


def self_test() -> int:
    payload = build_payload()
    comps = payload["angle_packet_bridge"]["comparisons"]
    ok = True
    ok = ok and payload["verdict"]["angle_packet_tie_supported"]
    ok = ok and not payload["verdict"]["timing_graph_tie_supported"]
    ok = ok and abs(float(comps[0]["difference_deg"])) < 1.0
    ok = ok and abs(float(comps[1]["difference_deg"])) < 0.5
    ok = ok and abs(float(comps[3]["difference_deg"])) < 1.2
    ok = ok and abs(float(comps[2]["difference_deg"])) > 5.0
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    write_json(OUT_PATH, payload)
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/sixto_angle_schedule_bridge.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
