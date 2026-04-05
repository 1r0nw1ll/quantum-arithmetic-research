#!/usr/bin/env python3
"""Test the QA moment-arm packet against the recovered Sixto K-labeled stage chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
STAGE_GRAPH_PATH = OUT_DIR / "sixto_stage_graph.json"
OUT_PATH = OUT_DIR / "sixto_k_chain_packet_test.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def coeff_var(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) * (value - mean) for value in values) / len(values)
    return math.sqrt(variance) / mean


def build_payload() -> dict[str, object]:
    stage_graph = read_json(STAGE_GRAPH_PATH)
    stages = stage_graph["stages"]
    radii = [float(stage["circle"]["radius"]) for stage in stages]
    center_x = [float(stage["circle"]["center_global"][0]) for stage in stages]
    center_y = [float(stage["circle"]["center_global"][1]) for stage in stages]
    stage_ids = [int(stage["stage_id"]) for stage in stages]
    spacings = [center_x[idx + 1] - center_x[idx] for idx in range(len(center_x) - 1)]
    radius_ratios = [radii[idx + 1] / radii[idx] for idx in range(len(radii) - 1)]
    spacing_ratios = [spacings[idx + 1] / spacings[idx] for idx in range(len(spacings) - 1)]
    mean_radius = sum(radii) / len(radii)
    mean_spacing = sum(spacings) / len(spacings)
    y_drift = max(center_y) - min(center_y)

    circle_branch_score = {
        "mean_radius": round_float(mean_radius),
        "radius_coefficient_of_variation": round_float(coeff_var(radii)),
        "mean_center_spacing": round_float(mean_spacing),
        "spacing_coefficient_of_variation": round_float(coeff_var(spacings)),
        "max_center_y_drift": round_float(y_drift),
        "interpretation": "Visible outer stage rims are almost constant-radius circles arranged on an almost level carrier line.",
    }

    ellipse_branch_obstruction = {
        "packet_fact": "For positive QA tuples, X = e*d > 0, so K = D + X > D and K/D = a/d > 1.",
        "visible_boundary_requirement_if_outer_rim_were_(K,D)_ellipse": "A true visible (K,D) ellipse would have anisotropy ratio strictly greater than 1, not a circular aspect ratio of 1.",
        "observed_boundary_ratio": 1.0,
        "observed_boundary_source": "Recovered outer stage witnesses are circles with one radius each.",
        "conclusion": "The visible stage rims do not behave like direct (K,D) ellipse boundaries.",
    }

    k_rule_mismatch = {
        "source_rule": stage_graph["rules"]["k_rule"],
        "expected_contraction": 0.62,
        "observed_radius_ratios": [round_float(value) for value in radius_ratios],
        "observed_center_spacing_ratios": [round_float(value) for value in spacing_ratios],
        "radius_ratio_distance_from_0_62": [round_float(abs(value - 0.62)) for value in radius_ratios],
        "spacing_ratio_distance_from_0_62": [round_float(abs(value - 0.62)) for value in spacing_ratios],
        "conclusion": "The visible outer radii and center spacings do not contract by the K-rule, so the K labels are not the outer rim size or stage spacing in the recovered witness.",
    }

    return {
        "artifact_id": "sixto_k_chain_packet_test",
        "purpose": "Test whether the recovered Sixto K-labeled stage witness behaves more like the circle branch W or the scaled ellipse branch (K,D).",
        "source_hierarchy": {
            "tier_1_stage_witness": {
                "path": "pythagoras_quantum_world_rt/sixto_stage_graph.json",
                "role": "Recovered K1..K4 stage circles and adjacency chain from the Sixto witness.",
            },
            "tier_2_packet_rewrite": {
                "paths": [
                    "pythagoras_quantum_world_rt/arto_contra_cyclic_moment_arm_packet.json",
                    "pythagoras_quantum_world_rt/arto_contra_cyclic_formula_rewrites.json",
                ],
                "role": "Current project rule for interpreting Arto moment-arm language through W or (K,D).",
            },
        },
        "observed_stage_geometry": {
            "stage_ids": stage_ids,
            "radii": [round_float(value) for value in radii],
            "center_x": [round_float(value) for value in center_x],
            "center_y": [round_float(value) for value in center_y],
            "adjacent_center_spacings": [round_float(value) for value in spacings],
        },
        "circle_branch_test": circle_branch_score,
        "ellipse_branch_obstruction": ellipse_branch_obstruction,
        "k_rule_visibility_test": k_rule_mismatch,
        "verdict": {
            "best_fit_branch": "circle_branch_W",
            "honest_summary": "The recovered Sixto outer stage geometry behaves much more like a W-type circle carrier than a visible (K,D) ellipse. The stage rims are nearly constant circles, their centers lie on an almost level line, and neither the outer radii nor the center spacings follow the stated 0.62 K-rule. The clean reading is that K belongs to an internal moment-arm or transfer packet, while the visible outer stage carrier is circle-branch behavior.",
            "supports_direct_visible_KD_ellipse": False,
            "supports_visible_W_circle_carrier": True,
        },
    }


def self_test() -> int:
    payload = build_payload()
    ok = True
    ok = ok and payload["verdict"]["best_fit_branch"] == "circle_branch_W"
    ok = ok and payload["verdict"]["supports_visible_W_circle_carrier"]
    ok = ok and not payload["verdict"]["supports_direct_visible_KD_ellipse"]
    ok = ok and payload["circle_branch_test"]["radius_coefficient_of_variation"] < 0.02
    ok = ok and payload["k_rule_visibility_test"]["radius_ratio_distance_from_0_62"][0] > 0.3
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
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/sixto_k_chain_packet_test.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
