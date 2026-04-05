#!/usr/bin/env python3
"""Build a source-native two-branch law packet for the Sixto timing graph."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
SUMMARY_PATH = OUT_DIR / "sixto_geogebra_extended_summary.json"
COMPRESSION_PATH = OUT_DIR / "sixto_graph_curve_compression_packet.json"
PACKET_PATH = OUT_DIR / "sixto_graph_two_branch_law_packet.json"

ANOMALY_THRESHOLD = 0.3


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def interp(points: list[tuple[float, float]], t_value: float) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if t_value <= xs[0]:
        return ys[0]
    if t_value >= xs[-1]:
        return ys[-1]
    index_value = bisect.bisect_left(xs, t_value)
    if xs[index_value] == t_value:
        return ys[index_value]
    x0 = xs[index_value - 1]
    x1 = xs[index_value]
    y0 = ys[index_value - 1]
    y1 = ys[index_value]
    u_value = (t_value - x0) / (x1 - x0)
    return y0 + (y1 - y0) * u_value


def cyan_right_branch(summary: dict[str, Any]) -> list[tuple[float, float]]:
    curve = next(item for item in summary["reconstructed_graph_curves"] if item["curve_id"] == "curve_graph_cyan")
    cross_x = float(curve["crossover_x"])
    peak_y = float(curve["peak"]["y"])
    branch = []
    for point in curve["sampled_points"]:
        x_value = float(point["x"])
        y_value = float(point["y"])
        if x_value >= cross_x:
            t_value = (x_value - cross_x) / (1212.0 - cross_x) if cross_x != 1212.0 else 0.0
            branch.append((t_value, y_value / peak_y if peak_y else 0.0))
    branch.sort()
    return branch


def build_payload() -> dict[str, Any]:
    summary = read_json(SUMMARY_PATH)
    compression = read_json(COMPRESSION_PATH)
    left_template = compression["branch_templates"]["left"]
    right_template = compression["branch_templates"]["right"]
    grid_t = [float(value) for value in right_template["grid_t"]]
    shared_positive_members = [
        item["curve_id"]
        for item in right_template["members"]
        if item["accepted_into_template_family"]
    ]
    anomaly_members = [
        item["curve_id"]
        for item in right_template["members"]
        if not item["accepted_into_template_family"]
    ]

    cyan_branch = cyan_right_branch(summary)
    template_profile = [float(value) for value in right_template["template_profile"]]
    anomaly_profile = [interp(cyan_branch, t_value) - template_value for t_value, template_value in zip(grid_t, template_profile)]
    support_indices = [index for index, value in enumerate(anomaly_profile) if abs(value) >= ANOMALY_THRESHOLD]
    if support_indices:
        start_index = support_indices[0]
        end_index = support_indices[-1]
    else:
        start_index = 0
        end_index = -1
    anomaly_window = {
        "t_start": round_float(grid_t[start_index]),
        "t_end": round_float(grid_t[end_index]) if end_index >= 0 else round_float(grid_t[0]),
        "support_count": len(support_indices),
    }
    anomaly_min = min(anomaly_profile)
    anomaly_max = max(anomaly_profile)
    anomaly_min_index = anomaly_profile.index(anomaly_min)
    anomaly_max_index = anomaly_profile.index(anomaly_max)

    curve_roles = []
    for curve in compression["curve_reconstructions"]:
        curve_id = curve["curve_id"]
        role = {
            "curve_id": curve_id,
            "negative_branch_family": "shared_left_template",
            "positive_branch_family": "shared_right_template" if curve["right_branch_template_member"] else "cyan_positive_anomaly",
            "anchors": curve["anchors"],
        }
        curve_roles.append(role)

    return {
        "artifact_id": "sixto_graph_two_branch_law_packet",
        "source_summary": "pythagoras_quantum_world_rt/sixto_geogebra_extended_summary.json",
        "source_compression": "pythagoras_quantum_world_rt/sixto_graph_curve_compression_packet.json",
        "shared_law_packet": {
            "negative_branch_family": {
                "family_id": "shared_left_template",
                "medoid_curve_id": left_template["medoid_curve_id"],
                "normalization": "t = x / crossover_x ; y_norm = y / abs(dip_y)",
                "grid_t": [round_float(value) for value in grid_t],
                "template_profile": [round_float(value) for value in left_template["template_profile"]],
                "member_curve_ids": [item["curve_id"] for item in left_template["members"] if item["accepted_into_template_family"]],
            },
            "positive_branch_family": {
                "family_id": "shared_right_template",
                "medoid_curve_id": right_template["medoid_curve_id"],
                "normalization": "t = (x - crossover_x) / (1212 - crossover_x) ; y_norm = y / peak_y",
                "grid_t": [round_float(value) for value in grid_t],
                "template_profile": [round_float(value) for value in template_profile],
                "member_curve_ids": shared_positive_members,
            },
        },
        "cyan_anomaly_lane": {
            "curve_id": "curve_graph_cyan",
            "family_id": "cyan_positive_anomaly",
            "base_family": "shared_right_template",
            "grid_t": [round_float(value) for value in grid_t],
            "delta_profile": [round_float(value) for value in anomaly_profile],
            "support_window": anomaly_window,
            "extrema": {
                "min_delta": round_float(anomaly_min),
                "min_t": round_float(grid_t[anomaly_min_index]),
                "max_delta": round_float(anomaly_max),
                "max_t": round_float(grid_t[anomaly_max_index]),
            },
            "interpretation": "The cyan positive branch follows the shared right template outside a mid-branch notch window and deviates strongly only on the interior interval where the residual is negative and large.",
        },
        "curve_roles": curve_roles,
        "verdict": {
            "two_branch_source_native_packet_ready": True,
            "anomaly_curve_ids": anomaly_members,
            "honest_summary": "The Sixto timing graph now supports a source-native two-branch law packet: one shared negative branch for all four traces, one shared positive branch for light-green/green/blue, and one explicit cyan positive-branch anomaly lane represented as a residual on top of the shared positive template.",
        },
    }


def self_test() -> int:
    payload = build_payload()
    anomaly = payload["cyan_anomaly_lane"]
    ok = True
    ok = ok and payload["shared_law_packet"]["negative_branch_family"]["member_curve_ids"] == [
        "curve_graph_light_green",
        "curve_graph_green",
        "curve_graph_cyan",
        "curve_graph_blue",
    ]
    ok = ok and payload["shared_law_packet"]["positive_branch_family"]["member_curve_ids"] == [
        "curve_graph_light_green",
        "curve_graph_green",
        "curve_graph_blue",
    ]
    ok = ok and payload["verdict"]["anomaly_curve_ids"] == ["curve_graph_cyan"]
    ok = ok and anomaly["support_window"]["t_start"] == 0.4
    ok = ok and anomaly["support_window"]["t_end"] == 0.5
    ok = ok and anomaly["support_window"]["support_count"] == 3
    ok = ok and anomaly["extrema"]["min_delta"] < -0.6
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    write_json(PACKET_PATH, payload)
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/sixto_graph_two_branch_law_packet.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
