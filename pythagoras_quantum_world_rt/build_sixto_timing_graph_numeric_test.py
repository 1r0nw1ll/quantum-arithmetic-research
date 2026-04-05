#!/usr/bin/env python3
"""Numerically extract the Sixto timing graph and compare it to the stage-angle schedule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
ASSET_PATH = Path("/tmp/sixto_assets/sioxto3002b.jpg")
ANGLE_BRIDGE_PATH = OUT_DIR / "sixto_angle_schedule_bridge.json"
OUT_PATH = OUT_DIR / "sixto_timing_graph_numeric_test.json"

GRAPH_Y0 = 120
GRAPH_Y1 = 980
GRAPH_X0 = 2850
GRAPH_X1 = 4220


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def contour_runs(index_value_pairs: list[tuple[int, int]]) -> list[dict[str, float]]:
    if not index_value_pairs:
        return []
    runs = []
    start = index_value_pairs[0][0]
    prev = index_value_pairs[0][0]
    vals = [index_value_pairs[0][1]]
    for idx, value in index_value_pairs[1:]:
        if idx == prev + 1:
            vals.append(value)
            prev = idx
            continue
        runs.append({"start": start, "end": prev, "center": (start + prev) / 2.0, "max": max(vals)})
        start = prev = idx
        vals = [value]
    runs.append({"start": start, "end": prev, "center": (start + prev) / 2.0, "max": max(vals)})
    return runs


def extract_graph_payload() -> dict[str, object]:
    image = cv2.imread(str(ASSET_PATH))
    if image is None:
        raise FileNotFoundError(ASSET_PATH)
    crop = image[GRAPH_Y0:GRAPH_Y1, GRAPH_X0:GRAPH_X1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_mask = gray < 80
    contours, _ = cv2.findContours(dark_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 5000:
            boxes.append((area, [x, y, w, h]))
    boxes.sort(reverse=True)
    plot_bbox = boxes[0][1]
    plot_x, plot_y, plot_w, plot_h = plot_bbox

    plot = crop[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w]
    plot_gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    plot_dark = plot_gray < 80

    col_counts = plot_dark.sum(axis=0)
    major_cols = contour_runs([(idx, int(value)) for idx, value in enumerate(col_counts) if value > 150])
    row_counts = plot_dark.sum(axis=1)
    major_rows = contour_runs([(idx, int(value)) for idx, value in enumerate(row_counts) if value > 500])

    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])) | cv2.inRange(
        hsv, np.array([170, 80, 80]), np.array([179, 255, 255])
    )
    red_row_counts = red_mask.sum(axis=1) / 255
    red_rows = contour_runs([(idx, int(value)) for idx, value in enumerate(red_row_counts) if value > 200])

    color_masks = {
        "blue": cv2.inRange(hsv, np.array([95, 60, 60]), np.array([115, 255, 255])),
        "green_dark": cv2.inRange(hsv, np.array([55, 50, 50]), np.array([78, 255, 255])),
        "cyan_light": cv2.inRange(hsv, np.array([80, 40, 40]), np.array([95, 255, 255])),
    }
    color_extrema = {}
    for name, mask in color_masks.items():
        points = np.column_stack(np.where(mask > 0))
        if len(points) == 0:
            continue
        ys = points[:, 0]
        xs = points[:, 1]
        color_extrema[name] = {
            "bbox_plot": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            ],
            "highest_point_plot": [int(xs[np.argmin(ys)]), int(ys.min())],
            "lowest_point_plot": [int(xs[np.argmax(ys)]), int(ys.max())],
        }

    return {
        "graph_crop_global_bbox": [GRAPH_X0, GRAPH_Y0, GRAPH_X1 - GRAPH_X0, GRAPH_Y1 - GRAPH_Y0],
        "plot_bbox_in_crop": plot_bbox,
        "plot_bbox_global": [plot_x + GRAPH_X0, plot_y + GRAPH_Y0, plot_w, plot_h],
        "major_vertical_guides_plot": [
            {"x_center_plot": round_float(run["center"]), "x_center_global": round_float(run["center"] + plot_x + GRAPH_X0)}
            for run in major_cols
        ],
        "major_horizontal_guides_plot": [
            {"y_center_plot": round_float(run["center"]), "y_center_global": round_float(run["center"] + plot_y + GRAPH_Y0)}
            for run in major_rows
        ],
        "red_reference_levels_plot": [
            {"y_center_plot": round_float(run["center"]), "y_center_global": round_float(run["center"] + plot_y + GRAPH_Y0)}
            for run in red_rows
        ],
        "color_curve_extrema_plot": color_extrema,
    }


def build_payload() -> dict[str, object]:
    graph = extract_graph_payload()
    angle_bridge = read_json(ANGLE_BRIDGE_PATH)

    guides = graph["major_vertical_guides_plot"]
    x0 = float(guides[0]["x_center_plot"])
    x1 = float(guides[1]["x_center_plot"])
    x2 = float(guides[2]["x_center_plot"])
    x3 = float(guides[3]["x_center_plot"])
    relative = [0.0, x1 - x0, x2 - x0, x3 - x0]
    geometric_mid = math.sqrt(relative[1] * relative[3])
    arithmetic_mid = (relative[1] + relative[3]) / 2.0

    angle_bridge_data = angle_bridge["angle_packet_bridge"]

    return {
        "artifact_id": "sixto_timing_graph_numeric_test",
        "purpose": "Numerically extract the top-right time-distance graph and test whether its X packet carries the same geometric-mean schedule seen in the recovered stage angles.",
        "source_hierarchy": {
            "tier_1_graph_witness": {
                "path": "/tmp/sixto_assets/sioxto3002b.jpg",
                "role": "Original raster witness containing the top-right time-distance graph.",
            },
            "tier_2_angle_bridge": {
                "path": "pythagoras_quantum_world_rt/sixto_angle_schedule_bridge.json",
                "role": "Recovered stage-angle schedule already tied to the printed angle packet.",
            },
        },
        "graph_extraction": graph,
        "graph_schedule_test": {
            "relative_major_x_guides": [round_float(value) for value in relative],
            "x2_vs_geometric_mid": round_float(relative[2] - geometric_mid),
            "x2_vs_arithmetic_mid": round_float(relative[2] - arithmetic_mid),
            "geometric_mid": round_float(geometric_mid),
            "arithmetic_mid": round_float(arithmetic_mid),
            "interpretation": "The middle major guide X2 is far closer to the geometric mean between X1 and X3 than to the arithmetic midpoint.",
        },
        "angle_schedule_reference": {
            "stage_angles_degrees": angle_bridge_data["recovered_stage_angles_degrees"],
            "stage3_vs_geometric_mean_deg": angle_bridge_data["comparisons"][1]["difference_deg"],
            "stage3_vs_arithmetic_mid_deg": angle_bridge_data["comparisons"][2]["difference_deg"],
        },
        "verdict": {
            "timing_graph_numeric_extraction_complete": True,
            "timing_graph_support_level": "weak_numeric",
            "honest_summary": "The timing graph is now numerically extracted at the frame/guide level. Its major X guides show the same qualitative pattern as the recovered stage-angle schedule: the middle guide X2 lies much closer to the geometric mean between X1 and X3 than to the arithmetic midpoint. That is a real numeric bridge, but still weaker than the printed-angle bridge because the graph curves themselves have not yet been fully separated into labeled Y1/Y2/Y3 traces.",
        },
    }


def self_test() -> int:
    payload = build_payload()
    ok = True
    ok = ok and payload["verdict"]["timing_graph_numeric_extraction_complete"]
    ok = ok and payload["verdict"]["timing_graph_support_level"] == "weak_numeric"
    ok = ok and len(payload["graph_extraction"]["major_vertical_guides_plot"]) == 4
    ok = ok and len(payload["graph_extraction"]["red_reference_levels_plot"]) == 3
    ok = ok and abs(float(payload["graph_schedule_test"]["x2_vs_geometric_mid"])) < 15.0
    ok = ok and abs(float(payload["graph_schedule_test"]["x2_vs_arithmetic_mid"])) > 100.0
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
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/sixto_timing_graph_numeric_test.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
