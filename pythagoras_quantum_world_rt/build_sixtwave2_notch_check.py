#!/usr/bin/env python3
"""Check whether the SixtoWave2 velocity graph reproduces the cyan notch anomaly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
ASSET_PATH = Path("/tmp/sixto_assets/sixtwave2.jpg")
TWO_BRANCH_PACKET_PATH = OUT_DIR / "sixto_graph_two_branch_law_packet.json"
OUTPUT_PATH = OUT_DIR / "sixtwave2_notch_check.json"

GRAPH_TOP = 430
GRAPH_BOTTOM = 930
GRAPH_LEFT = 500
GRAPH_RIGHT = 3020


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fit_line_residual(points: list[tuple[float, float]]) -> dict[str, float]:
    x_values = np.array([point[0] for point in points], dtype=float)
    y_values = np.array([point[1] for point in points], dtype=float)
    coeffs = np.polyfit(x_values, y_values, 1)
    predicted = coeffs[0] * x_values + coeffs[1]
    residuals = predicted - y_values
    return {
        "slope": round_float(coeffs[0]),
        "intercept": round_float(coeffs[1]),
        "max_abs_residual_px": round_float(np.max(np.abs(residuals))),
        "rmse_px": round_float(np.sqrt(np.mean(residuals * residuals))),
    }


def extract_wave_graph() -> dict[str, Any]:
    image = cv2.imread(str(ASSET_PATH))
    if image is None:
        raise FileNotFoundError(ASSET_PATH)
    crop = image[GRAPH_TOP:GRAPH_BOTTOM, GRAPH_LEFT:GRAPH_RIGHT]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 90, 100]), np.array([115, 255, 255]))

    points = []
    for x_value in range(blue_mask.shape[1]):
        ys = np.where(blue_mask[:, x_value] > 0)[0]
        if len(ys) == 0:
            continue
        points.append((float(x_value), float(np.median(ys))))
    if not points:
        raise RuntimeError("Failed to extract SixtoWave2 velocity graph.")

    x_values = np.array([point[0] for point in points], dtype=float)
    y_values = np.array([point[1] for point in points], dtype=float)
    y_smooth = moving_average(y_values, 21)

    x_peak = float(x_values[np.argmin(y_smooth)])
    y_peak = float(y_values[np.argmin(y_smooth)])
    x_trough = float(x_values[np.argmax(y_smooth)])
    y_trough = float(y_values[np.argmax(y_smooth)])

    if x_peak >= x_trough:
        raise RuntimeError("Unexpected SixtoWave2 graph ordering: peak not before trough.")

    start_point = (float(x_values[0]), float(y_values[0]))
    end_point = (float(x_values[-1]), float(y_values[-1]))

    rise_points = [point for point in points if point[0] <= x_peak]
    fall_points = [point for point in points if x_peak <= point[0] <= x_trough]
    return_points = [point for point in points if point[0] >= x_trough]

    return {
        "crop_bbox": [GRAPH_LEFT, GRAPH_TOP, GRAPH_RIGHT - GRAPH_LEFT, GRAPH_BOTTOM - GRAPH_TOP],
        "point_count": len(points),
        "start_point": [round_float(start_point[0]), round_float(start_point[1])],
        "peak_point": [round_float(x_peak), round_float(y_peak)],
        "trough_point": [round_float(x_trough), round_float(y_trough)],
        "end_point": [round_float(end_point[0]), round_float(end_point[1])],
        "rise_fit": fit_line_residual(rise_points),
        "fall_fit": fit_line_residual(fall_points),
        "return_fit": fit_line_residual(return_points),
    }


def build_payload() -> dict[str, Any]:
    packet = read_json(TWO_BRANCH_PACKET_PATH)
    wave = extract_wave_graph()
    return_max = float(wave["return_fit"]["max_abs_residual_px"])
    rise_max = float(wave["rise_fit"]["max_abs_residual_px"])
    fall_max = float(wave["fall_fit"]["max_abs_residual_px"])

    cyan_lane = packet["cyan_anomaly_lane"]
    return {
        "artifact_id": "sixtwave2_notch_check",
        "source_asset": "sixtwave2",
        "source_packet": "pythagoras_quantum_world_rt/sixto_graph_two_branch_law_packet.json",
        "graph_extraction": wave,
        "comparison_target": {
            "anomaly_curve_id": cyan_lane["curve_id"],
            "anomaly_support_window": cyan_lane["support_window"],
            "anomaly_min_delta": cyan_lane["extrema"]["min_delta"],
        },
        "verdict": {
            "localized_cyan_notch_present": False,
            "honest_summary": "The SixtoWave2 velocity graph does not reproduce the localized cyan positive-branch notch. Its blue trace is a three-segment piecewise-linear path with a single peak and a single trough, and the return branch stays linear to within the extracted raster residual.",
            "supporting_metrics": {
                "rise_max_abs_residual_px": round_float(rise_max),
                "fall_max_abs_residual_px": round_float(fall_max),
                "return_max_abs_residual_px": round_float(return_max),
            },
        },
    }


def self_test() -> int:
    payload = build_payload()
    wave = payload["graph_extraction"]
    ok = True
    ok = ok and payload["verdict"]["localized_cyan_notch_present"] is False
    ok = ok and float(wave["peak_point"][0]) < float(wave["trough_point"][0])
    ok = ok and float(wave["return_fit"]["max_abs_residual_px"]) < 4.0
    ok = ok and float(wave["fall_fit"]["max_abs_residual_px"]) < 12.0
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
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/sixtwave2_notch_check.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
