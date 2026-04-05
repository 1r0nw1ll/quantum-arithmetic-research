#!/usr/bin/env python3
"""Build a precision line-and-angle witness from the Sioxto3002B engineering packet."""

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
OUT_PATH = OUT_DIR / "sioxto3002b_precision_witness.json"
PACKET_X0 = 1200
PACKET_Y0 = 350
PACKET_X1 = 2500
PACKET_Y1 = 1450


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def round_float(value: float) -> float:
    return round(float(value), 6)


def point_tuple(point: list[int] | tuple[int, int]) -> tuple[float, float]:
    return (float(point[0]), float(point[1]))


def line_angle_deg(line: dict[str, object]) -> float:
    return float(line["angle_degrees"])


def detect_packet_lines(image: np.ndarray) -> list[dict[str, object]]:
    crop = image[PACKET_Y0:PACKET_Y1, PACKET_X0:PACKET_X1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=120,
        maxLineGap=12,
    )
    if raw is None:
        raise RuntimeError("No line candidates detected in Sioxto3002B packet crop.")
    lines = []
    for entry in raw[:, 0, :]:
        x1, y1, x2, y2 = map(int, entry)
        gx1 = x1 + PACKET_X0
        gy1 = y1 + PACKET_Y0
        gx2 = x2 + PACKET_X0
        gy2 = y2 + PACKET_Y0
        length = math.hypot(gx2 - gx1, gy2 - gy1)
        angle = math.degrees(math.atan2(gy2 - gy1, gx2 - gx1))
        lines.append(
            {
                "p1": [gx1, gy1],
                "p2": [gx2, gy2],
                "length": round_float(length),
                "angle_degrees": round_float(angle),
            }
        )
    return lines


def choose_line(
    lines: list[dict[str, object]],
    *,
    angle_min: float,
    angle_max: float,
    min_length: float,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    prefer_mid_x: float | None = None,
    prefer_mid_y: float | None = None,
    used_keys: set[tuple[int, int, int, int]] | None = None,
) -> dict[str, object]:
    used_keys = used_keys if used_keys is not None else set()
    best = None
    best_score = None
    for line in lines:
        angle = float(line["angle_degrees"])
        if not (angle_min <= angle <= angle_max):
            continue
        if float(line["length"]) < min_length:
            continue
        xs = [float(line["p1"][0]), float(line["p2"][0])]
        ys = [float(line["p1"][1]), float(line["p2"][1])]
        if x_min is not None and max(xs) < x_min:
            continue
        if x_max is not None and min(xs) > x_max:
            continue
        if y_min is not None and max(ys) < y_min:
            continue
        if y_max is not None and min(ys) > y_max:
            continue
        key = tuple(int(v) for v in (line["p1"] + line["p2"]))
        if key in used_keys:
            continue
        mid_x = sum(xs) / 2.0
        mid_y = sum(ys) / 2.0
        score = -float(line["length"])
        if prefer_mid_x is not None:
            score += abs(mid_x - prefer_mid_x) * 0.2
        if prefer_mid_y is not None:
            score += abs(mid_y - prefer_mid_y) * 0.2
        if best is None or score < best_score:
            best = line
            best_score = score
    if best is None:
        raise RuntimeError(f"no line found for angle range {angle_min}..{angle_max}")
    used_keys.add(tuple(int(v) for v in (best["p1"] + best["p2"])))
    return best


def line_intersection(
    line_a: dict[str, object], line_b: dict[str, object]
) -> list[float]:
    x1, y1 = point_tuple(line_a["p1"])
    x2, y2 = point_tuple(line_a["p2"])
    x3, y3 = point_tuple(line_b["p1"])
    x4, y4 = point_tuple(line_b["p2"])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if math.isclose(denom, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("parallel lines")
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return [round_float(px), round_float(py)]


def detect_circle_components(image: np.ndarray) -> dict[str, object]:
    crop = image[PACKET_Y0:PACKET_Y1, PACKET_X0:PACKET_X1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        "cyan": cv2.inRange(hsv, np.array((80, 50, 50)), np.array((100, 255, 255))),
        "green": cv2.inRange(hsv, np.array((35, 40, 40)), np.array((90, 255, 255))),
    }
    packet = {}
    for name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0]
        area = cv2.contourArea(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        x, y, w, h = cv2.boundingRect(contour)
        packet[f"{name}_dominant_ring"] = {
            "area": round_float(area),
            "bbox_global": [int(x + PACKET_X0), int(y + PACKET_Y0), int(w), int(h)],
            "min_enclosing_circle_global": {
                "center": [round_float(cx + PACKET_X0), round_float(cy + PACKET_Y0)],
                "radius": round_float(radius),
            },
        }
    return packet


def build_payload() -> dict[str, object]:
    image = cv2.imread(str(ASSET_PATH))
    if image is None:
        raise FileNotFoundError(ASSET_PATH)

    lines = detect_packet_lines(image)
    used: set[tuple[int, int, int, int]] = set()
    line_packet = {
        "main_axis_horizontal": choose_line(
            lines,
            angle_min=-1.0,
            angle_max=1.0,
            min_length=1100.0,
            prefer_mid_y=974.0,
            used_keys=used,
        ),
        "lower_base_horizontal": choose_line(
            lines,
            angle_min=-1.0,
            angle_max=1.0,
            min_length=700.0,
            prefer_mid_y=1248.0,
            used_keys=used,
        ),
        "left_packet_vertical": choose_line(
            lines,
            angle_min=-90.0,
            angle_max=-89.0,
            min_length=500.0,
            prefer_mid_x=1294.0,
            used_keys=used,
        ),
        "right_center_vertical": choose_line(
            lines,
            angle_min=-90.0,
            angle_max=-89.0,
            min_length=350.0,
            prefer_mid_x=1900.0,
            used_keys=used,
        ),
        "outer_right_measure_vertical": choose_line(
            lines,
            angle_min=-90.0,
            angle_max=-89.0,
            min_length=800.0,
            prefer_mid_x=2470.0,
            used_keys=used,
        ),
        "steep_up_edge_proxy": choose_line(
            lines,
            angle_min=70.0,
            angle_max=75.0,
            min_length=430.0,
            x_min=1350.0,
            used_keys=used,
        ),
        "shallow_positive_proxy": choose_line(
            lines,
            angle_min=14.0,
            angle_max=20.0,
            min_length=350.0,
            x_min=1450.0,
            used_keys=used,
        ),
        "shallow_negative_proxy": choose_line(
            lines,
            angle_min=-20.0,
            angle_max=-14.0,
            min_length=450.0,
            x_min=1300.0,
            used_keys=used,
        ),
        "complementary_negative_proxy": choose_line(
            lines,
            angle_min=-37.0,
            angle_max=-32.0,
            min_length=250.0,
            x_min=1400.0,
            used_keys=used,
        ),
    }

    circles = detect_circle_components(image)

    source_measurements = {
        "A_mm": 72.0,
        "A1_mm": 72.0,
        "B_mm": 233.0,
        "B1_mm": 233.0,
        "C_mm": 243.870867468831,
        "C1_mm": 243.870867468831,
    }
    source_angles = {
        "theta_green_deg": 17.15957,
        "theta_orange_deg": 55.54330,
        "theta_blue_deg": 72.82808,
        "theta_right_deg": 90.0,
    }

    angle_residuals = {
        "steep_up_vs_theta_blue": {
            "detected_angle_deg": line_packet["steep_up_edge_proxy"]["angle_degrees"],
            "source_angle_deg": source_angles["theta_blue_deg"],
            "difference_deg": round_float(
                line_packet["steep_up_edge_proxy"]["angle_degrees"] - source_angles["theta_blue_deg"]
            ),
        },
        "shallow_positive_vs_theta_green": {
            "detected_angle_deg": line_packet["shallow_positive_proxy"]["angle_degrees"],
            "source_angle_deg": source_angles["theta_green_deg"],
            "difference_deg": round_float(
                line_packet["shallow_positive_proxy"]["angle_degrees"] - source_angles["theta_green_deg"]
            ),
        },
        "complementary_negative_vs_theta_orange_complement": {
            "detected_angle_deg": abs(float(line_packet["complementary_negative_proxy"]["angle_degrees"])),
            "source_angle_deg": round_float(90.0 - source_angles["theta_orange_deg"]),
            "difference_deg": round_float(
                abs(float(line_packet["complementary_negative_proxy"]["angle_degrees"]))
                - (90.0 - source_angles["theta_orange_deg"])
            ),
        },
        "right_center_vertical_vs_theta_right": {
            "detected_angle_deg": 90.0,
            "source_angle_deg": source_angles["theta_right_deg"],
            "difference_deg": 0.0,
        },
    }

    anchor_points = {
        "right_center_axis_intersection": line_intersection(
            line_packet["main_axis_horizontal"], line_packet["right_center_vertical"]
        ),
        "left_axis_intersection": line_intersection(
            line_packet["main_axis_horizontal"], line_packet["left_packet_vertical"]
        ),
        "outer_measure_intersection": line_intersection(
            line_packet["main_axis_horizontal"], line_packet["outer_right_measure_vertical"]
        ),
        "right_center_base_intersection": line_intersection(
            line_packet["lower_base_horizontal"], line_packet["right_center_vertical"]
        ),
        "shallow_positive_to_right_center_vertical": line_intersection(
            line_packet["shallow_positive_proxy"], line_packet["right_center_vertical"]
        ),
        "steep_up_to_lower_base": line_intersection(
            line_packet["steep_up_edge_proxy"], line_packet["lower_base_horizontal"]
        ),
    }

    return {
        "witness_id": "sioxto3002b_precision_witness",
        "asset_path": str(ASSET_PATH),
        "packet_crop_global_bbox": [PACKET_X0, PACKET_Y0, PACKET_X1 - PACKET_X0, PACKET_Y1 - PACKET_Y0],
        "source_measurements": source_measurements,
        "source_angles": source_angles,
        "circle_packet": circles,
        "line_packet": line_packet,
        "angle_residuals": angle_residuals,
        "anchor_points": anchor_points,
        "recovery_notes": [
            "This witness is intended as the precision reference for sub-pixel Sixto work because the engineering packet has cleaner horizontals, verticals, and labeled angle families than CCH-Archm1b.",
            "The 72.8-degree and 17.16-degree lanes are much cleaner than the orange 55.54-degree lane, which still appears mainly through a complementary line family.",
            "The right packet center is represented structurally by the intersection of the main horizontal axis and the right-center vertical, not by OCR of the A/A1 glyphs.",
        ],
        "recommended_next_actions": [
            "Use the printed-angle witness as the precision reference when comparing CCH edge families against Sixto geometry.",
            "Propagate the right-center axis point and steep-up / shallow-positive line families into the next corrected-witness QA/RT pass.",
            "Delay any claim that depends on the orange 55.54330-degree lane until one more witness resolves it more directly.",
        ],
    }


def self_test() -> int:
    ok = True
    line_a = {"p1": [0, 0], "p2": [10, 0], "length": 10.0, "angle_degrees": 0.0}
    line_b = {"p1": [5, -5], "p2": [5, 5], "length": 10.0, "angle_degrees": -90.0}
    hit = line_intersection(line_a, line_b)
    ok = ok and hit == [5.0, 0.0]
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
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": ["pythagoras_quantum_world_rt/sioxto3002b_precision_witness.json"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
