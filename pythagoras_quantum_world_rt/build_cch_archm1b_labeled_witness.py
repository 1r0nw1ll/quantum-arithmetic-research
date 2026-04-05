#!/usr/bin/env python3
"""Build a richer labeled/source-anchored witness for CCH-Archm1b."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
ASSET_PATH = Path("/tmp/sixto_assets/cch-archm1b.png")
OUT_PATH = OUT_DIR / "cch_archm1b_labeled_witness.json"
ROI_X0 = 450
ROI_Y0 = 500


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def round_float(value: float) -> float:
    return round(float(value), 6)


def round_point(point: tuple[float, float]) -> list[float]:
    return [round_float(point[0]), round_float(point[1])]


def color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, np.array((0, 100, 80)), np.array((12, 255, 255))),
            cv2.inRange(hsv, np.array((168, 100, 80)), np.array((180, 255, 255))),
        ),
        "orange": cv2.inRange(hsv, np.array((8, 100, 80)), np.array((30, 255, 255))),
        "blue": cv2.inRange(hsv, np.array((95, 70, 50)), np.array((140, 255, 255))),
        "green": cv2.inRange(hsv, np.array((35, 70, 50)), np.array((90, 255, 255))),
    }


def largest_contours(mask: np.ndarray, topn: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[:topn]


def contour_summary(contour: np.ndarray, approx_eps: float = 0.02) -> dict[str, object]:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    m = cv2.moments(contour)
    if m["m00"] == 0.0:
        centroid = (float(x + w / 2.0), float(y + h / 2.0))
    else:
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, approx_eps * perimeter, True)
    hull = cv2.convexHull(contour)
    pts = contour[:, 0, :]
    left = tuple(int(v) for v in pts[pts[:, 0].argmin()])
    right = tuple(int(v) for v in pts[pts[:, 0].argmax()])
    top = tuple(int(v) for v in pts[pts[:, 1].argmin()])
    bottom = tuple(int(v) for v in pts[pts[:, 1].argmax()])
    circle_center, circle_radius = cv2.minEnclosingCircle(contour)
    return {
        "area": round_float(area),
        "bbox": [int(x), int(y), int(w), int(h)],
        "centroid": round_point((centroid[0], centroid[1])),
        "min_enclosing_circle": {
            "center": round_point((circle_center[0], circle_center[1])),
            "radius": round_float(circle_radius),
        },
        "extreme_points": {
            "left": [left[0], left[1]],
            "right": [right[0], right[1]],
            "top": [top[0], top[1]],
            "bottom": [bottom[0], bottom[1]],
        },
        "approx_polygon": [[int(xv), int(yv)] for xv, yv in approx[:, 0, :].tolist()],
        "convex_hull_size": int(len(hull)),
    }


def line_length(line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    return math.hypot(dx, dy)


def line_angle_deg(line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def detect_roi_lines(image: np.ndarray) -> list[dict[str, object]]:
    crop = image[ROI_Y0:1600, ROI_X0:2300]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=120,
        minLineLength=220,
        maxLineGap=20,
    )
    if raw is None:
        return []
    lines = []
    for entry in raw[:, 0, :]:
        x1, y1, x2, y2 = map(int, entry)
        global_line = (x1 + ROI_X0, y1 + ROI_Y0, x2 + ROI_X0, y2 + ROI_Y0)
        lines.append(
            {
                "p1": [global_line[0], global_line[1]],
                "p2": [global_line[2], global_line[3]],
                "length": round_float(line_length(global_line)),
                "angle_degrees": round_float(line_angle_deg(global_line)),
            }
        )
    lines.sort(key=lambda entry: entry["length"], reverse=True)
    return lines


def choose_line(
    lines: list[dict[str, object]],
    *,
    angle_min: float,
    angle_max: float,
    min_length: float,
    prefer_y: float | None = None,
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
        key = tuple(int(v) for v in (line["p1"] + line["p2"]))
        if key in used_keys:
            continue
        score = -float(line["length"])
        if prefer_y is not None:
            mid_y = (float(line["p1"][1]) + float(line["p2"][1])) / 2.0
            score += abs(mid_y - prefer_y) * 0.25
        if best is None or score < best_score:
            best = line
            best_score = score
    if best is None:
        raise RuntimeError(f"no line found for angle range {angle_min}..{angle_max}")
    used_keys.add(tuple(int(v) for v in (best["p1"] + best["p2"])))
    return best


def contour_band_points(contour: np.ndarray, y_value: int, tolerance: int) -> dict[str, list[int] | int]:
    pts = contour[:, 0, :]
    band = pts[np.abs(pts[:, 1] - y_value) <= tolerance]
    if len(band) == 0:
        raise RuntimeError(f"no contour points near y={y_value}")
    left = band[band[:, 0].argmin()]
    right = band[band[:, 0].argmax()]
    return {
        "band_y": int(y_value),
        "tolerance": int(tolerance),
        "left": [int(left[0]), int(left[1])],
        "right": [int(right[0]), int(right[1])],
    }


def build_payload() -> dict[str, object]:
    image = cv2.imread(str(ASSET_PATH))
    if image is None:
        raise FileNotFoundError(ASSET_PATH)
    masks = color_masks(image)

    red_contours = largest_contours(masks["red"], 4)
    orange_contours = largest_contours(masks["orange"], 4)
    blue_contours = largest_contours(masks["blue"], 4)
    green_contours = largest_contours(masks["green"], 4)

    component_packet = {
        "red_lower_lobe": contour_summary(red_contours[0]),
        "red_upper_arc_packet": contour_summary(red_contours[1]),
        "orange_main_lobe": contour_summary(orange_contours[0]),
        "orange_left_internal_lobe": contour_summary(orange_contours[1]),
        "blue_lower_wedge": contour_summary(blue_contours[0]),
        "blue_upper_wedge": contour_summary(blue_contours[1]),
        "blue_upper_circle_component": contour_summary(blue_contours[2]),
        "blue_mid_triangle_component": contour_summary(blue_contours[3]),
        "green_left_circle": contour_summary(green_contours[0]),
        "green_mid_lobe": contour_summary(green_contours[1]),
    }

    roi_lines = detect_roi_lines(image)
    used: set[tuple[int, int, int, int]] = set()
    line_packet = {
        "dominant_horizontal_upper_chord": choose_line(
            roi_lines, angle_min=-1.0, angle_max=1.0, min_length=1000.0, used_keys=used
        ),
        "dominant_red_axis_proxy": choose_line(
            roi_lines, angle_min=-24.0, angle_max=-18.0, min_length=600.0, used_keys=used
        ),
        "blue_lower_rising_edge_proxy": choose_line(
            roi_lines, angle_min=50.0, angle_max=56.0, min_length=250.0, used_keys=used
        ),
        "blue_lower_falling_edge_proxy": choose_line(
            roi_lines, angle_min=-40.0, angle_max=-34.0, min_length=250.0, used_keys=used
        ),
        "central_vertical_left_proxy": choose_line(
            roi_lines,
            angle_min=-90.0,
            angle_max=-89.0,
            min_length=180.0,
            prefer_y=1300.0,
            used_keys=used,
        ),
        "central_vertical_right_proxy": choose_line(
            roi_lines,
            angle_min=-90.0,
            angle_max=-89.0,
            min_length=160.0,
            prefer_y=1300.0,
            used_keys=used,
        ),
    }

    orange_g_anchor = tuple(component_packet["orange_main_lobe"]["extreme_points"]["right"])
    red_lower = red_contours[0]
    red_g_band = contour_band_points(red_lower, y_value=int(orange_g_anchor[1]), tolerance=2)

    blue_lower = component_packet["blue_lower_wedge"]
    labeled_points = {
        "D_source_band_anchor": {
            "source_label": "D",
            "point": red_g_band["left"],
            "construction": "leftmost red-lower contour point in the G-height band",
            "confidence": 0.68,
        },
        "red_lower_right_band_contact": {
            "source_label": "S?",
            "point": red_g_band["right"],
            "construction": "rightmost red-lower contour point in the G-height band",
            "confidence": 0.42,
        },
        "G_source_anchor": {
            "source_label": "G",
            "point": orange_g_anchor,
            "construction": "rightmost point of the dominant orange lobe where the source places G near the blue junction",
            "confidence": 0.82,
        },
        "blue_left_shoulder": {
            "source_label": "G-left-wedge",
            "point": blue_lower["extreme_points"]["left"],
            "construction": "leftmost point of the dominant blue lower wedge contour",
            "confidence": 0.88,
        },
        "blue_far_apex": {
            "source_label": "blue_far_apex",
            "point": blue_lower["extreme_points"]["right"],
            "construction": "rightmost point of the dominant blue lower wedge contour",
            "confidence": 0.95,
        },
        "blue_lower_apex": {
            "source_label": "blue_lower_apex",
            "point": blue_lower["extreme_points"]["bottom"],
            "construction": "lowest point of the dominant blue lower wedge contour",
            "confidence": 0.95,
        },
        "green_left_peak": {
            "source_label": "green_peak",
            "point": component_packet["green_left_circle"]["extreme_points"]["top"],
            "construction": "top point of the dominant left green circle",
            "confidence": 0.84,
        },
    }

    return {
        "witness_id": "cch_archm1b_labeled_witness",
        "asset_path": str(ASSET_PATH),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "component_packet": component_packet,
        "line_packet": line_packet,
        "labeled_points": labeled_points,
        "source_recovery_notes": [
            "This witness is stricter than the earlier circle-center packet: it preserves distinct dominant components rather than merging across guide lines.",
            "Only a small subset of source letters are promoted as anchors, and each anchor carries an explicit confidence because the raster still mixes fills, labels, and construction lines.",
            "The two central vertical proxies and the blue wedge edge proxies are intended as test inputs for the next QA/RT pass, not as theorem claims by themselves.",
        ],
        "recommended_next_tests": [
            "Use the blue wedge edge proxies plus G and the vertical proxies to test exact spreads before touching any law promotion.",
            "Compare the corrected red-lower D/G band against the Sixto stage-frame map; the earlier red-primary circle center was not stable enough for law work.",
            "Transcribe one more Sixto witness with cleaner linework if exact source letters beyond D/G are needed.",
        ],
    }


def self_test() -> int:
    ok = True
    triangle = np.array([[[0, 0]], [[10, 0]], [[0, 10]]], dtype=np.int32)
    summary = contour_summary(triangle, approx_eps=0.01)
    ok = ok and summary["bbox"] == [0, 0, 11, 11]
    band = contour_band_points(triangle, y_value=0, tolerance=0)
    ok = ok and band["left"] == [0, 0]
    ok = ok and band["right"] == [10, 0]
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
                "outputs": ["pythagoras_quantum_world_rt/cch_archm1b_labeled_witness.json"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
