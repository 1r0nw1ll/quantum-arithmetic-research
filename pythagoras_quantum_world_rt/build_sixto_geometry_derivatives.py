#!/usr/bin/env python3
"""Build CCH primitive vectorization and the Sixto H/N/K stage graph from local source assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
ASSET_DIR = Path("/tmp/sixto_assets")
CCH_PATH = ASSET_DIR / "cch-archm1b.png"
CHAIN_PATH = ASSET_DIR / "sioxto3002b.jpg"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def component_entries(mask: np.ndarray, area_floor: float) -> list[dict[str, object]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    entries = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_floor:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        bx, by, bw, bh = cv2.boundingRect(contour)
        entries.append(
            {
                "area": round(float(area), 2),
                "bbox": [int(bx), int(by), int(bw), int(bh)],
                "center": [round(float(cx), 2), round(float(cy), 2)],
                "radius": round(float(radius), 2),
            }
        )
    entries.sort(key=lambda entry: entry["area"], reverse=True)
    return entries


def extract_cch_vectorization() -> dict[str, object]:
    image = cv2.imread(str(CCH_PATH))
    if image is None:
        raise FileNotFoundError(CCH_PATH)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, np.array((0, 100, 100)), np.array((10, 255, 255))),
            cv2.inRange(hsv, np.array((170, 100, 100)), np.array((180, 255, 255))),
        ),
        "green": cv2.inRange(hsv, np.array((35, 70, 50)), np.array((90, 255, 255))),
        "blue": cv2.inRange(hsv, np.array((95, 70, 50)), np.array((140, 255, 255))),
        "orange": cv2.inRange(hsv, np.array((8, 100, 100)), np.array((30, 255, 255))),
    }
    components = {name: component_entries(mask, 500.0) for name, mask in masks.items()}

    red_mask = masks["red"]
    lines = cv2.HoughLinesP(
        red_mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=100,
        minLineLength=400,
        maxLineGap=20,
    )
    axis_line = None
    if lines is not None:
        best = None
        best_length = -1.0
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            length = ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) ** 0.5
            if length > best_length:
                best_length = length
                best = {"p1": [x1, y1], "p2": [x2, y2], "length": round(length, 2)}
        axis_line = best

    return {
        "vectorization_id": "cch_archm1b_vectorization",
        "asset_path": str(CCH_PATH),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "pixel_frame_primitives": {
            "left_red_primary_circle": components["red"][1],
            "left_red_lower_circle": components["red"][0],
            "upper_green_circle": components["green"][0],
            "upper_blue_circle": components["blue"][2],
            "central_orange_circle": components["orange"][0],
            "right_blue_wedge_component": {
                "bbox": components["blue"][1]["bbox"],
                "center": components["blue"][1]["center"],
            },
            "dominant_red_axis_line": axis_line,
        },
        "extraction_notes": [
            "Circle centers and radii are exact in image-pixel coordinates for the chosen mask-fit components.",
            "The blue right-hand wedge is recorded as a dominant component bounding box rather than a closed polygon because the raster fill is not a simple isolated triangle.",
        ],
    }


def extract_stage_graph() -> dict[str, object]:
    image = cv2.imread(str(CHAIN_PATH))
    if image is None:
        raise FileNotFoundError(CHAIN_PATH)
    full_h, full_w = image.shape[:2]
    y0 = int(full_h * 0.72)
    bottom = image[y0:full_h, :]
    x1 = int(bottom.shape[1] * 0.12)
    y1 = int(bottom.shape[0] * 0.08)
    chain = bottom[y1:bottom.shape[0], x1:bottom.shape[1]]
    gray = cv2.cvtColor(chain, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=500,
        param1=100,
        param2=30,
        minRadius=180,
        maxRadius=320,
    )
    if circles is None or len(circles[0]) < 4:
        raise RuntimeError("Unable to recover four stage circles from Sixto chain crop.")
    circle_rows = []
    for x, y, r in sorted(circles[0], key=lambda entry: entry[0])[:4]:
        circle_rows.append(
            {
                "center_crop": [round(float(x), 2), round(float(y), 2)],
                "center_global": [round(float(x + x1), 2), round(float(y + y0 + y1), 2)],
                "radius": round(float(r), 2),
            }
        )
    stages = []
    for idx, circle in enumerate(circle_rows, start=1):
        partition_type = {
            1: "vertical_diameter_split",
            2: "left_to_internal_apex_chevron",
            3: "left_to_internal_apex_chevron",
            4: "left_to_internal_apex_chevron",
        }[idx]
        stages.append(
            {
                "stage_id": idx,
                "k_label": f"K{idx}",
                "h_label": f"H{idx}",
                "n_label": f"N{idx}",
                "circle": circle,
                "partition_type": partition_type,
                "rotation_arc_color": "red",
            }
        )
    return {
        "graph_id": "sixto_stage_graph",
        "asset_path": str(CHAIN_PATH),
        "crop_offsets": {
            "chain_crop_x": x1,
            "chain_crop_y": y0 + y1,
        },
        "global_image_size": [full_w, full_h],
        "rules": {
            "k_rule": "K1 = Distance = Circumference/2; K2 = K1 x 0.62; K3 = K2 x 0.62; K4 = K3 x 0.62",
            "time_rule": "H1 = H2 = H3 = H4 = TimeH; N1 = N2 = N3 = N4 = TimeN; H + N = 1 Revolution in time",
        },
        "stages": stages,
        "edges": [
            {"from_stage": 1, "to_stage": 2, "edge_type": "single_diagonal_transfer"},
            {"from_stage": 2, "to_stage": 3, "edge_type": "single_diagonal_transfer"},
            {"from_stage": 3, "to_stage": 4, "edge_type": "single_diagonal_transfer"},
        ],
    }


def self_test() -> int:
    ok = True
    sample_components = component_entries(np.zeros((10, 10), dtype=np.uint8), 1.0)
    ok = ok and sample_components == []
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    cch_payload = extract_cch_vectorization()
    stage_payload = extract_stage_graph()
    write_json(OUT_DIR / "cch_archm1b_vectorization.json", cch_payload)
    write_json(OUT_DIR / "sixto_stage_graph.json", stage_payload)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    "pythagoras_quantum_world_rt/cch_archm1b_vectorization.json",
                    "pythagoras_quantum_world_rt/sixto_stage_graph.json",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
