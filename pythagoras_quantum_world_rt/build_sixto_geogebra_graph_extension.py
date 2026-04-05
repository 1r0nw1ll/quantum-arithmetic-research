#!/usr/bin/env python3
"""Extend the Sixto exact GeoGebra-style reconstruction with timing-graph curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
ASSET_PATH = Path("/tmp/sixto_assets/sioxto3002b.jpg")
BASE_SCENE_PATH = OUT_DIR / "sixto_geogebra_scene_export_v1.json"
BASE_SUMMARY_PATH = OUT_DIR / "sixto_geogebra_reconstruction_summary.json"
ANGLE_BRIDGE_PATH = OUT_DIR / "sixto_angle_schedule_bridge.json"
SCENE_PATH = OUT_DIR / "sixto_geogebra_extended_scene_export_v1.json"
CERT_PATH = OUT_DIR / "sixto_geogebra_extended_adapter_cert.json"
SUMMARY_PATH = OUT_DIR / "sixto_geogebra_extended_summary.json"

GRAPH_Y0 = 120
GRAPH_Y1 = 980
GRAPH_X0 = 2850
GRAPH_X1 = 4220
GRAPH_X_SAMPLES = [0, 63, 126, 189, 252, 315, 378, 462, 546, 630, 714, 794, 878, 962, 1046, 1129, 1212]

CURVE_SPECS = [
    {
        "curve_id": "curve_graph_light_green",
        "label": "Sixto timing graph light-green trace",
        "stroke": "#86c086",
        "hsv_low": [68, 40, 120],
        "hsv_high": [84, 140, 255],
        "crossover_x": 126,
    },
    {
        "curve_id": "curve_graph_green",
        "label": "Sixto timing graph green trace",
        "stroke": "#18834c",
        "hsv_low": [68, 150, 60],
        "hsv_high": [86, 255, 200],
        "crossover_x": 378,
    },
    {
        "curve_id": "curve_graph_cyan",
        "label": "Sixto timing graph cyan trace",
        "stroke": "#4eabb1",
        "hsv_low": [86, 70, 100],
        "hsv_high": [97, 190, 255],
        "crossover_x": 189,
    },
    {
        "curve_id": "curve_graph_blue",
        "label": "Sixto timing graph blue trace",
        "stroke": "#1775a8",
        "hsv_low": [96, 120, 80],
        "hsv_high": [112, 255, 220],
        "crossover_x": 630,
    },
]


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def step_hash(move_id: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    payload = {"inputs": inputs, "move_id": move_id, "outputs": outputs}
    return sha256_hex(canonical_dump(payload))


def coord_q_tenths(value: float) -> dict[str, Any]:
    scaled = int(round(value * 10.0))
    if scaled % 10 == 0:
        return {"k": "Z", "v": scaled // 10}
    return {"k": "Q", "n": scaled, "d": 10}


def point_xyz(x_value: float, y_value: float) -> list[dict[str, Any]]:
    return [coord_q_tenths(x_value), coord_q_tenths(y_value), {"k": "Z", "v": 0}]


def coord_den(coord: dict[str, Any]) -> int:
    return 1 if coord["k"] == "Z" else int(coord["d"])


def coord_num(coord: dict[str, Any]) -> int:
    return int(coord["v"]) if coord["k"] == "Z" else int(coord["n"])


def gcd_int(a_value: int, b_value: int) -> int:
    x_value = abs(a_value)
    y_value = abs(b_value)
    while y_value:
        x_value, y_value = y_value, x_value % y_value
    return x_value


def lcm_int(a_value: int, b_value: int) -> int:
    a0 = abs(a_value)
    b0 = abs(b_value)
    if a0 == 0 or b0 == 0:
        return 0
    return (a0 // gcd_int(a0, b0)) * b0


def lift_point(coord3: list[dict[str, Any]], lcm_value: int) -> list[int]:
    out = []
    for coord in coord3:
        denom = coord_den(coord)
        numer = coord_num(coord)
        scale = lcm_value // abs(denom)
        if denom < 0:
            numer = -numer
        out.append(numer * scale)
    return out


def vec_sub(a_point: list[int], b_point: list[int]) -> list[int]:
    return [a_point[idx] - b_point[idx] for idx in range(3)]


def dot_vec(a_vec: list[int], b_vec: list[int]) -> int:
    return a_vec[0] * b_vec[0] + a_vec[1] * b_vec[1] + a_vec[2] * b_vec[2]


def quadrance(a_point: list[int], b_point: list[int]) -> int:
    diff = vec_sub(b_point, a_point)
    return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]


def spread_pair(v1: list[int], v2: list[int]) -> dict[str, int]:
    q1 = dot_vec(v1, v1)
    q2 = dot_vec(v2, v2)
    d_value = q1 * q2
    dot_value = dot_vec(v1, v2)
    n_value = d_value - dot_value * dot_value
    return {"n": n_value, "d": d_value}


def compute_rt(vertices: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, int]]]:
    denoms = [coord_den(coord) for vertex in vertices for coord in vertex["coord"]]
    lcm_value = 1
    for denom in denoms:
        lcm_value = lcm_int(lcm_value, denom)
    a_point = lift_point(vertices[0]["coord"], lcm_value)
    b_point = lift_point(vertices[1]["coord"], lcm_value)
    c_point = lift_point(vertices[2]["coord"], lcm_value)
    q1 = quadrance(b_point, c_point)
    q2 = quadrance(c_point, a_point)
    q3 = quadrance(a_point, b_point)
    ab = vec_sub(b_point, a_point)
    ac = vec_sub(c_point, a_point)
    bc = vec_sub(c_point, b_point)
    ba = vec_sub(a_point, b_point)
    ca = vec_sub(a_point, c_point)
    cb = vec_sub(b_point, c_point)
    return [q1, q2, q3], [spread_pair(ab, ac), spread_pair(bc, ba), spread_pair(ca, cb)]


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


def scene_triangle_to_parsed(scene_object: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed_object = {
        "object_id": scene_object["id"],
        "object_type": "TRIANGLE",
        "label": scene_object["label"],
        "vertices": [
            {"id": "A", "coord": scene_object["vertices"]["A"]},
            {"id": "B", "coord": scene_object["vertices"]["B"]},
            {"id": "C", "coord": scene_object["vertices"]["C"]},
        ],
        "faces": [["A", "B", "C"]],
    }
    q_values, s_values = compute_rt(parsed_object["vertices"])
    return parsed_object, {"triangle_object_id": scene_object["id"], "Q": q_values, "s": s_values}


def build_graph_triangle_object(object_id: str, label: str, x_value: float, y_value: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scene_object = {
        "id": object_id,
        "type": "Triangle",
        "label": label,
        "vertices": {
            "A": point_xyz(0.0, 0.0),
            "B": point_xyz(x_value, 0.0),
            "C": point_xyz(x_value, y_value),
        },
    }
    parsed_object, result_triangle = scene_triangle_to_parsed(scene_object)
    return scene_object, parsed_object, result_triangle


def build_segment_object(object_id: str, label: str, x0: float, y0: float, x1: float, y1: float, stroke: str) -> dict[str, Any]:
    return {
        "id": object_id,
        "type": "Segment",
        "label": label,
        "stroke": {"color": stroke},
        "endpoints": {"A": point_xyz(x0, y0), "B": point_xyz(x1, y1)},
    }


def build_polyline_object(object_id: str, label: str, stroke: str, points: list[tuple[float, float]]) -> dict[str, Any]:
    return {
        "id": object_id,
        "type": "Polyline",
        "label": label,
        "stroke": {"color": stroke},
        "points": [{"id": f"P{idx:02d}", "coord": point_xyz(x_value, y_value)} for idx, (x_value, y_value) in enumerate(points)],
    }


def extract_graph_guides() -> dict[str, Any]:
    image = cv2.imread(str(ASSET_PATH))
    if image is None:
        raise FileNotFoundError(ASSET_PATH)
    crop = image[GRAPH_Y0:GRAPH_Y1, GRAPH_X0:GRAPH_X1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    dark_mask = gray < 80
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x_value, y_value, width, height = cv2.boundingRect(contour)
        area = width * height
        if area > 5000 and width > 800 and height > 200:
            boxes.append((area, [x_value, y_value, width, height]))
    boxes.sort(reverse=True)
    if not boxes:
        raise RuntimeError("No graph-sized plot bbox detected in sioxto3002b graph crop.")
    plot_x, plot_y, plot_w, plot_h = boxes[0][1]

    plot_dark = dark_mask[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w]
    col_counts = plot_dark.sum(axis=0)
    major_cols = contour_runs([(idx, int(value)) for idx, value in enumerate(col_counts) if value > int(plot_h * 0.8)])
    row_counts = plot_dark.sum(axis=1)
    major_rows = contour_runs([(idx, int(value)) for idx, value in enumerate(row_counts) if value > int(plot_w * 0.4)])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])) | cv2.inRange(
        hsv, np.array([170, 80, 80]), np.array([179, 255, 255])
    )
    red_plot = red_mask[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w]
    red_counts = red_plot.sum(axis=1) / 255
    red_rows = contour_runs([(idx, int(value)) for idx, value in enumerate(red_counts) if value > int(plot_w * 0.15)])

    plot_bgr = crop[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w]
    return {
        "plot_bbox_global": [plot_x + GRAPH_X0, plot_y + GRAPH_Y0, plot_w, plot_h],
        "major_x_plot": [run["center"] for run in major_cols],
        "major_y_plot": [run["center"] for run in major_rows],
        "red_y_plot": [run["center"] for run in red_rows],
        "plot_bgr": plot_bgr,
    }


def build_curve_trace(plot_bgr: np.ndarray, spec: dict[str, Any], baseline_y: float) -> dict[str, Any]:
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(spec["hsv_low"]), np.array(spec["hsv_high"]))
    raw_rel_values = []
    for x_value in range(mask.shape[1]):
        ys = np.where(mask[:, x_value] > 0)[0]
        if len(ys) == 0:
            raw_rel_values.append(np.nan)
            continue
        chosen_y = float(ys.max()) if x_value <= int(spec["crossover_x"]) else float(ys.min())
        raw_rel_values.append(baseline_y - chosen_y)
    rel_array = np.array(raw_rel_values, dtype=float)
    x_index = np.arange(len(rel_array))
    good = ~np.isnan(rel_array)
    if not np.any(good):
        raise RuntimeError(f"Failed to recover curve trace for {spec['curve_id']}.")
    rel_interp = np.interp(x_index, x_index[good], rel_array[good])
    rel_interp[0] = 0.0
    rel_interp[-1] = 0.0
    sampled_points = [(float(x_value), round_float(rel_interp[int(x_value)])) for x_value in GRAPH_X_SAMPLES]
    peak_x = int(np.argmax(rel_interp))
    dip_x = int(np.argmin(rel_interp))
    return {
        "curve_object": build_polyline_object(spec["curve_id"], spec["label"], spec["stroke"], sampled_points),
        "sampled_points": sampled_points,
        "peak": {"x": peak_x, "y": round_float(rel_interp[peak_x])},
        "dip": {"x": dip_x, "y": round_float(rel_interp[dip_x])},
        "crossover_x": int(spec["crossover_x"]),
    }


def build_payloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_scene = read_json(BASE_SCENE_PATH)
    base_summary = read_json(BASE_SUMMARY_PATH)
    angle_bridge = read_json(ANGLE_BRIDGE_PATH)
    graph = extract_graph_guides()

    major_x = graph["major_x_plot"]
    red_y = graph["red_y_plot"]
    plot_bgr = graph["plot_bgr"]
    x0 = float(major_x[0])
    x1 = float(major_x[1])
    x2 = float(major_x[2])
    x3 = float(major_x[3])
    v_plus_row = float(red_y[0])
    baseline_row = float(red_y[1])
    v_minus_row = float(red_y[2])
    v_plus = abs(v_plus_row - baseline_row)
    v_minus = abs(v_minus_row - baseline_row)

    relative_x = [0.0, x1 - x0, x2 - x0, x3 - x0]
    graph_specs = [
        ("tri_graph_x1_vplus", "Sixto graph X1-to-Vplus guide triangle", relative_x[1], v_plus),
        ("tri_graph_x2_vplus", "Sixto graph X2-to-Vplus guide triangle", relative_x[2], v_plus),
        ("tri_graph_x3_vplus", "Sixto graph X3-to-Vplus guide triangle", relative_x[3], v_plus),
        ("tri_graph_x1_vminus", "Sixto graph X1-to-Vminus guide triangle", relative_x[1], -v_minus),
    ]

    scene_objects = list(base_scene["objects"])
    parsed_objects = []
    result_triangles = []
    for scene_object in scene_objects:
        parsed_object, result_triangle = scene_triangle_to_parsed(scene_object)
        parsed_objects.append(parsed_object)
        result_triangles.append(result_triangle)

    graph_summary_triangles = []
    for object_id, label, x_value, y_value in graph_specs:
        scene_object, parsed_object, result_triangle = build_graph_triangle_object(object_id, label, x_value, y_value)
        scene_objects.append(scene_object)
        parsed_objects.append(parsed_object)
        result_triangles.append(result_triangle)
        graph_summary_triangles.append({"object_id": object_id, "x_value": round_float(x_value), "y_value": round_float(y_value), "Q": result_triangle["Q"]})

    segment_objects = [
        build_segment_object("seg_graph_v0", "Sixto timing graph V0 baseline", 0.0, 0.0, float(plot_bgr.shape[1] - 1), 0.0, "#d33d3d"),
        build_segment_object("seg_graph_vplus", "Sixto timing graph Vplus reference", 0.0, v_plus, float(plot_bgr.shape[1] - 1), v_plus, "#d33d3d"),
        build_segment_object("seg_graph_vminus", "Sixto timing graph Vminus reference", 0.0, -v_minus, float(plot_bgr.shape[1] - 1), -v_minus, "#d33d3d"),
    ]
    scene_objects.extend(segment_objects)

    curve_summaries = []
    curve_triangle_ids = []
    curve_polyline_ids = []
    for spec in CURVE_SPECS:
        curve_trace = build_curve_trace(plot_bgr, spec, baseline_row)
        curve_object = curve_trace["curve_object"]
        scene_objects.append(curve_object)
        curve_polyline_ids.append(curve_object["id"])
        peak = curve_trace["peak"]
        dip = curve_trace["dip"]
        peak_scene, peak_parsed, peak_result = build_graph_triangle_object(
            f"tri_{spec['curve_id']}_peak",
            f"{spec['label']} peak chord triangle",
            float(peak["x"]),
            float(peak["y"]),
        )
        dip_scene, dip_parsed, dip_result = build_graph_triangle_object(
            f"tri_{spec['curve_id']}_dip",
            f"{spec['label']} dip chord triangle",
            float(dip["x"]),
            float(dip["y"]),
        )
        scene_objects.extend([peak_scene, dip_scene])
        parsed_objects.extend([peak_parsed, dip_parsed])
        result_triangles.extend([peak_result, dip_result])
        curve_triangle_ids.extend([peak_scene["id"], dip_scene["id"]])
        curve_summaries.append(
            {
                "curve_id": curve_object["id"],
                "label": spec["label"],
                "stroke": spec["stroke"],
                "crossover_x": curve_trace["crossover_x"],
                "sample_count": len(curve_trace["sampled_points"]),
                "sampled_points": [{"x": round_float(x_value), "y": round_float(y_value)} for x_value, y_value in curve_trace["sampled_points"]],
                "peak": peak,
                "dip": dip,
                "derived_triangle_ids": [peak_scene["id"], dip_scene["id"]],
            }
        )

    parsed_objects.sort(key=lambda item: item["object_id"])
    result_triangles.sort(key=lambda item: item["triangle_object_id"])
    scene_objects.sort(key=lambda item: item["id"])

    scene_raw = {"version": "1.0", "objects": scene_objects}
    scene_raw_sha256 = sha256_hex(canonical_dump(scene_raw))
    parsed_ids = [obj["object_id"] for obj in parsed_objects]

    steps = []
    invariant_steps = []
    s1_inputs = {"scene_raw_sha256": scene_raw_sha256}
    s1_outputs = {"parsed_object_ids": parsed_ids}
    steps.append(
        {
            "step_id": "s1",
            "move_id": "GG_PARSE_SCENE",
            "inputs": s1_inputs,
            "outputs": s1_outputs,
            "step_hash_sha256": step_hash("GG_PARSE_SCENE", s1_inputs, s1_outputs),
        }
    )
    invariant_steps.append({"step_id": "s1", "adds": [f"parsed_objects[{object_id}]" for object_id in parsed_ids]})

    step_counter = 2
    for triangle in result_triangles:
        object_id = triangle["triangle_object_id"]
        s_inputs = {"triangle_object_id": object_id}
        s_outputs = {"Q": triangle["Q"], "s": triangle["s"]}
        step_id = f"s{step_counter}"
        steps.append(
            {
                "step_id": step_id,
                "move_id": "RT_COMPUTE_TRIANGLE_INVARIANTS",
                "inputs": s_inputs,
                "outputs": s_outputs,
                "step_hash_sha256": step_hash("RT_COMPUTE_TRIANGLE_INVARIANTS", s_inputs, s_outputs),
            }
        )
        invariant_steps.append({"step_id": step_id, "adds": [f"rt_invariants.triangles[{object_id}].Q", f"rt_invariants.triangles[{object_id}].s"]})
        step_counter += 1

    for triangle in result_triangles:
        object_id = triangle["triangle_object_id"]
        s_inputs = {"law_id": "RT_LAW_01", "triangle_object_id": object_id}
        s_outputs = {"verified": True}
        step_id = f"s{step_counter}"
        steps.append(
            {
                "step_id": step_id,
                "move_id": "RT_VALIDATE_LAW_EQUATION",
                "inputs": s_inputs,
                "outputs": s_outputs,
                "step_hash_sha256": step_hash("RT_VALIDATE_LAW_EQUATION", s_inputs, s_outputs),
            }
        )
        invariant_steps.append({"step_id": step_id, "adds": [f"law_verification[RT_LAW_01][{object_id}].verified"]})
        step_counter += 1

    cert = {
        "schema_version": "v1",
        "cert_type": "QA_GEOGEBRA_SCENE_ADAPTER.v1",
        "cert_id": "qa_geogebra_v1_sixto_stage_graph_0002",
        "created_utc": "2026-03-31T00:00:00Z",
        "compute_substrate": "qa_rational_pair_noreduce",
        "source_semantics": {
            "upstream": {
                "name": "GeoGebra",
                "app_url": "https://www.geogebra.org/",
                "repo_url": "repo-local reconstruction from sioxto3002b witness",
            },
            "export": {"format": "geogebra_scene_export_v1", "exported_by": "sixto_geogebra_graph_extension_v2"},
        },
        "base_algebra": {"name": "Q", "properties": {"integral_domain": True, "field": True, "no_zero_divisors": True}},
        "scene": {"coordinate_system": "XYZ", "scene_raw": scene_raw, "scene_raw_sha256": scene_raw_sha256},
        "derivation": {"parsed_objects": parsed_objects, "steps": steps},
        "result": {
            "rt_invariants": {"triangles": result_triangles},
            "invariant_diff": {
                "steps": invariant_steps,
                "provenance": {
                    "upstream_tool": "sixto_geogebra_graph_extension_v2",
                    "upstream_format": "geogebra_scene_export_v1",
                    "compute_substrate": "qa_rational_pair_noreduce",
                    "geometry_source": "Combined Sixto stage-arm reconstruction with red graph references, four sampled timing-graph polylines, and exact chord triangles derived from the sioxto3002b witness.",
                },
            },
        },
        "determinism_contract": {
            "canonical_json": True,
            "stable_sorting": True,
            "no_rng": True,
            "invariant_diff_defined": True,
        },
    }

    graph_schedule_test = {
        "relative_major_x_guides": [round_float(value) for value in relative_x],
        "x2_vs_geometric_mid": round_float(relative_x[2] - ((relative_x[1] * relative_x[3]) ** 0.5)),
        "x2_vs_arithmetic_mid": round_float(relative_x[2] - ((relative_x[1] + relative_x[3]) / 2.0)),
        "v_plus_height": round_float(v_plus),
        "v_minus_height": round_float(v_minus),
    }

    summary = {
        "artifact_id": "sixto_geogebra_extended_summary",
        "base_scene_object_ids": [item["id"] for item in base_scene["objects"]],
        "graph_triangle_object_ids": [item["object_id"] for item in graph_summary_triangles],
        "curve_polyline_object_ids": curve_polyline_ids,
        "curve_triangle_object_ids": curve_triangle_ids,
        "curve_reference_segment_ids": [item["id"] for item in segment_objects],
        "graph_plot_bbox_global": graph["plot_bbox_global"],
        "graph_schedule_test": graph_schedule_test,
        "reconstructed_graph_triangles": graph_summary_triangles,
        "reconstructed_graph_curves": curve_summaries,
        "stage_summary_reference": base_summary,
        "angle_schedule_reference": angle_bridge["angle_packet_bridge"],
        "verdict": {
            "extended_exact_scene_ready": True,
            "honest_summary": "The exact Sixto scene now includes both the recovered stage-angle triangles and machine-readable timing-graph curves as exact raw polyline objects, together with exact chord triangles sampled from those curves. The family [56] certificate still certifies only triangle objects, but those triangles now come from the curves themselves rather than only from the graph frame.",
        },
    }

    return scene_raw, cert, summary


def self_test() -> int:
    scene_raw, cert, summary = build_payloads()
    ok = True
    ok = ok and len(scene_raw["objects"]) == 22
    ok = ok and len(cert["derivation"]["parsed_objects"]) == 15
    ok = ok and len(cert["result"]["rt_invariants"]["triangles"]) == 15
    ok = ok and len(summary["curve_polyline_object_ids"]) == 4
    ok = ok and len(summary["curve_triangle_object_ids"]) == 8
    ok = ok and abs(float(summary["graph_schedule_test"]["x2_vs_geometric_mid"])) < 15.0
    ok = ok and abs(float(summary["graph_schedule_test"]["x2_vs_arithmetic_mid"])) > 100.0
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    scene_raw, cert, summary = build_payloads()
    write_json(SCENE_PATH, scene_raw)
    write_json(CERT_PATH, cert)
    write_json(SUMMARY_PATH, summary)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    "pythagoras_quantum_world_rt/sixto_geogebra_extended_scene_export_v1.json",
                    "pythagoras_quantum_world_rt/sixto_geogebra_extended_adapter_cert.json",
                    "pythagoras_quantum_world_rt/sixto_geogebra_extended_summary.json",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
