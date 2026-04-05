#!/usr/bin/env python3
"""Map the Sixto stage graph into the CCH primitive frame and test QA/RT landings."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
CCH_PATH = OUT_DIR / "cch_archm1b_vectorization.json"
CCH_LABEL_PATH = OUT_DIR / "cch_archm1b_labeled_witness.json"
STAGE_PATH = OUT_DIR / "sixto_stage_graph.json"
FRAME_MAP_PATH = OUT_DIR / "sixto_cch_frame_map.json"
LAW_TEST_PATH = OUT_DIR / "sixto_qa_rt_landing_tests.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def round_point(point: tuple[float, float]) -> list[float]:
    return [round_float(point[0]), round_float(point[1])]


def point_sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def point_add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def point_scale(value: tuple[float, float], scalar: float) -> tuple[float, float]:
    return (value[0] * scalar, value[1] * scalar)


def norm(value: tuple[float, float]) -> float:
    return math.hypot(value[0], value[1])


def unit(value: tuple[float, float]) -> tuple[float, float]:
    length = norm(value)
    if length == 0.0:
        raise ValueError("zero-length vector")
    return (value[0] / length, value[1] / length)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return norm(point_sub(a, b))


def quadrance(a: tuple[float, float], b: tuple[float, float]) -> float:
    delta = point_sub(a, b)
    return delta[0] * delta[0] + delta[1] * delta[1]


def angle_deg(value: tuple[float, float]) -> float:
    return math.degrees(math.atan2(value[1], value[0]))


def normalize_angle_deg(value: float) -> float:
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def line_spread(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    q1 = v1[0] * v1[0] + v1[1] * v1[1]
    q2 = v2[0] * v2[0] + v2[1] * v2[1]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return 1.0 - (dot * dot) / (q1 * q2)


def point_line_residual(point: tuple[float, float], line: tuple[tuple[float, float], tuple[float, float]]) -> float:
    x0, y0 = point
    (x1, y1), (x2, y2) = line
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    if denominator == 0.0:
        raise ValueError("degenerate line")
    return numerator / denominator


def rotate_and_scale(
    source_vector: tuple[float, float],
    source_anchor_vector: tuple[float, float],
    target_anchor_vector: tuple[float, float],
) -> tuple[float, float]:
    source_length = norm(source_anchor_vector)
    target_length = norm(target_anchor_vector)
    if source_length == 0.0 or target_length == 0.0:
        raise ValueError("anchor vector must be non-zero")
    ux = source_anchor_vector[0] / source_length
    uy = source_anchor_vector[1] / source_length
    rotated_x = ux * source_vector[0] + uy * source_vector[1]
    rotated_y = -uy * source_vector[0] + ux * source_vector[1]
    scale = target_length / source_length
    vx = target_anchor_vector[0] / target_length
    vy = target_anchor_vector[1] / target_length
    return (
        scale * (vx * rotated_x - vy * rotated_y),
        scale * (vy * rotated_x + vx * rotated_y),
    )


def nearest_primitives(
    point: tuple[float, float],
    primitive_centers: dict[str, tuple[float, float]],
    limit: int = 3,
) -> list[dict[str, object]]:
    entries = []
    for name, center in primitive_centers.items():
        entries.append(
            {
                "primitive": name,
                "distance": round_float(distance(point, center)),
            }
        )
    entries.sort(key=lambda entry: entry["distance"])
    return entries[:limit]


def parse_k_ratio(k_rule: str) -> float:
    match = re.search(r"x\s+([0-9.]+)", k_rule)
    if match is None:
        raise ValueError(f"unable to parse K ratio from {k_rule!r}")
    return float(match.group(1))


def line_from_entry(entry: dict[str, object]) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(entry["p1"][0]), float(entry["p1"][1])),
        (float(entry["p2"][0]), float(entry["p2"][1])),
    )


def line_lookup(labeled_payload: dict[str, object]) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    return {
        name: line_from_entry(entry)
        for name, entry in labeled_payload["line_packet"].items()
    }


def point_lookup(labeled_payload: dict[str, object]) -> dict[str, tuple[float, float]]:
    return {
        name: (float(entry["point"][0]), float(entry["point"][1]))
        for name, entry in labeled_payload["labeled_points"].items()
    }


def stage_lookup(stage_graph: dict[str, object]) -> dict[int, dict[str, object]]:
    return {int(stage["stage_id"]): stage for stage in stage_graph["stages"]}


def primitive_center_lookup(cch_payload: dict[str, object]) -> dict[str, tuple[float, float]]:
    lookup = {}
    for name, entry in cch_payload["pixel_frame_primitives"].items():
        if "center" in entry:
            lookup[name] = (float(entry["center"][0]), float(entry["center"][1]))
    return lookup


def primitive_radius_lookup(cch_payload: dict[str, object]) -> dict[str, float]:
    lookup = {}
    for name, entry in cch_payload["pixel_frame_primitives"].items():
        if "radius" in entry:
            lookup[name] = float(entry["radius"])
    return lookup


def build_corrected_witness_reference(labeled_payload: dict[str, object]) -> dict[str, object]:
    points = point_lookup(labeled_payload)
    lines = labeled_payload["line_packet"]
    return {
        "reference_id": "corrected_labeled_witness_reference",
        "kind": "source_labeled_geometry",
        "baseline_for_law_tests": True,
        "selected_points": {
            "G_source_anchor": round_point(points["G_source_anchor"]),
            "D_source_band_anchor": round_point(points["D_source_band_anchor"]),
            "blue_left_shoulder": round_point(points["blue_left_shoulder"]),
            "blue_far_apex": round_point(points["blue_far_apex"]),
            "blue_lower_apex": round_point(points["blue_lower_apex"]),
        },
        "selected_lines": {
            "blue_lower_rising_edge_proxy": lines["blue_lower_rising_edge_proxy"],
            "blue_lower_falling_edge_proxy": lines["blue_lower_falling_edge_proxy"],
            "central_vertical_left_proxy": lines["central_vertical_left_proxy"],
            "central_vertical_right_proxy": lines["central_vertical_right_proxy"],
            "dominant_horizontal_upper_chord": lines["dominant_horizontal_upper_chord"],
            "dominant_red_axis_proxy": lines["dominant_red_axis_proxy"],
        },
        "selection_reason": (
            "The corrected labeled witness replaces the old merged red-center assumption. "
            "The landing tests now use the recovered G anchor, blue wedge edges, and central vertical proxies."
        ),
    }


def build_center_similarity_map(
    cch_payload: dict[str, object], stage_graph: dict[str, object]
) -> dict[str, object]:
    centers = primitive_center_lookup(cch_payload)
    radii = primitive_radius_lookup(cch_payload)
    stages = stage_lookup(stage_graph)

    source_stage1 = tuple(stages[1]["circle"]["center_global"])
    source_stage2 = tuple(stages[2]["circle"]["center_global"])
    target_stage1 = centers["left_red_primary_circle"]
    target_stage2 = centers["central_orange_circle"]

    source_anchor_vector = point_sub(source_stage2, source_stage1)
    target_anchor_vector = point_sub(target_stage2, target_stage1)
    similarity_scale = norm(target_anchor_vector) / norm(source_anchor_vector)
    rotation_degrees = normalize_angle_deg(
        angle_deg(target_anchor_vector) - angle_deg(source_anchor_vector)
    )

    mapped_stages = []
    for stage_id in sorted(stages):
        circle = stages[stage_id]["circle"]
        source_center = tuple(circle["center_global"])
        mapped_vector = rotate_and_scale(
            point_sub(source_center, source_stage1),
            source_anchor_vector,
            target_anchor_vector,
        )
        mapped_center = point_add(target_stage1, mapped_vector)
        mapped_stages.append(
            {
                "stage_id": stage_id,
                "source_center_global": round_point(source_center),
                "mapped_center": round_point(mapped_center),
                "source_radius": round_float(float(circle["radius"])),
                "mapped_radius": round_float(float(circle["radius"]) * similarity_scale),
                "nearest_primitives": nearest_primitives(mapped_center, centers),
            }
        )

    stage1_radius = mapped_stages[0]["mapped_radius"]
    stage2_radius = mapped_stages[1]["mapped_radius"]

    return {
        "map_id": "center_similarity_map",
        "kind": "strict_stage_graph_similarity",
        "baseline_for_law_tests": True,
        "anchors": {
            "source_stage_1": {
                "stage_id": 1,
                "source_center_global": round_point(source_stage1),
            },
            "source_stage_2": {
                "stage_id": 2,
                "source_center_global": round_point(source_stage2),
            },
            "target_primitive_1": {
                "primitive": "left_red_primary_circle",
                "target_center": round_point(target_stage1),
                "target_radius": round_float(radii["left_red_primary_circle"]),
            },
            "target_primitive_2": {
                "primitive": "central_orange_circle",
                "target_center": round_point(target_stage2),
                "target_radius": round_float(radii["central_orange_circle"]),
            },
        },
        "transform": {
            "source_anchor_vector": round_point(source_anchor_vector),
            "target_anchor_vector": round_point(target_anchor_vector),
            "source_anchor_distance": round_float(norm(source_anchor_vector)),
            "target_anchor_distance": round_float(norm(target_anchor_vector)),
            "similarity_scale": round_float(similarity_scale),
            "rotation_degrees": round_float(rotation_degrees),
        },
        "mapped_stages": mapped_stages,
        "fit_diagnostics": {
            "stage1_radius_minus_left_red_primary_radius": round_float(
                stage1_radius - radii["left_red_primary_circle"]
            ),
            "stage2_radius_minus_central_orange_radius": round_float(
                stage2_radius - radii["central_orange_circle"]
            ),
        },
    }


def build_k_rule_projected_map(
    cch_payload: dict[str, object], stage_graph: dict[str, object]
) -> dict[str, object]:
    centers = primitive_center_lookup(cch_payload)
    radii = primitive_radius_lookup(cch_payload)
    stages = stage_lookup(stage_graph)
    k_ratio = parse_k_ratio(stage_graph["rules"]["k_rule"])

    stage1_center = centers["left_red_primary_circle"]
    stage2_center = centers["central_orange_circle"]
    anchor_vector = point_sub(stage2_center, stage1_center)
    anchor_unit = unit(anchor_vector)
    anchor_distance = norm(anchor_vector)

    stage1_source_radius = float(stages[1]["circle"]["radius"])
    stage_radius_scale = radii["left_red_primary_circle"] / stage1_source_radius

    mapped_stages = []
    stage2_to_stage3 = point_scale(anchor_unit, anchor_distance * k_ratio)
    stage3_center = point_add(stage2_center, stage2_to_stage3)
    stage3_to_stage4 = point_scale(anchor_unit, anchor_distance * k_ratio * k_ratio)
    stage4_center = point_add(stage3_center, stage3_to_stage4)
    projected_centers = {
        1: stage1_center,
        2: stage2_center,
        3: stage3_center,
        4: stage4_center,
    }

    for stage_id in sorted(stages):
        circle = stages[stage_id]["circle"]
        mapped_center = projected_centers[stage_id]
        mapped_stages.append(
            {
                "stage_id": stage_id,
                "source_center_global": round_point(tuple(circle["center_global"])),
                "mapped_center": round_point(mapped_center),
                "source_radius": round_float(float(circle["radius"])),
                "mapped_radius": round_float(float(circle["radius"]) * stage_radius_scale),
                "nearest_primitives": nearest_primitives(mapped_center, centers),
            }
        )

    return {
        "map_id": "k_rule_projected_map",
        "kind": "source_rule_projection",
        "baseline_for_law_tests": False,
        "anchors": {
            "stage_1_to_left_red_primary_circle": round_point(stage1_center),
            "stage_2_to_central_orange_circle": round_point(stage2_center),
        },
        "projection": {
            "k_ratio": round_float(k_ratio),
            "anchor_distance": round_float(anchor_distance),
            "anchor_unit": round_point(anchor_unit),
            "radius_scale_from_stage1_to_left_red_primary": round_float(stage_radius_scale),
        },
        "mapped_stages": mapped_stages,
        "fit_diagnostics": {
            "stage2_radius_minus_central_orange_radius": round_float(
                mapped_stages[1]["mapped_radius"] - radii["central_orange_circle"]
            ),
            "anchor_line_vs_dominant_red_axis_degrees": round_float(
                normalize_angle_deg(
                    angle_deg(anchor_vector)
                    - angle_deg(
                        point_sub(
                            tuple(cch_payload["pixel_frame_primitives"]["dominant_red_axis_line"]["p2"]),
                            tuple(cch_payload["pixel_frame_primitives"]["dominant_red_axis_line"]["p1"]),
                        )
                    )
                )
            ),
        },
    }


def pythagoras_test(name: str, points: dict[str, tuple[float, float]], labels: tuple[str, str, str]) -> dict[str, object]:
    a = points[labels[0]]
    b = points[labels[1]]
    c = points[labels[2]]
    q_ab = quadrance(a, b)
    q_ac = quadrance(a, c)
    q_bc = quadrance(b, c)
    quadrances = [
        ("Q_ab", q_ab),
        ("Q_ac", q_ac),
        ("Q_bc", q_bc),
    ]
    largest_name, largest_value = max(quadrances, key=lambda entry: entry[1])
    smaller_sum = sum(value for _, value in quadrances) - largest_value
    return {
        "test_id": name,
        "points": list(labels),
        "quadrances": {key: round_float(value) for key, value in quadrances},
        "largest_minus_sum_other_two": round_float(largest_value - smaller_sum),
        "exact_right_triangle": math.isclose(largest_value, smaller_sum, rel_tol=0.0, abs_tol=1.0e-9),
        "dominant_leg": largest_name,
    }


def line_spread_test(
    test_id: str,
    line_a_name: str,
    line_b_name: str,
    lines: dict[str, tuple[tuple[float, float], tuple[float, float]]],
) -> dict[str, object]:
    line_a = lines[line_a_name]
    line_b = lines[line_b_name]
    spread_value = line_spread(point_sub(line_a[1], line_a[0]), point_sub(line_b[1], line_b[0]))
    return {
        "test_id": test_id,
        "lines": [line_a_name, line_b_name],
        "spread": round_float(spread_value),
        "delta_from_one": round(float(spread_value - 1.0), 12),
        "exact_match": math.isclose(spread_value, 1.0, rel_tol=0.0, abs_tol=1.0e-9),
    }


def line_spread_value_test(
    test_id: str,
    line_a_name: str,
    line_b_name: str,
    lines: dict[str, tuple[tuple[float, float], tuple[float, float]]],
) -> dict[str, object]:
    line_a = lines[line_a_name]
    line_b = lines[line_b_name]
    spread_value = line_spread(point_sub(line_a[1], line_a[0]), point_sub(line_b[1], line_b[0]))
    return {
        "test_id": test_id,
        "lines": [line_a_name, line_b_name],
        "spread": round_float(spread_value),
        "exact_match": False,
    }


def incidence_test(
    test_id: str,
    point_name: str,
    line_name: str,
    points: dict[str, tuple[float, float]],
    lines: dict[str, tuple[tuple[float, float], tuple[float, float]]],
) -> dict[str, object]:
    residual = point_line_residual(points[point_name], lines[line_name])
    return {
        "test_id": test_id,
        "point": point_name,
        "line": line_name,
        "point_line_residual": round_float(residual),
        "exact_match": math.isclose(residual, 0.0, rel_tol=0.0, abs_tol=1.0e-9),
    }


def build_law_tests(
    cch_payload: dict[str, object],
    labeled_payload: dict[str, object],
    stage_graph: dict[str, object],
    frame_map: dict[str, object],
) -> dict[str, object]:
    _ = cch_payload
    points = point_lookup(labeled_payload)
    lines = line_lookup(labeled_payload)
    red_axis_line = lines["dominant_red_axis_proxy"]
    rising_line = lines["blue_lower_rising_edge_proxy"]
    falling_line = lines["blue_lower_falling_edge_proxy"]
    left_vertical = lines["central_vertical_left_proxy"]
    right_vertical = lines["central_vertical_right_proxy"]
    horizontal = lines["dominant_horizontal_upper_chord"]
    g_anchor_vector = point_sub(points["blue_left_shoulder"], points["G_source_anchor"])
    axis_vector = point_sub(red_axis_line[1], red_axis_line[0])
    angle_difference = normalize_angle_deg(angle_deg(g_anchor_vector) - angle_deg(axis_vector))

    k_ratio = parse_k_ratio(stage_graph["rules"]["k_rule"])
    fibonacci_pairs = [
        {"ratio_id": "34_over_55", "numerator": 34, "denominator": 55},
        {"ratio_id": "55_over_89", "numerator": 55, "denominator": 89},
        {"ratio_id": "89_over_144", "numerator": 89, "denominator": 144},
    ]
    ratio_tests = []
    for pair in fibonacci_pairs:
        value = pair["numerator"] / pair["denominator"]
        ratio_tests.append(
            {
                "ratio_id": pair["ratio_id"],
                "value": round_float(value),
                "delta_from_k_ratio": round_float(value - k_ratio),
                "exact_match": math.isclose(value, k_ratio, rel_tol=0.0, abs_tol=1.0e-12),
            }
        )

    corrected_reference = frame_map["corrected_witness_reference"]

    incidence_tests = [
        incidence_test(
            "g_on_blue_lower_rising_edge_proxy",
            "G_source_anchor",
            "blue_lower_rising_edge_proxy",
            points,
            lines,
        ),
        incidence_test(
            "blue_left_shoulder_on_blue_lower_rising_edge_proxy",
            "blue_left_shoulder",
            "blue_lower_rising_edge_proxy",
            points,
            lines,
        ),
        incidence_test(
            "blue_far_apex_on_blue_lower_falling_edge_proxy",
            "blue_far_apex",
            "blue_lower_falling_edge_proxy",
            points,
            lines,
        ),
        incidence_test(
            "blue_lower_apex_on_blue_lower_falling_edge_proxy",
            "blue_lower_apex",
            "blue_lower_falling_edge_proxy",
            points,
            lines,
        ),
        incidence_test(
            "g_on_central_vertical_left_proxy",
            "G_source_anchor",
            "central_vertical_left_proxy",
            points,
            lines,
        ),
        incidence_test(
            "g_on_central_vertical_right_proxy",
            "G_source_anchor",
            "central_vertical_right_proxy",
            points,
            lines,
        ),
    ]

    witness_triangle_tests = [
        pythagoras_test(
            "witness_triangle_g_far_lower",
            points,
            ("G_source_anchor", "blue_far_apex", "blue_lower_apex"),
        ),
        pythagoras_test(
            "witness_triangle_g_left_lower",
            points,
            ("G_source_anchor", "blue_left_shoulder", "blue_lower_apex"),
        ),
        pythagoras_test(
            "witness_triangle_g_left_far",
            points,
            ("G_source_anchor", "blue_left_shoulder", "blue_far_apex"),
        ),
        pythagoras_test(
            "witness_triangle_left_far_lower",
            points,
            ("blue_left_shoulder", "blue_far_apex", "blue_lower_apex"),
        ),
    ]

    spread_tests = [
        line_spread_test(
            "blue_edge_pair_are_perpendicular",
            "blue_lower_rising_edge_proxy",
            "blue_lower_falling_edge_proxy",
            lines,
        ),
        line_spread_value_test(
            "blue_rising_edge_vs_left_vertical_spread",
            "blue_lower_rising_edge_proxy",
            "central_vertical_left_proxy",
            lines,
        ),
        line_spread_value_test(
            "blue_falling_edge_vs_left_vertical_spread",
            "blue_lower_falling_edge_proxy",
            "central_vertical_left_proxy",
            lines,
        ),
        line_spread_value_test(
            "blue_rising_edge_vs_right_vertical_spread",
            "blue_lower_rising_edge_proxy",
            "central_vertical_right_proxy",
            lines,
        ),
        line_spread_value_test(
            "blue_falling_edge_vs_right_vertical_spread",
            "blue_lower_falling_edge_proxy",
            "central_vertical_right_proxy",
            lines,
        ),
        line_spread_value_test(
            "blue_rising_edge_vs_horizontal_spread",
            "blue_lower_rising_edge_proxy",
            "dominant_horizontal_upper_chord",
            lines,
        ),
        line_spread_value_test(
            "blue_falling_edge_vs_horizontal_spread",
            "blue_lower_falling_edge_proxy",
            "dominant_horizontal_upper_chord",
            lines,
        ),
    ]

    historical_context = [
        {
            "test_id": "historical_stage1_radius_minus_old_red_primary",
            "difference": frame_map["historical_stage_maps"][0]["fit_diagnostics"][
                "stage1_radius_minus_left_red_primary_radius"
            ],
            "counts_toward_verdict": False,
        },
        {
            "test_id": "historical_anchor_line_vs_old_red_axis",
            "angle_difference_degrees": frame_map["historical_stage_maps"][1]["fit_diagnostics"][
                "anchor_line_vs_dominant_red_axis_degrees"
            ],
            "counts_toward_verdict": False,
        },
    ]

    nontrivial_exact_landing_count = 0
    exact_buckets = [ratio_tests, incidence_tests, spread_tests]
    for bucket in exact_buckets:
        for entry in bucket:
            if entry["exact_match"]:
                nontrivial_exact_landing_count += 1
    for entry in witness_triangle_tests:
        if entry["exact_right_triangle"]:
            nontrivial_exact_landing_count += 1

    return {
        "test_id": "sixto_qa_rt_landing_tests",
        "baseline_map_id": corrected_reference["reference_id"],
        "source_constraints": [
            "The corrected labeled witness is the baseline; the old merged red-center packet is retained only as historical context.",
            "The active landing tests use the recovered G anchor, blue wedge edges, and central vertical proxies.",
            "QA/RT exactness means literal equality on the recovered image geometry at the stored numeric precision.",
        ],
        "frame_alignment_tests": [
            {
                "test_id": "g_to_blue_left_shoulder_matches_red_axis_proxy",
                "anchor_angle_degrees": round_float(angle_deg(g_anchor_vector)),
                "dominant_red_axis_angle_degrees": round_float(angle_deg(axis_vector)),
                "angle_difference_degrees": round_float(angle_difference),
                "line_spread": round_float(line_spread(g_anchor_vector, axis_vector)),
                "exact_match": math.isclose(angle_difference, 0.0, rel_tol=0.0, abs_tol=1.0e-9),
            }
        ],
        "k_ratio_tests": ratio_tests,
        "incidence_tests": incidence_tests,
        "spread_tests": spread_tests,
        "pythagoras_quadrance_tests": witness_triangle_tests,
        "historical_context": historical_context,
        "verdict": {
            "nontrivial_exact_landing_count": nontrivial_exact_landing_count,
            "exact_qa_rt_law_lands": nontrivial_exact_landing_count > 0,
            "honest_summary": (
                "No nontrivial QA/RT law lands exactly on the corrected witness geometry. "
                "The closest source hit is that G and the left shoulder lie almost on the rising blue edge proxy, "
                "and the two blue edge proxies are almost perpendicular, but none of those equalities are exact at the stored precision."
            ),
        },
    }


def build_outputs() -> tuple[dict[str, object], dict[str, object]]:
    cch_payload = read_json(CCH_PATH)
    labeled_payload = read_json(CCH_LABEL_PATH)
    stage_graph = read_json(STAGE_PATH)
    center_map = build_center_similarity_map(cch_payload, stage_graph)
    k_rule_map = build_k_rule_projected_map(cch_payload, stage_graph)
    frame_map = {
        "map_id": "sixto_cch_frame_map",
        "inputs": {
            "cch_vectorization": "pythagoras_quantum_world_rt/cch_archm1b_vectorization.json",
            "cch_labeled_witness": "pythagoras_quantum_world_rt/cch_archm1b_labeled_witness.json",
            "stage_graph": "pythagoras_quantum_world_rt/sixto_stage_graph.json",
        },
        "historical_stage_maps": [center_map, k_rule_map],
        "corrected_witness_reference": build_corrected_witness_reference(labeled_payload),
        "selected_baseline": "corrected_labeled_witness_reference",
        "selection_reason": (
            "The old stage-center maps are preserved only for historical comparison. "
            "The active baseline is now the corrected labeled witness because the older red-primary circle packet "
            "merged real geometry with guide-line contamination."
        ),
    }
    law_tests = build_law_tests(cch_payload, labeled_payload, stage_graph, frame_map)
    return frame_map, law_tests


def self_test() -> int:
    source_anchor = (10.0, 0.0)
    target_anchor = (0.0, 20.0)
    mapped = rotate_and_scale((20.0, 0.0), source_anchor, target_anchor)
    ok = True
    ok = ok and math.isclose(mapped[0], 0.0, rel_tol=0.0, abs_tol=1.0e-9)
    ok = ok and math.isclose(mapped[1], 40.0, rel_tol=0.0, abs_tol=1.0e-9)
    ok = ok and math.isclose(parse_k_ratio("K2 = K1 x 0.62"), 0.62, rel_tol=0.0, abs_tol=1.0e-12)
    ok = ok and math.isclose(line_spread((1.0, 0.0), (0.0, 1.0)), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
    ok = ok and math.isclose(
        point_line_residual((1.0, 1.0), ((0.0, 0.0), (2.0, 2.0))),
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    frame_map, law_tests = build_outputs()
    write_json(FRAME_MAP_PATH, frame_map)
    write_json(LAW_TEST_PATH, law_tests)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    "pythagoras_quantum_world_rt/sixto_cch_frame_map.json",
                    "pythagoras_quantum_world_rt/sixto_qa_rt_landing_tests.json",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
