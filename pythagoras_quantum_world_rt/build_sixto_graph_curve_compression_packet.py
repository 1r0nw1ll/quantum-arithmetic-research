#!/usr/bin/env python3
"""Build a compressed branch-family packet for the Sixto timing-graph curves."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
SUMMARY_PATH = OUT_DIR / "sixto_geogebra_extended_summary.json"
LAW_PACKET_PATH = OUT_DIR / "sixto_graph_curve_law_packet.json"
COMPRESSION_PATH = OUT_DIR / "sixto_graph_curve_compression_packet.json"

GRID_VALUES = [index / 20.0 for index in range(21)]
MEAN_ACCEPT = 0.14
MAX_ACCEPT = 0.30


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def interp(branch_points: list[tuple[float, float]], t_value: float) -> float:
    xs = [point[0] for point in branch_points]
    ys = [point[1] for point in branch_points]
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


def normalized_branches(curve: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    sampled_points = curve["sampled_points"]
    cross_x = float(curve["crossover_x"])
    peak_y = float(curve["peak"]["y"])
    dip_y = abs(float(curve["dip"]["y"]))
    left = []
    right = []
    for point in sampled_points:
        x_value = float(point["x"])
        y_value = float(point["y"])
        if x_value <= cross_x:
            t_value = x_value / cross_x if cross_x else 0.0
            left.append((t_value, y_value / dip_y if dip_y else 0.0))
        if x_value >= cross_x:
            t_value = (x_value - cross_x) / (1212.0 - cross_x) if cross_x != 1212.0 else 0.0
            right.append((t_value, y_value / peak_y if peak_y else 0.0))
    left.sort()
    right.sort()
    return {"left": left, "right": right}


def sampled_profile(branch_points: list[tuple[float, float]]) -> list[float]:
    return [interp(branch_points, t_value) for t_value in GRID_VALUES]


def pairwise_stats(profile_a: list[float], profile_b: list[float]) -> dict[str, float]:
    diffs = [abs(a_value - b_value) for a_value, b_value in zip(profile_a, profile_b)]
    return {"mean_abs": sum(diffs) / len(diffs), "max_abs": max(diffs)}


def medoid_id(profiles: dict[str, list[float]]) -> str:
    ids = sorted(profiles.keys())
    best_id = ids[0]
    best_score = None
    for candidate in ids:
        score = 0.0
        for other in ids:
            if other == candidate:
                continue
            score += pairwise_stats(profiles[candidate], profiles[other])["mean_abs"]
        if best_score is None or score < best_score:
            best_score = score
            best_id = candidate
    return best_id


def build_payload() -> dict[str, Any]:
    summary = read_json(SUMMARY_PATH)
    _ = read_json(LAW_PACKET_PATH)
    curves = summary["reconstructed_graph_curves"]

    branch_profiles = {"left": {}, "right": {}}
    curve_lookup = {}
    for curve in curves:
        curve_lookup[curve["curve_id"]] = curve
        branches = normalized_branches(curve)
        branch_profiles["left"][curve["curve_id"]] = sampled_profile(branches["left"])
        branch_profiles["right"][curve["curve_id"]] = sampled_profile(branches["right"])

    branch_templates = {}
    for branch_name in ("left", "right"):
        profiles = branch_profiles[branch_name]
        medoid = medoid_id(profiles)
        template_profile = profiles[medoid]
        members = []
        for curve_id, profile in profiles.items():
            stats = pairwise_stats(profile, template_profile)
            members.append(
                {
                    "curve_id": curve_id,
                    "mean_abs_residual": round_float(stats["mean_abs"]),
                    "max_abs_residual": round_float(stats["max_abs"]),
                    "accepted_into_template_family": bool(stats["mean_abs"] <= MEAN_ACCEPT and stats["max_abs"] <= MAX_ACCEPT),
                }
            )
        branch_templates[branch_name] = {
            "medoid_curve_id": medoid,
            "grid_t": [round_float(value) for value in GRID_VALUES],
            "template_profile": [round_float(value) for value in template_profile],
            "members": members,
        }

    curve_reconstructions = []
    for curve in curves:
        curve_id = curve["curve_id"]
        left_members = {item["curve_id"]: item for item in branch_templates["left"]["members"]}
        right_members = {item["curve_id"]: item for item in branch_templates["right"]["members"]}
        curve_reconstructions.append(
            {
                "curve_id": curve_id,
                "anchors": {
                    "start_x": 0.0,
                    "dip_x": curve["dip"]["x"],
                    "crossover_x": curve["crossover_x"],
                    "peak_x": curve["peak"]["x"],
                    "end_x": 1212.0,
                    "dip_y": curve["dip"]["y"],
                    "peak_y": curve["peak"]["y"],
                },
                "left_branch_template_member": left_members[curve_id]["accepted_into_template_family"],
                "right_branch_template_member": right_members[curve_id]["accepted_into_template_family"],
                "left_branch_residual": left_members[curve_id],
                "right_branch_residual": right_members[curve_id],
            }
        )

    return {
        "artifact_id": "sixto_graph_curve_compression_packet",
        "source_summary": "pythagoras_quantum_world_rt/sixto_geogebra_extended_summary.json",
        "source_law_packet": "pythagoras_quantum_world_rt/sixto_graph_curve_law_packet.json",
        "normalization_rule": {
            "left_branch": "t = x / crossover_x ; y_norm = y / abs(dip_y)",
            "right_branch": "t = (x - crossover_x) / (1212 - crossover_x) ; y_norm = y / peak_y",
            "accept_thresholds": {"mean_abs_residual_le": MEAN_ACCEPT, "max_abs_residual_le": MAX_ACCEPT},
        },
        "branch_templates": branch_templates,
        "curve_reconstructions": curve_reconstructions,
        "verdict": {
            "shared_branch_family_present": True,
            "honest_summary": "A compressed branch family exists under the exact interpolation packet. The medoid left-branch template is shared across all four curves within the configured residual envelope, while the right-branch template is shared by light-green, green, and blue but not by the cyan trace, whose positive branch remains an outlier under this normalization.",
        },
    }


def self_test() -> int:
    payload = build_payload()
    left_members = payload["branch_templates"]["left"]["members"]
    right_members = payload["branch_templates"]["right"]["members"]
    left_accept_count = sum(1 for item in left_members if item["accepted_into_template_family"])
    right_accept_count = sum(1 for item in right_members if item["accepted_into_template_family"])
    cyan_right = next(item for item in right_members if item["curve_id"] == "curve_graph_cyan")
    ok = True
    ok = ok and payload["branch_templates"]["left"]["medoid_curve_id"] == "curve_graph_green"
    ok = ok and payload["branch_templates"]["right"]["medoid_curve_id"] == "curve_graph_green"
    ok = ok and left_accept_count == 4
    ok = ok and right_accept_count == 3
    ok = ok and (not cyan_right["accepted_into_template_family"])
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    write_json(COMPRESSION_PATH, payload)
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/sixto_graph_curve_compression_packet.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
