#!/usr/bin/env python3
"""Build the stabilized bulls diagram layer from the validated BULLS equations."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def build_cycle_segments(answer_row: dict[str, object]) -> list[dict[str, object]]:
    y1 = int(answer_row["Y1"])
    x1 = int(answer_row["X1"])
    z1 = int(answer_row["Z1"])
    w1 = int(answer_row["W1"])
    segments = [
        {
            "discard_fraction": "1/6",
            "discard_value": int(Fraction(1, 6) * x1),
            "equation": "W1 = Y1 + 5/6 X1",
            "inner_ellipse_radius": w1,
            "kept_fraction": "5/6",
            "kept_value": int(Fraction(5, 6) * x1),
            "outer_boundary_radius": y1 + x1,
            "source_value": x1,
            "target_value": w1,
            "via_source": "X1",
        },
        {
            "discard_fraction": "11/20",
            "discard_value": int(Fraction(11, 20) * z1),
            "equation": "X1 = Y1 + 9/20 Z1",
            "inner_ellipse_radius": x1,
            "kept_fraction": "9/20",
            "kept_value": int(Fraction(9, 20) * z1),
            "outer_boundary_radius": y1 + z1,
            "source_value": z1,
            "target_value": x1,
            "via_source": "Z1",
        },
        {
            "discard_fraction": "29/42",
            "discard_value": int(Fraction(29, 42) * w1),
            "equation": "Z1 = Y1 + 13/42 W1",
            "inner_ellipse_radius": z1,
            "kept_fraction": "13/42",
            "kept_value": int(Fraction(13, 42) * w1),
            "outer_boundary_radius": y1 + w1,
            "source_value": w1,
            "target_value": z1,
            "via_source": "W1",
        },
    ]
    return segments


def linear_cycle_table(answer_row: dict[str, object]) -> list[dict[str, object]]:
    y1 = int(answer_row["Y1"])
    rows = []
    for segment in build_cycle_segments(answer_row):
        rows.append(
            {
                "equation": segment["equation"],
                "kept_fraction": segment["kept_fraction"],
                "kept_value": segment["kept_value"],
                "source": segment["via_source"],
                "target": segment["target_value"],
                "target_minus_Y1": segment["target_value"] - y1,
                "y1": y1,
            }
        )
    return rows


def nested_radii_table(answer_row: dict[str, object]) -> list[dict[str, object]]:
    y1 = int(answer_row["Y1"])
    rows = []
    for segment in build_cycle_segments(answer_row):
        rows.append(
            {
                "circle_to_inner_ellipse": segment["kept_value"],
                "figure27_source": segment["via_source"],
                "inner_circle_radius": y1,
                "inner_ellipse_radius": segment["inner_ellipse_radius"],
                "inner_to_outer_boundary": segment["discard_value"],
                "outer_boundary_radius": segment["outer_boundary_radius"],
            }
        )
    rows.sort(key=lambda row: row["figure27_source"])
    return rows


def cone_top_view_mapping(answer_row: dict[str, object]) -> dict[str, object]:
    y1 = int(answer_row["Y1"])
    return {
        "figure28_layers": [
            {"geometry_role": "inner_circle", "radius": y1},
            {"geometry_role": "intermediate_ellipse", "radii": [int(answer_row["Z1"]), int(answer_row["X1"]), int(answer_row["W1"])]},
            {
                "geometry_role": "outer_circle_or_outer_boundary",
                "radii": [y1 + int(answer_row["Z1"]), y1 + int(answer_row["X1"]), y1 + int(answer_row["W1"])],
            },
        ],
        "qualitative_claim": "The cone top-view analogy preserves the three-layer nesting but replaces the outer ellipse of Figure 27 with an outer circle.",
    }


def build_payload(bulls_program: dict[str, object], bulls_lane: dict[str, object]) -> dict[str, object]:
    answer_row = bulls_program["solution_family"]["answer_row"]
    cycle_rows = linear_cycle_table(answer_row)
    nested_rows = nested_radii_table(answer_row)
    return {
        "answer_row": answer_row,
        "figure26_linear_cycle": {
            "cycle_rows": cycle_rows,
            "qualitative_claim": "The three bulls equations form a circular proposition when laid out in shifted linear form with Y1 as the repeated additive constant.",
            "source_lines": [8170, 8187],
        },
        "figure27_nested_ellipse_model": {
            "nested_radii_rows": nested_rows,
            "qualitative_claim": "The page-121 ellipse sketch is modeled by an inner circle of radius Y1, an intermediate ellipse at the target values W1/X1/Z1, and an outer boundary reached by adding the full source values X1/Z1/W1 to Y1.",
            "source_lines": [8188, 8203],
        },
        "figure28_cone_top_view_model": {
            "mapping": cone_top_view_mapping(answer_row),
            "source_lines": [8230, 8268],
        },
        "source_ambiguities": [
            {
                "id": "pyth2_bulls_square_number_speculation",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the square-number comments isolated; the text is suggestive but the stable numeric diagram layer does not certify them.",
                "source_claim": "As shown in the drawing the value of W1 + Z1 will be a double square number ... But W1 + X1 may be a square number when the two radii of 891 are added to them.",
                "source_lines": [8204, 8221],
            },
            {
                "id": "pyth2_bulls_specific_ellipse_foci",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the exact-focus claim isolated until a concrete ellipse parameterization is derived from the stabilized bulls data.",
                "source_claim": "The two ellipses must be some specific ones, and their primary focus must be located exactly as he had the problem planned.",
                "source_lines": [8204, 8210],
            },
        ],
        "summary": {
            "figure26_row_count": len(cycle_rows),
            "figure27_row_count": len(nested_rows),
            "series": "Pyth-2",
            "source_ambiguity_count": 2,
        },
        "supporting_bulls_lane_ids": [item["id"] for item in bulls_lane["stable_items"]],
    }


def self_test() -> int:
    bulls_program = {
        "solution_family": {
            "answer_row": {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580}
        }
    }
    bulls_lane = {"stable_items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
    payload = build_payload(bulls_program, bulls_lane)
    rows = payload["figure27_nested_ellipse_model"]["nested_radii_rows"]
    ok = (
        payload["figure26_linear_cycle"]["cycle_rows"][0]["kept_value"] == 1335
        and rows[0]["figure27_source"] == "W1"
        and rows[0]["circle_to_inner_ellipse"] == 689
        and rows[2]["inner_ellipse_radius"] == 1602
        and payload["figure28_cone_top_view_model"]["mapping"]["figure28_layers"][0]["radius"] == 891
        and payload["summary"]["source_ambiguity_count"] == 2
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    bulls_program = read_json(OUT_DIR / "pyth2_bulls_program_artifacts.json")
    bulls_lane = read_json(OUT_DIR / "pyth2_bulls_reproduction_lane.json")
    payload = build_payload(bulls_program, bulls_lane)
    write_json(OUT_DIR / "pyth2_bulls_diagram_artifacts.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_bulls_diagram_artifacts.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
