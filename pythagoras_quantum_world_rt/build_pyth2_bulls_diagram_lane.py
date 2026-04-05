#!/usr/bin/env python3
"""Build the stabilized bulls diagram lane for Pyth-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


SELECTED_PRIOR_ART_KEYS = [
    "rational_trigonometry",
    "chromogeometry",
    "ptolemy_quadrance_and_uhg",
    "egyptian_fractions_and_hat",
]


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def prior_art_refs(prior_art_payload: dict[str, object]) -> list[dict[str, object]]:
    by_key = {entry["key"]: entry for entry in prior_art_payload["categories"]}
    return [by_key[key] for key in SELECTED_PRIOR_ART_KEYS]


def build_lane(diagram_artifacts: dict[str, object], prior_art_payload: dict[str, object]) -> dict[str, object]:
    refs = prior_art_refs(prior_art_payload)
    stable_items = [
        {
            "formal_claim": "Figure 26 stabilizes as a linear cyclic layout of the three bulls equations with Y1 = 891 as the repeated additive constant and the kept fractions 5/6, 9/20, and 13/42 carrying X1->W1, Z1->X1, and W1->Z1 respectively.",
            "id": "pyth2_bulls_figure26_linear_cycle",
            "issue_origin": "direct_validated",
            "normalized_answer": "Read the page-121 line diagram as a three-step cyclic transition table anchored by Y1 = 891.",
            "prior_art_refs": refs,
            "program_template_tags": ["bulls_linear_cycle", "fractional_transition_table"],
            "source_lines": diagram_artifacts["figure26_linear_cycle"]["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": diagram_artifacts["figure26_linear_cycle"]["cycle_rows"],
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "Figure 27 stabilizes as a nested-radius model with inner circle radius Y1 = 891, intermediate ellipse radii landing at W1/X1/Z1, and outer boundary radii given by Y1 plus the full source values X1/Z1/W1.",
            "id": "pyth2_bulls_figure27_nested_radii",
            "issue_origin": "direct_validated",
            "normalized_answer": "Use the stabilized nested-radii table rather than the OCR-damaged sketch itself.",
            "prior_art_refs": refs,
            "program_template_tags": ["bulls_nested_radii", "ellipse_annulus_split"],
            "source_lines": diagram_artifacts["figure27_nested_ellipse_model"]["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": diagram_artifacts["figure27_nested_ellipse_model"]["nested_radii_rows"],
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "Figure 28 stabilizes as a qualitative cone-top-view analogy with three layers: inner circle, intermediate ellipse, and outer circle/boundary. This preserves the nesting logic of Figure 27 while changing the outer shape.",
            "id": "pyth2_bulls_figure28_cone_top_view",
            "issue_origin": "direct_validated",
            "normalized_answer": "Treat the cone passage as a structural three-layer analogy, not yet as a fully parameterized ellipse certificate.",
            "prior_art_refs": refs,
            "program_template_tags": ["bulls_cone_top_view", "qualitative_geometry_bridge"],
            "source_lines": diagram_artifacts["figure28_cone_top_view_model"]["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": diagram_artifacts["figure28_cone_top_view_model"]["mapping"],
            "validation_status": "direct_validated",
        },
    ]
    unresolved_items = [
        {
            "id": item["id"],
            "issue_label": item["issue_label"],
            "normalized_answer": item["normalized_answer"],
            "prior_art_refs": refs,
            "source_claim": item["source_claim"],
            "source_lines": item["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
        }
        for item in diagram_artifacts["source_ambiguities"]
    ]
    return {
        "lane_id": "pyth2_bulls_diagram_lane",
        "stable_items": stable_items,
        "summary": {
            "series": "Pyth-2",
            "stable_count": len(stable_items),
            "unresolved_count": len(unresolved_items),
        },
        "supporting_bulls_lane_ids": diagram_artifacts["supporting_bulls_lane_ids"],
        "unresolved_items": unresolved_items,
    }


def self_test() -> int:
    diagram_artifacts = {
        "figure26_linear_cycle": {"cycle_rows": [{"y1": 891}], "source_lines": [1, 2]},
        "figure27_nested_ellipse_model": {"nested_radii_rows": [{"inner_circle_radius": 891}], "source_lines": [3, 4]},
        "figure28_cone_top_view_model": {"mapping": {"figure28_layers": [{"radius": 891}]}, "source_lines": [5, 6]},
        "source_ambiguities": [{"id": "amb", "issue_label": "source-layer ambiguity", "normalized_answer": "hold", "source_claim": "claim", "source_lines": [7, 8]}],
        "supporting_bulls_lane_ids": ["x", "y", "z"],
    }
    prior_art_payload = {"categories": [{"key": key, "refs": [], "summary": key} for key in SELECTED_PRIOR_ART_KEYS]}
    payload = build_lane(diagram_artifacts, prior_art_payload)
    ok = (
        payload["summary"]["stable_count"] == 3
        and payload["summary"]["unresolved_count"] == 1
        and payload["stable_items"][0]["validated_examples"][0]["y1"] == 891
        and payload["supporting_bulls_lane_ids"][2] == "z"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    diagram_artifacts = read_json(OUT_DIR / "pyth2_bulls_diagram_artifacts.json")
    prior_art_payload = read_json(OUT_DIR / "prior_art_bridge_map.json")
    payload = build_lane(diagram_artifacts, prior_art_payload)
    write_json(OUT_DIR / "pyth2_bulls_diagram_lane.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_bulls_diagram_lane.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
