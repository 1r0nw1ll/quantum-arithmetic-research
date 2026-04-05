#!/usr/bin/env python3
"""Build the validated BULLS program lane for Pyth-2."""

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


def build_lane(program_artifacts: dict[str, object], prior_art_payload: dict[str, object]) -> dict[str, object]:
    refs = prior_art_refs(prior_art_payload)
    stable_items = [
        {
            "formal_claim": "The BULLS equations stabilize as W1 = Y1 + 5/6 X1, X1 = Y1 + 9/20 Z1, and Z1 = Y1 + 13/42 W1. These generate the printed family and the stated answer row.",
            "id": "pyth2_bulls_equation_block",
            "issue_origin": "direct_validated",
            "normalized_answer": "Use the three bulls equations exactly as normalized in the source window.",
            "prior_art_refs": refs,
            "program_template_tags": ["bulls_equations", "fractional_mod_constraints"],
            "source_lines": [7991, 7996],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": program_artifacts["solution_family"]["first_three_rows"],
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "The reconstructed BULLS program core is stable after OCR normalization: it scans Y1 and X1 candidates, computes W1 and Z1, then checks the recovered X1 relation against the candidate C value.",
            "id": "pyth2_bulls_program_listing",
            "issue_origin": "corrected_source_validated",
            "normalized_answer": "Use the normalized base listing with Y1 = B, X1 = C, W1 = 5/6 X1 + Y1, Z1 = 13/42 W1 + Y1, and X_TEST = 9/20 Z1 + Y1.",
            "prior_art_refs": refs,
            "program_template_tags": ["bulls_basic_listing", "equation_search"],
            "source_lines": [8076, 8124],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": program_artifacts["reconstructed_listing"],
            "validation_status": "corrected_source_validated",
        },
        {
            "formal_claim": "The program prints a linear solution family whose first three rows are (742,534,297,1580/3), (1484,1068,594,3160/3), and (2226,1602,891,1580). The third row is the stated answer.",
            "id": "pyth2_bulls_solution_family",
            "issue_origin": "direct_validated",
            "normalized_answer": "The printed family scales linearly, and the third row gives the integer answer W1=2226, X1=1602, Y1=891, Z1=1580.",
            "prior_art_refs": refs,
            "program_template_tags": ["bulls_solution_family", "answer_row"],
            "source_lines": [7997, 8002, 8118, 8124],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "answer_row": program_artifacts["solution_family"]["answer_row"],
                "first_three_rows": program_artifacts["solution_family"]["first_three_rows"],
                "row_count_with_X1_le_10000": program_artifacts["solution_family"]["row_count_with_X1_le_10000"],
            },
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
        for item in program_artifacts["source_ambiguities"]
    ]
    return {
        "lane_id": "pyth2_bulls_reproduction_lane",
        "source_window": program_artifacts["source_window"],
        "stable_items": stable_items,
        "summary": {
            "series": "Pyth-2",
            "stable_count": len(stable_items),
            "unresolved_count": len(unresolved_items),
        },
        "unresolved_items": unresolved_items,
    }


def self_test() -> int:
    program_artifacts = {
        "reconstructed_listing": [{"normalized_code": "LET Y1 = B"}],
        "solution_family": {
            "answer_row": {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580},
            "first_three_rows": [{"W1": 742, "X1": 534, "Y1": 297, "Z1": "1580/3"}],
            "row_count_with_X1_le_10000": 18,
        },
        "source_ambiguities": [{"id": "amb", "issue_label": "source-layer ambiguity", "normalized_answer": "hold", "source_claim": "claim", "source_lines": [1, 2]}],
        "source_window": {"program_page": 118},
    }
    prior_art_payload = {"categories": [{"key": key, "refs": [], "summary": key} for key in SELECTED_PRIOR_ART_KEYS]}
    payload = build_lane(program_artifacts, prior_art_payload)
    ok = (
        payload["summary"]["stable_count"] == 3
        and payload["summary"]["unresolved_count"] == 1
        and payload["stable_items"][2]["validated_examples"]["answer_row"]["Z1"] == 1580
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    program_artifacts = read_json(OUT_DIR / "pyth2_bulls_program_artifacts.json")
    prior_art_payload = read_json(OUT_DIR / "prior_art_bridge_map.json")
    payload = build_lane(program_artifacts, prior_art_payload)
    write_json(OUT_DIR / "pyth2_bulls_reproduction_lane.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_bulls_reproduction_lane.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
