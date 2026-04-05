#!/usr/bin/env python3
"""Build the validated Program COPRIME reproduction lane for Pyth-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


SELECTED_PRIOR_ART_KEYS = [
    "pythagorean_triples_and_theorem",
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
    stable_items = [
        {
            "formal_claim": "The reconstructed Program COPRIME listing is stable after OCR normalization. In particular, line 40 must be 'LET Y = O' because the OCR reading 'LET Y = 0' makes Euclid's subtraction loop non-terminating. The source witness uses N and O as software-era surrogates for the underlying bead controls b and e.",
            "id": "pyth2_coprime_program_listing",
            "issue_origin": "corrected_source_validated",
            "normalized_answer": "Use the normalized 10-step BASIC listing ending with 'PLOT N,O', 'NEXT O', 'NEXT N', 'END', while reading N/O as source-program surrogates for bead controls b/e.",
            "letter_assignment_note": program_artifacts["letter_assignment_note"],
            "modernized_symbolic_listing": program_artifacts["modernized_symbolic_listing"],
            "prior_art_refs": prior_art_refs(prior_art_payload),
            "program_template_tags": ["coprime_basic_listing", "euclid_vii_1_subtraction"],
            "source_lines": [7153, 7209],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": program_artifacts["ocr_repair_validations"][0]["validation"],
            "validation_status": "corrected_source_validated",
        },
        {
            "formal_claim": "Figure 24 is reproduced by plotting all ordered pairs (N,O) with 1<=N,O<=20 for which the Euclid subtraction loop terminates at unity. This yields exactly 255 points.",
            "id": "pyth2_coprime_figure24_ordered_grid",
            "issue_origin": "direct_validated",
            "normalized_answer": "Figure 24 has 255 ordered coprime plot points on the 1..20 grid.",
            "prior_art_refs": prior_art_refs(prior_art_payload),
            "program_template_tags": ["coprime_grid", "figure24_reproduction"],
            "source_lines": [7132, 7136],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "axis_limit": 20,
                "ordered_point_count": program_artifacts["figure24"]["ordered_point_count"],
            },
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "The reproduced Figure 24 grid is exactly symmetric about the diagonal, so the unique upper-triangle count is 128 when the diagonal point (1,1) is counted once.",
            "id": "pyth2_coprime_figure24_symmetry",
            "issue_origin": "direct_validated",
            "normalized_answer": "The 1..20 coprime plot is diagonal-symmetric and has 128 unique upper-triangle placements.",
            "prior_art_refs": prior_art_refs(prior_art_payload),
            "program_template_tags": ["coprime_grid_symmetry", "upper_triangle_unique_count"],
            "source_lines": [7281, 7282],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "diagonal_symmetry": program_artifacts["figure24"]["diagonal_symmetry"],
                "upper_triangle_unique_count": program_artifacts["figure24"]["upper_triangle_unique_count"],
            },
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "Running the normalized program to the stated limit 43 produces 1167 ordered coprime pairs.",
            "id": "pyth2_coprime_program_run_43",
            "issue_origin": "direct_validated",
            "normalized_answer": "The normalized 43x43 Program COPRIME grid has 1167 ordered coprime points.",
            "prior_art_refs": prior_art_refs(prior_art_payload),
            "program_template_tags": ["coprime_grid_full_run"],
            "source_lines": [7151, 7209],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": program_artifacts["full_program_run"],
            "validation_status": "direct_validated",
        },
    ]
    unresolved_items = [
        {
            "best_supported_interpretation": ambiguity["best_supported_interpretation"],
            "candidate_counts": ambiguity["candidate_counts"],
            "id": ambiguity["id"],
            "issue_label": ambiguity["issue_label"],
            "normalized_answer": ambiguity["normalized_answer"],
            "prior_art_refs": prior_art_refs(prior_art_payload),
            "source_claim": ambiguity["source_claim"],
            "source_lines": ambiguity["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
        }
        for ambiguity in program_artifacts["source_ambiguities"]
    ]
    return {
        "lane_id": "pyth2_coprime_reproduction_lane",
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
        "figure24": {"diagonal_symmetry": True, "ordered_point_count": 255, "upper_triangle_unique_count": 128},
        "full_program_run": {"ordered_point_count": 1167},
        "letter_assignment_note": {
            "symbol_mapping": [{"qa_symbol": "b", "source_program_variable": "N"}]
        },
        "modernized_symbolic_listing": [{"line_number": 10, "normalized_code": "FOR B = 1 TO 43"}],
        "ocr_repair_validations": [{"validation": {"reconstructed_variant_coprime": True, "zero_variant_terminates_within_100_steps": False}}],
        "source_ambiguities": [
            {
                "best_supported_interpretation": "upper triangle",
                "candidate_counts": [{"count": 128, "description": "desc"}],
                "id": "amb",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "hold",
                "source_claim": "claim",
                "source_lines": [1, 2],
            }
        ],
        "source_window": {"program_page": 100},
    }
    prior_art_payload = {
        "categories": [
            {"key": key, "refs": [], "summary": key} for key in SELECTED_PRIOR_ART_KEYS
        ]
    }
    payload = build_lane(program_artifacts, prior_art_payload)
    ok = (
        payload["summary"]["stable_count"] == 4
        and payload["summary"]["unresolved_count"] == 1
        and payload["stable_items"][0]["letter_assignment_note"]["symbol_mapping"][0]["qa_symbol"] == "b"
        and payload["stable_items"][1]["validated_examples"]["ordered_point_count"] == 255
        and payload["unresolved_items"][0]["candidate_counts"][0]["count"] == 128
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    program_artifacts = read_json(OUT_DIR / "pyth2_coprime_program_artifacts.json")
    prior_art_payload = read_json(OUT_DIR / "prior_art_bridge_map.json")
    payload = build_lane(program_artifacts, prior_art_payload)
    write_json(OUT_DIR / "pyth2_coprime_reproduction_lane.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_coprime_reproduction_lane.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
