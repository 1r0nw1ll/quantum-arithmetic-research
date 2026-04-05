#!/usr/bin/env python3
"""Build the combined post-cows lane from the transposition branch and ellipse family material."""

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


def build_lane(
    branch_payload: dict[str, object],
    ellipse_payload: dict[str, object],
    figure29_payload: dict[str, object],
    cattle_test_payload: dict[str, object],
    slicing_layer_payload: dict[str, object],
    source_intended_family_payload: dict[str, object],
    ratio_bead_chain_payload: dict[str, object],
    prior_art_payload: dict[str, object],
) -> dict[str, object]:
    refs = prior_art_refs(prior_art_payload)
    stable_items = [
        {
            "formal_claim": "For C = 84, the page-130 ellipse family mechanics stabilize exactly: the coprime factor pairs of 42 are (1,42), (2,21), (3,14), and (6,7), yielding the stated radius intervals 1722-1806, 399-483, 154-238, and 7-91.",
            "id": "pyth2_ellipse_family_c84",
            "issue_origin": "direct_validated",
            "normalized_answer": "Use the C = 84 factor-pair family as the first stable ellipse-family mechanics packet.",
            "prior_art_refs": refs,
            "program_template_tags": ["ellipse_family_c84", "integer_radius_interval"],
            "source_lines": ellipse_payload["c_84_family"]["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": ellipse_payload["c_84_family"]["coprime_factor_pairs"],
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "The page-130 general rule stabilizes: for C = 2de with gcd(e,d)=1, the integer radius count is 2C and the radius interval is D ± C/2 with D = d*d.",
            "id": "pyth2_ellipse_family_general_rule",
            "issue_origin": "direct_validated",
            "normalized_answer": "Use the 2C integer-radius rule and the D ± C/2 interval rule as the stable post-cows ellipse mechanics.",
            "prior_art_refs": refs,
            "program_template_tags": ["ellipse_family_rule", "radius_interval_rule"],
            "source_lines": [8530, 8558],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": ellipse_payload["general_rules"],
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "For Figure 29, the stable C = 5040 mechanics are a fixed-C ellipse family: there are 24 factor pairs of C/2 = 2520 with e <= d, 8 of them coprime, and every family member has 2C = 10080 integer radius points.",
            "id": "pyth2_figure29_c5040_family",
            "issue_origin": "direct_validated",
            "normalized_answer": "Use Figure 29 first as a fixed-C family packet with 24 total factor pairs, 8 coprime pairs, and 10080 integer radius points.",
            "prior_art_refs": refs,
            "program_template_tags": ["figure29_c5040_family", "fixed_c_family"],
            "source_lines": figure29_payload["figure_29_fixed_c_family"]["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "coprime_factor_pairs": figure29_payload["figure_29_fixed_c_family"]["coprime_factor_pairs"],
                "integer_radius_point_count": figure29_payload["figure_29_fixed_c_family"]["integer_radius_point_count"],
                "stable_narrative_claims": figure29_payload["stable_narrative_claims"],
                "total_factor_pair_count": figure29_payload["figure_29_fixed_c_family"]["all_factor_pair_count"],
            },
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "The direct G-route from the validated C = 5040 coprime subfamily does not recover the stabilized bulls packet: neither the raw bulls values nor the one-seventh-lifted bulls values match any coprime-family G value, and the lifted values fail the square conditions G ± C.",
            "id": "pyth2_figure29_c5040_direct_g_obstruction",
            "issue_origin": "direct_validated",
            "normalized_answer": cattle_test_payload["verdict"]["normalized_answer"],
            "prior_art_refs": refs,
            "program_template_tags": ["figure29_c5040_test", "direct_g_obstruction"],
            "source_lines": [8620, 8651],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "coprime_g_values": cattle_test_payload["coprime_c5040_family"]["coprime_g_values"],
                "raw_bulls_as_g_candidates": cattle_test_payload["raw_bulls_as_g_candidates"],
                "scaled_bulls_times_7_as_g_candidates": cattle_test_payload["scaled_bulls_times_7_as_g_candidates"],
                "summary": cattle_test_payload["summary"],
            },
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "The Figure 29 Gaussian-integer rhetoric is better explained by the earlier QA slicing/quaternion layer than by the direct coprime-G identity layer: the relevant formulas are x = n d/e and y = sqrt(F*(DE - n*n))/e, and the source explicitly says the second square root may be considered the same as Gaussian integers. But applying that slice layer to the validated C = 5040 coprime subfamily still does not recover the stabilized bulls packet.",
            "id": "pyth2_figure29_slicing_layer_bridge",
            "issue_origin": "corrected_source_validated",
            "normalized_answer": slicing_layer_payload["verdict"]["normalized_answer"],
            "prior_art_refs": refs,
            "program_template_tags": ["figure29_slicing_layer", "gaussian_bridge"],
            "source_lines": [8591, 8604],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "figure29_gaussian_bridge": slicing_layer_payload["figure29_gaussian_bridge"],
                "coprime_c5040_slice_rows": slicing_layer_payload["coprime_c5040_slice_rows"],
                "raw_bulls_slice_candidates": slicing_layer_payload["raw_bulls_slice_candidates"],
                "scaled_bulls_times_7_slice_candidates": slicing_layer_payload["scaled_bulls_times_7_slice_candidates"],
                "summary": slicing_layer_payload["summary"],
            },
            "validation_status": "corrected_source_validated",
        },
        {
            "formal_claim": "The page-132 count of 9 ellipses is best explained by the explicit source-intended list itself: it equals the validated 8 coprime C = 5040 pairs plus one extra noncoprime pair (2,1260). Adding that ninth source-only pair does not change the cattle-generation verdict.",
            "id": "pyth2_figure29_source_intended_nine_family",
            "issue_origin": "corrected_source_validated",
            "normalized_answer": source_intended_family_payload["verdict"]["normalized_answer"],
            "prior_art_refs": refs,
            "program_template_tags": ["figure29_source_intended_family", "count_reconciliation"],
            "source_lines": [8627, 8653],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "source_intended_family": source_intended_family_payload["source_intended_family"],
                "validated_family_overlap": source_intended_family_payload["validated_family_overlap"],
                "summary": source_intended_family_payload["summary"],
            },
            "validation_status": "corrected_source_validated",
        },
        {
            "formal_claim": "The page-133 ratio/proportion alternative is a distinct construction lane: the four cattle fractions generate one continuous Fibonacci-type bead chain 2/7 -> 1/3 -> 2/5 -> 1/2 -> 2/3, with overlapping endpoints and a common integerization scale 420.",
            "id": "pyth2_ratio_proportion_bead_chain",
            "issue_origin": "corrected_source_validated",
            "normalized_answer": ratio_bead_chain_payload["verdict"]["normalized_answer"],
            "prior_art_refs": refs,
            "program_template_tags": ["ratio_proportion_bead_chain", "fraction_to_bead_chain"],
            "source_lines": [8690, 8722],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": {
                "bead_chain_rows": ratio_bead_chain_payload["bead_chain_rows"],
                "continuous_division_summary": ratio_bead_chain_payload["continuous_division_summary"],
                "overlap_chain": ratio_bead_chain_payload["overlap_chain"],
                "summary": ratio_bead_chain_payload["summary"],
            },
            "validation_status": "corrected_source_validated",
        },
    ]
    unresolved_items = [
        {
            "id": "pyth2_cows_transposition_branch",
            "issue_label": "unresolved_branch",
            "normalized_answer": "The explicit transposition branch is now formalized and testable, but it still yields nonintegral cattle counts.",
            "prior_art_refs": refs,
            "source_claim": branch_payload["branch_assumption"]["description"],
            "source_lines": branch_payload["branch_assumption"]["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": branch_payload["branch_consequences"],
        }
    ]
    unresolved_items.extend(
        {
            "id": item["id"],
            "issue_label": item["issue_label"],
            "normalized_answer": item["normalized_answer"],
            "prior_art_refs": refs,
            "source_claim": item["source_claim"],
            "source_lines": item["source_lines"],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": item.get("tested_candidate_orders"),
        }
        for item in figure29_payload["unresolved_claims"]
        if item["id"] not in {"pyth2_figure29_gaussian_integer_family", "pyth2_figure29_5040_count"}
    )
    unresolved_items.append(
        {
            "id": "pyth2_figure29_multiple_answer_sets",
            "issue_label": "source-layer ambiguity",
            "normalized_answer": "The multiple-answer-sets rhetoric remains unvalidated even after the slicing-layer bridge and the page-133 ratio/proportion bead-chain reconstruction. The source now supports a coupled bead-derivation lane, but not explicit multiple cattle-answer families.",
            "prior_art_refs": refs,
            "source_claim": "So there could be nine different sets of answers to the Cattle Problem which are seemingly unrelated to each other, except through the value of C.",
            "source_lines": [8594, 8601],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
        }
    )
    return {
        "lane_id": "pyth2_post_cows_lane",
        "stable_items": stable_items,
        "summary": {
            "series": "Pyth-2",
            "stable_count": len(stable_items),
            "unresolved_count": len(unresolved_items),
        },
        "unresolved_items": unresolved_items,
    }


def self_test() -> int:
    branch_payload = {
        "branch_assumption": {"description": "swap", "source_lines": [1, 2]},
        "branch_consequences": {"exact_solution": {"Y2": "9801/19"}},
    }
    ellipse_payload = {
        "c_84_family": {"coprime_factor_pairs": [{"radius_low": 1722}], "source_lines": [3, 4]},
        "general_rules": [{"rule": "r"}],
    }
    figure29_payload = {
        "figure_29_fixed_c_family": {
            "all_factor_pair_count": 24,
            "coprime_factor_pairs": [{"d": 2520}],
            "integer_radius_point_count": 10080,
            "source_lines": [7, 8],
        },
        "stable_narrative_claims": [{"claim": "c5040"}],
        "unresolved_claims": [
            {"id": "amb1", "issue_label": "source-layer ambiguity", "normalized_answer": "hold", "source_claim": "claim", "source_lines": [9, 10]},
            {"id": "amb2", "issue_label": "source-layer ambiguity", "normalized_answer": "hold", "source_claim": "claim", "source_lines": [11, 12]},
        ],
    }
    cattle_test_payload = {
        "coprime_c5040_family": {"coprime_g_values": [1, 2, 3]},
        "raw_bulls_as_g_candidates": [{"bull_id": "W1"}],
        "scaled_bulls_times_7_as_g_candidates": [{"bull_id": "W1"}],
        "summary": {"scaled_direct_g_match_count": 0, "scaled_valid_square_condition_count": 0},
        "verdict": {"normalized_answer": "no direct construction"},
    }
    slicing_layer_payload = {
        "coprime_c5040_slice_rows": [{"integer_slice_hit_count": 2}],
        "figure29_gaussian_bridge": {"normalized_answer": "slice bridge"},
        "raw_bulls_slice_candidates": [{"bull_id": "W1"}],
        "scaled_bulls_times_7_slice_candidates": [{"bull_id": "W1"}],
        "summary": {"scaled_slice_integer_hit_count": 0},
        "verdict": {"normalized_answer": "gaussian bridge but no cattle construction"},
    }
    source_intended_family_payload = {
        "source_intended_family": {"listed_pair_count": 9},
        "validated_family_overlap": {"extra_source_only_pairs": [{"e": 2}], "validated_overlap_count": 8},
        "summary": {"extra_source_only_pair_count": 1},
        "verdict": {"normalized_answer": "nine-family reconciliation"},
    }
    ratio_bead_chain_payload = {
        "bead_chain_rows": [{"integerized_bead_numbers": [120, 10, 130, 140]}],
        "continuous_division_summary": {"common_scale": 420},
        "overlap_chain": [{"values_match": True}],
        "summary": {"fibonacci_row_count": 4},
        "verdict": {"normalized_answer": "continuous bead chain"},
    }
    prior_art_payload = {"categories": [{"key": key, "refs": [], "summary": key} for key in SELECTED_PRIOR_ART_KEYS]}
    payload = build_lane(
        branch_payload,
        ellipse_payload,
        figure29_payload,
        cattle_test_payload,
        slicing_layer_payload,
        source_intended_family_payload,
        ratio_bead_chain_payload,
        prior_art_payload,
    )
    ok = (
        payload["summary"]["stable_count"] == 7
        and payload["summary"]["unresolved_count"] == 4
        and payload["stable_items"][0]["validated_examples"][0]["radius_low"] == 1722
        and payload["stable_items"][2]["validated_examples"]["total_factor_pair_count"] == 24
        and payload["stable_items"][3]["validated_examples"]["summary"]["scaled_direct_g_match_count"] == 0
        and payload["stable_items"][4]["validated_examples"]["summary"]["scaled_slice_integer_hit_count"] == 0
        and payload["stable_items"][5]["validated_examples"]["summary"]["extra_source_only_pair_count"] == 1
        and payload["stable_items"][6]["validated_examples"]["continuous_division_summary"]["common_scale"] == 420
        and payload["unresolved_items"][0]["validated_examples"]["exact_solution"]["Y2"] == "9801/19"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    branch_payload = read_json(OUT_DIR / "pyth2_cows_transposition_branch.json")
    ellipse_payload = read_json(OUT_DIR / "pyth2_ellipse_family_artifacts.json")
    figure29_payload = read_json(OUT_DIR / "pyth2_figure29_narrative.json")
    cattle_test_payload = read_json(OUT_DIR / "pyth2_c5040_cattle_test.json")
    slicing_layer_payload = read_json(OUT_DIR / "pyth2_c5040_slicing_layer.json")
    source_intended_family_payload = read_json(OUT_DIR / "pyth2_c5040_source_intended_family.json")
    ratio_bead_chain_payload = read_json(OUT_DIR / "pyth2_ratio_proportion_bead_chain.json")
    prior_art_payload = read_json(OUT_DIR / "prior_art_bridge_map.json")
    payload = build_lane(
        branch_payload,
        ellipse_payload,
        figure29_payload,
        cattle_test_payload,
        slicing_layer_payload,
        source_intended_family_payload,
        ratio_bead_chain_payload,
        prior_art_payload,
    )
    write_json(OUT_DIR / "pyth2_post_cows_lane.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_post_cows_lane.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
