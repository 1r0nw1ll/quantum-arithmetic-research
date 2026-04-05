#!/usr/bin/env python3
"""Build the validated Pyth-2 workset from stable batch items and page-110 leftovers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"

THEORY_LANE_BY_CLASS = {
    "counting_constraint": "counting_and_admissibility",
    "general_theory_item": "general_qa_theory",
    "periodicity_rule": "fibonacci_periodicity",
    "prime_factor_encoding": "prime_factor_encoding",
    "symmetry_rule": "reflection_symmetry",
}


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def validation_status(item: dict[str, object]) -> str:
    if item["issue_label"] == "none":
        return "direct_validated"
    return "corrected_source_validated"


def program_template_tags(item: dict[str, object]) -> list[str]:
    if item["id"] == "pyth2_p110_l7760":
        return ["bounded_pair_counts"]
    if item["id"] == "pyth2_p110_l7765":
        return ["ed_pair_counts"]
    if item["id"] == "pyth2_p075_l5658":
        return ["prime_power_factorization_5040"]
    if item["id"] == "pyth2_p075_l5663":
        return ["prime_numbering_5040"]
    if item["id"] == "pyth2_p075_l5668":
        return ["prime_power_regrouping_5040"]
    if item["id"] == "pyth2_p075_l5680":
        return ["prime_power_factorization_1000"]
    intake_class = str(item["classification"]["intake_class"])
    if intake_class == "counting_constraint":
        return ["fixed_e_counting"]
    if intake_class == "symmetry_rule":
        return ["reflection_pairs"]
    if intake_class == "periodicity_rule":
        if item["id"] == "pyth2_p110_l7743":
            return ["fibonacci_periodicity_even"]
        if item["id"] == "pyth2_p110_l7754":
            return ["fibonacci_periodicity_five"]
        return ["fibonacci_periodicity"]
    return [intake_class]


def stable_item(item: dict[str, object]) -> dict[str, object]:
    return {
        "classification": item["classification"],
        "formal_claim": item["formal_claim"],
        "id": item["id"],
        "issue_origin": item["issue_label"],
        "normalized_answer": item["normalized_answer"],
        "next_step": item["next_step"],
        "prior_art_keys": item["prior_art_keys"],
        "prior_art_refs": item["prior_art_refs"],
        "program_template_tags": program_template_tags(item),
        "qa_formulas": item["qa_formulas"],
        "rt_bridge": item["rt_bridge"],
        "chromo_bridge": item["chromo_bridge"],
        "source": item["source"],
        "source_answer": item["source_answer"],
        "source_question": item["source_question"],
        "source_verdict": item["source_verdict"],
        "theory_lane": THEORY_LANE_BY_CLASS[item["classification"]["intake_class"]],
        "validated_examples": item["validated_examples"],
        "validation_status": validation_status(item),
    }


def holdout_item(item: dict[str, object]) -> dict[str, object]:
    return {
        "classification": item["classification"],
        "formal_claim": item["formal_claim"],
        "id": item["id"],
        "issue_label": item["issue_label"],
        "next_step": item["next_step"],
        "normalized_answer": item["normalized_answer"],
        "prior_art_keys": item["prior_art_keys"],
        "prior_art_refs": item["prior_art_refs"],
        "qa_formulas": item["qa_formulas"],
        "rt_bridge": item["rt_bridge"],
        "source": item["source"],
        "source_answer": item["source_answer"],
        "source_question": item["source_question"],
        "source_verdict": item["source_verdict"],
        "theory_lane": THEORY_LANE_BY_CLASS[item["classification"]["intake_class"]],
        "validated_examples": item["validated_examples"],
        "chromo_bridge": item["chromo_bridge"],
    }


def build_workset(
    batch: dict[str, object],
    unresolved_batch: dict[str, object],
    page75_batch: dict[str, object],
) -> dict[str, object]:
    stable_items = [stable_item(item) for item in batch["items"]]
    stable_items.extend(stable_item(item) for item in unresolved_batch["items"] if item["issue_label"] == "none")
    stable_items.extend(stable_item(item) for item in page75_batch["items"] if item["issue_label"] != "OCR corruption")
    holdouts = [holdout_item(item) for item in unresolved_batch["items"] if item["issue_label"] != "none"]
    holdouts.extend(holdout_item(item) for item in page75_batch["items"] if item["issue_label"] == "OCR corruption")
    counts: dict[str, int] = {}
    for item in stable_items:
        status = str(item["validation_status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "holdouts": holdouts,
        "stable_items": stable_items,
        "summary": {
            "counts_by_validation_status": counts,
            "holdout_count": len(holdouts),
            "series": "Pyth-2",
            "stable_total": len(stable_items),
        },
    }


def self_test() -> int:
    batch = {
        "items": [
            {
                "classification": {"intake_class": "counting_constraint"},
                "chromo_bridge": "cg",
                "formal_claim": "claim",
                "id": "x",
                "issue_label": "none",
                "next_step": "next",
                "normalized_answer": "answer",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": [],
                "rt_bridge": "rt",
                "source": {"page": 110},
                "source_answer": "sa",
                "source_question": "sq",
                "source_verdict": "faithful",
                "validated_examples": [],
            },
            {
                "classification": {"intake_class": "periodicity_rule"},
                "chromo_bridge": "cg",
                "formal_claim": "claim2",
                "id": "y",
                "issue_label": "likely typo",
                "next_step": "next2",
                "normalized_answer": "answer2",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": [],
                "rt_bridge": "rt2",
                "source": {"page": 110},
                "source_answer": "sa2",
                "source_question": "sq2",
                "source_verdict": "corrected",
                "validated_examples": [],
            },
        ]
    }
    unresolved_batch = {
        "items": [
            {
                "classification": {"intake_class": "counting_constraint"},
                "chromo_bridge": "cg3",
                "formal_claim": "claim3",
                "id": "pyth2_p110_l7760",
                "issue_label": "none",
                "next_step": "next3",
                "normalized_answer": "answer3",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": [],
                "rt_bridge": "rt3",
                "source": {"page": 110},
                "source_answer": "sa3",
                "source_question": "sq3",
                "source_verdict": "corrected",
                "validated_examples": [],
            },
            {
                "classification": {"intake_class": "periodicity_rule"},
                "chromo_bridge": "cg4",
                "formal_claim": "claim4",
                "id": "z",
                "issue_label": "source-layer ambiguity",
                "next_step": "next4",
                "normalized_answer": "answer4",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": [],
                "rt_bridge": "rt4",
                "source": {"page": 110},
                "source_answer": "sa4",
                "source_question": "sq4",
                "source_verdict": "corrected",
                "validated_examples": [],
            },
        ]
    }
    page75_batch = {
        "items": [
            {
                "classification": {"intake_class": "prime_factor_encoding"},
                "chromo_bridge": "cg5",
                "formal_claim": "claim5",
                "id": "pyth2_p075_l5658",
                "issue_label": "source-layer ambiguity",
                "next_step": "next5",
                "normalized_answer": "answer5",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": [],
                "rt_bridge": "rt5",
                "source": {"page": 75},
                "source_answer": "sa5",
                "source_question": "sq5",
                "source_verdict": "corrected",
                "validated_examples": [],
            },
            {
                "classification": {"intake_class": "prime_factor_encoding"},
                "chromo_bridge": "cg6",
                "formal_claim": "claim6",
                "id": "hold75",
                "issue_label": "OCR corruption",
                "next_step": "next6",
                "normalized_answer": "answer6",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": [],
                "rt_bridge": "rt6",
                "source": {"page": 75},
                "source_answer": "sa6",
                "source_question": "sq6",
                "source_verdict": "flagged",
                "validated_examples": [],
            },
        ]
    }
    payload = build_workset(batch, unresolved_batch, page75_batch)
    ok = (
        payload["summary"]["stable_total"] == 4
        and payload["summary"]["holdout_count"] == 2
        and payload["stable_items"][0]["program_template_tags"] == ["fixed_e_counting"]
        and payload["stable_items"][1]["validation_status"] == "corrected_source_validated"
        and payload["stable_items"][2]["program_template_tags"] == ["bounded_pair_counts"]
        and payload["stable_items"][3]["program_template_tags"] == ["prime_power_factorization_5040"]
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    batch = read_json(OUT_DIR / "pyth2_reinterpretation_batch_001.json")
    unresolved_batch = read_json(OUT_DIR / "pyth2_page110_unresolved_reinterpretation_batch_001.json")
    page75_batch = read_json(OUT_DIR / "pyth2_page75_prime_factor_batch_001.json")
    payload = build_workset(batch, unresolved_batch, page75_batch)
    write_json(OUT_DIR / "pyth2_validated_workset.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_validated_workset.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
