#!/usr/bin/env python3
"""Build the stabilized cows continuation lane for Pyth-2."""

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


def build_lane(cows_artifacts: dict[str, object], prior_art_payload: dict[str, object]) -> dict[str, object]:
    refs = prior_art_refs(prior_art_payload)
    stable_items = [
        {
            "formal_claim": "Substituting the stabilized bulls answer row into the four cow equations yields a fixed affine system with offsets 1869/2, 711, 3267/10, and 689.",
            "id": "pyth2_cows_substituted_affine_system",
            "issue_origin": "direct_validated",
            "normalized_answer": "Use the bulls-anchored affine cow system before attempting any ellipse reinterpretation.",
            "prior_art_refs": refs,
            "program_template_tags": ["cows_affine_system", "bulls_anchor_substitution"],
            "source_lines": [8278, 8292],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": cows_artifacts["substituted_equations"],
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "The original cows system has a unique rational solution with denominator 4657, so the fixed bulls anchor does not produce an integral cow solution under the equations exactly as stated.",
            "id": "pyth2_cows_rational_nonintegral_solution",
            "issue_origin": "direct_validated",
            "normalized_answer": "The literal cow system solves to a nonintegral rational point, not an integer cattle count.",
            "prior_art_refs": refs,
            "program_template_tags": ["cows_rational_solution", "integrality_obstruction"],
            "source_lines": [8278, 8301],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": cows_artifacts["exact_original_system_solution"],
            "validation_status": "direct_validated",
        },
        {
            "formal_claim": "The substituted cow equations impose definite congruence conditions: X2 ≡ 6 mod 12, Z2 ≡ 0 mod 20, Y2 ≡ 9 mod 30, and W2 ≡ 0 mod 42.",
            "id": "pyth2_cows_modular_constraints",
            "issue_origin": "direct_validated",
            "normalized_answer": "Read the page-124 cow continuation first as a modular constraint packet.",
            "prior_art_refs": refs,
            "program_template_tags": ["cows_mod_constraints", "fractional_divisibility"],
            "source_lines": [8286, 8301],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "validated_examples": cows_artifacts["modular_constraints"],
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
        for item in cows_artifacts["source_ambiguities"]
    ]
    return {
        "lane_id": "pyth2_cows_continuation_lane",
        "stable_items": stable_items,
        "summary": {
            "series": "Pyth-2",
            "stable_count": len(stable_items),
            "unresolved_count": len(unresolved_items),
        },
        "supporting_bulls_diagram_summary": cows_artifacts["supporting_bulls_diagram_summary"],
        "unresolved_items": unresolved_items,
    }


def self_test() -> int:
    cows_artifacts = {
        "substituted_equations": [{"normalized_affine_form": "W2 = 1869/2 + 7/12 X2"}],
        "exact_original_system_solution": {"all_integral": False, "denominator": 4657},
        "modular_constraints": [{"constraint": "X2 ≡ 6 (mod 12)"}],
        "source_ambiguities": [{"id": "amb", "issue_label": "source-layer ambiguity", "normalized_answer": "hold", "source_claim": "claim", "source_lines": [1, 2]}],
        "supporting_bulls_diagram_summary": {"figure26_row_count": 3},
    }
    prior_art_payload = {"categories": [{"key": key, "refs": [], "summary": key} for key in SELECTED_PRIOR_ART_KEYS]}
    payload = build_lane(cows_artifacts, prior_art_payload)
    ok = (
        payload["summary"]["stable_count"] == 3
        and payload["summary"]["unresolved_count"] == 1
        and payload["stable_items"][1]["validated_examples"]["denominator"] == 4657
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    cows_artifacts = read_json(OUT_DIR / "pyth2_cows_equation_artifacts.json")
    prior_art_payload = read_json(OUT_DIR / "prior_art_bridge_map.json")
    payload = build_lane(cows_artifacts, prior_art_payload)
    write_json(OUT_DIR / "pyth2_cows_continuation_lane.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_cows_continuation_lane.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
