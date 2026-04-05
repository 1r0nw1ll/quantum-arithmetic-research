#!/usr/bin/env python3
"""Build the validated Pyth-1 workset for downstream table/program reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"

THEORY_LANE_BY_CLASS = {
    "base_factorization": "generator_to_quadrance_inversion",
    "combinatorial_constraint": "generator_admissibility",
    "general_theory_item": "general_qa_theory",
    "symbolic_conversion": "derived_invariant_normalization",
    "table_graph_lookup": "table_and_graph_dynamics",
    "trace_sequence": "koenig_trace_dynamics",
    "tuple_reconstruction": "generator_reconstruction",
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
    if item["id"] == "pyth1_p066_l5029":
        return "validated_core_with_raw_holdout"
    return "corrected_source_validated"


def validation_scope(item: dict[str, object]) -> str:
    if item["id"] == "pyth1_p066_l5029":
        return "core_values_through_K_only"
    return "full_item"


def program_template_tags(item: dict[str, object]) -> list[str]:
    intake_class = str(item["classification"]["intake_class"])
    tags: list[str] = []
    if intake_class == "tuple_reconstruction":
        tags.append("tuple_completion")
    if intake_class == "combinatorial_constraint":
        tags.append("generator_constraint")
    if intake_class == "symbolic_conversion":
        tags.append("derived_identity")
    if intake_class == "table_graph_lookup":
        tags.append("table3_block")
    if intake_class == "trace_sequence":
        tags.append("koenig_trace_reference")
    if intake_class == "general_theory_item":
        tags.append("construction_geometry")
    formulas = list(item.get("qa_formulas", []))
    if "C=2de" in formulas and "quadrance_level_set" not in tags:
        tags.append("quadrance_level_set")
    if "F=ab" in formulas and "triangle_generation" not in tags:
        tags.append("triangle_generation")
    return tags


def stable_item(item: dict[str, object]) -> dict[str, object]:
    normalized_answer = item.get("normalized_answer", item["source_answer"])
    return {
        "id": item["id"],
        "source": item["source"],
        "source_question": item["source_question"],
        "source_answer": item["source_answer"],
        "normalized_answer": normalized_answer,
        "formal_claim": item["formal_claim"],
        "issue_origin": item["issue_label"],
        "source_verdict": item["source_verdict"],
        "validation_status": validation_status(item),
        "validation_scope": validation_scope(item),
        "classification": item["classification"],
        "theory_lane": THEORY_LANE_BY_CLASS[item["classification"]["intake_class"]],
        "program_template_tags": program_template_tags(item),
        "qa_formulas": item["qa_formulas"],
        "validated_examples": item["validated_examples"],
        "prior_art_keys": item["prior_art_keys"],
        "prior_art_refs": item["prior_art_refs"],
        "rt_bridge": item["rt_bridge"],
        "chromo_bridge": item["chromo_bridge"],
        "next_step": item["next_step"],
    }


def holdout_item(item: dict[str, object]) -> dict[str, object]:
    return {
        "id": item["id"],
        "source": item["source"],
        "source_question": item["source_question"],
        "source_answer": item["source_answer"],
        "formal_claim": item["formal_claim"],
        "issue_label": item["issue_label"],
        "severity": item["severity"],
        "priority_lane": item["priority_lane"],
        "resolution_path": item["resolution_path"],
        "theory_lane": item["theory_lane"],
        "qa_formulas": item["qa_formulas"],
        "validated_examples": item["validated_examples"],
        "project_anchor_paths": item["project_anchor_paths"],
        "open_brain_timestamps": item["open_brain_timestamps"],
        "next_step": item["next_step"],
    }


def build_workset(
    accepted_batch: dict[str, object],
    unresolved_batch: dict[str, object],
    issue_register: dict[str, object],
) -> dict[str, object]:
    stable_direct = [stable_item(item) for item in accepted_batch["items"] if item["issue_label"] == "none"]
    stable_corrected = [stable_item(item) for item in unresolved_batch["items"]]
    holdouts = [holdout_item(item) for item in issue_register["items"]]
    stable_items = stable_direct + stable_corrected
    counts_by_status: dict[str, int] = {}
    for item in stable_items:
        status = str(item["validation_status"])
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
    return {
        "holdouts": holdouts,
        "stable_items": stable_items,
        "summary": {
            "corrected_validated_count": len(stable_corrected),
            "counts_by_validation_status": counts_by_status,
            "direct_validated_count": len(stable_direct),
            "holdout_count": len(holdouts),
            "series": "Pyth-1",
            "stable_total": len(stable_items),
        },
    }


def self_test() -> int:
    accepted_batch = {
        "items": [
            {
                "classification": {"intake_class": "tuple_reconstruction"},
                "formal_claim": "claim",
                "id": "a",
                "issue_label": "none",
                "next_step": "next",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": ["C=2de", "F=ab"],
                "rt_bridge": "rt",
                "chromo_bridge": "cg",
                "source": {"page": 1},
                "source_answer": "answer",
                "source_question": "question",
                "source_verdict": "faithful",
                "validated_examples": [],
            }
        ]
    }
    unresolved_batch = {
        "items": [
            {
                "classification": {"intake_class": "table_graph_lookup"},
                "formal_claim": "claim2",
                "id": "pyth1_p066_l5029",
                "issue_label": "OCR corruption",
                "next_step": "next2",
                "normalized_answer": "normalized",
                "prior_art_keys": [],
                "prior_art_refs": [],
                "qa_formulas": ["I=|C-F|"],
                "rt_bridge": "rt2",
                "chromo_bridge": "cg2",
                "source": {"page": 2},
                "source_answer": "answer2",
                "source_question": "question2",
                "source_verdict": "corrected",
                "validated_examples": [],
            }
        ]
    }
    issue_register = {
        "items": [
            {
                "formal_claim": "holdout",
                "id": "h1",
                "issue_label": "true contradiction requiring investigation",
                "next_step": "inspect",
                "open_brain_timestamps": [],
                "priority_lane": "hold",
                "project_anchor_paths": [],
                "qa_formulas": [],
                "resolution_path": {"action": "search"},
                "severity": "critical",
                "source": {"page": 3},
                "source_answer": "sa",
                "source_question": "sq",
                "theory_lane": "generator_admissibility",
                "validated_examples": [],
            }
        ]
    }
    payload = build_workset(accepted_batch, unresolved_batch, issue_register)
    ok = (
        payload["summary"]["stable_total"] == 2
        and payload["summary"]["holdout_count"] == 1
        and payload["stable_items"][0]["program_template_tags"] == ["tuple_completion", "quadrance_level_set", "triangle_generation"]
        and payload["stable_items"][1]["validation_scope"] == "core_values_through_K_only"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    accepted_batch = read_json(OUT_DIR / "pyth1_reinterpretation_batch_001.json")
    unresolved_batch = read_json(OUT_DIR / "pyth1_unresolved_reinterpretation_batch_001.json")
    issue_register = read_json(OUT_DIR / "pyth1_issue_register.json")
    payload = build_workset(accepted_batch, unresolved_batch, issue_register)
    write_json(OUT_DIR / "pyth1_validated_workset.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth1_validated_workset.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
