#!/usr/bin/env python3
"""Build Pyth-1 issue and prior-art alignment registers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"

ISSUE_SEVERITY = {
    "OCR corruption": "high",
    "none": "none",
    "source-layer ambiguity": "medium",
    "true contradiction requiring investigation": "critical",
}

THEORY_LANE_BY_CLASS = {
    "base_factorization": "generator_to_quadrance_inversion",
    "combinatorial_constraint": "generator_admissibility",
    "general_theory_item": "general_qa_theory",
    "symbolic_conversion": "derived_invariant_normalization",
    "table_graph_lookup": "table_and_graph_dynamics",
    "trace_sequence": "koenig_trace_dynamics",
    "tuple_reconstruction": "generator_reconstruction",
}

STACK_ROLE_MAP = {
    "pythagorean_triples_and_theorem": "Provides the tuple-to-triple arithmetic witness for the claimed bead reconstruction or exclusion.",
    "rational_trigonometry": "Supplies the polynomial RT layer once the tuple has been translated into quadrance language.",
    "chromogeometry": "Interprets (C,F,G) as green/red/blue quadrances and anchors the metric meaning of the item.",
    "ptolemy_quadrance_and_uhg": "Places the tuple or quadrance claim in the parent cyclic-quadrilateral/null-point layer above Pythagoras.",
    "egyptian_fractions_and_hat": "Connects direction ratios, Koenig traces, and tree navigation to the HAT/Egyptian-fraction line of prior art.",
}

LANE_PRIORITY = {
    "OCR corruption": "transcription_first",
    "none": "ready_for_table_and_program_encoding",
    "source-layer ambiguity": "stabilize_before_program_reproduction",
    "true contradiction requiring investigation": "hold_out_of_validated_theory_layer",
}


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def stack_alignment_entries(item: dict[str, object]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for ref in item["prior_art_refs"]:
        key = str(ref["key"])
        entries.append(
            {
                "key": key,
                "role": STACK_ROLE_MAP[key],
                "summary": ref["summary"],
                "refs": ref["refs"],
            }
        )
    return entries


def project_anchor_paths(stack_entries: list[dict[str, object]]) -> list[str]:
    paths: list[str] = []
    for entry in stack_entries:
        for ref in entry["refs"]:
            if ref.get("path") and ref["path"] not in paths:
                paths.append(ref["path"])
    return paths


def open_brain_timestamps(stack_entries: list[dict[str, object]]) -> list[str]:
    timestamps: list[str] = []
    for entry in stack_entries:
        for ref in entry["refs"]:
            timestamp = ref.get("open_brain_timestamp")
            if timestamp and timestamp not in timestamps:
                timestamps.append(str(timestamp))
    return timestamps


def resolution_path(item: dict[str, object]) -> dict[str, str]:
    issue_label = str(item["issue_label"])
    intake_class = str(item["classification"]["intake_class"])
    if issue_label == "OCR corruption":
        return {
            "action": "correct_source_witness_and_rebuild_trace",
            "owner_lane": "transcription_cleanup",
            "reason": "The source answer conflicts with the question context and adjacent corroborating items.",
        }
    if issue_label == "true contradiction requiring investigation":
        return {
            "action": "search_for_hidden_generator_constraint",
            "owner_lane": "mathematical_review",
            "reason": "The source uniqueness claim is not recovered by the current tuple laws or RT/chromogeometry bridge.",
        }
    if issue_label == "source-layer ambiguity" and intake_class == "symbolic_conversion":
        return {
            "action": "normalize_symbolic_identity_against_tuple_substitutions",
            "owner_lane": "symbolic_validation",
            "reason": "The algebra can be stabilized by explicit substitution through d=b+e and a=b+2e.",
        }
    if issue_label == "source-layer ambiguity" and intake_class == "tuple_reconstruction":
        return {
            "action": "stabilize_factorization_language",
            "owner_lane": "generator_reconstruction_review",
            "reason": "The source answer is mathematically recoverable but needs cleaned factor-pair wording before reuse.",
        }
    if issue_label == "source-layer ambiguity":
        return {
            "action": "verify_table_reference_against_local_context",
            "owner_lane": "table_lookup_review",
            "reason": "The statement is likely recoverable but depends on nearby source conventions and table wording.",
        }
    return {
        "action": "promote_to_validated_queue",
        "owner_lane": "validated_theory",
        "reason": "The reinterpretation is already stable enough for downstream table or program work.",
    }


def issue_entry(item: dict[str, object]) -> dict[str, object]:
    stack_entries = stack_alignment_entries(item)
    return {
        "id": item["id"],
        "issue_label": item["issue_label"],
        "severity": ISSUE_SEVERITY[item["issue_label"]],
        "priority_lane": LANE_PRIORITY[item["issue_label"]],
        "theory_lane": THEORY_LANE_BY_CLASS[item["classification"]["intake_class"]],
        "source": item["source"],
        "source_question": item["source_question"],
        "source_answer": item["source_answer"],
        "formal_claim": item["formal_claim"],
        "source_verdict": item["source_verdict"],
        "qa_formulas": item["qa_formulas"],
        "validated_examples": item["validated_examples"],
        "classification": item["classification"],
        "resolution_path": resolution_path(item),
        "stack_alignment": stack_entries,
        "project_anchor_paths": project_anchor_paths(stack_entries),
        "open_brain_timestamps": open_brain_timestamps(stack_entries),
        "next_step": item["next_step"],
    }


def accepted_entry(item: dict[str, object]) -> dict[str, object]:
    stack_entries = stack_alignment_entries(item)
    stable_status = "stable" if item["issue_label"] == "none" else "flagged"
    return {
        "id": item["id"],
        "stable_status": stable_status,
        "issue_label": item["issue_label"],
        "theory_lane": THEORY_LANE_BY_CLASS[item["classification"]["intake_class"]],
        "classification": item["classification"],
        "source": item["source"],
        "formal_claim": item["formal_claim"],
        "qa_formulas": item["qa_formulas"],
        "rt_bridge": item["rt_bridge"],
        "chromo_bridge": item["chromo_bridge"],
        "stack_alignment": stack_entries,
        "project_anchor_paths": project_anchor_paths(stack_entries),
        "open_brain_timestamps": open_brain_timestamps(stack_entries),
        "implementation_readiness": LANE_PRIORITY[item["issue_label"]],
        "next_step": item["next_step"],
    }


def build_issue_register(batch: dict[str, object]) -> dict[str, object]:
    items = [issue_entry(item) for item in batch["items"] if item["issue_label"] != "none"]
    counts_by_label: dict[str, int] = {}
    for item in items:
        label = str(item["issue_label"])
        counts_by_label[label] = counts_by_label.get(label, 0) + 1
    return {
        "items": items,
        "source_batch_id": batch["batch_id"],
        "summary": {
            "counts_by_label": counts_by_label,
            "flagged_count": len(items),
            "series": batch["summary"]["series"],
        },
    }


def build_alignment_register(batch: dict[str, object]) -> dict[str, object]:
    items = [accepted_entry(item) for item in batch["items"]]
    counts_by_lane: dict[str, int] = {}
    counts_by_status: dict[str, int] = {}
    for item in items:
        lane = str(item["theory_lane"])
        counts_by_lane[lane] = counts_by_lane.get(lane, 0) + 1
        status = str(item["stable_status"])
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
    return {
        "items": items,
        "source_batch_id": batch["batch_id"],
        "summary": {
            "counts_by_stability": counts_by_status,
            "counts_by_theory_lane": counts_by_lane,
            "item_count": len(items),
            "series": batch["summary"]["series"],
        },
    }


def self_test() -> int:
    batch = {
        "batch_id": "pyth1_reinterpretation_batch_001",
        "items": [
            {
                "classification": {
                    "intake_class": "tuple_reconstruction",
                    "next_operation": "formalize_tuple_constraint",
                    "rt_bridge_mode": "direct_quadrance_bridge",
                },
                "chromo_bridge": "c",
                "formal_claim": "claim",
                "id": "pyth1_test",
                "issue_label": "source-layer ambiguity",
                "next_step": "next",
                "prior_art_refs": [
                    {
                        "key": "chromogeometry",
                        "refs": [
                            {"kind": "doc", "path": "docs/families/125_qa_chromogeometry.md"},
                            {"kind": "open_brain", "open_brain_timestamp": "2026-03-29T05:09:31.997Z"},
                        ],
                        "summary": "summary",
                    }
                ],
                "qa_formulas": ["d=b+e"],
                "rt_bridge": "rt",
                "source": {"page": 16},
                "source_answer": "answer",
                "source_question": "question",
                "source_verdict": "corrected",
                "validated_examples": [{"tuple": {"a": 7}}],
            }
        ],
        "summary": {"series": "Pyth-1"},
    }
    issue_register = build_issue_register(batch)
    alignment_register = build_alignment_register(batch)
    ok = (
        issue_register["summary"]["flagged_count"] == 1
        and issue_register["items"][0]["resolution_path"]["owner_lane"] == "generator_reconstruction_review"
        and alignment_register["summary"]["counts_by_stability"]["flagged"] == 1
        and alignment_register["items"][0]["project_anchor_paths"] == ["docs/families/125_qa_chromogeometry.md"]
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    batch = read_json(OUT_DIR / "pyth1_reinterpretation_batch_001.json")
    issue_register = build_issue_register(batch)
    alignment_register = build_alignment_register(batch)

    write_json(OUT_DIR / "pyth1_issue_register.json", issue_register)
    write_json(OUT_DIR / "pyth1_prior_art_alignment.json", alignment_register)

    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    str(OUT_DIR / "pyth1_issue_register.json"),
                    str(OUT_DIR / "pyth1_prior_art_alignment.json"),
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
