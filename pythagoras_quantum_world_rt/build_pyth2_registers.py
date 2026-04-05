#!/usr/bin/env python3
"""Build Pyth-2 issue and prior-art alignment registers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"

ISSUE_SEVERITY = {
    "likely typo": "low",
    "none": "none",
}

NON_BLOCKING_ISSUE_LABELS = {"likely typo", "none"}

THEORY_LANE_BY_CLASS = {
    "counting_constraint": "counting_and_admissibility",
    "general_theory_item": "general_qa_theory",
    "periodicity_rule": "fibonacci_periodicity",
    "prime_factor_encoding": "prime_factor_encoding",
    "symmetry_rule": "reflection_symmetry",
}

STACK_ROLE_MAP = {
    "pythagorean_triples_and_theorem": "Provides the tuple-generation background beneath the counting rule or symmetry claim.",
    "rational_trigonometry": "Supplies the RT interpretation layer when the counting rule is pushed into quadrance language.",
    "chromogeometry": "Anchors any later C/F/G reading of the admissible tuple families.",
    "ptolemy_quadrance_and_uhg": "Places fixed-generator counting claims inside the wider Pythagorean/UHG parent structure.",
    "egyptian_fractions_and_hat": "Connects Fibonacci and direction-ratio structure to the HAT/Egyptian-fraction prior-art line.",
}


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def stack_alignment_entries(item: dict[str, object]) -> list[dict[str, object]]:
    entries = []
    for ref in item["prior_art_refs"]:
        key = str(ref["key"])
        entries.append({"key": key, "refs": ref["refs"], "role": STACK_ROLE_MAP[key], "summary": ref["summary"]})
    return entries


def project_anchor_paths(stack_entries: list[dict[str, object]]) -> list[str]:
    paths: list[str] = []
    for entry in stack_entries:
        for ref in entry["refs"]:
            path = ref.get("path")
            if path and path not in paths:
                paths.append(path)
    return paths


def open_brain_timestamps(stack_entries: list[dict[str, object]]) -> list[str]:
    timestamps: list[str] = []
    for entry in stack_entries:
        for ref in entry["refs"]:
            ts = ref.get("open_brain_timestamp")
            if ts and ts not in timestamps:
                timestamps.append(str(ts))
    return timestamps


def resolution_path(item: dict[str, object]) -> dict[str, str]:
    if item["issue_label"] == "likely typo":
        return {
            "action": "normalize_source_wording_and_promote",
            "owner_lane": "corrected_source_validation",
            "reason": "The mathematics is stable and only the surface wording is damaged.",
        }
    return {
        "action": "promote_to_validated_queue",
        "owner_lane": "validated_theory",
        "reason": "The reinterpretation is already stable enough for downstream work.",
    }


def issue_entry(item: dict[str, object]) -> dict[str, object]:
    stack_entries = stack_alignment_entries(item)
    return {
        "classification": item["classification"],
        "formal_claim": item["formal_claim"],
        "id": item["id"],
        "issue_label": item["issue_label"],
        "next_step": item["next_step"],
        "open_brain_timestamps": open_brain_timestamps(stack_entries),
        "priority_lane": "correct_and_keep" if item["issue_label"] == "likely typo" else "validated",
        "project_anchor_paths": project_anchor_paths(stack_entries),
        "qa_formulas": item["qa_formulas"],
        "resolution_path": resolution_path(item),
        "severity": ISSUE_SEVERITY[item["issue_label"]],
        "source": item["source"],
        "source_answer": item["source_answer"],
        "source_question": item["source_question"],
        "stack_alignment": stack_entries,
        "theory_lane": THEORY_LANE_BY_CLASS[item["classification"]["intake_class"]],
        "validated_examples": item["validated_examples"],
    }


def accepted_entry(item: dict[str, object]) -> dict[str, object]:
    stack_entries = stack_alignment_entries(item)
    stable_status = "stable"
    if item["issue_label"] == "likely typo":
        stable_status = "corrected_source_stable"
    elif item["issue_label"] != "none":
        stable_status = "flagged"
    return {
        "classification": item["classification"],
        "formal_claim": item["formal_claim"],
        "id": item["id"],
        "implementation_readiness": "ready" if item["issue_label"] == "none" else "corrected_source_ready",
        "issue_label": item["issue_label"],
        "next_step": item["next_step"],
        "open_brain_timestamps": open_brain_timestamps(stack_entries),
        "project_anchor_paths": project_anchor_paths(stack_entries),
        "qa_formulas": item["qa_formulas"],
        "rt_bridge": item["rt_bridge"],
        "chromo_bridge": item["chromo_bridge"],
        "source": item["source"],
        "stable_status": stable_status,
        "stack_alignment": stack_entries,
        "theory_lane": THEORY_LANE_BY_CLASS[item["classification"]["intake_class"]],
    }


def build_issue_register(batch: dict[str, object]) -> dict[str, object]:
    items = [issue_entry(item) for item in batch["items"] if item["issue_label"] not in NON_BLOCKING_ISSUE_LABELS]
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
    counts_by_status: dict[str, int] = {}
    for item in items:
        status = str(item["stable_status"])
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
    return {
        "items": items,
        "source_batch_id": batch["batch_id"],
        "summary": {
            "counts_by_stability": counts_by_status,
            "item_count": len(items),
            "series": batch["summary"]["series"],
        },
    }


def self_test() -> int:
    batch = {
        "batch_id": "pyth2_reinterpretation_batch_001",
        "items": [
            {
                "classification": {"intake_class": "periodicity_rule"},
                "chromo_bridge": "cg",
                "formal_claim": "claim",
                "id": "x",
                "issue_label": "likely typo",
                "next_step": "next",
                "prior_art_refs": [{"key": "egyptian_fractions_and_hat", "refs": [{"path": "docs/QA_PRIOR_ART_CONVERGENCE.md"}], "summary": "summary"}],
                "qa_formulas": ["5 divides F_n iff 5 divides n"],
                "rt_bridge": "rt",
                "source": {"page": 110},
                "source_answer": "answer",
                "source_question": "question",
                "validated_examples": [],
            }
        ],
        "summary": {"series": "Pyth-2"},
    }
    issue_register = build_issue_register(batch)
    alignment = build_alignment_register(batch)
    ok = (
        issue_register["summary"]["flagged_count"] == 0
        and alignment["items"][0]["implementation_readiness"] == "corrected_source_ready"
        and alignment["items"][0]["stable_status"] == "corrected_source_stable"
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
    issue_register = build_issue_register(batch)
    alignment = build_alignment_register(batch)
    write_json(OUT_DIR / "pyth2_issue_register.json", issue_register)
    write_json(OUT_DIR / "pyth2_prior_art_alignment.json", alignment)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    str(OUT_DIR / "pyth2_issue_register.json"),
                    str(OUT_DIR / "pyth2_prior_art_alignment.json"),
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
