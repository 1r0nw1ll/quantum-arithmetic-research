#!/usr/bin/env python3
"""Build the isolated page-110 issue register for Pyth-2 convention-sensitive leftovers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def conflict_index(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(payload["id"]): payload} if payload else {}


def issue_entry(item: dict[str, object], conflict_payload: dict[str, object] | None = None) -> dict[str, object]:
    entry = {
        "classification": item["classification"],
        "formal_claim": item["formal_claim"],
        "id": item["id"],
        "issue_label": item["issue_label"],
        "next_step": item["next_step"],
        "normalized_answer": item["normalized_answer"],
        "priority_lane": "page110_convention_holdout",
        "qa_formulas": item["qa_formulas"],
        "resolution_path": {
            "action": "document_local_term_or_counting_convention",
            "owner_lane": "page110_source_convention_review",
            "reason": "The mathematics is plausible, but the local source convention is not yet stabilized.",
        },
        "severity": "medium",
        "source": item["source"],
        "source_answer": item["source_answer"],
        "source_question": item["source_question"],
        "validated_examples": item["validated_examples"],
    }
    if conflict_payload is not None and item["id"] == conflict_payload.get("id"):
        entry["conflict_analysis"] = {
            "conflict_status": conflict_payload["conflict_status"],
            "counts": conflict_payload["counts"],
            "holdout_decision": conflict_payload["holdout_decision"],
            "interpretations": [
                {
                    "count": candidate["count"],
                    "include_unity_root_pair": candidate["include_unity_root_pair"],
                    "label": candidate["label"],
                }
                for candidate in conflict_payload["interpretations"]
            ],
        }
        entry["resolution_path"]["reason"] = (
            "The nearby e-d table convention and the explicit source answer disagree on whether (1,1) is counted."
        )
        entry["severity"] = "high"
    return entry


def build_register(batch: dict[str, object], conflict_payload: dict[str, object] | None = None) -> dict[str, object]:
    items = [
        issue_entry(item, conflict_payload) for item in batch["items"] if item["issue_label"] != "none"
    ]
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
            "page": 110,
            "series": "Pyth-2",
        },
    }


def self_test() -> int:
    batch = {
        "batch_id": "b",
        "items": [
            {
                "classification": {"intake_class": "periodicity_rule"},
                "formal_claim": "claim",
                "id": "x",
                "issue_label": "source-layer ambiguity",
                "next_step": "next",
                "normalized_answer": "answer",
                "qa_formulas": ["3 divides F_n iff 4 divides n"],
                "source": {"page": 110},
                "source_answer": "sa",
                "source_question": "sq",
                "validated_examples": [],
            }
        ],
    }
    conflict_payload = {
        "conflict_status": "unresolved_local_convention_conflict",
        "counts": {"with_unity_root_pair": 18, "without_unity_root_pair": 17},
        "holdout_decision": {"decision": "keep_isolated"},
        "id": "x",
        "interpretations": [
            {"count": 18, "include_unity_root_pair": True, "label": "count_the_unity_root_pair"},
            {"count": 17, "include_unity_root_pair": False, "label": "blank_the_lower_left_ed_cell"},
        ],
    }
    payload = build_register(batch, conflict_payload)
    ok = (
        payload["summary"]["flagged_count"] == 1
        and payload["items"][0]["priority_lane"] == "page110_convention_holdout"
        and payload["items"][0]["severity"] == "high"
        and payload["items"][0]["conflict_analysis"]["counts"]["without_unity_root_pair"] == 17
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    batch = read_json(OUT_DIR / "pyth2_page110_unresolved_reinterpretation_batch_001.json")
    conflict_payload = None
    conflict_path = OUT_DIR / "pyth2_page110_ed_conflict_analysis.json"
    if conflict_path.exists():
        conflict_payload = read_json(conflict_path)
    payload = build_register(batch, conflict_payload)
    write_json(OUT_DIR / "pyth2_page110_issue_register.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_page110_issue_register.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
