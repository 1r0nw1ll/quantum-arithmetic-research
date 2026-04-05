#!/usr/bin/env python3
"""Build the isolated page-75 issue register for the unrecovered Quantum Prime notation item."""

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


def recovery_index(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(item["id"]): item for item in payload.get("items", [])}


def build_register(
    batch: dict[str, object], recovery_payload: dict[str, object] | None = None
) -> dict[str, object]:
    recovery_by_id = recovery_index(recovery_payload or {})
    items = []
    for item in batch["items"]:
        if item["issue_label"] != "OCR corruption":
            continue
        recovery = recovery_by_id.get(str(item["id"]))
        next_step = item["next_step"]
        resolution_reason = "The notation string itself is too damaged to validate from this witness alone."
        if recovery is not None:
            best = recovery["best_candidate"]
            next_step = (
                "Keep isolated. Local evidence now supports the probable compact recovery "
                f"{best['compact_prime_numbering']} ({best['expanded_form']}), "
                "but do not promote it without a cleaner witness."
            )
            resolution_reason = (
                "Local evidence now supports a probable reconstruction, but it still rests on "
                "one damaged OCR witness plus convention inference."
            )
        register_item = {
            "classification": item["classification"],
            "formal_claim": item["formal_claim"],
            "id": item["id"],
            "issue_label": item["issue_label"],
            "next_step": next_step,
            "normalized_answer": item["normalized_answer"],
            "priority_lane": "page75_notation_holdout",
            "qa_formulas": item["qa_formulas"],
            "resolution_path": {
                "action": "recover_quantum_prime_notation_witness",
                "owner_lane": "page75_ocr_recovery",
                "reason": resolution_reason,
            },
            "severity": "high",
            "source": item["source"],
            "source_answer": item["source_answer"],
            "source_question": item["source_question"],
            "validated_examples": item["validated_examples"],
        }
        if recovery is not None:
            register_item["notation_recovery"] = {
                "best_candidate": recovery["best_candidate"],
                "candidate_status": recovery["candidate_status"],
                "promotion_decision": recovery["promotion_decision"],
                "supporting_candidates": recovery["supporting_candidates"],
                "unresolved_points": recovery["unresolved_points"],
            }
        items.append(register_item)
    return {
        "items": items,
        "source_batch_id": batch["batch_id"],
        "summary": {
            "flagged_count": len(items),
            "page": 75,
            "series": "Pyth-2",
        },
    }


def self_test() -> int:
    batch = {
        "batch_id": "b",
        "items": [
            {
                "classification": {"intake_class": "prime_factor_encoding"},
                "formal_claim": "claim",
                "id": "x",
                "issue_label": "OCR corruption",
                "next_step": "next",
                "normalized_answer": "hold",
                "qa_formulas": ["5040 = 2^4 * 3^2 * 5 * 7"],
                "source": {"page": 75},
                "source_answer": "sa",
                "source_question": "sq",
                "validated_examples": [],
            }
        ],
    }
    recovery_payload = {
        "items": [
            {
                "best_candidate": {
                    "compact_prime_numbering": "24325171",
                    "expanded_form": "2^4 3^2 5^1 7^1",
                },
                "candidate_status": "probable_recovery_not_validated",
                "id": "x",
                "promotion_decision": {"decision": "keep_isolated"},
                "supporting_candidates": [{"compact_prime_numbering": "243257"}],
                "unresolved_points": ["u1"],
            }
        ]
    }
    payload = build_register(batch, recovery_payload)
    ok = (
        payload["summary"]["flagged_count"] == 1
        and payload["items"][0]["priority_lane"] == "page75_notation_holdout"
        and payload["items"][0]["notation_recovery"]["best_candidate"]["compact_prime_numbering"] == "24325171"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    batch = read_json(OUT_DIR / "pyth2_page75_prime_factor_batch_001.json")
    recovery_payload = None
    recovery_path = OUT_DIR / "pyth2_page75_notation_recovery_candidates.json"
    if recovery_path.exists():
        recovery_payload = read_json(recovery_path)
    payload = build_register(batch, recovery_payload)
    write_json(OUT_DIR / "pyth2_page75_issue_register.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_page75_issue_register.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
