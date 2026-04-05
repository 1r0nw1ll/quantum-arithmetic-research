#!/usr/bin/env python3
"""Build the resolved analysis for the Pyth-2 page-110 e-d count item."""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def coprime_ed_pairs(max_d_exclusive: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for d in range(1, max_d_exclusive):
        for e in range(1, d + 1):
            if gcd(e, d) == 1:
                rows.append([e, d])
    return rows


def build_payload(batch: dict[str, object]) -> dict[str, object]:
    target = None
    for item in batch["items"]:
        if item["id"] == "pyth2_p110_l7765":
            target = item
            break
    if target is None:
        raise ValueError("pyth2_p110_l7765 not present in page-110 reinterpretation batch")

    with_root = coprime_ed_pairs(8)
    without_root = [row for row in with_root if row != [1, 1]]
    return {
        "conflict_status": "resolved_by_local_generation_rule",
        "counts": {
            "with_unity_root_pair": len(with_root),
            "without_unity_root_pair": len(without_root),
        },
        "evidence": [
            {
                "kind": "question_answer",
                "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                "quote_excerpt": "How many pairs of bead number e-d are there with d less than 8? Ans: 18.",
                "source_lines": [7764, 7766],
            },
            {
                "kind": "table_convention",
                "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                "quote_excerpt": "if the vertical axis represents e, and the horizontal axis represents d ... the lower left space (1,1) will be left blank.",
                "source_lines": [7288, 7291],
            },
            {
                "kind": "generation_rule",
                "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                "quote_excerpt": "b and e must be prime to each other except when either one or both are equal to unity. They may not be equal in any other case.",
                "source_lines": [873, 874],
            },
            {
                "kind": "coprime_definition",
                "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                "quote_excerpt": "Any group of numbers is co-prime when they have no common factor ... other than unity.",
                "source_lines": [2061, 2063],
            },
        ],
        "holdout_decision": {
            "decision": "promote_to_validated_lane",
            "reason": "The source answer 18 matches the admissible-pair generation rule including unity; the blank (1,1) note is a display convention for the triangular table.",
        },
        "id": "pyth2_p110_l7765",
        "interpretations": [
            {
                "count": len(with_root),
                "decision_basis": "matches the source answer and the local rule allowing equality when one or both entries are unity",
                "include_unity_root_pair": True,
                "label": "admissible_pair_count",
                "pairs": with_root,
            },
            {
                "count": len(without_root),
                "decision_basis": "matches the visible plotted cells if the lower-left display corner is left blank",
                "include_unity_root_pair": False,
                "label": "visible_triangle_cells_only",
                "pairs": without_root,
            },
        ],
        "normalized_answer": target["normalized_answer"],
        "source": target["source"],
        "source_question": target["source_question"],
        "summary": {
            "page": 110,
            "series": "Pyth-2",
            "supporting_evidence_count": 4,
        },
    }


def self_test() -> int:
    batch = {
        "items": [
            {
                "id": "pyth2_p110_l7765",
                "normalized_answer": "hold",
                "source": {"page": 110},
                "source_question": "q",
            }
        ]
    }
    payload = build_payload(batch)
    ok = (
        payload["counts"]["with_unity_root_pair"] == 18
        and payload["counts"]["without_unity_root_pair"] == 17
        and payload["interpretations"][0]["pairs"][0] == [1, 1]
        and payload["holdout_decision"]["decision"] == "promote_to_validated_lane"
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
    payload = build_payload(batch)
    write_json(OUT_DIR / "pyth2_page110_ed_conflict_analysis.json", payload)
    print(
        canonical_dump(
            {"ok": True, "outputs": [str(OUT_DIR / "pyth2_page110_ed_conflict_analysis.json")]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
