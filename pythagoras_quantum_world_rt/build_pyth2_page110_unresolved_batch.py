#!/usr/bin/env python3
"""Build a controlled reinterpretation batch for the remaining Pyth-2 page-110 items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


UNRESOLVED_SPECS = {
    "pyth2_p110_l7743": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "In the Fibonacci sequence, even values occur exactly at indices divisible by 3, so the even terms appear every third number.",
        "normalized_answer": "Every third number.",
        "qa_formulas": [
            "2 divides F_n iff 3 divides n",
        ],
        "validated_examples": [
            {"indices": [3, 6, 9, 12], "values": [2, 8, 34, 144]}
        ],
        "rt_bridge": "No direct RT law; this is a Fibonacci periodicity rule in the discrete counting layer.",
        "chromo_bridge": "None directly; the claim stays in integer periodicity before any quadrance translation.",
        "next_step": "Promote together with the 3-, 4-, and 5-divisibility timing rules as a periodicity mini-table.",
    },
    "pyth2_p110_l7747": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "Using the local tri-number convention, Fibonacci terms in the 3-tri class occur exactly at indices divisible by 4, so they appear every fourth term.",
        "normalized_answer": "Every fourth number.",
        "qa_formulas": [
            "3 divides F_n iff 4 divides n",
        ],
        "validated_examples": [
            {"indices": [4, 8, 12, 16], "values": [3, 21, 144, 987]}
        ],
        "rt_bridge": "No direct RT law; this is a discrete divisibility cadence in the Fibonacci layer.",
        "chromo_bridge": "None directly; this remains a pre-quadrance periodicity rule.",
        "next_step": "Promote with the adjacent Fibonacci divisibility timing rules as a stabilized periodicity row.",
    },
    "pyth2_p110_l7751": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "Using the local 4-par convention, Fibonacci terms divisible by 4 occur exactly at indices divisible by 6, so they appear every sixth term.",
        "normalized_answer": "Every sixth number.",
        "qa_formulas": [
            "4 divides F_n iff 6 divides n",
        ],
        "validated_examples": [
            {"indices": [6, 12, 18], "values": [8, 144, 2584]}
        ],
        "rt_bridge": "No direct RT law; this is another Fibonacci divisibility cadence in the discrete counting layer.",
        "chromo_bridge": "None directly; this remains a pre-quadrance periodicity rule.",
        "next_step": "Promote with the adjacent Fibonacci divisibility timing rules as a stabilized periodicity row.",
    },
    "pyth2_p110_l7760": {
        "issue_label": "none",
        "source_verdict": "corrected",
        "formal_claim": "The counts 15, 13, and 35 are recovered by counting ordered coprime bead pairs (b,e) under the stated bounds. With b restricted to odd values and both entries less than 7, there are 15 such pairs; with both less than 6 there are 13; and with both less than 8 and b allowed odd or even there are 35.",
        "normalized_answer": "There are 15 coprime ordered pairs with both entries less than 7 and b odd, 13 with both entries less than 6 and b odd, and 35 with both entries less than 8 when b may be odd or even.",
        "qa_formulas": [
            "count pairs (b,e) with gcd(b,e)=1",
            "b odd restriction for the first two counts",
        ],
        "validated_examples": [
            {"bound": 7, "b_parity": "odd", "count": 15},
            {"bound": 6, "b_parity": "odd", "count": 13},
            {"bound": 8, "b_parity": "any", "count": 35},
        ],
        "rt_bridge": "This is a generator counting problem before any quadrance objects are formed.",
        "chromo_bridge": "Indirect only: the counts constrain which tuple directions are available to the chromogeometric layer.",
        "next_step": "Promote as a counted admissibility example for bounded bead pairs.",
    },
    "pyth2_p110_l7765": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "For bead pairs (e,d) with 1<=e<=d<8 and gcd(e,d)=1, there are 18 admissible pairs when the unity root pair (1,1) is included. The nearby note that the lower-left (1,1) space is left blank applies to the triangular e-d table display, not to the underlying admissible-pair count asked in the question.",
        "normalized_answer": "18 admissible e-d bead pairs with d less than 8, counting the unity root pair (1,1).",
        "qa_formulas": [
            "count pairs (e,d) with gcd(e,d)=1",
            "1<=e<=d<8",
            "include the unity root pair (1,1)",
        ],
        "validated_examples": [
            {"bound_on_d": 8, "count": 18, "include_unity_root_pair": True},
            {"bound_on_d": 8, "visible_triangle_cells_if_blank_corner": 17},
        ],
        "rt_bridge": "This is a direction-pair counting rule at the QA tuple layer before any quadrance translation.",
        "chromo_bridge": "Indirect only: the count constrains the candidate direction pairs available before any chromogeometric mapping.",
        "next_step": "Promote as the stabilized e-d admissible-pair count, with the blank lower-left table cell retained only as a display convention note.",
    },
}


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_items(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def build_item(workbook_item: dict[str, object]) -> dict[str, object]:
    item_id = str(workbook_item["id"])
    spec = UNRESOLVED_SPECS[item_id]
    return {
        "classification": workbook_item["classification"],
        "formal_claim": spec["formal_claim"],
        "id": item_id,
        "issue_label": spec["issue_label"],
        "next_step": spec["next_step"],
        "normalized_answer": spec["normalized_answer"],
        "prior_art_keys": workbook_item["prior_art_keys"],
        "prior_art_refs": workbook_item["prior_art_refs"],
        "qa_formulas": spec["qa_formulas"],
        "rt_bridge": spec["rt_bridge"],
        "source": workbook_item["source"],
        "source_answer": workbook_item["source_answer"],
        "source_question": workbook_item["source_question"],
        "source_verdict": spec["source_verdict"],
        "status": "reinterpreted_from_page110_leftovers",
        "validated_examples": spec["validated_examples"],
        "chromo_bridge": spec["chromo_bridge"],
    }


def self_test() -> int:
    sample = {
        "classification": {"intake_class": "counting_constraint"},
        "id": "pyth2_p110_l7765",
        "prior_art_keys": ["chromogeometry"],
        "prior_art_refs": [{"key": "chromogeometry"}],
        "source": {"page": 110},
        "source_answer": "dummy",
        "source_question": "dummy",
    }
    item = build_item(sample)
    ok = (
        item["issue_label"] == "none"
        and item["validated_examples"][0]["count"] == 18
        and item["status"] == "reinterpreted_from_page110_leftovers"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    ocr_queue = read_items(OUT_DIR / "pyth2_ocr_cleanup_queue.json")
    manual_queue = read_items(OUT_DIR / "pyth2_manual_review_queue.json")
    page110_items = [item for item in (ocr_queue + manual_queue) if item["source"]["page"] == 110]
    batch = [build_item(item) for item in page110_items]
    payload = {
        "batch_id": "pyth2_page110_unresolved_reinterpretation_batch_001",
        "items": batch,
        "summary": {
            "flagged_count": sum(1 for item in batch if item["issue_label"] != "none"),
            "page": 110,
            "series": "Pyth-2",
            "total_items": len(batch),
        },
    }
    write_json(OUT_DIR / "pyth2_page110_unresolved_reinterpretation_batch_001.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_page110_unresolved_reinterpretation_batch_001.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
