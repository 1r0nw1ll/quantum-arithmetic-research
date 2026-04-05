#!/usr/bin/env python3
"""Build the first controlled Pyth-2 reinterpretation batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


INTERPRETATION_SPECS = {
    "pyth2_p110_l7731": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "Fixing e=60 and requiring b<60 with primitive admissibility leaves exactly one tuple for each prime or prime power b below 60 that is coprime to 60, namely b in {1,7,11,13,17,19,23,29,31,37,41,43,47,49,53,59}. The exclusions 2,3,5 and their multiples are exactly the values sharing a factor with 60.",
        "normalized_answer": "There is one admissible set for each prime number other than 2, 3, and 5, together with the prime power 49, below 60.",
        "qa_formulas": [
            "d=b+e",
            "a=b+2e",
            "primitive admissibility requires gcd(b,e)=1",
            "e=60",
        ],
        "validated_examples": [
            {"b_values": [1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59], "count": 16}
        ],
        "rt_bridge": "This is a generator-admissibility sieve before quadrance formation, with e fixed and b ranging over the coprime odd residues.",
        "chromo_bridge": "Indirect only: the item constrains which direction vectors may enter the chromogeometric layer for fixed e.",
        "next_step": "Promote as the first fixed-e admissibility table for Pyth-2.",
    },
    "pyth2_p110_l7738": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "The prime-compatible b-values for e=60 occur in complementary pairs around 30: whenever b is admissible, 60-b is also admissible, so the valid odd residues are symmetric under reflection through 30.",
        "normalized_answer": "The allowed prime-compatible b-values pair symmetrically about 30, with each pair summing to 60.",
        "qa_formulas": [
            "b + b' = 60",
            "gcd(b,60)=1 implies gcd(60-b,60)=1",
        ],
        "validated_examples": [
            {"pairs": [[1, 59], [7, 53], [11, 49], [13, 47], [17, 43], [19, 41], [23, 37], [29, 31]]}
        ],
        "rt_bridge": "This is a generator reflection symmetry on the admissible fixed-e slice.",
        "chromo_bridge": "Indirect only: the symmetry is imposed at the tuple layer before C/F/G are computed.",
        "next_step": "Use as the reflection-symmetry companion to the fixed-e admissibility table.",
    },
    "pyth2_p110_l7754": {
        "issue_label": "likely typo",
        "source_verdict": "corrected",
        "formal_claim": "In the standard Fibonacci sequence modulo the pentagonal divisibility rule, numbers divisible by 5 appear every fifth term. The source wording 'Every fifth rumber' is a typo-level OCR defect, not a mathematical issue.",
        "normalized_answer": "Every fifth number.",
        "qa_formulas": [
            "5 divides F_n iff 5 divides n",
        ],
        "validated_examples": [
            {"indices": [5, 10, 15, 20], "values": [5, 55, 610, 6765]}
        ],
        "rt_bridge": "No direct RT law; this is a Fibonacci periodicity rule feeding the QA counting layer.",
        "chromo_bridge": "None directly; the content stays in the integer-periodicity layer.",
        "next_step": "Keep as a clean periodicity rule and group with the adjacent every-third/every-fourth/every-sixth items later.",
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
    spec = INTERPRETATION_SPECS[item_id]
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
        "status": "reinterpreted",
        "validated_examples": spec["validated_examples"],
        "chromo_bridge": spec["chromo_bridge"],
    }


def self_test() -> int:
    sample = {
        "classification": {"intake_class": "periodicity_rule"},
        "id": "pyth2_p110_l7754",
        "prior_art_keys": ["chromogeometry"],
        "prior_art_refs": [{"key": "chromogeometry"}],
        "source": {"page": 110},
        "source_answer": "dummy",
        "source_question": "dummy",
    }
    item = build_item(sample)
    ok = (
        item["issue_label"] == "likely typo"
        and item["normalized_answer"] == "Every fifth number."
        and item["validated_examples"][0]["indices"][0] == 5
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    workbook = read_items(OUT_DIR / "pyth2_theory_workbook.json")
    accepted = [item for item in workbook if item["status"] == "queued"]
    batch = [build_item(item) for item in accepted]
    payload = {
        "batch_id": "pyth2_reinterpretation_batch_001",
        "items": batch,
        "summary": {
            "accepted_reinterpreted_count": len(batch),
            "flagged_issue_count": sum(1 for item in batch if item["issue_label"] != "none"),
            "series": "Pyth-2",
        },
    }
    write_json(OUT_DIR / "pyth2_reinterpretation_batch_001.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_reinterpretation_batch_001.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
