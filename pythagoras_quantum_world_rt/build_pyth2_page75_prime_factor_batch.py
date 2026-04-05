#!/usr/bin/env python3
"""Build the first controlled reinterpretation batch for the noisier Pyth-2 page-75 prime-factor material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


PAGE75_SPECS = {
    "pyth2_p075_l5658": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "The wavelength decomposition of 5040 is the prime-power factorization 2^4 * 3^2 * 5 * 7, so the normalized wavelength factors are 16, 9, 5, and 7.",
        "normalized_answer": "The combined wavelengths are 16, 9, 5, and 7, obtained by collecting the powers of the primes in 5040.",
        "qa_formulas": [
            "5040 = 2^4 * 3^2 * 5 * 7",
            "group equal prime factors into prime powers",
        ],
        "validated_examples": [
            {"factorization": [16, 9, 5, 7], "product": 5040}
        ],
        "rt_bridge": "No direct RT law; this is a prime-factor encoding step in the QA combinatorial layer.",
        "chromo_bridge": "None directly; the item stays in discrete factor organization.",
        "next_step": "Keep as the normalized seed decomposition for the page-75 prime-factor lane.",
    },
    "pyth2_p075_l5663": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "The compact Quantum Prime numbering for 5040 is 24325171, encoding the prime-power factorization 2^4 * 3^2 * 5^1 * 7^1 in explicit prime-exponent tokens.",
        "normalized_answer": "In compact Quantum Prime numbering, 5040 is written as 24325171, meaning 2^4 3^2 5^1 7^1.",
        "qa_formulas": [
            "5040 = 2^4 * 3^2 * 5 * 7",
            "24325171 -> 2^4 3^2 5^1 7^1",
        ],
        "validated_examples": [
            {
                "compact_prime_numbering": "24325171",
                "factorization": [16, 9, 5, 7],
                "token_pairs": ["24", "32", "51", "71"],
                "value": 5040,
            }
        ],
        "rt_bridge": "No direct RT law; this is a compact symbolic encoding of the QA prime-power layer.",
        "chromo_bridge": "None.",
        "next_step": "Promote as corrected-source validated notation while preserving the OCR-recovery evidence trail.",
    },
    "pyth2_p075_l5668": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "Alternative wavelength groupings are allowed by combining coprime prime-power blocks. For 5040, combining 16 and 5 gives 80, leaving the coprime factors 9 and 7, so 5040 can be represented as 80 * 9 * 7.",
        "normalized_answer": "Yes. For example, combining 16 and 5 gives 80, so one valid regrouping is 80, 9, and 7.",
        "qa_formulas": [
            "16 * 5 = 80",
            "80 * 9 * 7 = 5040",
        ],
        "validated_examples": [
            {"grouping": [80, 9, 7], "product": 5040}
        ],
        "rt_bridge": "No direct RT law; this is regrouping within the prime-factor encoding layer.",
        "chromo_bridge": "None directly.",
        "next_step": "Promote as the first explicit regrouping rule once the page-75 lane is stabilized.",
    },
    "pyth2_p075_l5680": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "For a combined wave of 1000 units, the zero count forces equal powers of 2 and 5: 1000 = 2^3 * 5^3 = 8 * 125, so the normalized wavelengths are 8 and 125.",
        "normalized_answer": "The wavelengths are 8 and 125 because 1000 = 2^3 * 5^3.",
        "qa_formulas": [
            "1000 = 2^3 * 5^3",
            "8 * 125 = 1000",
        ],
        "validated_examples": [
            {"factorization": [8, 125], "product": 1000}
        ],
        "rt_bridge": "No direct RT law; this is a prime-power decomposition in the QA combinatorial layer.",
        "chromo_bridge": "None directly.",
        "next_step": "Use as the cleanest small target in the page-75 prime-factor lane.",
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
    spec = PAGE75_SPECS[item_id]
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
        "status": "reinterpreted_from_page75_prime_factor_lane",
        "validated_examples": spec["validated_examples"],
        "chromo_bridge": spec["chromo_bridge"],
    }


def self_test() -> int:
    sample = {
        "classification": {"intake_class": "prime_factor_encoding"},
        "id": "pyth2_p075_l5680",
        "prior_art_keys": ["pythagorean_triples_and_theorem"],
        "prior_art_refs": [{"key": "pythagorean_triples_and_theorem"}],
        "source": {"page": 75},
        "source_answer": "dummy",
        "source_question": "dummy",
    }
    item = build_item(sample)
    ok = (
        item["issue_label"] == "source-layer ambiguity"
        and item["validated_examples"][0]["factorization"] == [8, 125]
        and item["status"] == "reinterpreted_from_page75_prime_factor_lane"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    manual_queue = read_items(OUT_DIR / "pyth2_manual_review_queue.json")
    page75_items = [item for item in manual_queue if item["source"]["page"] == 75]
    batch = [build_item(item) for item in page75_items]
    payload = {
        "batch_id": "pyth2_page75_prime_factor_batch_001",
        "items": batch,
        "summary": {
            "flagged_count": sum(1 for item in batch if item["issue_label"] != "none"),
            "page": 75,
            "series": "Pyth-2",
            "total_items": len(batch),
        },
    }
    write_json(OUT_DIR / "pyth2_page75_prime_factor_batch_001.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_page75_prime_factor_batch_001.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
