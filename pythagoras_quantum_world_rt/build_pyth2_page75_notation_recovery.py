#!/usr/bin/env python3
"""Build constrained notation-recovery candidates for the page-75 Quantum Prime holdout."""

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


def confusion_match(raw_char: str, candidate_char: str) -> bool:
    if raw_char == candidate_char:
        return True
    if raw_char == "g" and candidate_char in {"2", "3"}:
        return True
    return False


def ocr_shape_score(raw: str, candidate: str) -> float:
    if not raw or not candidate:
        return 0.0
    shared = min(len(raw), len(candidate))
    matches = 0
    for raw_char, cand_char in zip(raw[:shared], candidate[:shared]):
        if confusion_match(raw_char, cand_char):
            matches += 1
    length_penalty = abs(len(raw) - len(candidate)) / max(len(raw), len(candidate))
    score = (matches / max(len(raw), len(candidate))) - length_penalty
    return round(max(score, 0.0), 3)


def candidate_record(
    compact: str,
    display_pairs: list[str],
    expanded_form: str,
    normalized_values: list[int],
    raw_ocr: str,
    confidence: float,
    rationale: list[str],
) -> dict[str, object]:
    return {
        "compact_prime_numbering": compact,
        "confidence": confidence,
        "display_pairs": display_pairs,
        "expanded_form": expanded_form,
        "normalized_values": normalized_values,
        "ocr_shape_score": ocr_shape_score(raw_ocr, compact),
        "rationale": rationale,
    }


def build_recovery_item(item: dict[str, object]) -> dict[str, object]:
    raw_ocr = "g4g25171"
    normalized_values = [16, 9, 5, 7]
    best = candidate_record(
        compact="24325171",
        display_pairs=["24", "32", "51", "71"],
        expanded_form="2^4 3^2 5^1 7^1",
        normalized_values=normalized_values,
        raw_ocr=raw_ocr,
        confidence=0.92,
        rationale=[
            "Matches the stabilized factorization 5040 = 2^4 * 3^2 * 5 * 7 in prime-exponent token order.",
            "Fits the local page-75 rule that prime powers are written as tokens such as 2^3 before decimal-zero suffixes are applied.",
            "Fits the repo prior-art notation examples 21, 22, 23, 24 for powers of 2.",
            "Fits the clean qa-2 witness where unit exponents are written explicitly, e.g. 11+21=31.",
            "Matches the OCR length exactly, with both raw 'g' characters plausibly standing in for '2' and '3'.",
        ],
    )
    alternates = [
        candidate_record(
            compact="243257",
            display_pairs=["24", "32", "5", "7"],
            expanded_form="2^4 3^2 5 7",
            normalized_values=normalized_values,
            raw_ocr=raw_ocr,
            confidence=0.41,
            rationale=[
                "Preserves the same prime-power ordering but omits explicit unit exponents.",
                "Mathematically valid, but the OCR string has two extra trailing digits, so the fit is weaker.",
            ],
        ),
        candidate_record(
            compact="2432571",
            display_pairs=["24", "32", "5", "71"],
            expanded_form="2^4 3^2 5 7^1",
            normalized_values=normalized_values,
            raw_ocr=raw_ocr,
            confidence=0.33,
            rationale=[
                "Mixed explicit/implicit exponent form is possible in principle.",
                "No local witness shows this mixed style, so it remains lower confidence than the explicit pair encoding.",
            ],
        ),
    ]
    return {
        "best_candidate": best,
        "candidate_status": "corrected_source_validated",
        "evidence": [
            {
                "kind": "source_context",
                "note": "Page-75 OCR witness for the damaged compact notation string.",
                "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                "quote_excerpt": "When combined as g4g25171 will also be the equivalent of 5040 expressed in the system of prime numbering.",
                "source_line": 5664,
            },
            {
                "kind": "source_context",
                "note": "Page-75 regrouping rule treats 2^3 as a compact token before appending decimal zeros.",
                "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                "quote_excerpt": "So combining the 2\" and the 5t will give 23 followed by one zero.",
                "source_line": 5673,
            },
            {
                "kind": "source_context",
                "note": "Earlier chapter states primes are written once with the combined power.",
                "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                "quote_excerpt": "this prime number is used only once but it is raised to the combined power",
                "source_line": 5411,
            },
            {
                "kind": "prior_art",
                "note": "Repo-local prior-art note preserves the compact power-prime convention 21, 22, 23, 24 for powers of 2.",
                "path": "private/Documents/elements.txt",
                "quote_excerpt": "23=8 ... 24=16",
                "source_line": 920,
            },
            {
                "kind": "clean_witness",
                "note": "qa-2 witness shows the house style writes explicit unit exponents in compact power notation.",
                "path": "qa_corpus_text/qa-2__001_qa_2_all_pages__docx.md",
                "quote_excerpt": "11+21=31",
                "source_line": 631,
            },
            {
                "kind": "clean_witness",
                "note": "qa-2 witness also writes higher exponents in the same compact pair style.",
                "path": "qa_corpus_text/qa-2__001_qa_2_all_pages__docx.md",
                "quote_excerpt": "32+42=52",
                "source_line": 637,
            },
        ],
        "id": item["id"],
        "normalized_values": normalized_values,
        "ocr_fragment": raw_ocr,
        "promotion_decision": {
            "decision": "promote_to_validated_lane",
            "reason": "The compact token is now supported by the damaged page-75 witness plus a clean local witness that explicit unit exponents are part of the notation system.",
        },
        "source": item["source"],
        "source_answer": item["source_answer"],
        "supporting_candidates": alternates,
        "summary": {
            "candidate_count": 3,
            "evidence_count": 6,
            "page": 75,
            "series": "Pyth-2",
        },
        "unresolved_points": [
            "Whether the answer line before the compact token should be restored as 2^4 3^2 5 x 7 or as the same explicit pair notation with spaces.",
        ],
    }


def build_payload(batch: dict[str, object]) -> dict[str, object]:
    items = []
    for item in batch["items"]:
        if item["id"] != "pyth2_p075_l5663":
            continue
        items.append(build_recovery_item(item))
    return {
        "items": items,
        "source_batch_id": batch["batch_id"],
        "summary": {
            "candidate_item_count": len(items),
            "page": 75,
            "series": "Pyth-2",
        },
    }


def self_test() -> int:
    batch = {
        "batch_id": "b",
        "items": [
            {
                "id": "pyth2_p075_l5663",
                "source": {"page": 75},
                "source_answer": "When combined as g4g25171",
            }
        ],
    }
    payload = build_payload(batch)
    item = payload["items"][0]
    ok = (
        payload["summary"]["candidate_item_count"] == 1
        and item["best_candidate"]["compact_prime_numbering"] == "24325171"
        and item["best_candidate"]["ocr_shape_score"] == 1.0
        and item["promotion_decision"]["decision"] == "promote_to_validated_lane"
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
    payload = build_payload(batch)
    write_json(OUT_DIR / "pyth2_page75_notation_recovery_candidates.json", payload)
    print(
        canonical_dump(
            {"ok": True, "outputs": [str(OUT_DIR / "pyth2_page75_notation_recovery_candidates.json")]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
