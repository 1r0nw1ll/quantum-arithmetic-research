#!/usr/bin/env python3
"""Build a controlled reinterpretation batch for unresolved Pyth-1 items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


UNRESOLVED_SPECS = {
    "pyth1_p016_l1086": {
        "issue_label": "OCR corruption",
        "source_verdict": "corrected",
        "formal_claim": "The bead numbers 1, 7, and 9 are best read as (e,b,a)=(1,7,9), so the missing value is d=(7+9)/2=8 and the completed tuple is (b,e,d,a)=(7,1,8,9).",
        "normalized_answer": "The numbers represent e=1, b=7, and a=9. The missing bead number is d=8 because b+e=7+1=8 and a=b+2e=7+2=9.",
        "qa_formulas": [
            "d=b+e",
            "a=b+2e",
            "d=(b+a)/2",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 9, "b": 7, "d": 8, "e": 1},
                "triple": {"C": 16, "F": 63, "G": 65},
            }
        ],
        "rt_bridge": "Direction vector (d,e)=(8,1) gives the QA/RT triple (C,F,G)=(16,63,65).",
        "chromo_bridge": "This is a direct chromogeometric instance with C=Q_green, F=Q_red, and G=Q_blue.",
        "next_step": "Promote as the cleaned introductory tuple-reconstruction example and retire the OCR-damaged wording.",
    },
    "pyth1_p034_l2243": {
        "issue_label": "OCR corruption",
        "source_verdict": "corrected",
        "formal_claim": "For C=60, solving de=30 yields four admissible factor pairs: (d,e)=(30,1),(15,2),(10,3),(6,5). These generate the tuples (29,1,30,31), (13,2,15,17), (7,3,10,13), and (1,5,6,11), hence the four triangles (60,899,901), (60,221,229), (60,91,109), and (60,11,61).",
        "normalized_answer": "Since C=2de, we solve de=30. The four factor pairs are 30x1, 15x2, 10x3, and 6x5. They give bead numbers (29,1,30,31), (13,2,15,17), (7,3,10,13), and (1,5,6,11), with triangles (60,899,901), (60,221,229), (60,91,109), and (60,11,61).",
        "qa_formulas": [
            "C=2de",
            "de=30",
            "b=d-e",
            "a=d+e",
            "F=ab",
            "G=d*d+e*e",
        ],
        "validated_examples": [
            {"tuple": {"a": 31, "b": 29, "d": 30, "e": 1}, "triple": {"C": 60, "F": 899, "G": 901}},
            {"tuple": {"a": 17, "b": 13, "d": 15, "e": 2}, "triple": {"C": 60, "F": 221, "G": 229}},
            {"tuple": {"a": 13, "b": 7, "d": 10, "e": 3}, "triple": {"C": 60, "F": 91, "G": 109}},
            {"tuple": {"a": 11, "b": 1, "d": 6, "e": 5}, "triple": {"C": 60, "F": 11, "G": 61}},
        ],
        "rt_bridge": "This is a fixed-green-quadrance inversion problem with four integer directions on the same C-level set.",
        "chromo_bridge": "The four tuples are precisely the integer directions with Q_green(d,e)=60 in the source window.",
        "next_step": "Use as the first four-solution C-level-set table entry when table reproduction starts.",
    },
    "pyth1_p058_l4423": {
        "issue_label": "OCR corruption",
        "source_verdict": "corrected",
        "formal_claim": "The damaged OCR derivation should be normalized as H=d*d+2*d*e-e*e with d=(a+b)/2 and e=(a-b)/2, giving H=(a*a+2*a*b-b*b)/2.",
        "normalized_answer": "Start with H=d*d+2*d*e-e*e. Substitute d=(a+b)/2 and e=(a-b)/2, expand, and combine terms to obtain H=(a*a+2*a*b-b*b)/2.",
        "qa_formulas": [
            "H=d*d+2*d*e-e*e",
            "d=(a+b)/2",
            "e=(a-b)/2",
            "H=(a*a+2*a*b-b*b)/2",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 3, "b": 1, "d": 2, "e": 1},
                "values": {"H": 7},
            }
        ],
        "rt_bridge": "H remains a derived QA invariant over the same generator ring used to produce the RT quadrances.",
        "chromo_bridge": "Indirect only: the cleanup stabilizes a higher-level polynomial observable above C/F/G.",
        "next_step": "Use this cleaned identity as the canonical source text for symbolic table encoding.",
    },
    "pyth1_p066_l5029": {
        "issue_label": "OCR corruption",
        "source_verdict": "corrected",
        "formal_claim": "The completed Table 3 block for b=7 and e=6 is the tuple (b,e,d,a)=(7,6,13,19) with normalized rows B=49, E=36, A=361, D=169; C=156, F=133, G=205; H=289, I=23, J=91, K=247. The raw source also records L=1729, which should be preserved as an unvalidated source cell until its table meaning is re-derived locally.",
        "normalized_answer": "Complete the block with b=7, e=6, a=19, d=13. Then B=49, E=36, A=361, D=169; C=156, F=133, G=205; H=289, I=23, J=91, K=247. Preserve the source cell L=1729 but do not treat it as locally validated yet.",
        "qa_formulas": [
            "d=b+e",
            "a=b+2e",
            "B=b*b",
            "E=e*e",
            "A=a*a",
            "D=d*d",
            "C=2de",
            "F=ab",
            "G=d*d+e*e",
            "H=C+F",
            "I=|C-F|",
            "J=b*d",
            "K=d*a",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 19, "b": 7, "d": 13, "e": 6},
                "values": {
                    "A": 361,
                    "B": 49,
                    "C": 156,
                    "D": 169,
                    "E": 36,
                    "F": 133,
                    "G": 205,
                    "H": 289,
                    "I": 23,
                    "J": 91,
                    "K": 247,
                    "L_raw_source": 1729,
                },
            }
        ],
        "rt_bridge": "The corrected row stabilizes the C/F/G quadrance core of the Table 3 block and leaves only the higher cell L outside the current validated bridge.",
        "chromo_bridge": "C=156, F=133, and G=205 certify the chromogeometric core of the source block.",
        "next_step": "Promote the validated cells into the table-reproduction program and keep L on a separate re-derivation list.",
    },
    "pyth1_p046_l2854": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "For the 20-21-29 triangle with tuple (b,e,d,a)=(3,2,5,7), F=ab=21 is the 3-by-7 rectangle, and the gnomon identity is F=D-E=25-4. The source line giving e=3 and E=9 is inconsistent with the rest of the triangle data and should be corrected to e=2 and E=4.",
        "normalized_answer": "For this triangle, F=ab=3x7=21, so the rectangle has side lengths 3 and 7. The gnomon of F is D-E with D=25 and E=4, since d=5 and e=2. The gnomon of F squared is G squared minus C squared, leaving outer legs of 29 and inner legs of 20, with width 9.",
        "qa_formulas": [
            "F=ab",
            "F=D-E",
            "D=d*d",
            "E=e*e",
            "G*G-C*C=F*F",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 7, "b": 3, "d": 5, "e": 2},
                "triple": {"C": 20, "F": 21, "G": 29},
                "values": {"D": 25, "E": 4},
            }
        ],
        "rt_bridge": "This is the red-quadrance construction companion to the earlier green-quadrance C-construction item.",
        "chromo_bridge": "F is the red quadrance of the same direction vector, so the rectangle and gnomon are a red-metric construction.",
        "next_step": "Pair with the C-construction item and treat the e=3/E=9 wording as a corrected source typo, not validated theory.",
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
        "status": "reinterpreted_from_unresolved_queue",
        "validated_examples": spec["validated_examples"],
        "chromo_bridge": spec["chromo_bridge"],
    }


def self_test() -> int:
    sample = {
        "classification": {"intake_class": "general_theory_item"},
        "id": "pyth1_p016_l1086",
        "prior_art_keys": ["chromogeometry"],
        "prior_art_refs": [{"key": "chromogeometry"}],
        "source": {"page": 16},
        "source_answer": "dummy",
        "source_question": "dummy",
    }
    item = build_item(sample)
    ok = (
        item["issue_label"] == "OCR corruption"
        and item["status"] == "reinterpreted_from_unresolved_queue"
        and item["validated_examples"][0]["tuple"]["d"] == 8
        and "normalized_answer" in item
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    ocr_queue = read_items(OUT_DIR / "pyth1_ocr_cleanup_queue.json")
    manual_queue = read_items(OUT_DIR / "pyth1_manual_review_queue.json")
    batch_items = [build_item(item) for item in ocr_queue + manual_queue]
    payload = {
        "batch_id": "pyth1_unresolved_reinterpretation_batch_001",
        "items": batch_items,
        "summary": {
            "flagged_count": sum(1 for item in batch_items if item["issue_label"] != "none"),
            "ocr_corrected_count": sum(1 for item in batch_items if item["issue_label"] == "OCR corruption"),
            "manual_review_items": len(manual_queue),
            "series": "Pyth-1",
            "total_items": len(batch_items),
        },
    }
    write_json(OUT_DIR / "pyth1_unresolved_reinterpretation_batch_001.json", payload)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [str(OUT_DIR / "pyth1_unresolved_reinterpretation_batch_001.json")],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
