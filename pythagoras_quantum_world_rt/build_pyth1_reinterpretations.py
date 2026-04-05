#!/usr/bin/env python3
"""Build the first controlled Pyth-1 reinterpretation batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


INTERPRETATION_SPECS = {
    "pyth1_p016_l1095": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "Interpreting the given bead numbers as (e,d,a)=(1,60,61), the unique missing bead is b=d-e=59, which is consistent with a=d+e and the odd-b/odd-a primitive bead convention.",
        "qa_formulas": [
            "d=b+e",
            "a=b+2e=d+e",
            "b=d-e",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 61, "b": 59, "d": 60, "e": 1},
                "triple": {"C": 120, "F": 3599, "G": 3601},
            }
        ],
        "rt_bridge": "Direction vector (d,e)=(60,1) gives blue/green/red quadrances G=3601, C=120, F=3599.",
        "chromo_bridge": "This is a direct instance of (C,F,G)=(Q_green,Q_red,Q_blue).",
        "next_step": "Promote as a tuple-reconstruction rule example in the Pyth-1 intake ledger.",
    },
    "pyth1_p016_l1103": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "No primitive bead set lies entirely in the interval [7,21]: if the smallest bead is at least 7, then a=b+2e exceeds 21 under the primitive non-root bead constraints.",
        "qa_formulas": [
            "a=b+2e",
            "For non-root primitive tuples, a>3*min(b,e)",
        ],
        "validated_examples": [],
        "rt_bridge": "This is a generator-admissibility constraint before any RT object is formed.",
        "chromo_bridge": "The impossibility occurs at the tuple layer prior to quadrance construction.",
        "next_step": "Keep as a finite-window exclusion rule for later table validation.",
    },
    "pyth1_p016_l1109": {
        "issue_label": "true contradiction requiring investigation",
        "source_verdict": "flagged",
        "formal_claim": "The source claims uniqueness for the interval [7,23], but the bare tuple laws admit at least two candidate primitive tuples: (b,e,d,a)=(7,8,15,23) and (9,7,16,23). This item must remain flagged until the source convention causing exclusion is identified.",
        "qa_formulas": [
            "d=b+e",
            "a=b+2e",
            "a=23 => b+2e=23",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 23, "b": 7, "d": 15, "e": 8},
                "triple": {"C": 240, "F": 161, "G": 289},
            },
            {
                "tuple": {"a": 23, "b": 9, "d": 16, "e": 7},
                "triple": {"C": 224, "F": 207, "G": 305},
            },
        ],
        "rt_bridge": "Both candidate directions produce valid RT/QA triples, so uniqueness is not recovered from the RT bridge alone.",
        "chromo_bridge": "Both candidates define valid chromogeometric direction vectors.",
        "next_step": "Move to issue register and inspect whether a hidden parity/order convention excludes one branch.",
    },
    "pyth1_p034_l2226": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "For base C=20, the bead reconstruction problem is equivalent to solving de=10. The two positive factor pairs (d,e)=(10,1) and (5,2) yield the tuples (59? no) actually (9,1,10,11) and (3,2,5,7), hence the triangles (20,99,101) and (20,21,29).",
        "qa_formulas": [
            "C=2de",
            "de=10",
            "b=d-e",
            "a=d+e",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 11, "b": 9, "d": 10, "e": 1},
                "triple": {"C": 20, "F": 99, "G": 101},
            },
            {
                "tuple": {"a": 7, "b": 3, "d": 5, "e": 2},
                "triple": {"C": 20, "F": 21, "G": 29},
            },
        ],
        "rt_bridge": "This is a direct quadrance inversion problem: recover QA generators from fixed green quadrance C.",
        "chromo_bridge": "C is Q_green(d,e), so the problem reduces to enumerating integer directions with fixed green quadrance.",
        "next_step": "Use as the first direct C->(d,e)->(F,G) reconstruction example in the batch.",
    },
    "pyth1_p046_l2845": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "For the 20-21-29 triangle with (d,e)=(5,2), the base invariant C=20 is geometrically represented as two de-rectangles of size 2 by 5, i.e. a doubled rectangle model of Q_green(d,e)=2de.",
        "qa_formulas": [
            "C=2de",
            "Q_green(d,e)=2de",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 7, "b": 3, "d": 5, "e": 2},
                "triple": {"C": 20, "F": 21, "G": 29},
            }
        ],
        "rt_bridge": "This is the area-style geometric realization of the green quadrance side of the RT/chromogeometry bridge.",
        "chromo_bridge": "The rectangle model is exactly the green quadrance visualization.",
        "next_step": "Keep as the construction-side companion to the 20-base reconstruction item.",
    },
    "pyth1_p059_l4435": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "The invariant H may be written in three equivalent polynomial forms: H=d*d+2*d*e-e*e, H=(a*a+2*a*b-b*b)/2, and H=b*b+4*b*e+2*e*e after substituting d=b+e and a=b+2e.",
        "qa_formulas": [
            "H=d*d+2*d*e-e*e",
            "a=b+2e",
            "d=b+e",
            "H=(a*a+2*a*b-b*b)/2",
            "H=b*b+4*b*e+2*e*e",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 3, "b": 1, "d": 2, "e": 1},
                "values": {"H": 7},
            }
        ],
        "rt_bridge": "H is not a primary RT invariant, but it is a deterministic polynomial over the same generator ring and can be treated as a derived QA observable.",
        "chromo_bridge": "Indirect only: H sits above the primary C/F/G chromogeometric quadrances.",
        "next_step": "Normalize this identity family before any symbolic-table encoding.",
    },
    "pyth1_p066_l5053": {
        "issue_label": "source-layer ambiguity",
        "source_verdict": "corrected",
        "formal_claim": "The lookup 'H=31 appears twice as I' is the integer-solution statement |2*e*e-b*b|=31, whose Table 3 solutions in range are (b,e)=(1,4) and (7,3). The note about Table 4/Figure 6 extends the same I-value into the Koenig layer with (I,G,H)=(31,221,311).",
        "qa_formulas": [
            "I=|C-F|",
            "C-F=2e*e-b*b",
            "|2e*e-b*b|=31",
        ],
        "validated_examples": [
            {
                "tuple": {"a": 9, "b": 1, "d": 5, "e": 4},
                "values": {"C": 40, "F": 9, "I": 31},
            },
            {
                "tuple": {"a": 13, "b": 7, "d": 10, "e": 3},
                "values": {"C": 60, "F": 91, "I": 31},
            },
        ],
        "rt_bridge": "This is a level-set problem on the conic discriminant, not a direct side-length theorem.",
        "chromo_bridge": "The two solutions land on opposite sign branches of C-F, so the same I-value spans two conic regimes.",
        "next_step": "Use as the first explicit discriminant-search example in the Pyth-1 batch.",
    },
    "pyth1_p080_l6103": {
        "issue_label": "OCR corruption",
        "source_verdict": "corrected",
        "formal_claim": "The backward Koenig trace from I=193 to unity is best read as 193,(137),17,(13),7,(5),1. The leading '198' in the OCR output is inconsistent with the question and the following items, which all use I=193.",
        "qa_formulas": [
            "Koenig rule: previous H becomes next I",
            "Node sequence: I -> G -> H -> ...",
        ],
        "validated_examples": [
            {
                "sequence": [1, 5, 7, 13, 17, 137, 193],
                "sequence_format": "I,(G),H,(G),H,(G),H",
            }
        ],
        "rt_bridge": "QA-first only; this is a Koenig graph trace over derived invariants rather than a primary RT law.",
        "chromo_bridge": "Indirect via I as positive discriminant difference.",
        "next_step": "Carry the corrected sequence into the Koenig trace ledger.",
    },
    "pyth1_p080_l6106": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "The backward trace from I=193 to unity passes through three Table 4 block transitions: 1->7, 7->17, and 17->193, so only the first three blocks are used.",
        "qa_formulas": [
            "Table-4 block rule: terminal H of one block becomes I of the next block",
        ],
        "validated_examples": [
            {
                "blocks": [1, 7, 17, 193],
            }
        ],
        "rt_bridge": "QA-first only; this is block-graph bookkeeping for derived invariants.",
        "chromo_bridge": "Indirect via the I/H chain.",
        "next_step": "Encode as a Table 4 transition-count rule.",
    },
    "pyth1_p080_l6108": {
        "issue_label": "none",
        "source_verdict": "faithful",
        "formal_claim": "At the Koenig node I=193, Table 4 exposes at least two admissible forward H-successors, 497 and 599, so upward growth is branching rather than unique.",
        "qa_formulas": [
            "Koenig tree forward rule: fixed I may admit multiple H children",
        ],
        "validated_examples": [
            {
                "branch_point": 193,
                "successors": [497, 599],
            }
        ],
        "rt_bridge": "No direct RT theorem; this is QA tree dynamics over derived invariants.",
        "chromo_bridge": "Indirect only through I as discriminant seed for higher-level branches.",
        "next_step": "Treat as the first explicit Koenig branching rule in the Pyth-1 dynamic layer.",
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
        "classification": {"intake_class": "tuple_reconstruction"},
        "id": "pyth1_p016_l1095",
        "prior_art_keys": ["rational_trigonometry"],
        "prior_art_refs": [{"key": "rational_trigonometry"}],
        "source": {"page": 16},
        "source_answer": "dummy",
        "source_question": "dummy",
    }
    item = build_item(sample)
    ok = (
        item["issue_label"] == "none"
        and "b=d-e" in item["qa_formulas"]
        and item["source_verdict"] == "faithful"
        and item["status"] == "reinterpreted"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    workbook = read_items(OUT_DIR / "pyth1_theory_workbook.json")
    accepted = [item for item in workbook if item["status"] == "queued"]
    batch = [build_item(item) for item in accepted]
    payload = {
        "batch_id": "pyth1_reinterpretation_batch_001",
        "items": batch,
        "summary": {
            "accepted_reinterpreted_count": len(batch),
            "flagged_issue_count": sum(1 for item in batch if item["issue_label"] != "none"),
            "series": "Pyth-1",
        },
    }
    write_json(OUT_DIR / "pyth1_reinterpretation_batch_001.json", payload)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [str(OUT_DIR / "pyth1_reinterpretation_batch_001.json")],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
