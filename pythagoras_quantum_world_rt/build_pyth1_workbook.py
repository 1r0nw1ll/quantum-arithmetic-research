#!/usr/bin/env python3
"""Build a controlled Pyth-1 intake workbook from triaged inventory outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
PRIOR_ART_MAP = DEFAULT_OUT_DIR / "prior_art_bridge_map.json"

QUESTION_PREFIX_RE = re.compile(r"^(?:QUESTIONS:\s*)?", re.IGNORECASE)


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_items(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def load_prior_art_map() -> dict[str, dict[str, object]]:
    payload = json.loads(PRIOR_ART_MAP.read_text(encoding="utf-8"))
    return {entry["key"]: entry for entry in payload["categories"]}


def clean_question_text(context_before: list[dict[str, object]]) -> str:
    text = " ".join(str(item["text"]) for item in context_before).strip()
    return QUESTION_PREFIX_RE.sub("", text).strip()


def intake_class(question_text: str, answer_text: str) -> str:
    lowered = f"{question_text} {answer_text}".lower()
    if "trace" in lowered or "sequence" in lowered or "backward" in lowered:
        return "trace_sequence"
    if "convert" in lowered or "formula" in lowered:
        return "symbolic_conversion"
    if "missing bead number" in lowered or "what are the bead numbers" in lowered:
        return "tuple_reconstruction"
    if "how many sets" in lowered or "how many" in lowered:
        return "combinatorial_constraint"
    if "table 3" in lowered or "table 4" in lowered or "figure 6" in lowered:
        return "table_graph_lookup"
    if "base" in lowered or "factors into" in lowered:
        return "base_factorization"
    return "general_theory_item"


def qa_hooks(question_text: str, answer_text: str) -> list[str]:
    lowered = f"{question_text} {answer_text}".lower()
    hooks: list[str] = []
    if "bead" in lowered or " b " in f" {lowered} " or " e " in f" {lowered} " or " d " in f" {lowered} ":
        hooks.append("bead_numbers_(b,e,d,a)")
    if "c =" in lowered or " base" in lowered:
        hooks.append("C_equals_2de")
    if "g" in answer_text or " g " in f" {lowered} ":
        hooks.append("G_blue_quadrance")
    if "h" in answer_text or " h " in f" {lowered} ":
        hooks.append("H_invariant")
    if "i" in answer_text or " i " in f" {lowered} ":
        hooks.append("I_positive_difference")
    if "triangle" in lowered or "triangles" in lowered:
        hooks.append("triangle_generation")
    if "table 3" in lowered or "table 4" in lowered or "figure 6" in lowered:
        hooks.append("table_or_graph_lookup")
    deduped: list[str] = []
    for hook in hooks:
        if hook not in deduped:
            deduped.append(hook)
    return deduped


def rt_bridge_mode(question_text: str, answer_text: str) -> str:
    lowered = f"{question_text} {answer_text}".lower()
    if "c =" in lowered or " base" in lowered or "triangle" in lowered:
        return "direct_quadrance_bridge"
    if "h" in answer_text or " i " in f" {lowered} ":
        return "indirect_via_qa_invariants"
    return "qa_first_then_rt_bridge"


def next_operation_label(entry: dict[str, object], intake_kind: str, bridge_mode: str) -> str:
    if entry.get("ocr_risk_flag"):
        return "transcription_cleanup"
    if intake_kind == "symbolic_conversion":
        return "normalize_symbolic_identity"
    if intake_kind == "trace_sequence":
        return "encode_table_trace_rule"
    if intake_kind in {"tuple_reconstruction", "base_factorization", "combinatorial_constraint"}:
        return "formalize_tuple_constraint"
    if bridge_mode == "direct_quadrance_bridge":
        return "map_to_rt_quadrance_language"
    return "manual_math_review"


def prior_art_keys_for_item(intake_kind: str, bridge_mode: str, question_text: str, answer_text: str) -> list[str]:
    keys = [
        "pythagorean_triples_and_theorem",
        "rational_trigonometry",
        "chromogeometry",
    ]
    lowered = f"{question_text} {answer_text}".lower()
    if intake_kind in {"trace_sequence", "table_graph_lookup"} or "koenig" in lowered or "figure 6" in lowered:
        keys.append("egyptian_fractions_and_hat")
    if intake_kind in {"tuple_reconstruction", "base_factorization", "combinatorial_constraint"}:
        keys.append("ptolemy_quadrance_and_uhg")
    if bridge_mode == "indirect_via_qa_invariants" and "egyptian_fractions_and_hat" not in keys:
        keys.append("egyptian_fractions_and_hat")
    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def workbook_item(entry: dict[str, object], status: str, prior_art_map: dict[str, dict[str, object]]) -> dict[str, object]:
    question_text = clean_question_text(list(entry.get("context_before", [])))
    answer_text = str(entry.get("answer_text", "")).strip()
    intake_kind = intake_class(question_text, answer_text)
    bridge_mode = rt_bridge_mode(question_text, answer_text)
    prior_art_keys = prior_art_keys_for_item(intake_kind, bridge_mode, question_text, answer_text)
    return {
        "id": f"pyth1_p{int(entry['page']):03d}_l{int(entry['line_no']):04d}",
        "prior_art_keys": prior_art_keys,
        "prior_art_refs": [prior_art_map[key] for key in prior_art_keys],
        "status": status,
        "source": {
            "line_no": entry["line_no"],
            "line_span": entry["line_span"],
            "page": entry["page"],
            "series": entry["series"],
            "source_path": entry["source_path"],
            "witness_id": entry["witness_id"],
            "witness_kind": entry["witness_kind"],
        },
        "source_question": question_text,
        "source_answer": answer_text,
        "classification": {
            "actionability": entry["actionability"],
            "confidence": entry["confidence"],
            "intake_class": intake_kind,
            "next_operation": next_operation_label(entry, intake_kind, bridge_mode),
            "qa_hooks": qa_hooks(question_text, answer_text),
            "rt_bridge_mode": bridge_mode,
            "semantic_tags": entry["semantic_tags"],
            "source_issue_label": "OCR corruption" if entry.get("ocr_risk_flag") else "none",
        },
    }


def self_test() -> int:
    prior_art_map = {
        "pythagorean_triples_and_theorem": {"key": "pythagorean_triples_and_theorem"},
        "rational_trigonometry": {"key": "rational_trigonometry"},
        "chromogeometry": {"key": "chromogeometry"},
        "ptolemy_quadrance_and_uhg": {"key": "ptolemy_quadrance_and_uhg"},
        "egyptian_fractions_and_hat": {"key": "egyptian_fractions_and_hat"},
    }
    sample = {
        "actionability": "needs_mathematical_reinterpretation",
        "answer_text": "Since C = 2de, then de = 10. This factors into 1 and 10.",
        "confidence": 0.72,
        "context_before": [
            {"line_no": 1, "page": 7, "text": "QUESTIONS:"},
            {"line_no": 2, "page": 7, "text": "(1) What are the bead numbers when C = 20?"},
        ],
        "line_no": 3,
        "line_span": [3, 3],
        "ocr_risk_flag": False,
        "page": 7,
        "semantic_tags": ["qa_answer", "qa_exercise", "theory_bearing"],
        "series": "Pyth-1",
        "source_path": "qa_corpus_text/sample.md",
        "witness_id": "sample",
        "witness_kind": "ocr_markdown",
    }
    item = workbook_item(sample, "queued", prior_art_map)
    ok = (
        item["classification"]["intake_class"] == "tuple_reconstruction"
        and item["classification"]["rt_bridge_mode"] == "direct_quadrance_bridge"
        and item["classification"]["next_operation"] == "formalize_tuple_constraint"
        and "rational_trigonometry" in item["prior_art_keys"]
        and item["source_question"] == "(1) What are the bead numbers when C = 20?"
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    out_dir = Path(args.out_dir)
    accepted = read_items(out_dir / "accepted_theory_items.json")
    ambiguity = read_items(out_dir / "ocr_or_ambiguity_items.json")
    manual = read_items(out_dir / "needs_manual_review.json")
    prior_art_map = load_prior_art_map()

    pyth1_accepted = [item for item in accepted if item["series"] == "Pyth-1"]
    pyth1_ambiguity = [item for item in ambiguity if item["series"] == "Pyth-1"]
    pyth1_manual = [item for item in manual if item["series"] == "Pyth-1"]

    accepted_queue = [workbook_item(item, "queued", prior_art_map) for item in pyth1_accepted]
    ambiguity_queue = [workbook_item(item, "ocr_cleanup", prior_art_map) for item in pyth1_ambiguity]
    manual_queue = [workbook_item(item, "manual_review", prior_art_map) for item in pyth1_manual]

    workbook = {
        "items": accepted_queue + ambiguity_queue + manual_queue,
        "summary": {
            "accepted_count": len(accepted_queue),
            "manual_review_count": len(manual_queue),
            "ocr_cleanup_count": len(ambiguity_queue),
            "series": "Pyth-1",
        },
    }

    write_json(out_dir / "pyth1_accepted_theory_queue.json", {"items": accepted_queue})
    write_json(out_dir / "pyth1_ocr_cleanup_queue.json", {"items": ambiguity_queue})
    write_json(out_dir / "pyth1_manual_review_queue.json", {"items": manual_queue})
    write_json(out_dir / "pyth1_theory_workbook.json", workbook)

    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    str(out_dir / "pyth1_accepted_theory_queue.json"),
                    str(out_dir / "pyth1_ocr_cleanup_queue.json"),
                    str(out_dir / "pyth1_manual_review_queue.json"),
                    str(out_dir / "pyth1_theory_workbook.json"),
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
