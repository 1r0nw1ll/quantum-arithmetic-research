#!/usr/bin/env python3
"""Build a controlled Pyth-2 intake workbook from triaged inventory outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
PRIOR_ART_MAP = DEFAULT_OUT_DIR / "prior_art_bridge_map.json"

QUESTION_PREFIX_RE = re.compile(r"^(?:QUESTIONS:\s*|PROBLEMS\s*)?", re.IGNORECASE)


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
    text = QUESTION_PREFIX_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


def intake_class(question_text: str, answer_text: str) -> str:
    lowered = f"{question_text} {answer_text}".lower()
    if "how often" in lowered or "every third" in lowered or "every fourth" in lowered or "every fifth" in lowered or "every sixth" in lowered:
        return "periodicity_rule"
    if "how many bead numbers" in lowered or "how many pairs" in lowered:
        return "counting_constraint"
    if "symmetrical" in lowered or "sum of each pair" in lowered:
        return "symmetry_rule"
    if "quantum prime numbering system" in lowered or "wavelength" in lowered or "factor" in lowered:
        return "prime_factor_encoding"
    return "general_theory_item"


def qa_hooks(question_text: str, answer_text: str) -> list[str]:
    lowered = f"{question_text} {answer_text}".lower()
    hooks: list[str] = []
    if "bead" in lowered or "b," in lowered or "e =" in lowered:
        hooks.append("bead_number_constraints")
    if "fibonacci" in lowered:
        hooks.append("fibonacci_periodicity")
    if "prime" in lowered:
        hooks.append("prime_structure")
    if "pairs" in lowered:
        hooks.append("pair_counting")
    if "symmetrical" in lowered:
        hooks.append("mirror_symmetry")
    deduped: list[str] = []
    for hook in hooks:
        if hook not in deduped:
            deduped.append(hook)
    return deduped


def rt_bridge_mode(question_text: str, answer_text: str) -> str:
    lowered = f"{question_text} {answer_text}".lower()
    if "bead" in lowered:
        return "qa_first_then_rt_bridge"
    if "fibonacci" in lowered or "prime" in lowered or "pairs" in lowered:
        return "qa_first_combinatorial"
    return "qa_first_then_rt_bridge"


def next_operation_label(entry: dict[str, object], intake_kind: str) -> str:
    if entry.get("ocr_risk_flag"):
        return "transcription_cleanup"
    if intake_kind in {"periodicity_rule", "counting_constraint", "symmetry_rule"}:
        return "formalize_counting_rule"
    if intake_kind == "prime_factor_encoding":
        return "manual_math_review"
    return "manual_math_review"


def prior_art_keys_for_item(intake_kind: str, question_text: str, answer_text: str) -> list[str]:
    keys = ["pythagorean_triples_and_theorem", "rational_trigonometry", "chromogeometry"]
    lowered = f"{question_text} {answer_text}".lower()
    if intake_kind in {"periodicity_rule", "symmetry_rule"} or "fibonacci" in lowered:
        keys.append("egyptian_fractions_and_hat")
    if intake_kind == "counting_constraint":
        keys.append("ptolemy_quadrance_and_uhg")
    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def workbook_item(entry: dict[str, object], status: str, prior_art_map: dict[str, dict[str, object]]) -> dict[str, object]:
    question_text = clean_question_text(list(entry.get("context_before", [])))
    answer_text = str(entry.get("answer_text", "")).strip()
    intake_kind = intake_class(question_text, answer_text)
    prior_art_keys = prior_art_keys_for_item(intake_kind, question_text, answer_text)
    return {
        "id": f"pyth2_p{int(entry['page']):03d}_l{int(entry['line_no']):04d}",
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
            "next_operation": next_operation_label(entry, intake_kind),
            "qa_hooks": qa_hooks(question_text, answer_text),
            "rt_bridge_mode": rt_bridge_mode(question_text, answer_text),
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
        "answer_text": "Every third number.",
        "confidence": 0.72,
        "context_before": [
            {"text": "3 How often do even numbers appear in the original Fibonacci Series?"}
        ],
        "line_no": 1,
        "line_span": [1, 1],
        "ocr_risk_flag": False,
        "page": 110,
        "semantic_tags": ["qa_answer"],
        "series": "Pyth-2",
        "source_path": "qa_corpus_text/sample.md",
        "witness_id": "sample",
        "witness_kind": "ocr_markdown",
    }
    item = workbook_item(sample, "queued", prior_art_map)
    ok = (
        item["classification"]["intake_class"] == "periodicity_rule"
        and item["classification"]["next_operation"] == "formalize_counting_rule"
        and "egyptian_fractions_and_hat" in item["prior_art_keys"]
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    out_dir = DEFAULT_OUT_DIR
    accepted = read_items(out_dir / "accepted_theory_items.json")
    ambiguity = read_items(out_dir / "ocr_or_ambiguity_items.json")
    manual = read_items(out_dir / "needs_manual_review.json")
    prior_art_map = load_prior_art_map()

    pyth2_accepted = [item for item in accepted if item["series"] == "Pyth-2"]
    pyth2_ambiguity = [item for item in ambiguity if item["series"] == "Pyth-2"]
    pyth2_manual = [item for item in manual if item["series"] == "Pyth-2"]

    accepted_queue = [workbook_item(item, "queued", prior_art_map) for item in pyth2_accepted]
    ambiguity_queue = [workbook_item(item, "ocr_cleanup", prior_art_map) for item in pyth2_ambiguity]
    manual_queue = [workbook_item(item, "manual_review", prior_art_map) for item in pyth2_manual]

    workbook = {
        "items": accepted_queue + ambiguity_queue + manual_queue,
        "summary": {
            "accepted_count": len(accepted_queue),
            "manual_review_count": len(manual_queue),
            "ocr_cleanup_count": len(ambiguity_queue),
            "series": "Pyth-2",
        },
    }

    write_json(out_dir / "pyth2_accepted_theory_queue.json", {"items": accepted_queue})
    write_json(out_dir / "pyth2_ocr_cleanup_queue.json", {"items": ambiguity_queue})
    write_json(out_dir / "pyth2_manual_review_queue.json", {"items": manual_queue})
    write_json(out_dir / "pyth2_theory_workbook.json", workbook)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    str(out_dir / "pyth2_accepted_theory_queue.json"),
                    str(out_dir / "pyth2_ocr_cleanup_queue.json"),
                    str(out_dir / "pyth2_manual_review_queue.json"),
                    str(out_dir / "pyth2_theory_workbook.json"),
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
