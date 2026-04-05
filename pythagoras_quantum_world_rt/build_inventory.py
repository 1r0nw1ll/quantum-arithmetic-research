#!/usr/bin/env python3
"""Build a witness-aware work inventory for the Pythagoras and the Quantum World subproject."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path
from typing import Deque


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "pythagoras_quantum_world_rt"

SOURCES = [
    {
        "path": "qa_corpus_text/pyth_1__ocr__pyth1.md",
        "priority": 1,
        "series": "Pyth-1",
        "witness_id": "pyth_1__ocr__pyth1",
        "witness_kind": "ocr_markdown",
    },
    {
        "path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
        "priority": 2,
        "series": "Pyth-2",
        "witness_id": "pyth_2__ocr__pyth2",
        "witness_kind": "ocr_markdown",
    },
    {
        "path": "qa_corpus_text/pyth-3__pyth_3_all_pages_smaller__docx.md",
        "priority": 3,
        "series": "Pyth-3",
        "witness_id": "pyth_3__all_pages_smaller__docx",
        "witness_kind": "docx_markdown",
    },
    {
        "path": "qa_corpus_text/pyth-3__pythagoras_vol3_enneagram__docx.md",
        "priority": 4,
        "series": "Pyth-3",
        "witness_id": "pyth_3__enneagram__docx",
        "witness_kind": "docx_markdown",
    },
]

PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")
ANSWER_RE = re.compile(r"\bAns:\s*(.*)")
TABLE_RE = re.compile(r"\btable(?:s)?\b", re.IGNORECASE)
PROGRAM_RE = re.compile(r"\bprogram(?:s)?\b", re.IGNORECASE)
QUESTION_RE = re.compile(r"\?")
TRIMMED_WHITESPACE_RE = re.compile(r"\s+")
EXERCISE_RE = re.compile(r"^\(\d+\)")
OCR_NOISE_RE = re.compile(r"[?]|ain which|nometric|[A-Za-z]\?[A-Za-z]?")
THEORY_RE = re.compile(
    r"\b(identity|identities|bead|triangle|triangles|base|altitude|rectangle|gnomon|formula|formulas|C\b|F\b|G\b|H\b|I\b)\b",
    re.IGNORECASE,
)
DRAWING_RE = re.compile(r"\b(draw|sketch|copy|work out)\b", re.IGNORECASE)
INSTRUCTION_RE = re.compile(r"\b(finish|set up|run|write|use|produce|generate|computer generated)\b", re.IGNORECASE)
SECTION_BREAK_RE = re.compile(r"^(?:QUESTIONS:|PROJECT:|Chapter\b|\(\d+\)|\d+\.?$)", re.IGNORECASE)


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_text(text: str) -> str:
    return TRIMMED_WHITESPACE_RE.sub(" ", text.strip())


def context_window(recent_lines: Deque[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "line_no": int(item["line_no"]),
            "page": item["page"],
            "text": str(item["text"]),
        }
        for item in recent_lines
        if str(item["text"]).strip()
    ]


def build_line_records(lines: list[str]) -> list[dict[str, object]]:
    current_page: int | None = None
    records: list[dict[str, object]] = []
    for line_no, raw_line in enumerate(lines, start=1):
        page_match = PAGE_RE.search(raw_line)
        if page_match:
            current_page = int(page_match.group(1))
        records.append(
            {
                "line_no": line_no,
                "page": current_page,
                "raw": raw_line,
                "text": normalize_text(raw_line),
            }
        )
    return records


def context_from_records(records: list[dict[str, object]], index: int, step: int) -> list[dict[str, object]]:
    window: list[dict[str, object]] = []
    i = index + step
    while 0 <= i < len(records) and len(window) < 3:
        text = str(records[i]["text"])
        if text and not PAGE_RE.search(str(records[i]["raw"])):
            window.append(
                {
                    "line_no": int(records[i]["line_no"]),
                    "page": records[i]["page"],
                    "text": text,
                }
            )
        i += step
    if step < 0:
        window.reverse()
    return window


def answer_continuation(records: list[dict[str, object]], index: int) -> list[str]:
    parts: list[str] = []
    for next_index in range(index + 1, min(index + 5, len(records))):
        record = records[next_index]
        raw = str(record["raw"])
        text = str(record["text"])
        if PAGE_RE.search(raw):
            continue
        if not text:
            continue
        if ANSWER_RE.search(raw) or SECTION_BREAK_RE.search(text):
            break
        parts.append(text)
        if text.endswith((".", "?", "!", ";")):
            break
    return parts


def ocr_risk_for_text(text: str, witness_kind: str) -> dict[str, object]:
    reasons: list[str] = []
    if not text:
        reasons.append("empty_text")
    if OCR_NOISE_RE.search(text):
        reasons.append("ocr_noise_pattern")
    if len(text.split()) <= 2:
        reasons.append("short_fragment")
    if text.endswith(("the", "and", "or", "is", "=")):
        reasons.append("truncated_ending")
    return {
        "ocr_risk_flag": bool(reasons),
        "ocr_risk_reasons": reasons,
        "witness_is_ocr": "ocr" in witness_kind,
    }


def classify_qa_item(item: dict[str, object]) -> dict[str, object]:
    context_text = " ".join(
        [str(part["text"]) for part in item["context_before"]] + [str(part["text"]) for part in item["context_after"]]
    )
    combined_text = f"{context_text} {item['answer_text']}".strip()
    semantic_tags = ["qa_answer"]
    if EXERCISE_RE.search(context_text) or "?" in context_text:
        semantic_tags.append("qa_exercise")
    if THEORY_RE.search(combined_text):
        semantic_tags.append("theory_bearing")
    if DRAWING_RE.search(combined_text):
        semantic_tags.append("construction_exercise")

    ocr_meta = ocr_risk_for_text(str(item["answer_text"]), str(item["witness_kind"]))
    needs_manual = False
    bucket = "needs_manual_review"
    actionability = "needs_mathematical_reinterpretation"
    confidence = 0.65 if not ocr_meta["witness_is_ocr"] else 0.58

    if ocr_meta["ocr_risk_flag"]:
        bucket = "ocr_or_ambiguity_items"
        actionability = "needs_transcription_cleanup"
        confidence = 0.25 if ocr_meta["witness_is_ocr"] else 0.35
        semantic_tags.append("likely_ocr_noise")
    elif "theory_bearing" in semantic_tags and "construction_exercise" not in semantic_tags:
        bucket = "accepted_theory_items"
        actionability = "needs_mathematical_reinterpretation"
        confidence = 0.72 if ocr_meta["witness_is_ocr"] else 0.88
    else:
        needs_manual = True
        if "construction_exercise" in semantic_tags:
            actionability = "needs_manual_review"
            confidence = 0.5

    return {
        "actionability": actionability,
        "confidence": confidence,
        "needs_manual_review": needs_manual,
        "semantic_tags": semantic_tags,
        "triage_bucket": bucket,
        **ocr_meta,
    }


def classify_reference_item(item: dict[str, object]) -> dict[str, object]:
    text = str(item["text"])
    lowered = text.lower()
    semantic_tags = [str(item["kind"])]
    if INSTRUCTION_RE.search(text):
        semantic_tags.append("instruction")
    if "quantize" in lowered or "babthe" in lowered or "coprime" in lowered:
        semantic_tags.append("named_program")
    if "table below" in lowered or "finish" in lowered:
        semantic_tags.append("table_instruction")
    elif "table" in lowered and "shown" in lowered:
        semantic_tags.append("narrative_reference")
    elif "program" in lowered and ("run" in lowered or "set up" in lowered or "write" in lowered):
        semantic_tags.append("program_instruction")

    ocr_meta = ocr_risk_for_text(text, str(item["witness_kind"]))
    confidence = 0.72 if not ocr_meta["witness_is_ocr"] else 0.6
    if ocr_meta["ocr_risk_flag"]:
        confidence = 0.35
    if "instruction" in semantic_tags or "named_program" in semantic_tags:
        actionability = "needs_implementation_reproduction"
    elif "narrative_reference" in semantic_tags:
        actionability = "low_value_reference"
    else:
        actionability = "needs_manual_review"

    return {
        "actionability": actionability,
        "confidence": confidence,
        "semantic_tags": semantic_tags,
        **ocr_meta,
    }


def extract_from_source(source: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    path = ROOT / str(source["path"])
    lines = path.read_text(encoding="utf-8").splitlines()
    records = build_line_records(lines)
    recent_nonempty: Deque[dict[str, object]] = deque(maxlen=4)
    qa_items: list[dict[str, object]] = []
    table_program_items: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        line_no = int(record["line_no"])
        raw_line = str(record["raw"])
        text = str(record["text"])
        current_page = record["page"]
        if PAGE_RE.search(raw_line):
            continue
        if text:
            recent_nonempty.append({"line_no": line_no, "page": current_page, "text": text})

        answer_match = ANSWER_RE.search(raw_line)
        if answer_match:
            counts["qa_pair_candidates"] += 1
            answer_parts = [normalize_text(answer_match.group(1))]
            answer_parts.extend(answer_continuation(records, index))
            item = {
                "answer_excerpt": normalize_text(answer_match.group(1)),
                "answer_text": normalize_text(" ".join(part for part in answer_parts if part)),
                "context_after": context_from_records(records, index, 1),
                "context_before": context_window(recent_nonempty)[:-1],
                "line_span": [line_no, line_no],
                "line_no": line_no,
                "page": current_page,
                "series": source["series"],
                "source_path": source["path"],
                "witness_id": source["witness_id"],
                "witness_kind": source["witness_kind"],
            }
            item.update(classify_qa_item(item))
            qa_items.append(item)

        table_hit = TABLE_RE.search(raw_line)
        if table_hit:
            counts["table_references"] += 1
            item = {
                "kind": "table_reference",
                "context_after": context_from_records(records, index, 1),
                "context_before": context_window(recent_nonempty)[:-1],
                "line_span": [line_no, line_no],
                "line_no": line_no,
                "page": current_page,
                "series": source["series"],
                "source_path": source["path"],
                "text": text,
                "witness_id": source["witness_id"],
                "witness_kind": source["witness_kind"],
            }
            item.update(classify_reference_item(item))
            table_program_items.append(item)

        program_hit = PROGRAM_RE.search(raw_line)
        if program_hit:
            counts["program_references"] += 1
            item = {
                "kind": "program_reference",
                "context_after": context_from_records(records, index, 1),
                "context_before": context_window(recent_nonempty)[:-1],
                "line_span": [line_no, line_no],
                "line_no": line_no,
                "page": current_page,
                "series": source["series"],
                "source_path": source["path"],
                "text": text,
                "witness_id": source["witness_id"],
                "witness_kind": source["witness_kind"],
            }
            item.update(classify_reference_item(item))
            table_program_items.append(item)

        if QUESTION_RE.search(raw_line):
            counts["question_marks"] += 1

    summary = {
        "line_count": len(lines),
        "path": source["path"],
        "priority": source["priority"],
        "series": source["series"],
        "witness_id": source["witness_id"],
        "witness_kind": source["witness_kind"],
        "counts": {
            "program_references": counts["program_references"],
            "qa_pair_candidates": counts["qa_pair_candidates"],
            "question_marks": counts["question_marks"],
            "table_references": counts["table_references"],
        },
    }
    return summary, qa_items, table_program_items


def build_qa_triage(qa_items: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    triage = {
        "accepted_theory_items": [],
        "ocr_or_ambiguity_items": [],
        "needs_manual_review": [],
    }
    for item in qa_items:
        triage[str(item["triage_bucket"])].append(item)
    return triage


def build_priority_queue(
    qa_items: list[dict[str, object]],
    table_program_items: list[dict[str, object]],
) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []

    for item in qa_items:
        if item["triage_bucket"] == "accepted_theory_items":
            priority_reason = "Direct answer marker with theory-bearing context."
        elif item["triage_bucket"] == "ocr_or_ambiguity_items":
            priority_reason = "Direct answer marker but OCR/ambiguity cleanup needed first."
        else:
            priority_reason = "Direct answer marker requiring manual semantic review."
        queue.append(
            {
                "kind": "qa_pair_candidate",
                "confidence": item["confidence"],
                "line_no": item["line_no"],
                "page": item["page"],
                "priority_reason": priority_reason,
                "series": item["series"],
                "source_path": item["source_path"],
                "triage_bucket": item["triage_bucket"],
                "text": item["answer_text"],
                "witness_id": item["witness_id"],
            }
        )

    for item in table_program_items:
        text = str(item["text"])
        lowered = text.lower()
        if "table below" in lowered or "finish" in lowered or "quantize" in lowered:
            priority_reason = "Likely actionable table/program passage for Ben-continuation work."
        elif item["kind"] == "program_reference":
            priority_reason = "Program mention likely maps to reproducible implementation work."
        else:
            priority_reason = "Table mention likely maps to build-out work."
        queue.append(
            {
                "kind": item["kind"],
                "confidence": item["confidence"],
                "line_no": item["line_no"],
                "page": item["page"],
                "priority_reason": priority_reason,
                "series": item["series"],
                "source_path": item["source_path"],
                "semantic_tags": item["semantic_tags"],
                "text": text,
                "witness_id": item["witness_id"],
            }
        )

    def score(entry: dict[str, object]) -> tuple[int, int, float, int]:
        series_rank = {"Pyth-1": 1, "Pyth-2": 2, "Pyth-3": 3}.get(str(entry["series"]), 9)
        kind_rank = {
            "qa_pair_candidate": 1,
            "program_reference": 2,
            "table_reference": 3,
        }.get(str(entry["kind"]), 9)
        confidence_rank = -float(entry.get("confidence", 0.0))
        return (series_rank, kind_rank, confidence_rank, int(entry["line_no"]))

    queue.sort(key=score)
    return queue[:200]


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def self_test() -> int:
    sample = {
        "path": "inline",
        "priority": 1,
        "series": "Pyth-1",
        "witness_id": "sample",
        "witness_kind": "inline",
    }
    temp_path = ROOT / "pythagoras_quantum_world_rt" / "_self_test_sample.md"
    temp_path.write_text(
        "\n".join(
            [
                "<!-- page 7 -->",
                "Question prompt",
                "Ans: Example answer",
                "Finish out the table below.",
                "Program Quantize will help.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        sample["path"] = str(temp_path.relative_to(ROOT))
        summary, qa_items, table_program_items = extract_from_source(sample)
        triage = build_qa_triage(qa_items)
        ok = (
            summary["counts"]["qa_pair_candidates"] == 1
            and summary["counts"]["table_references"] == 1
            and summary["counts"]["program_references"] == 1
            and qa_items[0]["page"] == 7
            and qa_items[0]["context_after"][0]["text"] == "Finish out the table below."
            and sum(len(items) for items in triage.values()) == 1
            and table_program_items[0]["page"] == 7
        )
        print(canonical_dump({"ok": ok}))
        return 0 if ok else 1
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_inventory: list[dict[str, object]] = []
    qa_pair_candidates: list[dict[str, object]] = []
    table_program_inventory: list[dict[str, object]] = []

    for source in SOURCES:
        summary, qa_items, table_program_items = extract_from_source(source)
        corpus_inventory.append(summary)
        qa_pair_candidates.extend(qa_items)
        table_program_inventory.extend(table_program_items)

    qa_triage = build_qa_triage(qa_pair_candidates)
    priority_queue = build_priority_queue(qa_pair_candidates, table_program_inventory)

    write_json(out_dir / "corpus_inventory.json", {"sources": corpus_inventory})
    write_json(out_dir / "qa_pair_candidates.json", {"items": qa_pair_candidates})
    write_json(out_dir / "table_program_inventory.json", {"items": table_program_inventory})
    write_json(out_dir / "priority_work_queue.json", {"items": priority_queue})
    write_json(out_dir / "accepted_theory_items.json", {"items": qa_triage["accepted_theory_items"]})
    write_json(out_dir / "ocr_or_ambiguity_items.json", {"items": qa_triage["ocr_or_ambiguity_items"]})
    write_json(out_dir / "needs_manual_review.json", {"items": qa_triage["needs_manual_review"]})

    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    str(out_dir / "corpus_inventory.json"),
                    str(out_dir / "qa_pair_candidates.json"),
                    str(out_dir / "table_program_inventory.json"),
                    str(out_dir / "priority_work_queue.json"),
                    str(out_dir / "accepted_theory_items.json"),
                    str(out_dir / "ocr_or_ambiguity_items.json"),
                    str(out_dir / "needs_manual_review.json"),
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
