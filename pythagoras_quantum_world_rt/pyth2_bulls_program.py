#!/usr/bin/env python3
"""Reconstruct and validate the Pyth-2 BULLS program lane."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def source_listing() -> list[dict[str, object]]:
    return [
        {
            "confidence": "corrected_source_validated",
            "line_number": 10,
            "normalized_code": "FOR B = 1 TO 10000",
            "repair_note": "The narrative says the upper limit in lines 1 and 3 is one myriad, so OCR '12600' normalizes to 10000.",
            "source_fragments": ["1g", "FOR B= 1 TO 12600"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 20,
            "normalized_code": "LET Y1 = B",
            "repair_note": "OCR 'V1' is a damaged Y1.",
            "source_fragments": ["26", "LET V1 =.B"],
        },
        {
            "confidence": "source-layer ambiguity",
            "line_number": 30,
            "normalized_code": "FOR C = 1 TO 10000",
            "repair_note": "The upper limit is stabilized by the narrative, but the OCR start value is damaged and later optimization text suggests a STEP 6 variant.",
            "source_fragments": ["3", "=FORC =", "TO 19090"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 40,
            "normalized_code": "LET X1 = C",
            "repair_note": "Faithful except for spacing/punctuation.",
            "source_fragments": ["40)", "LED Xt =.¢"],
        },
        {
            "confidence": "faithful",
            "line_number": 50,
            "normalized_code": "LET W1 = 5/6 * X1 + Y1",
            "repair_note": "Matches the narrative equation (1).",
            "source_fragments": ["56", "LET Wi= 5/6 * X1 + Y1"],
        },
        {
            "confidence": "faithful",
            "line_number": 60,
            "normalized_code": "LET Z1 = 13/42 * W1 + Y1",
            "repair_note": "Matches the narrative equation (3).",
            "source_fragments": ["60", "LET Z1 = 13/42 * Wl + Y1"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 70,
            "normalized_code": "LET X_TEST = 9/20 * Z1 + Y1",
            "repair_note": "The OCR denominator '28' conflicts with the narrative equation (2), which clearly gives 9/20.",
            "source_fragments": ["7@", "LET X1 = 9/28* Z1 + Y1"],
        },
        {
            "confidence": "faithful",
            "line_number": 90,
            "normalized_code": "IF X_TEST <> C THEN GOTO 300",
            "repair_note": "The branch target is the NEXT C line.",
            "source_fragments": ["96", "IF X1 <> C THEN GOTO 259"],
        },
        {
            "confidence": "source-layer ambiguity",
            "line_number": 200,
            "normalized_code": "PRINT C, B, W1, X1, Y1, Z1",
            "repair_note": "The print order is clear but the TAB column layout is OCR-damaged.",
            "source_fragments": ["206", "PRINT TAB 12;C;TAB26;B;TAB $;W1;TAB16;X1;TABO;Y1;TAB16;Z1"],
        },
        {
            "confidence": "faithful",
            "line_number": 300,
            "normalized_code": "NEXT C",
            "repair_note": "Faithful.",
            "source_fragments": ["308", "NEXT C"],
        },
        {
            "confidence": "faithful",
            "line_number": 310,
            "normalized_code": "NEXT B",
            "repair_note": "Faithful.",
            "source_fragments": ["319", "NEXT B"],
        },
    ]


def optional_optimizations() -> list[dict[str, object]]:
    return [
        {
            "confidence": "faithful",
            "line_number": 80,
            "normalized_code": "IF C > (X_TEST + 2) THEN GOTO 310",
            "repair_note": "The narrative explicitly gives this optimization line as an optional addition.",
            "source_lines": [8078, 8088],
        },
        {
            "confidence": "source-layer ambiguity",
            "line_number": 30,
            "normalized_code": "FOR C = 6 TO 10000 STEP 6",
            "repair_note": "The optimization narrative clearly calls for a STEP 6 scan, but the OCR around the changed line number is damaged.",
            "source_lines": [8090, 8098],
        },
        {
            "confidence": "source-layer ambiguity",
            "line_number": 250,
            "normalized_code": "IF C < (X_TEST - 2) THEN LET C = INT (X_TEST/6) * 6",
            "repair_note": "The jump-ahead optimization is described clearly, but the OCR line number is damaged.",
            "source_lines": [8098, 8103],
        },
    ]


def source_equations() -> list[dict[str, object]]:
    return [
        {"equation": "W1 = Y1 + 5/6 X1", "source_lines": [7991, 7996]},
        {"equation": "X1 = Y1 + 9/20 Z1", "source_lines": [7991, 7996]},
        {"equation": "Z1 = Y1 + 13/42 W1", "source_lines": [7991, 7996]},
    ]


def bulls_family(limit_x: int) -> list[dict[str, object]]:
    rows = []
    max_m = limit_x // 534
    for m in range(1, max_m + 1):
        y1 = 297 * m
        x1 = 534 * m
        w1 = 742 * m
        z1 = Fraction(1580 * m, 3)
        rows.append(
            {
                "W1": w1,
                "X1": x1,
                "Y1": y1,
                "Z1": str(z1) if z1.denominator != 1 else z1.numerator,
                "integer_Z1": z1.denominator == 1,
                "m": m,
            }
        )
    return rows


def equation_holds(row: dict[str, object]) -> bool:
    w1 = Fraction(int(row["W1"]), 1)
    x1 = Fraction(int(row["X1"]), 1)
    y1 = Fraction(int(row["Y1"]), 1)
    z_value = row["Z1"]
    if isinstance(z_value, int):
        z1 = Fraction(z_value, 1)
    else:
        z1 = Fraction(z_value)
    return (
        w1 == y1 + Fraction(5, 6) * x1
        and x1 == y1 + Fraction(9, 20) * z1
        and z1 == y1 + Fraction(13, 42) * w1
    )


def answer_row() -> dict[str, object]:
    return {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580, "integer_Z1": True, "m": 3}


def build_payload() -> dict[str, object]:
    family = bulls_family(10000)
    return {
        "normalized_equations": source_equations(),
        "optional_optimizations": optional_optimizations(),
        "reconstructed_listing": source_listing(),
        "solution_family": {
            "answer_row": answer_row(),
            "first_three_rows": family[:3],
            "row_count_with_X1_le_10000": len(family),
            "rows": family,
        },
        "source_ambiguities": [
            {
                "id": "pyth2_bulls_line30_scan_range",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the base scan range and the STEP 6 optimization distinct until the damaged OCR around line 30 is reconciled.",
                "source_claim": "Change line 34 to read: 36 FOR C = ... TO 109% STEP 6",
                "source_lines": [8090, 8098],
            },
            {
                "id": "pyth2_bulls_print_layout",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "The print order is stable, but the TAB column formatting remains OCR-damaged.",
                "source_claim": "PRINT TAB 12;C;TAB26;B;TAB $;W1;TAB16;X1;TABO;Y1;TAB16;Z1",
                "source_lines": [8110, 8117],
            },
        ],
        "source_window": {
            "equation_pages": [115, 117],
            "program_page": 118,
            "series": "Pyth-2",
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
        },
        "validations": [
            {
                "claim": "The first three printed rows satisfy the three bulls equations.",
                "holds": all(equation_holds(row) for row in family[:3]),
                "source_lines": [7991, 7996, 8118, 8124],
            },
            {
                "claim": "The stated answer row is the third member of the printed solution family and gives integer Z1 = 1580.",
                "holds": equation_holds(answer_row()) and family[2] == answer_row(),
                "source_lines": [7997, 8002, 8122, 8124],
            },
        ],
        "summary": {
            "equation_count": 3,
            "row_count_with_X1_le_10000": len(family),
            "series": "Pyth-2",
            "source_ambiguity_count": 2,
        },
    }


def self_test() -> int:
    payload = build_payload()
    ok = (
        payload["reconstructed_listing"][1]["normalized_code"] == "LET Y1 = B"
        and payload["reconstructed_listing"][6]["normalized_code"] == "LET X_TEST = 9/20 * Z1 + Y1"
        and payload["solution_family"]["row_count_with_X1_le_10000"] == 18
        and payload["solution_family"]["first_three_rows"][0]["Z1"] == "1580/3"
        and payload["solution_family"]["answer_row"]["Z1"] == 1580
        and all(item["holds"] for item in payload["validations"])
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    if args.emit_json:
        write_json(OUT_DIR / "pyth2_bulls_program_artifacts.json", payload)
        print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_bulls_program_artifacts.json")]}))
        return 0

    print(canonical_dump(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
