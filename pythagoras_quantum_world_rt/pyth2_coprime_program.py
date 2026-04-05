#!/usr/bin/env python3
"""Reconstruct and validate Program COPRIME and Figure 24 for Pyth-2."""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def reconstructed_listing() -> list[dict[str, object]]:
    return [
        {
            "confidence": "corrected_source_validated",
            "line_number": 10,
            "normalized_code": "FOR N = 1 TO 43",
            "repair_note": "OCR splits the upper bound across 'FORN=1' and '70 43'.",
            "source_fragments": ["19", "FORN=1", "70 43"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 20,
            "normalized_code": "FOR O = 1 TO 43",
            "repair_note": "OCR line number '26' normalizes to 20 by the program's 10-step cadence.",
            "source_fragments": ["26", "FOR O = 1 TO 43"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 30,
            "normalized_code": "LET X = N",
            "repair_note": "Faithful except for spacing.",
            "source_fragments": ["39", "LET", "X =N"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 40,
            "normalized_code": "LET Y = O",
            "repair_note": "OCR shows 'LET Y = 0', but zero makes the subtraction loop non-terminating; the paired loop variable O is required.",
            "source_fragments": ["4g", "LET", "Y = 0"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 50,
            "normalized_code": "IF X > Y THEN LET X = X - Y",
            "repair_note": "Faithful except for spacing.",
            "source_fragments": ["56", "IF X > Y THEN LETX=X-Y"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 60,
            "normalized_code": "IF Y > X THEN LET Y = Y - X",
            "repair_note": "Faithful except for spacing.",
            "source_fragments": ["66", "IF Y > X THEN LET Y = Y - X"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 70,
            "normalized_code": "IF X <> Y THEN GOTO 50",
            "repair_note": "The OCR line number is intact; the target follows the Euclid subtraction loop.",
            "source_fragments": ["79", "IF X <> Y THEN GOTO 56"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 80,
            "normalized_code": "IF X > 1 THEN GOTO 200",
            "repair_note": "OCR '244' is a damaged loop-exit target; NEXT O must be the branch target.",
            "source_fragments": ["4)", "IF X > 1 THEN GOTO 244"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 90,
            "normalized_code": "PLOT N,O",
            "repair_note": "Faithful except for OCR punctuation damage in the line number.",
            "source_fragments": ["9%", "PLOTN,O"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 200,
            "normalized_code": "NEXT O",
            "repair_note": "Faithful after restoring spacing.",
            "source_fragments": ["20", "NEXTO"],
        },
        {
            "confidence": "corrected_source_validated",
            "line_number": 210,
            "normalized_code": "NEXT N",
            "repair_note": "OCR '219' normalizes to 210 by the program's 10-step cadence.",
            "source_fragments": ["219", "NEXTN"],
        },
        {
            "confidence": "faithful",
            "line_number": 220,
            "normalized_code": "END",
            "repair_note": "Faithful.",
            "source_fragments": ["220", "END"],
        },
    ]


def modernized_symbolic_listing() -> list[dict[str, object]]:
    return [
        {"line_number": 10, "normalized_code": "FOR B = 1 TO 43"},
        {"line_number": 20, "normalized_code": "FOR E = 1 TO 43"},
        {"line_number": 30, "normalized_code": "LET X = B"},
        {"line_number": 40, "normalized_code": "LET Y = E"},
        {"line_number": 50, "normalized_code": "IF X > Y THEN LET X = X - Y"},
        {"line_number": 60, "normalized_code": "IF Y > X THEN LET Y = Y - X"},
        {"line_number": 70, "normalized_code": "IF X <> Y THEN GOTO 50"},
        {"line_number": 80, "normalized_code": "IF X > 1 THEN GOTO 200"},
        {"line_number": 90, "normalized_code": "PLOT B,E"},
        {"line_number": 200, "normalized_code": "NEXT E"},
        {"line_number": 210, "normalized_code": "NEXT B"},
        {"line_number": 220, "normalized_code": "END"},
    ]


def subtraction_coprime(n: int, o: int) -> bool:
    x = n
    y = o
    while x != y:
        if x > y:
            x = x - y
        elif y > x:
            y = y - x
    return x == 1


def zero_variant_terminates(n: int, max_steps: int = 100) -> bool:
    x = n
    y = 0
    for _ in range(max_steps):
        if x == y:
            return True
        if x > y:
            x = x - y
        elif y > x:
            y = y - x
    return False


def ordered_coprime_points(limit: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for n in range(1, limit + 1):
        for o in range(1, limit + 1):
            if subtraction_coprime(n, o):
                rows.append([n, o])
    return rows


def upper_triangle_unique_pairs(limit: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for n in range(1, limit + 1):
        for o in range(n, limit + 1):
            if subtraction_coprime(n, o):
                rows.append([n, o])
    return rows


def row_hit_counts(points: list[list[int]], limit: int) -> list[dict[str, int]]:
    counts = []
    for n in range(1, limit + 1):
        counts.append({"count": sum(1 for point in points if point[0] == n), "n": n})
    return counts


def diagonal_symmetry(points: list[list[int]]) -> bool:
    point_set = {(n, o) for n, o in points}
    return all((o, n) in point_set for n, o in point_set)


def twin_prime_band_examples(limit: int) -> list[dict[str, object]]:
    points = ordered_coprime_points(limit)
    point_set = {(n, o) for n, o in points}
    rows = []
    for left, right in [(5, 7), (11, 13), (17, 19)]:
        shared = [o for o in range(1, limit + 1) if (left, o) in point_set and (right, o) in point_set]
        rows.append(
            {
                "pair": [left, right],
                "shared_columns": shared,
                "shared_count": len(shared),
            }
        )
    return rows


def build_payload() -> dict[str, object]:
    figure24_points = ordered_coprime_points(20)
    upper_triangle = upper_triangle_unique_pairs(20)
    full_program_points = ordered_coprime_points(43)
    odd_b_20 = sum(1 for n, _o in figure24_points if n % 2 == 1)
    odd_b_17 = sum(1 for n, o in ordered_coprime_points(17) if n % 2 == 1 for _ in [o])
    return {
        "figure24": {
            "axis_limit": 20,
            "diagonal_symmetry": diagonal_symmetry(figure24_points),
            "ordered_coprime_points": figure24_points,
            "ordered_point_count": len(figure24_points),
            "row_hit_counts": row_hit_counts(figure24_points, 20),
            "twin_prime_band_examples": twin_prime_band_examples(20),
            "upper_triangle_unique_count": len(upper_triangle),
            "upper_triangle_unique_pairs": upper_triangle,
        },
        "full_program_run": {
            "axis_limit": 43,
            "ordered_point_count": len(full_program_points),
            "row_hit_counts": row_hit_counts(full_program_points, 43),
        },
        "letter_assignment_note": {
            "modern_comment": "Modern reproductions do not need to preserve the source-era software workaround that replaced the bead-control letters with N and O.",
            "source_comment": "The source program uses loop variables N and O, while the surrounding Figure 24 narrative identifies the control axes as bead numbers b and e.",
            "symbol_mapping": [
                {"qa_symbol": "b", "source_program_variable": "N"},
                {"qa_symbol": "e", "source_program_variable": "O"},
                {"qa_symbol": "working copy of b", "source_program_variable": "X"},
                {"qa_symbol": "working copy of e", "source_program_variable": "Y"},
            ],
        },
        "modernized_symbolic_listing": modernized_symbolic_listing(),
        "normalized_listing": reconstructed_listing(),
        "ocr_repair_validations": [
            {
                "claim": "Line 40 must be 'LET Y = O', not 'LET Y = 0'.",
                "evidence_type": "execution",
                "source_lines": [7187, 7190],
                "validation": {
                    "example_input": {"n": 5, "o": 7},
                    "zero_variant_terminates_within_100_steps": zero_variant_terminates(5, 100),
                    "reconstructed_variant_coprime": subtraction_coprime(5, 7),
                },
            },
            {
                "claim": "The page-99 any-parity count for the first 20 integers is stable.",
                "evidence_type": "counting",
                "source_lines": [7132, 7136],
                "validation": {"ordered_point_count_limit_20": len(figure24_points)},
            },
            {
                "claim": "The figure's diagonal symmetry is stable.",
                "evidence_type": "symmetry",
                "source_lines": [7281, 7282],
                "validation": {"diagonal_symmetry_limit_20": diagonal_symmetry(figure24_points)},
            },
        ],
        "source_ambiguities": [
            {
                "best_supported_interpretation": "The stable count 128 matches the unique upper-triangle coprime pairs for axes 1..20, exploiting the figure's diagonal symmetry.",
                "candidate_counts": [
                    {
                        "count": len(upper_triangle),
                        "description": "Unique unordered coprime pairs for 1..20, counting the diagonal (1,1) once.",
                    },
                    {
                        "count": odd_b_20,
                        "description": "Ordered coprime pairs for 1..20 with odd first coordinate.",
                    },
                    {
                        "count": odd_b_17,
                        "description": "Ordered coprime pairs for 1..17 with odd first coordinate.",
                    },
                ],
                "id": "pyth2_coprime_page99_128_count",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep 128 as a corrected-source figure-count candidate, not yet as a faithful literal reading of the OCR wording.",
                "source_claim": "There are 128 combinations within this range if b is considered to be odd.",
                "source_lines": [7132, 7136],
            }
        ],
        "source_window": {
            "end_page": 103,
            "figure_page": 101,
            "program_page": 100,
            "series": "Pyth-2",
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
            "start_page": 99,
        },
        "summary": {
            "figure24_ordered_point_count": len(figure24_points),
            "figure24_unique_upper_triangle_count": len(upper_triangle),
            "full_program_point_count": len(full_program_points),
            "normalized_listing_line_count": len(reconstructed_listing()),
            "series": "Pyth-2",
            "source_ambiguity_count": 1,
        },
    }


def self_test() -> int:
    payload = build_payload()
    figure24 = payload["figure24"]
    listing = payload["normalized_listing"]
    ok = (
        listing[3]["normalized_code"] == "LET Y = O"
        and payload["modernized_symbolic_listing"][0]["normalized_code"] == "FOR B = 1 TO 43"
        and payload["letter_assignment_note"]["symbol_mapping"][1]["qa_symbol"] == "e"
        and payload["ocr_repair_validations"][0]["validation"]["zero_variant_terminates_within_100_steps"] is False
        and payload["ocr_repair_validations"][0]["validation"]["reconstructed_variant_coprime"] is True
        and figure24["ordered_point_count"] == 255
        and figure24["upper_triangle_unique_count"] == 128
        and payload["full_program_run"]["ordered_point_count"] == 1167
        and figure24["diagonal_symmetry"] is True
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
        write_json(OUT_DIR / "pyth2_coprime_program_artifacts.json", payload)
        print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_coprime_program_artifacts.json")]}))
        return 0

    print(canonical_dump(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
