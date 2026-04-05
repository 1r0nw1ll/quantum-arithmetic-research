#!/usr/bin/env python3
"""Reconstruct the page-132 source-intended 9-ellipse family for C=5040."""

from __future__ import annotations

import argparse
import json
from math import gcd, isqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"

SOURCE_LISTED_PAIRS = [(1, 2520), (2, 1260), (5, 504), (7, 360), (8, 315), (9, 280), (35, 72), (40, 63), (45, 56)]


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def is_square(value: int) -> tuple[bool, int | None]:
    if value < 0:
        return (False, None)
    root = isqrt(value)
    return (root * root == value, root)


def pair_row(e: int, d: int) -> dict[str, int | bool]:
    d2 = d * d
    e2 = e * e
    return {
        "C": 5040,
        "D": d2,
        "E": e2,
        "F": d2 - e2,
        "G": d2 + e2,
        "J": d2 - d * e,
        "K": d2 + d * e,
        "X": d * e,
        "coprime": gcd(e, d) == 1,
        "d": d,
        "e": e,
    }


def slice_hits(row: dict[str, int | bool], target: int) -> list[dict[str, int | bool | None]]:
    hits = []
    d2 = int(row["D"])
    x_value = int(row["X"])
    for n_value in [target - d2, d2 - target]:
        if n_value < 0 or n_value > x_value:
            continue
        numerator = int(row["F"]) * ((int(row["D"]) * int(row["E"])) - (n_value * n_value))
        denominator = int(row["E"])
        if numerator % denominator == 0:
            y2 = numerator // denominator
            ok, y_value = is_square(y2)
        else:
            ok, y_value = (False, None)
        hits.append({"n": n_value, "y_integer": ok, "y": y_value})
    return hits


def candidate_tests(rows: list[dict[str, int | bool]], bulls: dict[str, int], scale_factor: int) -> list[dict[str, object]]:
    items = []
    for bull_id, bull_value in bulls.items():
        candidate_value = bull_value * scale_factor
        row_hits = []
        for row in rows:
            direct_g = int(row["G"]) == candidate_value
            interval = int(row["J"]) <= candidate_value <= int(row["K"])
            slices = slice_hits(row, candidate_value)
            if direct_g or interval or slices:
                row_hits.append(
                    {
                        "d": int(row["d"]),
                        "e": int(row["e"]),
                        "coprime": bool(row["coprime"]),
                        "direct_g_match": direct_g,
                        "interval_hit": interval,
                        "slice_hits": slices,
                    }
                )
        items.append(
            {
                "bull_id": bull_id,
                "bull_value": bull_value,
                "candidate_value": candidate_value,
                "scale_factor": scale_factor,
                "row_hits": row_hits,
            }
        )
    return items


def build_payload(figure29_payload: dict[str, object], bulls_program: dict[str, object]) -> dict[str, object]:
    validated_pairs = {
        (int(row["e"]), int(row["d"])) for row in figure29_payload["figure_29_fixed_c_family"]["coprime_factor_pairs"]
    }
    rows = [pair_row(e, d) for e, d in SOURCE_LISTED_PAIRS]
    answer_row = bulls_program["solution_family"]["answer_row"]
    bulls = {key: int(answer_row[key]) for key in ["W1", "X1", "Y1", "Z1"]}
    raw_tests = candidate_tests(rows, bulls, 1)
    scaled_tests = candidate_tests(rows, bulls, 7)
    extra_rows = [row for row in rows if (int(row["e"]), int(row["d"])) not in validated_pairs]
    return {
        "source_intended_family": {
            "listed_pair_count": len(rows),
            "listed_pairs": rows,
            "source_lines": [8627, 8643],
            "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
        },
        "validated_family_overlap": {
            "extra_source_only_pairs": extra_rows,
            "validated_coprime_pair_count": len(validated_pairs),
            "validated_overlap_count": len(rows) - len(extra_rows),
        },
        "raw_bulls_candidates": raw_tests,
        "scaled_bulls_times_7_candidates": scaled_tests,
        "summary": {
            "extra_source_only_pair_count": len(extra_rows),
            "raw_hit_count": sum(int(bool(item["row_hits"])) for item in raw_tests),
            "scaled_hit_count": sum(int(bool(item["row_hits"])) for item in scaled_tests),
            "scaled_integer_slice_hit_count": sum(
                int(
                    any(
                        any(hit["y_integer"] for hit in row["slice_hits"])
                        for row in item["row_hits"]
                    )
                )
                for item in scaled_tests
            ),
            "series": "Pyth-2",
            "source_listed_pair_count": len(rows),
        },
        "verdict": {
            "normalized_answer": "The page-132 count of 9 is best explained by the explicit source list itself: the validated 8 coprime pairs plus one extra source-only noncoprime pair (2,1260). But adding that ninth pair still does not recover explicit cattle counts.",
            "source_intended_nine_family_exists": True,
            "source_only_pair_changes_cattle_verdict": False,
        },
    }


def self_test() -> int:
    figure29_payload = {
        "figure_29_fixed_c_family": {
            "coprime_factor_pairs": [
                {"e": 1, "d": 2520},
                {"e": 5, "d": 504},
                {"e": 7, "d": 360},
                {"e": 8, "d": 315},
                {"e": 9, "d": 280},
                {"e": 35, "d": 72},
                {"e": 40, "d": 63},
                {"e": 45, "d": 56},
            ]
        }
    }
    bulls_program = {"solution_family": {"answer_row": {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580}}}
    payload = build_payload(figure29_payload, bulls_program)
    ok = (
        payload["source_intended_family"]["listed_pair_count"] == 9
        and payload["summary"]["extra_source_only_pair_count"] == 1
        and payload["validated_family_overlap"]["extra_source_only_pairs"][0]["e"] == 2
        and payload["summary"]["scaled_integer_slice_hit_count"] == 0
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    figure29_payload = read_json(OUT_DIR / "pyth2_figure29_narrative.json")
    bulls_program = read_json(OUT_DIR / "pyth2_bulls_program_artifacts.json")
    payload = build_payload(figure29_payload, bulls_program)
    write_json(OUT_DIR / "pyth2_c5040_source_intended_family.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_c5040_source_intended_family.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
