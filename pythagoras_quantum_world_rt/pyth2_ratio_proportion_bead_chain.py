#!/usr/bin/env python3
"""Build the page-133 ratio/proportion bead-chain lane for Pyth-2."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import lcm
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


SOURCE_FRACTIONS = [
    {
        "equation_id": "eq7",
        "generated_bead_numbers": [Fraction(2, 7), Fraction(1, 42), Fraction(13, 42), Fraction(1, 3)],
        "source_fraction_a": Fraction(1, 6),
        "source_fraction_b": Fraction(1, 7),
        "source_lines": [8708, 8714],
    },
    {
        "equation_id": "eq6",
        "generated_bead_numbers": [Fraction(1, 3), Fraction(1, 30), Fraction(11, 30), Fraction(2, 5)],
        "source_fraction_a": Fraction(1, 5),
        "source_fraction_b": Fraction(1, 6),
        "source_lines": [8704, 8707],
    },
    {
        "equation_id": "eq5",
        "generated_bead_numbers": [Fraction(2, 5), Fraction(1, 20), Fraction(9, 20), Fraction(1, 2)],
        "source_fraction_a": Fraction(1, 4),
        "source_fraction_b": Fraction(1, 5),
        "source_lines": [8699, 8702],
    },
    {
        "equation_id": "eq4",
        "generated_bead_numbers": [Fraction(1, 2), Fraction(1, 12), Fraction(7, 12), Fraction(2, 3)],
        "source_fraction_a": Fraction(1, 3),
        "source_fraction_b": Fraction(1, 4),
        "source_lines": [8690, 8696],
    },
]


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def bead_row(entry: dict[str, object], common_scale: int) -> dict[str, object]:
    beads = list(entry["generated_bead_numbers"])
    return {
        "equation_id": entry["equation_id"],
        "fraction_sum": str(entry["source_fraction_a"] + entry["source_fraction_b"]),
        "fraction_difference": str(abs(entry["source_fraction_a"] - entry["source_fraction_b"])),
        "generated_bead_numbers": [str(value) for value in beads],
        "integerized_bead_numbers": [int(value * common_scale) for value in beads],
        "source_fractions": [str(entry["source_fraction_a"]), str(entry["source_fraction_b"])],
        "source_lines": entry["source_lines"],
        "valid_fibonacci_chain": beads[0] + beads[1] == beads[2] and beads[1] + beads[2] == beads[3],
    }


def overlap_chain(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    overlaps = []
    for left, right in zip(rows, rows[1:]):
        left_end = Fraction(left["generated_bead_numbers"][3])
        right_start = Fraction(right["generated_bead_numbers"][0])
        overlaps.append(
            {
                "from_equation": left["equation_id"],
                "shared_value": str(left_end),
                "to_equation": right["equation_id"],
                "values_match": left_end == right_start,
            }
        )
    return overlaps


def build_payload() -> dict[str, object]:
    common_scale = lcm(*[value.denominator for entry in SOURCE_FRACTIONS for value in entry["generated_bead_numbers"]])
    rows = [bead_row(entry, common_scale) for entry in SOURCE_FRACTIONS]
    overlaps = overlap_chain(rows)
    return {
        "bead_chain_rows": rows,
        "continuous_division_summary": {
            "common_scale": common_scale,
            "normalized_answer": "The page-133 ratio/proportion construction gives a single overlapping bead chain running from 2/7 to 2/3, not nine unrelated answer families.",
            "range_endpoints": ["2/7", "2/3"],
            "source_lines": [8715, 8722],
        },
        "overlap_chain": overlaps,
        "summary": {
            "fibonacci_row_count": sum(int(row["valid_fibonacci_chain"]) for row in rows),
            "overlap_count": len(overlaps),
            "series": "Pyth-2",
        },
        "verdict": {
            "distinct_ratio_proportion_lane_exists": True,
            "normalized_answer": "The source's ratio/proportion alternative is a distinct lane: the four cattle fractions generate one continuous Fibonacci-type bead chain 2/7 -> 1/3 -> 2/5 -> 1/2 -> 2/3. This supports a coupled bead-derivation reading rather than multiple unrelated answer sets, but it still does not extract explicit cattle counts by itself.",
        },
    }


def self_test() -> int:
    payload = build_payload()
    ok = (
        payload["continuous_division_summary"]["common_scale"] == 420
        and payload["bead_chain_rows"][0]["integerized_bead_numbers"] == [120, 10, 130, 140]
        and payload["bead_chain_rows"][3]["integerized_bead_numbers"] == [210, 35, 245, 280]
        and all(row["valid_fibonacci_chain"] for row in payload["bead_chain_rows"])
        and all(item["values_match"] for item in payload["overlap_chain"])
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    write_json(OUT_DIR / "pyth2_ratio_proportion_bead_chain.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_ratio_proportion_bead_chain.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
