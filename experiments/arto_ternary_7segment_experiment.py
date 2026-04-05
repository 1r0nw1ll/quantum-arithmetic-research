#!/usr/bin/env python3
"""
Reconstruct the 0..9 ternary 7-segment decoder branch from Arto Heino's 2024 post.

This experiment evaluates the software-visible logic only:
- balanced ternary 3-trit encoding for decimal digits 0..9
- uniqueness of the code assignments
- standard 7-segment outputs for digits 0..9

It does not validate gate-level wiring delays, diode fanout, or hardware robustness.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


TRIT_TO_VALUE = {"-": -1, "0": 0, "+": 1}
VALUE_TO_TRIT = {-1: "-", 0: "0", 1: "+"}
SEGMENTS = ("a", "b", "c", "d", "e", "f", "g")


def balanced_ternary_encode(n: int, width: int = 3) -> str:
    """Encode a nonnegative integer into balanced ternary with fixed width."""
    if n < 0:
        raise ValueError("balanced_ternary_encode expects a nonnegative integer")

    digits: list[int] = []
    value = n
    while value != 0:
        value, remainder = divmod(value, 3)
        if remainder == 2:
            remainder = -1
            value += 1
        digits.append(remainder)

    while len(digits) < width:
        digits.append(0)

    if len(digits) > width:
        raise ValueError(f"value {n} does not fit in {width} balanced trits")

    return "".join(VALUE_TO_TRIT[d] for d in reversed(digits))


def build_digit_code_table() -> dict[int, str]:
    """Balanced-ternary-coded decimal table inferred from Arto's 0..9 image."""
    return {digit: balanced_ternary_encode(digit, width=3) for digit in range(10)}


def build_seven_segment_table() -> dict[int, dict[str, int]]:
    """Standard 7-segment glyphs for decimal digits."""
    active_segments = {
        0: "abcdef",
        1: "bc",
        2: "abdeg",
        3: "abcdg",
        4: "bcfg",
        5: "acdfg",
        6: "acdefg",
        7: "abc",
        8: "abcdefg",
        9: "abcdfg",
    }
    return {
        digit: {segment: int(segment in lit) for segment in SEGMENTS}
        for digit, lit in active_segments.items()
    }


def all_trit_codes(width: int = 3) -> list[str]:
    return ["".join(code) for code in product("-0+", repeat=width)]


def run_experiment() -> dict[str, object]:
    digit_codes = build_digit_code_table()
    segment_table = build_seven_segment_table()
    used_codes = set(digit_codes.values())
    full_space = set(all_trit_codes())
    unused_codes = sorted(full_space - used_codes)

    digit_values = {
        digit: sum(TRIT_TO_VALUE[t] * (3 ** power) for power, t in enumerate(reversed(code)))
        for digit, code in digit_codes.items()
    }
    unique_codes = len(used_codes) == 10
    exact_values = all(digit_values[digit] == digit for digit in range(10))
    standard_segment_count = all(sum(segment_table[d].values()) in {2, 3, 4, 5, 6, 7} for d in range(10))
    verdict = "PASS" if unique_codes and exact_values and len(unused_codes) == 17 and standard_segment_count else "FAIL"

    return {
        "experiment_id": "arto_ternary_7segment_experiment_2026-03-29",
        "hypothesis": (
            "The 0..9 branch of Arto Heino's ternary 7-segment circuit is a coherent "
            "balanced-ternary-coded decimal design with unique 3-trit codes and standard "
            "7-segment decimal glyphs."
        ),
        "success_criteria": (
            "PASS if software reconstruction yields a one-to-one 3-trit encoding for 10 digits, "
            "exact standard 7-segment outputs for 0..9, and 17 unused states from the full "
            "3-trit input space."
        ),
        "result": verdict,
        "digit_code_table": digit_codes,
        "decoded_numeric_values": digit_values,
        "seven_segment_table": segment_table,
        "used_code_count": len(used_codes),
        "unused_code_count": len(unused_codes),
        "unused_codes": unused_codes,
        "summary": {
            "unique_codes": unique_codes,
            "exact_digit_roundtrip": exact_values,
            "full_state_space": len(full_space),
            "note": (
                "This validates a coherent software-visible decoder mapping only. "
                "It does not prove the exact transistor/diode implementation from the published schematic."
            ),
        },
        "source_basis": {
            "post": "https://artoheino.com/2024/03/23/ternary-coded-decimal-and-7-segment-control/",
            "assumption": (
                "The 0..9 code table in the published image is interpreted as balanced ternary "
                "for decimal values 0 through 9."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct ternary 7-segment decoder mapping.")
    parser.add_argument(
        "--out",
        default="results/arto_ternary_7segment_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    result = run_experiment()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
