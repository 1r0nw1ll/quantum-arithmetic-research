#!/usr/bin/env python3
"""
Evaluate balanced-ternary addition as a software-side proxy for Arto Heino's ternary
addition claim.

This experiment validates exact arithmetic coherence only:
- one-digit balanced-ternary full adder
- fixed-width ripple addition across the full representable range

It does not validate transistor counts, gate delays, or physical hardware quality.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


TRITS = (-1, 0, 1)
VALUE_TO_TRIT = {-1: "-", 0: "0", 1: "+"}
TRIT_TO_VALUE = {"-": -1, "0": 0, "+": 1}


def full_adder_digit(x: int, y: int, carry_in: int) -> tuple[int, int]:
    """
    Balanced-ternary one-digit full adder.

    Returns (sum_digit, carry_out) such that:
      x + y + carry_in = sum_digit + 3 * carry_out
    with both outputs in {-1, 0, +1}.
    """
    total = x + y + carry_in
    if total > 1:
        return total - 3, 1
    if total < -1:
        return total + 3, -1
    return total, 0


def encode_balanced_ternary(n: int, width: int) -> list[int]:
    if width <= 0:
        raise ValueError("width must be positive")

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
    return digits


def decode_balanced_ternary(digits: list[int]) -> int:
    return sum(digit * (3**power) for power, digit in enumerate(digits))


def digit_string_lsd_last(digits: list[int]) -> str:
    return "".join(VALUE_TO_TRIT[d] for d in reversed(digits))


def ripple_add(a_digits: list[int], b_digits: list[int]) -> tuple[list[int], int]:
    if len(a_digits) != len(b_digits):
        raise ValueError("digit widths must match")

    carry = 0
    out: list[int] = []
    for a_digit, b_digit in zip(a_digits, b_digits):
        sum_digit, carry = full_adder_digit(a_digit, b_digit, carry)
        out.append(sum_digit)
    return out, carry


def representable_range(width: int) -> tuple[int, int]:
    max_abs = (3**width - 1) // 2
    return -max_abs, max_abs


def equivalent_signed_binary_bits(max_abs: int) -> int:
    bits = 1
    while (2 ** (bits - 1)) - 1 < max_abs:
        bits += 1
    return bits


def run_experiment(width: int) -> dict[str, object]:
    digit_cases: list[dict[str, object]] = []
    all_digit_cases_ok = True

    for x, y, carry_in in product(TRITS, repeat=3):
        sum_digit, carry_out = full_adder_digit(x, y, carry_in)
        total = x + y + carry_in
        reconstructed = sum_digit + 3 * carry_out
        ok = reconstructed == total and sum_digit in TRITS and carry_out in TRITS
        all_digit_cases_ok = all_digit_cases_ok and ok
        digit_cases.append(
            {
                "x": x,
                "y": y,
                "carry_in": carry_in,
                "total": total,
                "sum_digit": sum_digit,
                "carry_out": carry_out,
                "ok": ok,
            }
        )

    lo, hi = representable_range(width)
    exhaustive_pairs_checked = 0
    exhaustive_pairs_in_range = 0
    all_ripple_cases_ok = True
    sample_failures: list[dict[str, object]] = []

    for a in range(lo, hi + 1):
        for b in range(lo, hi + 1):
            exhaustive_pairs_checked += 1
            expected = a + b
            if not (lo <= expected <= hi):
                continue
            exhaustive_pairs_in_range += 1

            a_digits = encode_balanced_ternary(a, width)
            b_digits = encode_balanced_ternary(b, width)
            sum_digits, carry_out = ripple_add(a_digits, b_digits)
            actual = decode_balanced_ternary(sum_digits) + carry_out * (3**width)
            ok = actual == expected
            all_ripple_cases_ok = all_ripple_cases_ok and ok
            if not ok and len(sample_failures) < 10:
                sample_failures.append(
                    {
                        "a": a,
                        "b": b,
                        "expected": expected,
                        "actual": actual,
                        "a_digits": digit_string_lsd_last(a_digits),
                        "b_digits": digit_string_lsd_last(b_digits),
                        "sum_digits": digit_string_lsd_last(sum_digits),
                        "carry_out": carry_out,
                    }
                )

    max_abs = hi
    binary_bits = equivalent_signed_binary_bits(max_abs)
    verdict = "PASS" if all_digit_cases_ok and all_ripple_cases_ok else "FAIL"

    return {
        "experiment_id": "arto_ternary_adder_experiment_2026-03-30",
        "hypothesis": (
            "A balanced-ternary full adder gives an exact local decomposition of x+y+carry_in "
            "into sum_digit + 3*carry_out, and a ripple adder built from that slice exactly "
            "reproduces integer addition across the full in-range representable space."
        ),
        "success_criteria": (
            "PASS if all 27 one-digit full-adder input triples satisfy exact decomposition and "
            "the fixed-width ripple adder exhaustively matches integer addition over all in-range cases."
        ),
        "result": verdict,
        "width_trits": width,
        "representable_range": {"min": lo, "max": hi},
        "full_adder_cases": digit_cases,
        "full_adder_all_cases_ok": all_digit_cases_ok,
        "ripple_addition": {
            "pairs_checked_total": exhaustive_pairs_checked,
            "pairs_checked_in_range": exhaustive_pairs_in_range,
            "all_cases_ok": all_ripple_cases_ok,
            "sample_failures": sample_failures,
        },
        "range_comparison": {
            "balanced_ternary_trits": width,
            "balanced_ternary_symmetric_range": [-max_abs, max_abs],
            "minimum_signed_binary_bits_for_same_max_abs": binary_bits,
            "binary_symmetric_range_at_that_width": [-(2 ** (binary_bits - 1)), (2 ** (binary_bits - 1)) - 1],
        },
        "summary": {
            "note": (
                "This validates arithmetic coherence of balanced-ternary addition in software. "
                "It does not by itself prove a specific hardware implementation, gate economy, or speed advantage."
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate balanced-ternary adder coherence.")
    parser.add_argument(
        "--width",
        type=int,
        default=4,
        help="Number of balanced trits in the ripple adder.",
    )
    parser.add_argument(
        "--out",
        default="results/arto_ternary_adder_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    result = run_experiment(width=args.width)
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
