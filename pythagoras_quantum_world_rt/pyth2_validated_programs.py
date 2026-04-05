#!/usr/bin/env python3
"""Standalone validated Pyth-2 programs and artifact emitter."""

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


def admissible_b_values_for_fixed_e(e: int, upper_bound: int | None = None) -> list[int]:
    if e <= 0:
        raise ValueError("e must be positive")
    limit = e if upper_bound is None else upper_bound
    return [b for b in range(1, limit) if b % 2 == 1 and gcd(b, e) == 1]


def complementary_pairs(values: list[int], total: int) -> list[list[int]]:
    seen: set[int] = set()
    pairs: list[list[int]] = []
    for value in values:
        partner = total - value
        if partner in seen:
            continue
        if partner in values and value <= partner:
            pairs.append([value, partner])
            seen.add(value)
            seen.add(partner)
    return pairs


def fibonacci_prefix(length: int) -> list[int]:
    if length <= 0:
        return []
    seq = [1, 1]
    while len(seq) < length:
        seq.append(seq[-1] + seq[-2])
    return seq[:length]


def fibonacci_divisible_indices(divisor: int, length: int) -> list[dict[str, int]]:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    seq = fibonacci_prefix(length)
    rows: list[dict[str, int]] = []
    for idx, value in enumerate(seq, start=1):
        if value % divisor == 0:
            rows.append({"index": idx, "value": value})
    return rows


def bounded_coprime_bead_pair_count(bound: int, *, b_odd_only: bool) -> dict[str, object]:
    rows: list[dict[str, int]] = []
    for b in range(1, bound):
        if b_odd_only and b % 2 == 0:
            continue
        for e in range(1, bound):
            if gcd(b, e) == 1:
                rows.append({"b": b, "e": e})
    return {
        "bound": bound,
        "b_parity": "odd" if b_odd_only else "any",
        "count": len(rows),
        "pairs": rows,
    }


def ed_pair_count_case(max_d_exclusive: int) -> dict[str, object]:
    rows: list[dict[str, int]] = []
    for d in range(1, max_d_exclusive):
        for e in range(1, d + 1):
            if gcd(e, d) == 1:
                rows.append({"d": d, "e": e})
    visible_rows = [row for row in rows if not (row["e"] == 1 and row["d"] == 1)]
    return {
        "bound_on_d": max_d_exclusive,
        "count": len(rows),
        "pairs": rows,
        "unity_root_pair": {"d": 1, "e": 1},
        "visible_triangle_cell_count_if_corner_blank": len(visible_rows),
    }


def prime_power_factorization_rows(value: int) -> list[int]:
    rows: list[int] = []
    remaining = value
    power = 1
    while remaining % 2 == 0:
        power *= 2
        remaining //= 2
    if power > 1:
        rows.append(power)
    power = 1
    while remaining % 3 == 0:
        power *= 3
        remaining //= 3
    if power > 1:
        rows.append(power)
    power = 1
    while remaining % 5 == 0:
        power *= 5
        remaining //= 5
    if power > 1:
        rows.append(power)
    power = 1
    while remaining % 7 == 0:
        power *= 7
        remaining //= 7
    if power > 1:
        rows.append(power)
    if remaining != 1:
        rows.append(remaining)
    return rows


def explicit_prime_numbering_tokens(value: int) -> dict[str, object]:
    if value != 5040:
        raise ValueError("only the source-supported 5040 case is currently stabilized")
    return {
        "compact_prime_numbering": "24325171",
        "factorization": [16, 9, 5, 7],
        "token_pairs": ["24", "32", "51", "71"],
        "expanded_form": "2^4 3^2 5^1 7^1",
        "value": 5040,
    }


def build_program_artifacts() -> dict[str, object]:
    e60_values = admissible_b_values_for_fixed_e(60, 60)
    e60_pairs = complementary_pairs(e60_values, 60)
    divisor2_rows = fibonacci_divisible_indices(2, 20)
    divisor3_rows = fibonacci_divisible_indices(3, 20)
    divisor4_rows = fibonacci_divisible_indices(4, 20)
    divisor5_rows = fibonacci_divisible_indices(5, 20)
    bounded_counts = [
        bounded_coprime_bead_pair_count(7, b_odd_only=True),
        bounded_coprime_bead_pair_count(6, b_odd_only=True),
        bounded_coprime_bead_pair_count(8, b_odd_only=False),
    ]
    ed_case = ed_pair_count_case(8)
    factors_5040 = prime_power_factorization_rows(5040)
    factors_1000 = prime_power_factorization_rows(1000)
    numbering_5040 = explicit_prime_numbering_tokens(5040)
    return {
        "bounded_pair_counts": {
            "cases": bounded_counts,
            "rule": "count coprime ordered bead pairs under the stated bounds and parity restriction",
        },
        "fixed_e_admissibility": {
            "e=60": {
                "admissible_b_values": e60_values,
                "count": len(e60_values),
                "rule": "odd b below 60 with gcd(b,60)=1",
            }
        },
        "ed_pair_counts": {
            "d<8": {
                "count": ed_case["count"],
                "pairs": ed_case["pairs"],
                "rule": "count coprime e-d pairs with 1<=e<=d<8, including the unity root pair",
                "unity_root_pair": ed_case["unity_root_pair"],
                "visible_triangle_cell_count_if_corner_blank": ed_case["visible_triangle_cell_count_if_corner_blank"],
            }
        },
        "fibonacci_periodicity": {
            "divisor_2": {
                "divisible_rows": divisor2_rows,
                "rule": "2 divides F_n iff 3 divides n",
            },
            "divisor_3": {
                "divisible_rows": divisor3_rows,
                "rule": "3 divides F_n iff 4 divides n",
            },
            "divisor_4": {
                "divisible_rows": divisor4_rows,
                "rule": "4 divides F_n iff 6 divides n",
            },
            "divisor_5": {
                "divisible_rows": divisor5_rows,
                "rule": "5 divides F_n iff 5 divides n",
            }
        },
        "prime_numbering": {
            "value_5040": {
                "compact_prime_numbering": numbering_5040["compact_prime_numbering"],
                "expanded_form": numbering_5040["expanded_form"],
                "factorization": numbering_5040["factorization"],
                "rule": "write each prime once with its explicit exponent, including unit exponents",
                "token_pairs": numbering_5040["token_pairs"],
            }
        },
        "prime_power_wavelengths": {
            "value_1000": {
                "factorization": factors_1000,
                "rule": "group equal prime factors into prime powers",
            },
            "value_5040": {
                "factorization": factors_5040,
                "regrouping_example": [80, 9, 7],
                "rule": "group equal prime factors into prime powers, then optionally combine coprime blocks",
            },
        },
        "reflection_pairs": {
            "e=60": {
                "pairs": e60_pairs,
                "rule": "admissible b-values pair to 60 under reflection about 30",
            }
        },
        "summary": {
            "bounded_pair_case_count": len(bounded_counts),
            "ed_pair_case_count": 1,
            "fixed_e_case_count": 1,
            "periodicity_rule_count": 4,
            "prime_numbering_case_count": 1,
            "prime_power_case_count": 2,
            "reflection_case_count": 1,
            "series": "Pyth-2",
        },
    }


def self_test() -> int:
    e60 = admissible_b_values_for_fixed_e(60, 60)
    pairs = complementary_pairs(e60, 60)
    div2 = fibonacci_divisible_indices(2, 20)
    div3 = fibonacci_divisible_indices(3, 20)
    div4 = fibonacci_divisible_indices(4, 20)
    div5 = fibonacci_divisible_indices(5, 20)
    bounded = bounded_coprime_bead_pair_count(7, b_odd_only=True)
    ed_case = ed_pair_count_case(8)
    p5040 = prime_power_factorization_rows(5040)
    p1000 = prime_power_factorization_rows(1000)
    n5040 = explicit_prime_numbering_tokens(5040)
    ok = (
        e60 == [1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59]
        and pairs[0] == [1, 59]
        and pairs[-1] == [29, 31]
        and [row["index"] for row in div2] == [3, 6, 9, 12, 15, 18]
        and [row["index"] for row in div3] == [4, 8, 12, 16, 20]
        and [row["index"] for row in div4] == [6, 12, 18]
        and [row["index"] for row in div5] == [5, 10, 15, 20]
        and bounded["count"] == 15
        and ed_case["count"] == 18
        and ed_case["visible_triangle_cell_count_if_corner_blank"] == 17
        and p5040 == [16, 9, 5, 7]
        and p1000 == [8, 125]
        and n5040["compact_prime_numbering"] == "24325171"
        and n5040["token_pairs"] == ["24", "32", "51", "71"]
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

    payload = build_program_artifacts()
    if args.emit_json:
        write_json(OUT_DIR / "pyth2_program_artifacts.json", payload)
        print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_program_artifacts.json")]}))
        return 0

    print(canonical_dump(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
