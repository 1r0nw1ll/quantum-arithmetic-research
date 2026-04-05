#!/usr/bin/env python3
"""Formalize the unresolved cows transposition branch and its consequences."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def branch_solution(answer_row: dict[str, int]) -> dict[str, object]:
    w1 = int(answer_row["W1"])
    x1 = int(answer_row["X1"])
    y1 = int(answer_row["Y1"])
    z1 = int(answer_row["Z1"])
    y2 = Fraction(11 * y1, 30 - 11)
    a = Fraction(7, 12)
    b = a * x1
    c = Fraction(9, 20)
    d = c * z1
    e = Fraction(13, 42)
    f = e * w1
    const = b + a * (d + c * f)
    coeff = a * c * e
    w2 = const / (1 - coeff)
    z2 = f + e * w2
    x2 = d + c * z2
    return {
        "W2": str(w2),
        "X2": str(x2),
        "Y2": str(y2),
        "Z2": str(z2),
        "all_integral": all(value.denominator == 1 for value in [w2, x2, y2, z2]),
    }


def build_payload(cows_artifacts: dict[str, object]) -> dict[str, object]:
    answer_row = cows_artifacts["bulls_anchor"]
    branch = branch_solution(answer_row)
    return {
        "branch_assumption": {
            "description": "Swap the right-hand sides of equations (6) and (7), as suggested in the source prose.",
            "equations": [
                "W2 = 7/12 (X1 + X2)",
                "X2 = 9/20 (Z1 + Z2)",
                "Z2 = 13/42 (W1 + W2)",
                "Y2 = 11/30 (Y1 + Y2)",
            ],
            "source_lines": [8338, 8361],
        },
        "branch_consequences": {
            "exact_solution": branch,
            "y2_numerator": 11 * int(answer_row["Y1"]),
            "y2_symbolic": "Y2 = 9801/19",
        },
        "summary": {
            "all_integral": branch["all_integral"],
            "series": "Pyth-2",
        },
    }


def self_test() -> int:
    cows_artifacts = {"bulls_anchor": {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580}}
    payload = build_payload(cows_artifacts)
    ok = (
        payload["branch_consequences"]["exact_solution"]["Y2"] == "9801/19"
        and payload["branch_consequences"]["y2_numerator"] == 9801
        and payload["summary"]["all_integral"] is False
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    cows_artifacts = read_json(OUT_DIR / "pyth2_cows_equation_artifacts.json")
    payload = build_payload(cows_artifacts)
    write_json(OUT_DIR / "pyth2_cows_transposition_branch.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_cows_transposition_branch.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
