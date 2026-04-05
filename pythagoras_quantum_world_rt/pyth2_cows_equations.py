#!/usr/bin/env python3
"""Build the stabilized cow-equation continuation from the bulls answer row."""

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


def source_equations() -> list[dict[str, object]]:
    return [
        {"equation": "W2 = 7/12 (X1 + X2)", "source_lines": [8278, 8285]},
        {"equation": "X2 = 9/20 (Z1 + Z2)", "source_lines": [8278, 8285]},
        {"equation": "Z2 = 11/30 (Y1 + Y2)", "source_lines": [8278, 8285]},
        {"equation": "Y2 = 13/42 (W1 + W2)", "source_lines": [8278, 8285]},
    ]


def substitute_bulls(answer_row: dict[str, int]) -> list[dict[str, object]]:
    w1 = int(answer_row["W1"])
    x1 = int(answer_row["X1"])
    y1 = int(answer_row["Y1"])
    z1 = int(answer_row["Z1"])
    return [
        {
            "equation": "W2 = 7/12 (1602 + X2)",
            "fractional_offset": "1869/2",
            "normalized_affine_form": "W2 = 1869/2 + 7/12 X2",
            "source_lines": [8286, 8292],
        },
        {
            "equation": "X2 = 9/20 (1580 + Z2)",
            "fractional_offset": "711",
            "normalized_affine_form": "X2 = 711 + 9/20 Z2",
            "source_lines": [8286, 8292],
        },
        {
            "equation": "Z2 = 11/30 (891 + Y2)",
            "fractional_offset": "3267/10",
            "normalized_affine_form": "Z2 = 3267/10 + 11/30 Y2",
            "source_lines": [8286, 8292],
        },
        {
            "equation": "Y2 = 13/42 (2226 + W2)",
            "fractional_offset": "689",
            "normalized_affine_form": "Y2 = 689 + 13/42 W2",
            "source_lines": [8286, 8292],
        },
    ]


def exact_solution(answer_row: dict[str, int]) -> dict[str, object]:
    w1 = int(answer_row["W1"])
    x1 = int(answer_row["X1"])
    y1 = int(answer_row["Y1"])
    z1 = int(answer_row["Z1"])
    a = Fraction(7, 12)
    b = Fraction(7, 12) * x1
    c = Fraction(9, 20)
    d = Fraction(9, 20) * z1
    e = Fraction(11, 30)
    f = Fraction(11, 30) * y1
    g = Fraction(13, 42)
    h = Fraction(13, 42) * w1
    const = b + a * (d + c * (f + e * h))
    coeff = a * c * e * g
    w2 = const / (1 - coeff)
    y2 = h + g * w2
    z2 = f + e * y2
    x2 = d + c * z2
    return {
        "W2": str(w2),
        "X2": str(x2),
        "Y2": str(y2),
        "Z2": str(z2),
        "all_integral": all(value.denominator == 1 for value in [w2, x2, y2, z2]),
        "denominator": w2.denominator,
    }


def modular_constraints(answer_row: dict[str, int]) -> list[dict[str, object]]:
    x1 = int(answer_row["X1"])
    y1 = int(answer_row["Y1"])
    z1 = int(answer_row["Z1"])
    w1 = int(answer_row["W1"])
    return [
        {
            "constraint": "X2 ≡ 6 (mod 12)",
            "derivation": "1602 + X2 must be divisible by 12",
            "modulus": 12,
            "residue": x1 % 12,
            "source_lines": [8286, 8298],
        },
        {
            "constraint": "Z2 ≡ 0 (mod 20)",
            "derivation": "1580 + Z2 must be divisible by 20",
            "modulus": 20,
            "residue": z1 % 20,
            "source_lines": [8286, 8298],
        },
        {
            "constraint": "Y2 ≡ 9 (mod 30)",
            "derivation": "891 + Y2 must be divisible by 30",
            "modulus": 30,
            "residue": (-y1) % 30,
            "source_lines": [8292, 8301],
        },
        {
            "constraint": "W2 ≡ 0 (mod 42)",
            "derivation": "2226 + W2 must be divisible by 42",
            "modulus": 42,
            "residue": w1 % 42,
            "source_lines": [8286, 8298],
        },
    ]


def build_payload(bulls_diagram_artifacts: dict[str, object]) -> dict[str, object]:
    answer_row = bulls_diagram_artifacts["answer_row"]
    return {
        "bulls_anchor": answer_row,
        "exact_original_system_solution": exact_solution(answer_row),
        "modular_constraints": modular_constraints(answer_row),
        "source_ambiguities": [
            {
                "id": "pyth2_cows_equation_transposition_hypothesis",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the proposed swap of equations (6) and (7) isolated until a validated ellipse-based derivation supports it.",
                "source_claim": "Quantum Arithmetic actually dictates that equations (6) and (7) must be transposed with each other.",
                "source_lines": [8338, 8372],
            },
            {
                "id": "pyth2_cows_9801_claim",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the 9801-yellow-cows claim isolated; the literal switched equation gives 9801/19 rather than 9801.",
                "source_claim": "In this form, the three nested ellipses do apply, and there would be 9801 yellow cows.",
                "source_lines": [8354, 8361],
            },
            {
                "id": "pyth2_cows_triangular_requirement",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the triangular-number requirement isolated until the cows block is tied to a stabilized QA triangular-number lane.",
                "source_claim": "There is one additional requirement that Y1 plus Z1 should be a triangular number.",
                "source_lines": [8373, 8394],
            },
        ],
        "substituted_equations": substitute_bulls(answer_row),
        "summary": {
            "mod_constraint_count": 4,
            "series": "Pyth-2",
            "source_ambiguity_count": 3,
        },
        "supporting_bulls_diagram_summary": bulls_diagram_artifacts["summary"],
    }


def self_test() -> int:
    bulls_diagram_artifacts = {
        "answer_row": {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580},
        "summary": {"figure26_row_count": 3},
    }
    payload = build_payload(bulls_diagram_artifacts)
    ok = (
        payload["substituted_equations"][0]["normalized_affine_form"] == "W2 = 1869/2 + 7/12 X2"
        and payload["exact_original_system_solution"]["all_integral"] is False
        and payload["exact_original_system_solution"]["denominator"] == 4657
        and payload["modular_constraints"][0]["constraint"] == "X2 ≡ 6 (mod 12)"
        and payload["modular_constraints"][2]["residue"] == 9
        and payload["summary"]["source_ambiguity_count"] == 3
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    bulls_diagram_artifacts = read_json(OUT_DIR / "pyth2_bulls_diagram_artifacts.json")
    payload = build_payload(bulls_diagram_artifacts)
    write_json(OUT_DIR / "pyth2_cows_equation_artifacts.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_cows_equation_artifacts.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
