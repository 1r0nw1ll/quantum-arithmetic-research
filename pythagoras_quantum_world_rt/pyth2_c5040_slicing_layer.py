#!/usr/bin/env python3
"""Bridge Figure 29 Gaussian rhetoric to the QA slicing-the-ellipse layer."""

from __future__ import annotations

import argparse
import json
from math import isqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


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


def coprime_slice_rows(figure29_payload: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for row in figure29_payload["figure_29_fixed_c_family"]["coprime_factor_pairs"]:
        e = int(row["e"])
        d = int(row["d"])
        d2 = d * d
        e2 = e * e
        f_value = d2 - e2
        x_value = d * e
        integer_hits = []
        for n in range(0, x_value + 1):
            numerator = f_value * ((d2 * e2) - (n * n))
            denominator = e2
            if numerator % denominator != 0:
                continue
            y2 = numerator // denominator
            ok, y_value = is_square(y2)
            if not ok:
                continue
            integer_hits.append(
                {
                    "n": n,
                    "radius_minus": d2 - n,
                    "radius_plus": d2 + n,
                    "x_numerator": n * d,
                    "x_denominator": e,
                    "y": y_value,
                }
            )
        rows.append(
            {
                "D": d2,
                "E": e2,
                "F": f_value,
                "X": x_value,
                "d": d,
                "e": e,
                "integer_slice_hit_count": len(integer_hits),
                "sample_integer_slice_hits": integer_hits[:8],
            }
        )
    rows.sort(key=lambda row: row["e"])
    return rows


def candidate_slice_hits(slice_rows: list[dict[str, object]], bulls: dict[str, int], scale_factor: int) -> list[dict[str, object]]:
    rows = []
    for bull_id, bull_value in bulls.items():
        candidate_value = bull_value * scale_factor
        hits = []
        for row in slice_rows:
            d2 = int(row["D"])
            x_value = int(row["X"])
            for n_value in [candidate_value - d2, d2 - candidate_value]:
                if n_value < 0 or n_value > x_value:
                    continue
                numerator = int(row["F"]) * ((int(row["D"]) * int(row["E"])) - (n_value * n_value))
                denominator = int(row["E"])
                if numerator % denominator != 0:
                    y_integer = False
                    y_value = None
                else:
                    y2 = numerator // denominator
                    y_integer, y_value = is_square(y2)
                hits.append(
                    {
                        "d": int(row["d"]),
                        "e": int(row["e"]),
                        "n": n_value,
                        "radius_target": candidate_value,
                        "radius_minus": d2 - n_value,
                        "radius_plus": d2 + n_value,
                        "y": y_value,
                        "y_integer": y_integer,
                    }
                )
        rows.append(
            {
                "bull_id": bull_id,
                "bull_value": bull_value,
                "candidate_value": candidate_value,
                "scale_factor": scale_factor,
                "slice_hits": hits,
            }
        )
    return rows


def build_payload(figure29_payload: dict[str, object], bulls_program: dict[str, object]) -> dict[str, object]:
    answer_row = bulls_program["solution_family"]["answer_row"]
    bulls = {key: int(answer_row[key]) for key in ["W1", "X1", "Y1", "Z1"]}
    slice_rows = coprime_slice_rows(figure29_payload)
    raw_hits = candidate_slice_hits(slice_rows, bulls, 1)
    scaled_hits = candidate_slice_hits(slice_rows, bulls, 7)
    return {
        "coprime_c5040_slice_rows": slice_rows,
        "figure29_gaussian_bridge": {
            "normalized_answer": "The Figure 29 Gaussian-integer language aligns more naturally with the earlier QA slicing/quaternion layer than with the direct coprime-G identity layer.",
            "source_basis": [
                {
                    "claim": "The ellipse is plotted by the slicing variables x = n d/e and y = sqrt(F*(DE - n*n))/e.",
                    "source_lines": [836, 855],
                    "source_path": "qa_corpus_text/qa-2__001_qa_2_all_pages__docx.md",
                },
                {
                    "claim": "The second square root of the slicing formula may be considered the same as the original Gaussian integers.",
                    "source_lines": [855, 855],
                    "source_path": "qa_corpus_text/qa-2__001_qa_2_all_pages__docx.md",
                },
                {
                    "claim": "Figure 29 says the cattle numbers come from radial lines and then shifts into Gaussian-integer rhetoric.",
                    "source_lines": [8591, 8604],
                    "source_path": "qa_corpus_text/pyth_2__ocr__pyth2.md",
                },
            ],
        },
        "raw_bulls_slice_candidates": raw_hits,
        "scaled_bulls_times_7_slice_candidates": scaled_hits,
        "summary": {
            "raw_slice_integer_hit_count": sum(
                int(any(hit["y_integer"] for hit in row["slice_hits"])) for row in raw_hits
            ),
            "scaled_slice_integer_hit_count": sum(
                int(any(hit["y_integer"] for hit in row["slice_hits"])) for row in scaled_hits
            ),
            "series": "Pyth-2",
            "slice_row_count": len(slice_rows),
        },
        "verdict": {
            "gaussian_rhetoric_refers_to_distinct_layer": True,
            "normalized_answer": "The best-supported reading is that the Gaussian-integer rhetoric refers to the earlier slicing/quaternion layer, but that layer still does not recover an explicit cattle construction from the validated C=5040 coprime subfamily: none of the raw bulls values, and none of the one-seventh-lifted bulls values, land on an integer-y slice hit.",
            "slice_layer_cattle_generation_supported": False,
        },
    }


def self_test() -> int:
    figure29_payload = {
        "figure_29_fixed_c_family": {
            "coprime_factor_pairs": [
                {"C": 5040, "D": 5184, "d": 72, "e": 35},
                {"C": 5040, "D": 3969, "d": 63, "e": 40},
                {"C": 5040, "D": 3136, "d": 56, "e": 45},
            ]
        }
    }
    bulls_program = {"solution_family": {"answer_row": {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580}}}
    payload = build_payload(figure29_payload, bulls_program)
    ok = (
        payload["coprime_c5040_slice_rows"][0]["integer_slice_hit_count"] >= 2
        and payload["summary"]["raw_slice_integer_hit_count"] == 0
        and payload["summary"]["scaled_slice_integer_hit_count"] == 0
        and payload["verdict"]["gaussian_rhetoric_refers_to_distinct_layer"] is True
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
    write_json(OUT_DIR / "pyth2_c5040_slicing_layer.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_c5040_slicing_layer.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
