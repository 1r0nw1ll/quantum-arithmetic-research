#!/usr/bin/env python3
"""Build the Figure 29 C=5040 narrative packet for Pyth-2."""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"

SOURCE_PATH = "qa_corpus_text/pyth_2__ocr__pyth2.md"
SOURCE_LINES = [8586, 8601]


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def factor_family_rows(c_value: int) -> list[dict[str, int | bool]]:
    target = c_value // 2
    rows = []
    for e in range(1, target + 1):
        if target % e != 0:
            continue
        d = target // e
        if e > d:
            continue
        d2 = d * d
        rows.append(
            {
                "C": c_value,
                "D": d2,
                "d": d,
                "e": e,
                "coprime": gcd(e, d) == 1,
                "radius_high": d2 + (c_value // 2),
                "radius_low": d2 - (c_value // 2),
            }
        )
    rows.sort(key=lambda row: row["e"])
    return rows


def candidate_orderings(rows: list[dict[str, int | bool]]) -> list[dict[str, object]]:
    orderings = []
    candidates = [
        ("ascending_e", lambda row: row["e"]),
        ("ascending_radius_low", lambda row: row["radius_low"]),
        ("descending_d", lambda row: -int(row["d"])),
    ]
    for label, key_fn in candidates:
        ordered = sorted(rows, key=key_fn)
        shown = ordered[:6]
        noncoprime_positions = [
            index
            for index, row in enumerate(shown, start=1)
            if not bool(row["coprime"])
        ]
        orderings.append(
            {
                "candidate_order": label,
                "first_six_rows": [
                    {
                        "coprime": row["coprime"],
                        "d": row["d"],
                        "e": row["e"],
                        "radius_low": row["radius_low"],
                        "radius_high": row["radius_high"],
                    }
                    for row in shown
                ],
                "matches_inner_and_fourth_noncoprime": noncoprime_positions == [1, 4],
                "noncoprime_positions_within_first_six": noncoprime_positions,
            }
        )
    return orderings


def build_payload() -> dict[str, object]:
    rows = factor_family_rows(5040)
    coprime_rows = [row for row in rows if bool(row["coprime"])]
    tested_orderings = candidate_orderings(rows)
    return {
        "figure_29_fixed_c_family": {
            "all_factor_pairs": rows,
            "all_factor_pair_count": len(rows),
            "c_value": 5040,
            "coprime_factor_pairs": coprime_rows,
            "coprime_factor_pair_count": len(coprime_rows),
            "integer_radius_point_count": 10080,
            "source_lines": SOURCE_LINES,
            "source_path": SOURCE_PATH,
        },
        "stable_narrative_claims": [
            {
                "claim": "Figure 29 is a fixed-C family with C = 5040 rather than a single ellipse.",
                "source_lines": [8586, 8588],
            },
            {
                "claim": "For fixed C = 5040, there are 24 factor pairs of C/2 = 2520 with e <= d, of which 8 are coprime.",
                "source_lines": [8587, 8601],
            },
            {
                "claim": "Every ellipse in the C = 5040 family has 2C = 10080 integer radius points.",
                "source_lines": [8530, 8538],
            },
        ],
        "tested_display_orderings": tested_orderings,
        "unresolved_claims": [
            {
                "id": "pyth2_figure29_5040_count",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the source claim of 9 coprime ellipses quarantined; direct reconstruction of the coprime factor-pair family gives 8.",
                "source_claim": "There would be a total of 9 ellipses in this series for which the bead numbers would be coprime.",
                "source_lines": [8588, 8601],
            },
            {
                "id": "pyth2_figure29_visibility_order",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the 'six shown / outer five omitted / inner and fourth noncoprime' display wording quarantined; the OCR witness does not preserve the figure and the tested simple orderings do not reproduce that pattern.",
                "source_claim": "Figure 29 shows only six concentric ellipses ... The outer 5 are not shown, and the inner ellipse does not have coprime bead numbers, nor does the fourth.",
                "source_lines": [8586, 8591],
                "tested_candidate_orders": tested_orderings,
            },
            {
                "id": "pyth2_figure29_gaussian_integer_family",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the Gaussian-integer and 'nine different answer sets' language quarantined until a validated reconstruction links the C=5040 family to explicit cattle-count generations.",
                "source_claim": "The numbers of the various cattle will be given by the lengths of radial lines ... and the numbers in each case will be a family of Gaussian integers. So there could be nine different sets of answers ...",
                "source_lines": [8592, 8601],
            },
        ],
        "summary": {
            "all_factor_pair_count": len(rows),
            "coprime_factor_pair_count": len(coprime_rows),
            "series": "Pyth-2",
            "source_ambiguity_count": 3,
            "tested_candidate_order_count": len(tested_orderings),
        },
    }


def self_test() -> int:
    payload = build_payload()
    all_rows = payload["figure_29_fixed_c_family"]["all_factor_pairs"]
    coprime_rows = payload["figure_29_fixed_c_family"]["coprime_factor_pairs"]
    tested_orderings = payload["tested_display_orderings"]
    ok = (
        payload["figure_29_fixed_c_family"]["all_factor_pair_count"] == 24
        and payload["figure_29_fixed_c_family"]["coprime_factor_pair_count"] == 8
        and all_rows[0]["e"] == 1
        and coprime_rows[-1]["d"] == 56
        and not any(item["matches_inner_and_fourth_noncoprime"] for item in tested_orderings)
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
    write_json(OUT_DIR / "pyth2_figure29_narrative.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_figure29_narrative.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
