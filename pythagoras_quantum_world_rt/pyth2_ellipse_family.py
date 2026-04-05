#!/usr/bin/env python3
"""Build the page-130/page-131 ellipse family mechanics for Pyth-2."""

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


def c_factor_pairs(c_value: int) -> list[dict[str, int]]:
    target = c_value // 2
    rows = []
    for e in range(1, int(target**0.5) + 1):
        if target % e == 0:
            d = target // e
            if gcd(e, d) == 1:
                d2 = d * d
                rows.append(
                    {
                        "C": c_value,
                        "D": d2,
                        "d": d,
                        "e": e,
                        "radius_high": d2 + (c_value // 2),
                        "radius_low": d2 - (c_value // 2),
                    }
                )
    rows.sort(key=lambda row: row["e"])
    return rows


def build_payload() -> dict[str, object]:
    c84 = c_factor_pairs(84)
    c5040 = c_factor_pairs(5040)
    return {
        "c_5040_family": {
            "coprime_factor_pairs": c5040,
            "coprime_factor_pair_count": len(c5040),
            "source_lines": [8548, 8613],
        },
        "c_84_family": {
            "coprime_factor_pairs": c84,
            "coprime_factor_pair_count": len(c84),
            "source_lines": [8530, 8558],
        },
        "general_rules": [
            {
                "rule": "The number of integer radius points is 2C.",
                "source_lines": [8530, 8538],
            },
            {
                "rule": "For a factorization C = 2de, the integer radius interval runs from D - C/2 to D + C/2 with D = d*d.",
                "source_lines": [8551, 8558],
            },
        ],
        "source_ambiguities": [
            {
                "id": "pyth2_ellipse_family_5040_count",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "The source claims 9 coprime factorization ellipses for C = 5040, but the direct coprime factor-pair count is 8.",
                "source_claim": "There would be a total of 9 ellipses in this series for which the bead numbers would be coprime.",
                "source_lines": [8588, 8601],
            },
            {
                "id": "pyth2_ellipse_family_figure29_visibility",
                "issue_label": "source-layer ambiguity",
                "normalized_answer": "Keep the figure-visibility remarks isolated because the OCR witness does not preserve the actual drawing.",
                "source_claim": "Figure 29 shows only six concentric ellipses ... The outer 5 are not shown ...",
                "source_lines": [8586, 8591],
            },
        ],
        "summary": {
            "c_5040_coprime_factor_pair_count": len(c5040),
            "c_84_coprime_factor_pair_count": len(c84),
            "series": "Pyth-2",
            "source_ambiguity_count": 2,
        },
    }


def self_test() -> int:
    payload = build_payload()
    c84 = payload["c_84_family"]["coprime_factor_pairs"]
    ok = (
        payload["c_84_family"]["coprime_factor_pair_count"] == 4
        and c84[0]["radius_low"] == 1722
        and c84[-1]["radius_high"] == 91
        and payload["c_5040_family"]["coprime_factor_pair_count"] == 8
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
    write_json(OUT_DIR / "pyth2_ellipse_family_artifacts.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_ellipse_family_artifacts.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
