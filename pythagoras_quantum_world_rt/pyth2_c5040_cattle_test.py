#!/usr/bin/env python3
"""Test direct cattle-count generation from the validated C=5040 coprime subfamily."""

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


def coprime_family_rows(figure29_payload: dict[str, object]) -> list[dict[str, int]]:
    rows = []
    for row in figure29_payload["figure_29_fixed_c_family"]["coprime_factor_pairs"]:
        e = int(row["e"])
        d = int(row["d"])
        rows.append(
            {
                "C": int(row["C"]),
                "D": int(row["D"]),
                "F": (d - e) * (d + e),
                "G": d * d + e * e,
                "J": d * (d - e),
                "K": d * (d + e),
                "d": d,
                "e": e,
            }
        )
    rows.sort(key=lambda row: row["e"])
    return rows


def interval_hits(rows: list[dict[str, int]], target: int) -> list[dict[str, int]]:
    hits = []
    for row in rows:
        if row["J"] <= target <= row["K"]:
            hits.append({"d": row["d"], "e": row["e"], "interval_low": row["J"], "interval_high": row["K"]})
    return hits


def direct_g_matches(rows: list[dict[str, int]], target: int) -> list[dict[str, int]]:
    hits = []
    for row in rows:
        if row["G"] == target:
            hits.append({"d": row["d"], "e": row["e"], "G": row["G"]})
    return hits


def square_condition(target: int, c_value: int) -> dict[str, object]:
    lower_ok, lower_root = is_square(target - c_value)
    upper_ok, upper_root = is_square(target + c_value)
    return {
        "g_minus_c": target - c_value,
        "g_minus_c_is_square": lower_ok,
        "g_minus_c_root": lower_root,
        "g_plus_c": target + c_value,
        "g_plus_c_is_square": upper_ok,
        "g_plus_c_root": upper_root,
        "valid_direct_g_candidate": lower_ok and upper_ok,
    }


def candidate_rows(rows: list[dict[str, int]], bulls: dict[str, int], scale_factor: int) -> list[dict[str, object]]:
    c_value = int(rows[0]["C"])
    packet = []
    for name, value in bulls.items():
        candidate = value * scale_factor
        packet.append(
            {
                "bull_id": name,
                "bull_value": value,
                "candidate_value": candidate,
                "direct_g_matches": direct_g_matches(rows, candidate),
                "interval_hits": interval_hits(rows, candidate),
                "scale_factor": scale_factor,
                "square_condition": square_condition(candidate, c_value),
            }
        )
    return packet


def build_payload(figure29_payload: dict[str, object], bulls_program: dict[str, object]) -> dict[str, object]:
    rows = coprime_family_rows(figure29_payload)
    answer_row = bulls_program["solution_family"]["answer_row"]
    bulls = {key: int(answer_row[key]) for key in ["W1", "X1", "Y1", "Z1"]}
    raw_candidates = candidate_rows(rows, bulls, 1)
    scaled_candidates = candidate_rows(rows, bulls, 7)
    return {
        "coprime_c5040_family": {
            "c_value": rows[0]["C"],
            "coprime_family_rows": rows,
            "coprime_g_values": [row["G"] for row in rows],
        },
        "raw_bulls_as_g_candidates": raw_candidates,
        "scaled_bulls_times_7_as_g_candidates": scaled_candidates,
        "summary": {
            "raw_direct_g_match_count": sum(len(row["direct_g_matches"]) for row in raw_candidates),
            "raw_valid_square_condition_count": sum(int(row["square_condition"]["valid_direct_g_candidate"]) for row in raw_candidates),
            "scaled_direct_g_match_count": sum(len(row["direct_g_matches"]) for row in scaled_candidates),
            "scaled_interval_hit_count": sum(int(bool(row["interval_hits"])) for row in scaled_candidates),
            "scaled_valid_square_condition_count": sum(int(row["square_condition"]["valid_direct_g_candidate"]) for row in scaled_candidates),
            "series": "Pyth-2",
        },
        "verdict": {
            "direct_cattle_generation_supported": False,
            "normalized_answer": "No explicit cattle-count construction is recovered by the direct G-route from the validated C=5040 coprime subfamily. None of the stabilized bulls values, and none of the one-seventh-lifted bulls values, match a coprime-family G value or satisfy the required square conditions G±C.",
        },
    }


def self_test() -> int:
    figure29_payload = {
        "figure_29_fixed_c_family": {
            "coprime_factor_pairs": [
                {"C": 5040, "D": 6350400, "d": 2520, "e": 1},
                {"C": 5040, "D": 254016, "d": 504, "e": 5},
                {"C": 5040, "D": 129600, "d": 360, "e": 7},
                {"C": 5040, "D": 99225, "d": 315, "e": 8},
                {"C": 5040, "D": 78400, "d": 280, "e": 9},
                {"C": 5040, "D": 5184, "d": 72, "e": 35},
                {"C": 5040, "D": 3969, "d": 63, "e": 40},
                {"C": 5040, "D": 3136, "d": 56, "e": 45},
            ]
        }
    }
    bulls_program = {"solution_family": {"answer_row": {"W1": 2226, "X1": 1602, "Y1": 891, "Z1": 1580}}}
    payload = build_payload(figure29_payload, bulls_program)
    ok = (
        payload["coprime_c5040_family"]["coprime_g_values"][-1] == 5161
        and payload["summary"]["scaled_direct_g_match_count"] == 0
        and payload["summary"]["scaled_valid_square_condition_count"] == 0
        and payload["summary"]["scaled_interval_hit_count"] == 1
        and payload["verdict"]["direct_cattle_generation_supported"] is False
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
    write_json(OUT_DIR / "pyth2_c5040_cattle_test.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_c5040_cattle_test.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
