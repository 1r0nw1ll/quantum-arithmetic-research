#!/usr/bin/env python3
"""Build flat validated Pyth-2 tables from the stable workset and program artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def stable_item_table(workset: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for item in workset["stable_items"]:
        rows.append(
            {
                "id": item["id"],
                "issue_origin": item["issue_origin"],
                "program_template_tags": item["program_template_tags"],
                "source_page": item["source"]["page"],
                "theory_lane": item["theory_lane"],
                "validation_status": item["validation_status"],
            }
        )
    return rows


def admissibility_table(program_artifacts: dict[str, object]) -> list[dict[str, int]]:
    rows = []
    payload = program_artifacts["fixed_e_admissibility"]["e=60"]
    for b in payload["admissible_b_values"]:
        rows.append({"b": b, "e": 60, "sum_to_60_partner": 60 - b})
    return rows


def reflection_pair_table(program_artifacts: dict[str, object]) -> list[dict[str, int]]:
    rows = []
    for left, right in program_artifacts["reflection_pairs"]["e=60"]["pairs"]:
        rows.append({"left_b": left, "right_b": right, "sum": left + right})
    return rows


def bounded_pair_count_table(program_artifacts: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for case in program_artifacts["bounded_pair_counts"]["cases"]:
        rows.append(
            {
                "b_parity": case["b_parity"],
                "bound": case["bound"],
                "count": case["count"],
            }
        )
    return rows


def ed_pair_count_table(program_artifacts: dict[str, object]) -> list[dict[str, int]]:
    rows = []
    for label, payload in program_artifacts["ed_pair_counts"].items():
        rows.append(
            {
                "bound_on_d_exclusive": int(label.split("<")[1]),
                "count": payload["count"],
                "visible_triangle_cell_count_if_corner_blank": payload["visible_triangle_cell_count_if_corner_blank"],
            }
        )
    rows.sort(key=lambda row: row["bound_on_d_exclusive"])
    return rows


def prime_power_wavelength_table(program_artifacts: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for label, payload in program_artifacts["prime_power_wavelengths"].items():
        value = int(label.split("_")[1])
        row = {
            "factorization": payload["factorization"],
            "value": value,
        }
        if "regrouping_example" in payload:
            row["regrouping_example"] = payload["regrouping_example"]
        rows.append(row)
    rows.sort(key=lambda row: row["value"])
    return rows


def prime_numbering_table(program_artifacts: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for label, payload in program_artifacts["prime_numbering"].items():
        value = int(label.split("_")[1])
        rows.append(
            {
                "compact_prime_numbering": payload["compact_prime_numbering"],
                "expanded_form": payload["expanded_form"],
                "factorization": payload["factorization"],
                "token_pairs": payload["token_pairs"],
                "value": value,
            }
        )
    rows.sort(key=lambda row: row["value"])
    return rows


def fibonacci_periodicity_table(program_artifacts: dict[str, object]) -> list[dict[str, int]]:
    rows = []
    for divisor_key, payload in program_artifacts["fibonacci_periodicity"].items():
        divisor = int(divisor_key.split("_")[1])
        for row in payload["divisible_rows"]:
            rows.append({"divisor": divisor, "index": row["index"], "value": row["value"]})
    rows.sort(key=lambda row: (row["divisor"], row["index"]))
    return rows


def build_tables(workset: dict[str, object], program_artifacts: dict[str, object]) -> dict[str, object]:
    stable_rows = stable_item_table(workset)
    admissibility_rows = admissibility_table(program_artifacts)
    reflection_rows = reflection_pair_table(program_artifacts)
    bounded_rows = bounded_pair_count_table(program_artifacts)
    ed_rows = ed_pair_count_table(program_artifacts)
    numbering_rows = prime_numbering_table(program_artifacts)
    wavelength_rows = prime_power_wavelength_table(program_artifacts)
    periodicity_rows = fibonacci_periodicity_table(program_artifacts)
    return {
        "admissibility_table": admissibility_rows,
        "bounded_pair_count_table": bounded_rows,
        "ed_pair_count_table": ed_rows,
        "fibonacci_periodicity_table": periodicity_rows,
        "prime_numbering_table": numbering_rows,
        "prime_power_wavelength_table": wavelength_rows,
        "reflection_pair_table": reflection_rows,
        "stable_item_table": stable_rows,
        "summary": {
            "admissibility_row_count": len(admissibility_rows),
            "bounded_pair_count_row_count": len(bounded_rows),
            "ed_pair_count_row_count": len(ed_rows),
            "fibonacci_periodicity_row_count": len(periodicity_rows),
            "prime_numbering_row_count": len(numbering_rows),
            "prime_power_wavelength_row_count": len(wavelength_rows),
            "reflection_pair_row_count": len(reflection_rows),
            "series": "Pyth-2",
            "stable_item_row_count": len(stable_rows),
        },
    }


def self_test() -> int:
    workset = {
        "stable_items": [
            {
                "id": "x",
                "issue_origin": "none",
                "program_template_tags": ["fixed_e_counting"],
                "source": {"page": 110},
                "theory_lane": "counting_and_admissibility",
                "validation_status": "direct_validated",
            }
        ]
    }
    program_artifacts = {
        "bounded_pair_counts": {"cases": [{"b_parity": "odd", "bound": 7, "count": 15}]},
        "ed_pair_counts": {
            "d<8": {
                "count": 18,
                "visible_triangle_cell_count_if_corner_blank": 17,
            }
        },
        "fixed_e_admissibility": {"e=60": {"admissible_b_values": [1, 7, 11]}},
        "prime_numbering": {
            "value_5040": {
                "compact_prime_numbering": "24325171",
                "expanded_form": "2^4 3^2 5^1 7^1",
                "factorization": [16, 9, 5, 7],
                "token_pairs": ["24", "32", "51", "71"],
            }
        },
        "prime_power_wavelengths": {
            "value_1000": {"factorization": [8, 125]},
            "value_5040": {"factorization": [16, 9, 5, 7], "regrouping_example": [80, 9, 7]},
        },
        "reflection_pairs": {"e=60": {"pairs": [[1, 59]]}},
        "fibonacci_periodicity": {
            "divisor_2": {"divisible_rows": [{"index": 3, "value": 2}]},
            "divisor_5": {"divisible_rows": [{"index": 5, "value": 5}]},
        },
    }
    payload = build_tables(workset, program_artifacts)
    ok = (
        payload["summary"]["admissibility_row_count"] == 3
        and payload["summary"]["bounded_pair_count_row_count"] == 1
        and payload["summary"]["ed_pair_count_row_count"] == 1
        and payload["reflection_pair_table"][0]["sum"] == 60
        and payload["summary"]["prime_power_wavelength_row_count"] == 2
        and payload["summary"]["prime_numbering_row_count"] == 1
        and payload["prime_numbering_table"][0]["compact_prime_numbering"] == "24325171"
        and payload["ed_pair_count_table"][0]["count"] == 18
        and payload["fibonacci_periodicity_table"][0]["index"] == 3
        and payload["fibonacci_periodicity_table"][1]["index"] == 5
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    workset = read_json(OUT_DIR / "pyth2_validated_workset.json")
    program_artifacts = read_json(OUT_DIR / "pyth2_program_artifacts.json")
    payload = build_tables(workset, program_artifacts)
    write_json(OUT_DIR / "pyth2_validated_tables.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth2_validated_tables.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
