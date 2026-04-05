#!/usr/bin/env python3
"""Build flat validated Pyth-1 tables from the workset and program artifacts."""

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
                "validation_scope": item["validation_scope"],
                "validation_status": item["validation_status"],
            }
        )
    return rows


def quadrance_level_set_table(program_artifacts: dict[str, object]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for c_label, entries in program_artifacts["quadrance_level_sets"].items():
        c_value = int(c_label.split("=")[1])
        for entry in entries:
            t = entry["tuple"]
            q = entry["quadrances"]
            rows.append(
                {
                    "C": c_value,
                    "F": q["F"],
                    "G": q["G"],
                    "a": t["a"],
                    "b": t["b"],
                    "d": t["d"],
                    "e": t["e"],
                }
            )
    rows.sort(key=lambda row: (row["C"], row["e"]))
    return rows


def table3_core_table(program_artifacts: dict[str, object]) -> list[dict[str, object]]:
    example = program_artifacts["table3_block_examples"]["b=7,e=6"]
    core = example["core_values"]
    row = {
        "A": core["A"],
        "B": core["B"],
        "C": core["C"],
        "D": core["D"],
        "E": core["E"],
        "F": core["F"],
        "G": core["G"],
        "H": core["H"],
        "I": core["I"],
        "J": core["J"],
        "K": core["K"],
        "a": core["a"],
        "b": core["b"],
        "d": core["d"],
        "e": core["e"],
    }
    if "raw_source_holdout" in example:
        row["holdout_L"] = example["raw_source_holdout"]["L"]
        row["holdout_status"] = example["raw_source_holdout"]["status"]
    return [row]


def koenig_reference_table(program_artifacts: dict[str, object]) -> list[dict[str, object]]:
    refs = program_artifacts["koenig_references"]
    return [
        {"kind": "backward_trace_to_193", "values": refs["backward_trace_to_193"]},
        {"kind": "block_transition_roots", "values": refs["block_transition_roots"]},
        {"kind": "forward_successors_from_193", "values": refs["forward_successors_from_193"]},
    ]


def build_tables(workset: dict[str, object], program_artifacts: dict[str, object]) -> dict[str, object]:
    stable_rows = stable_item_table(workset)
    quadrance_rows = quadrance_level_set_table(program_artifacts)
    table3_rows = table3_core_table(program_artifacts)
    koenig_rows = koenig_reference_table(program_artifacts)
    return {
        "koenig_reference_table": koenig_rows,
        "quadrance_level_set_table": quadrance_rows,
        "stable_item_table": stable_rows,
        "summary": {
            "koenig_reference_row_count": len(koenig_rows),
            "quadrance_level_set_row_count": len(quadrance_rows),
            "series": "Pyth-1",
            "stable_item_row_count": len(stable_rows),
            "table3_core_row_count": len(table3_rows),
        },
        "table3_core_table": table3_rows,
    }


def self_test() -> int:
    workset = {
        "stable_items": [
            {
                "id": "x",
                "issue_origin": "none",
                "program_template_tags": ["tuple_completion"],
                "source": {"page": 16},
                "theory_lane": "generator_reconstruction",
                "validation_scope": "full_item",
                "validation_status": "direct_validated",
            }
        ]
    }
    program_artifacts = {
        "koenig_references": {
            "backward_trace_to_193": [1, 5, 7],
            "block_transition_roots": [1, 7],
            "forward_successors_from_193": [497, 599],
        },
        "quadrance_level_sets": {
            "C=20": [
                {
                    "quadrances": {"F": 21, "G": 29},
                    "tuple": {"a": 7, "b": 3, "d": 5, "e": 2},
                }
            ]
        },
        "table3_block_examples": {
            "b=7,e=6": {
                "core_values": {"A": 361, "B": 49, "C": 156, "D": 169, "E": 36, "F": 133, "G": 205, "H": 289, "I": 23, "J": 91, "K": 247, "a": 19, "b": 7, "d": 13, "e": 6},
                "raw_source_holdout": {"L": 1729, "status": "unvalidated_raw_source_cell"},
            }
        },
    }
    payload = build_tables(workset, program_artifacts)
    ok = (
        payload["summary"]["quadrance_level_set_row_count"] == 1
        and payload["table3_core_table"][0]["holdout_L"] == 1729
        and payload["koenig_reference_table"][2]["values"] == [497, 599]
    )
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    workset = read_json(OUT_DIR / "pyth1_validated_workset.json")
    program_artifacts = read_json(OUT_DIR / "pyth1_program_artifacts.json")
    payload = build_tables(workset, program_artifacts)
    write_json(OUT_DIR / "pyth1_validated_tables.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth1_validated_tables.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
