#!/usr/bin/env python3
"""Build the validated Pyth-1 Table 4 / Koenig layer from stable items only."""

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


def _find_item(stable_items: list[dict[str, object]], target_id: str) -> dict[str, object]:
    for item in stable_items:
        if item["id"] == target_id:
            return item
    raise KeyError(target_id)


def build_layer(workset: dict[str, object], program_artifacts: dict[str, object]) -> dict[str, object]:
    stable_items = workset["stable_items"]
    transition_item = _find_item(stable_items, "pyth1_p080_l6106")
    successor_item = _find_item(stable_items, "pyth1_p080_l6108")
    refs = program_artifacts["koenig_references"]
    roots = list(refs["block_transition_roots"])
    successors = list(refs["forward_successors_from_193"])

    transition_rows = []
    for start, end in zip(roots, roots[1:]):
        transition_rows.append(
            {
                "from_I": start,
                "kind": "validated_block_transition",
                "source_id": transition_item["id"],
                "to_H": end,
                "validation_status": transition_item["validation_status"],
            }
        )
    for successor in successors:
        transition_rows.append(
            {
                "from_I": 193,
                "kind": "validated_forward_successor",
                "source_id": successor_item["id"],
                "to_H": successor,
                "validation_status": successor_item["validation_status"],
            }
        )

    node_values = sorted({row["from_I"] for row in transition_rows} | {row["to_H"] for row in transition_rows})
    node_rows = []
    for value in node_values:
        outgoing = [row["to_H"] for row in transition_rows if row["from_I"] == value]
        incoming = [row["from_I"] for row in transition_rows if row["to_H"] == value]
        node_rows.append(
            {
                "incoming_count": len(incoming),
                "node_value": value,
                "outgoing_count": len(outgoing),
                "outgoing_targets": outgoing,
                "role": "branch_point" if len(outgoing) > 1 else "chain_node",
            }
        )

    return {
        "graph": {
            "nodes": node_rows,
            "transitions": transition_rows,
        },
        "summary": {
            "node_count": len(node_rows),
            "series": "Pyth-1",
            "transition_count": len(transition_rows),
            "validated_source_ids": [transition_item["id"], successor_item["id"]],
        },
        "table4_validated_rows": transition_rows,
    }


def self_test() -> int:
    workset = {
        "stable_items": [
            {"id": "pyth1_p080_l6106", "validation_status": "direct_validated"},
            {"id": "pyth1_p080_l6108", "validation_status": "direct_validated"},
        ]
    }
    program_artifacts = {
        "koenig_references": {
            "block_transition_roots": [1, 7, 17, 193],
            "forward_successors_from_193": [497, 599],
        }
    }
    payload = build_layer(workset, program_artifacts)
    ok = (
        payload["summary"]["transition_count"] == 5
        and payload["graph"]["nodes"][-1]["node_value"] == 599
        and payload["graph"]["nodes"][3]["role"] == "branch_point"
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
    payload = build_layer(workset, program_artifacts)
    write_json(OUT_DIR / "pyth1_koenig_validated_layer.json", payload)
    print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth1_koenig_validated_layer.json")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
