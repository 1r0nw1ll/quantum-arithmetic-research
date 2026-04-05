#!/usr/bin/env python3
"""Standalone validated Pyth-1 programs and artifact emitter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def canonical_tuple_from_d_e(d: int, e: int) -> dict[str, int]:
    if d <= 0 or e <= 0 or d <= e:
        raise ValueError("Require d > e > 0 for the canonical primitive direction.")
    b = d - e
    a = d + e
    return {"a": a, "b": b, "d": d, "e": e}


def core_values_from_tuple(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    B = b * b
    E = e * e
    A = a * a
    D = d * d
    C = 2 * d * e
    F = a * b
    G = d * d + e * e
    H = C + F
    I = abs(C - F)
    J = b * d
    K = d * a
    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "E": E,
        "F": F,
        "G": G,
        "H": H,
        "I": I,
        "J": J,
        "K": K,
        "a": a,
        "b": b,
        "d": d,
        "e": e,
    }


def core_values_from_d_e(d: int, e: int) -> dict[str, int]:
    t = canonical_tuple_from_d_e(d, e)
    return core_values_from_tuple(t["b"], t["e"])


def complete_tuple_from_known_beads(*, e: int | None = None, b: int | None = None, d: int | None = None, a: int | None = None) -> dict[str, int]:
    if e is not None and b is not None and a is not None:
        inferred_d = b + e
        if a != b + 2 * e:
            raise ValueError("Inconsistent bead values for (b,e,a).")
        return {"a": a, "b": b, "d": inferred_d, "e": e}
    if e is not None and d is not None and a is not None:
        inferred_b = d - e
        if a != d + e:
            raise ValueError("Inconsistent bead values for (e,d,a).")
        return {"a": a, "b": inferred_b, "d": d, "e": e}
    raise ValueError("Unsupported bead completion pattern.")


def tuples_for_green_quadrance(c_value: int) -> list[dict[str, object]]:
    if c_value <= 0 or c_value % 2 != 0:
        raise ValueError("Green quadrance C must be a positive even integer.")
    target = c_value // 2
    rows: list[dict[str, object]] = []
    for e in range(1, target + 1):
        if target % e != 0:
            continue
        d = target // e
        if d <= e:
            continue
        tuple_values = canonical_tuple_from_d_e(d, e)
        rows.append(
            {
                "quadrances": {
                    "C": c_value,
                    "F": tuple_values["a"] * tuple_values["b"],
                    "G": d * d + e * e,
                },
                "tuple": tuple_values,
            }
        )
    rows.sort(key=lambda row: row["tuple"]["e"])
    return rows


def h_identity_forms(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    via_d_e = d * d + 2 * d * e - e * e
    via_a_b = (a * a + 2 * a * b - b * b) // 2
    via_b_e = b * b + 4 * b * e + 2 * e * e
    return {
        "via_a_b": via_a_b,
        "via_b_e": via_b_e,
        "via_d_e": via_d_e,
    }


def table3_block(b: int, e: int) -> dict[str, object]:
    core = core_values_from_tuple(b, e)
    result = {
        "core_values": core,
        "row_A_to_E": {"A": core["A"], "B": core["B"], "D": core["D"], "E": core["E"]},
        "row_C_to_G": {"C": core["C"], "F": core["F"], "G": core["G"]},
        "row_H_to_K": {"H": core["H"], "I": core["I"], "J": core["J"], "K": core["K"]},
    }
    if b == 7 and e == 6:
        result["raw_source_holdout"] = {"L": 1729, "status": "unvalidated_raw_source_cell"}
    return result


def construction_reference_payload() -> dict[str, object]:
    return {
        "C_construction": {
            "claim": "C=20 is represented by two 2-by-5 rectangles for the 20-21-29 triangle.",
            "tuple": {"a": 7, "b": 3, "d": 5, "e": 2},
            "values": {"C": 20, "rectangle_dimensions": [2, 5]},
        },
        "F_construction": {
            "claim": "F=21 is represented by a 3-by-7 rectangle and the corrected gnomon identity F=D-E=25-4.",
            "tuple": {"a": 7, "b": 3, "d": 5, "e": 2},
            "values": {"D": 25, "E": 4, "F": 21, "rectangle_dimensions": [3, 7]},
        },
    }


def koenig_reference_payload() -> dict[str, object]:
    return {
        "backward_trace_to_193": [1, 5, 7, 13, 17, 137, 193],
        "block_transition_roots": [1, 7, 17, 193],
        "forward_successors_from_193": [497, 599],
    }


def build_program_artifacts() -> dict[str, object]:
    bead_completion_examples = [
        {
            "input": {"a": 9, "b": 7, "e": 1},
            "output": complete_tuple_from_known_beads(a=9, b=7, e=1),
        },
        {
            "input": {"a": 61, "d": 60, "e": 1},
            "output": complete_tuple_from_known_beads(a=61, d=60, e=1),
        },
    ]
    quadrance_examples = {
        "C=20": tuples_for_green_quadrance(20),
        "C=60": tuples_for_green_quadrance(60),
    }
    h_example = {
        "tuple": {"a": 3, "b": 1, "d": 2, "e": 1},
        "forms": h_identity_forms(1, 1),
    }
    block_example = table3_block(7, 6)
    return {
        "bead_completion_examples": bead_completion_examples,
        "construction_references": construction_reference_payload(),
        "h_identity_example": h_example,
        "koenig_references": koenig_reference_payload(),
        "quadrance_level_sets": quadrance_examples,
        "table3_block_examples": {"b=7,e=6": block_example},
        "summary": {
            "bead_completion_example_count": len(bead_completion_examples),
            "construction_reference_count": 2,
            "koenig_reference_count": 3,
            "quadrance_level_set_count": len(quadrance_examples),
            "series": "Pyth-1",
            "table3_block_count": 1,
        },
    }


def self_test() -> int:
    tuple_1086 = complete_tuple_from_known_beads(a=9, b=7, e=1)
    tuple_1095 = complete_tuple_from_known_beads(a=61, d=60, e=1)
    c60_rows = tuples_for_green_quadrance(60)
    block = table3_block(7, 6)
    h_forms = h_identity_forms(1, 1)
    ok = (
        tuple_1086 == {"a": 9, "b": 7, "d": 8, "e": 1}
        and tuple_1095["b"] == 59
        and len(c60_rows) == 4
        and c60_rows[-1]["quadrances"]["G"] == 61
        and block["row_C_to_G"] == {"C": 156, "F": 133, "G": 205}
        and block["raw_source_holdout"]["L"] == 1729
        and h_forms["via_d_e"] == h_forms["via_a_b"] == h_forms["via_b_e"] == 7
        and koenig_reference_payload()["forward_successors_from_193"] == [497, 599]
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

    if args.emit_json:
        payload = build_program_artifacts()
        write_json(OUT_DIR / "pyth1_program_artifacts.json", payload)
        print(canonical_dump({"ok": True, "outputs": [str(OUT_DIR / "pyth1_program_artifacts.json")]}))
        return 0

    print(canonical_dump(build_program_artifacts()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
