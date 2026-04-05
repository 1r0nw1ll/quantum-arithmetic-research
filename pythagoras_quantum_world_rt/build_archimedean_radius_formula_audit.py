#!/usr/bin/env python3
"""Audit the quarantined Archimedean radius formulas against corpus-native QA notation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "archimedean_radius_formula_audit.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def round_float(value: float) -> float:
    return round(float(value), 12)


def make_tuple(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = d + e
    return {"b": b, "e": e, "d": d, "a": a}


def add_qa_values(item: dict[str, int]) -> dict[str, object]:
    b = item["b"]
    e = item["e"]
    d = item["d"]
    a = item["a"]
    f_value = a * b
    d_value = d * d
    a_value = a * a
    j_value = b * d
    x_value = e * d
    k_value = d * a
    radius_formula = d_value - d * math.sqrt(f_value)
    return {
        "tuple": item,
        "qa_values": {
            "F": f_value,
            "D": d_value,
            "A": a_value,
            "J": j_value,
            "X": x_value,
            "K": k_value,
        },
        "local_radius_formula": {
            "r": round_float(radius_formula),
            "r_squared": round_float(radius_formula * radius_formula),
        },
        "corpus_radius_if_r_squared_equals_D_squared": {
            "r": d_value,
            "r_squared": d_value * d_value,
        },
        "difference": {
            "r_minus_D": round_float(radius_formula - d_value),
            "r_squared_minus_D_squared": round_float(radius_formula * radius_formula - d_value * d_value),
        },
        "is_F_square": int(math.isqrt(f_value)) * int(math.isqrt(f_value)) == f_value,
    }


def build_payload() -> dict[str, object]:
    sample_tuples = [
        make_tuple(1, 1),
        make_tuple(1, 2),
        make_tuple(1, 3),
        make_tuple(2, 3),
        make_tuple(3, 5),
    ]
    sample_rows = [add_qa_values(item) for item in sample_tuples]

    scanned_total = 0
    f_square_count = 0
    local_equals_corpus_r_count = 0
    local_equals_corpus_rsq_count = 0
    scaled_tuple_matches_a_count = 0

    for b in range(1, 13):
        for e in range(1, 13):
            scanned_total += 1
            item = add_qa_values(make_tuple(b, e))
            f_value = int(item["qa_values"]["F"])
            if item["is_F_square"]:
                f_square_count += 1
            if math.isclose(float(item["local_radius_formula"]["r"]), float(item["corpus_radius_if_r_squared_equals_D_squared"]["r"]), rel_tol=0.0, abs_tol=1.0e-12):
                local_equals_corpus_r_count += 1
            if math.isclose(float(item["local_radius_formula"]["r_squared"]), float(item["corpus_radius_if_r_squared_equals_D_squared"]["r_squared"]), rel_tol=0.0, abs_tol=1.0e-12):
                local_equals_corpus_rsq_count += 1
            if item["qa_values"]["K"] == item["qa_values"]["A"]:
                scaled_tuple_matches_a_count += 1

    return {
        "artifact_id": "archimedean_radius_formula_audit",
        "scope": {
            "question": "Can the local formulas `r = d*d - d*sqrt(F)` and `r^2 = D^2` coexist under corpus-native QA notation?",
            "domain": "positive QA tuples with b>=1 and e>=1",
            "scanned_b_range": [1, 12],
            "scanned_e_range": [1, 12],
        },
        "corpus_native_packet": {
            "definitions": [
                {"symbol": "D", "equation": "D = d*d", "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:212"},
                {"symbol": "A", "equation": "A = a*a", "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:218"},
                {"symbol": "X", "equation": "X = e*d", "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:219"},
                {"symbol": "J", "equation": "J = b*d", "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:229"},
                {"symbol": "K", "equation": "K = d*a", "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:230"},
                {"symbol": "F", "equation": "F = b*a", "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:221"},
            ],
            "stable_reading": "The corpus-native uppercase packet is B,E,D,A as squares, and J,X,K as mixed products.",
        },
        "local_note_claims": {
            "radius_formula": {
                "equation": "r = d*d - d*sqrt(F)",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:14",
            },
            "quadrance_claim": {
                "equation": "r^2 = D^2",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
            },
            "scaling_claim": {
                "equation": "((b,e,d,a)*d)^2 = (J,X,D,A)^2",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
            },
        },
        "sample_rows": sample_rows,
        "domain_scan": {
            "scanned_total": scanned_total,
            "F_square_count": f_square_count,
            "local_equals_corpus_r_count": local_equals_corpus_r_count,
            "local_equals_corpus_r_squared_count": local_equals_corpus_rsq_count,
            "scaled_tuple_matches_A_count": scaled_tuple_matches_a_count,
        },
        "findings": [
            {
                "id": "radius_conflict_01",
                "claim": "Under corpus-native QA notation, `r^2 = D^2` forces `r = D = d*d` for positive radii.",
                "status": "stable",
            },
            {
                "id": "radius_conflict_02",
                "claim": "For every scanned positive QA tuple, the local radius formula fails both `r = D` and `r^2 = D^2`.",
                "status": "stable",
                "evidence": {
                    "local_equals_corpus_r_count": local_equals_corpus_r_count,
                    "local_equals_corpus_r_squared_count": local_equals_corpus_rsq_count,
                    "scanned_total": scanned_total,
                },
            },
            {
                "id": "radius_conflict_03",
                "claim": "The scaling sentence mixes packet types: `(b*d,e*d,d*d,a*d)` corresponds to `(J,X,D,K)`, not `(J,X,D,A)` under corpus-native QA notation.",
                "status": "stable",
                "evidence": {
                    "scaled_tuple_matches_A_count": scaled_tuple_matches_a_count,
                    "scanned_total": scanned_total,
                },
            },
            {
                "id": "radius_conflict_04",
                "claim": "The local radius formula is usually irrational because `F = a*b` is not usually a square.",
                "status": "stable",
                "evidence": {
                    "F_square_count": f_square_count,
                    "scanned_total": scanned_total,
                },
            },
        ],
        "verdict": {
            "qa_consistent_reconciliation_found": False,
            "honest_summary": (
                "No nontrivial reconciliation was found. The corpus-native QA packet supports `D = d*d` and the mixed-product lift `(J,X,D,K)`, "
                "while the local note merges that with a different radius formula. The clean result is that `r = d*d - d*sqrt(F)` and `r^2 = D^2` cannot both stand "
                "under the project's native QA notation for positive tuples."
            ),
        },
        "salvage_rule": {
            "next_step": "Keep both local formulas quarantined unless a new source witness changes the meanings of `r` or `D`.",
            "allowed_reinterpretations": [
                "Treat `r = d*d - d*sqrt(F)` as a different radius symbol than the one used in `r^2 = D^2`.",
                "Treat the scaling claim as a mixed packet `(J,X,D,K)` instead of `(J,X,D,A)`.",
            ],
        },
    }


def self_test() -> int:
    row = add_qa_values(make_tuple(1, 2))
    ok = True
    ok = ok and row["qa_values"]["D"] == 9
    ok = ok and row["qa_values"]["A"] == 25
    ok = ok and row["qa_values"]["K"] == 15
    ok = ok and row["qa_values"]["K"] != row["qa_values"]["A"]
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    payload = build_payload()
    write_json(OUT_PATH, payload)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": ["pythagoras_quantum_world_rt/archimedean_radius_formula_audit.json"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
