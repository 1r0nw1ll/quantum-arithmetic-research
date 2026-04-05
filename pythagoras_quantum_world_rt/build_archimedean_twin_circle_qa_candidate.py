#!/usr/bin/env python3
"""Build a QA-native replacement for the inconsistent Archimedean twin-circle radius note."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "archimedean_twin_circle_qa_candidate.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def make_tuple(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = d + e
    return {"a": a, "b": b, "d": d, "e": e}


def add_packet(item: dict[str, int]) -> dict[str, object]:
    b = item["b"]
    e = item["e"]
    d = item["d"]
    a = item["a"]
    j_value = b * d
    x_value = e * d
    d_value = d * d
    k_value = a * d
    w_value = x_value + k_value
    p_value = 2 * w_value
    return {
        "tuple": item,
        "scaled_tuple": [b * d, e * d, d * d, a * d],
        "packet": {
            "J": j_value,
            "X": x_value,
            "D": d_value,
            "K": k_value,
            "W": w_value,
            "P": p_value,
        },
        "checks": {
            "scaled_tuple_matches_jxdk": [j_value, x_value, d_value, k_value],
            "W_equals_X_plus_K": w_value == x_value + k_value,
            "P_equals_2W": p_value == 2 * w_value,
            "W_equals_d_times_e_plus_a": w_value == d * (e + a),
        },
    }


def build_payload() -> dict[str, object]:
    sample_rows = [add_packet(make_tuple(b, e)) for (b, e) in ((1, 1), (1, 2), (2, 3), (3, 5))]

    return {
        "artifact_id": "archimedean_twin_circle_qa_candidate",
        "purpose": "Replace the inconsistent local Archimedean radius note with a corpus-native QA packet stated only in stable variables.",
        "source_hierarchy": {
            "tier_1_corpus_packet": {
                "path": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md",
                "role": "Stable uppercase QA packet and ellipse/circle meanings.",
            },
            "tier_2_qa_first_mapping": {
                "path": "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md",
                "role": "Local QA-first interpretation of the Archimedean / contra-cyclic lane.",
            },
            "tier_3_quarantined_note": {
                "path": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md",
                "role": "Rejected merged radius note kept only for provenance.",
            },
        },
        "corpus_native_packet": {
            "definitions": [
                {
                    "meaning": "Half width of ellipse; locus to apex on ellipse.",
                    "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:215",
                    "symbol": "D",
                    "equation": "D = d*d",
                },
                {
                    "meaning": "Quarter width of ellipse; half length of right triangle.",
                    "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:219",
                    "symbol": "X",
                    "equation": "X = e*d",
                },
                {
                    "meaning": "Distance of loci to outer width of ellipse.",
                    "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:229",
                    "symbol": "J",
                    "equation": "J = b*d",
                },
                {
                    "meaning": "Furthest loci to outer width of ellipse.",
                    "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:230",
                    "symbol": "K",
                    "equation": "K = d*a",
                },
                {
                    "meaning": "Side of equilateral triangle.",
                    "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:232",
                    "symbol": "W",
                    "equation": "W = X + K = d*(e+a)",
                },
                {
                    "meaning": "Diameter of circle.",
                    "source_ref": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:240",
                    "symbol": "P",
                    "equation": "P = 2*W",
                },
            ],
            "stable_lift": {
                "claim": "Scaling the canonical tuple by d lands on the mixed packet, not the square packet.",
                "equation": "d*(b,e,d,a) = (J,X,D,K)",
                "status": "stable",
            },
        },
        "replacement_statement": {
            "summary": "Read the Archimedean twin-circle lane through the corpus-native width packet D/X/J/K/W/P. Do not use `r = d*d - d*sqrt(F)` as the default radius law.",
            "candidate_rules": [
                {
                    "id": "arch_qa_01",
                    "claim": "The ellipse scaffold is fixed by D and X, with D as half-width and X as quarter-width.",
                    "status": "stable",
                },
                {
                    "id": "arch_qa_02",
                    "claim": "The outer-width loci are tracked by J and K, so the natural scaled packet is (J,X,D,K).",
                    "status": "stable",
                },
                {
                    "id": "arch_qa_03",
                    "claim": "If the source asks for a circle diameter, the corpus-native candidate is P = 2*W.",
                    "status": "stable",
                },
                {
                    "id": "arch_qa_04",
                    "claim": "If the source asks for the corresponding circle radius, the stable corpus-native candidate is W = P/2.",
                    "status": "stable",
                },
                {
                    "id": "arch_qa_05",
                    "claim": "D may still serve as an apex or half-width radius-like measure on the ellipse packet, but it should not be conflated with the circle radius unless a witness says so.",
                    "status": "stable",
                },
            ],
            "operational_rule": "Prefer D/X/J/K/W/P when reconstructing Arto’s Archimedean twin-circle formulas. Only reintroduce sqrt(F) if a stronger witness explicitly defines a different radius symbol.",
        },
        "replaced_local_claims": [
            {
                "claim": "r = d*d - d*sqrt(F)",
                "disposition": "rejected_as_default_reading",
                "reason": "Conflicts with the corpus-native packet and usually introduces irrational output.",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:14",
            },
            {
                "claim": "r^2 = D^2 together with the above radius law",
                "disposition": "rejected_as_merged_note",
                "reason": "The pair is QA-inconsistent under stable corpus-native notation.",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
            },
            {
                "claim": "((b,e,d,a)*d)^2 = (J,X,D,A)^2",
                "disposition": "rewritten",
                "reason": "The stable mixed-product lift is `(J,X,D,K)`, not `(J,X,D,A)`.",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
            },
        ],
        "sample_rows": sample_rows,
        "verdict": {
            "qa_native_candidate_ready": True,
            "honest_summary": "The replacement statement is not a proof of Arto’s geometry. It is a disciplined QA-native candidate packet: D/X/J/K/W/P from the corpus, the mixed lift d*(b,e,d,a)=(J,X,D,K), and the explicit removal of the unstable `sqrt(F)` radius note from default use.",
        },
    }


def self_test() -> int:
    row = add_packet(make_tuple(1, 2))
    ok = True
    ok = ok and row["packet"]["D"] == 9
    ok = ok and row["packet"]["J"] == 3
    ok = ok and row["packet"]["X"] == 6
    ok = ok and row["packet"]["K"] == 15
    ok = ok and row["packet"]["W"] == 21
    ok = ok and row["packet"]["P"] == 42
    ok = ok and row["checks"]["scaled_tuple_matches_jxdk"] == [3, 6, 9, 15]
    ok = ok and row["checks"]["W_equals_X_plus_K"]
    ok = ok and row["checks"]["P_equals_2W"]
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
                "outputs": ["pythagoras_quantum_world_rt/archimedean_twin_circle_qa_candidate.json"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
