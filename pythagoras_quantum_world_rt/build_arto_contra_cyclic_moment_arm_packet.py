#!/usr/bin/env python3
"""Build a packet-derived replacement for the contra-cyclic moment-arm fragment."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "arto_contra_cyclic_moment_arm_packet.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def frac_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def make_tuple(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = d + e
    return {"a": a, "b": b, "d": d, "e": e}


def packet(item: dict[str, int]) -> dict[str, int]:
    b = item["b"]
    e = item["e"]
    d = item["d"]
    a = item["a"]
    x_value = e * d
    j_value = b * d
    d_value = d * d
    k_value = d * a
    w_value = x_value + k_value
    p_value = 2 * w_value
    c_value = 2 * e * d
    f_value = a * b
    return {
        "C": c_value,
        "D": d_value,
        "F": f_value,
        "J": j_value,
        "K": k_value,
        "P": p_value,
        "W": w_value,
        "X": x_value,
    }


def rationalized_ellipse_branch(values: dict[str, int], t_value: Fraction) -> dict[str, object]:
    one = Fraction(1, 1)
    denom = one + t_value * t_value
    x_coord = Fraction(values["K"], 1) * (one - t_value * t_value) / denom
    y_coord = Fraction(2 * values["D"], 1) * t_value / denom
    quadrance = x_coord * x_coord + y_coord * y_coord
    closed_form = (
        Fraction(values["K"] * values["K"], 1) * (one - t_value * t_value) * (one - t_value * t_value)
        + Fraction(4 * values["D"] * values["D"], 1) * t_value * t_value
    ) / (denom * denom)
    return {
        "t": frac_text(t_value),
        "x": frac_text(x_coord),
        "y": frac_text(y_coord),
        "quadrance": frac_text(quadrance),
        "closed_form_quadrance": frac_text(closed_form),
        "matches_closed_form": quadrance == closed_form,
    }


def sample_row(item: dict[str, int]) -> dict[str, object]:
    values = packet(item)
    t_samples = [Fraction(0, 1), Fraction(1, 1), Fraction(2, 1)]
    return {
        "tuple": item,
        "packet": values,
        "packet_identities": {
            "D_equals_J_plus_X": values["D"] == values["J"] + values["X"],
            "K_equals_D_plus_X": values["K"] == values["D"] + values["X"],
            "K_equals_J_plus_C": values["K"] == values["J"] + values["C"],
            "W_equals_K_plus_X": values["W"] == values["K"] + values["X"],
            "W_equals_D_plus_C": values["W"] == values["D"] + values["C"],
            "P_equals_2W": values["P"] == 2 * values["W"],
        },
        "circle_branch": {
            "radius": values["W"],
            "quadrance": values["W"] * values["W"],
            "diameter": values["P"],
        },
        "ellipse_branch_samples": [rationalized_ellipse_branch(values, t_value) for t_value in t_samples],
        "drive_law": {
            "gear_ratio_a_over_d": frac_text(Fraction(item["a"], item["d"])),
            "velocity_proxy_F_over_C": frac_text(Fraction(values["F"], values["C"])),
        },
    }


def build_payload() -> dict[str, object]:
    rows = [sample_row(make_tuple(b, e)) for (b, e) in ((1, 1), (1, 2), (2, 3), (3, 5))]
    return {
        "artifact_id": "arto_contra_cyclic_moment_arm_packet",
        "purpose": "Replace the geometry-dependent `r(Θ)` fragment with packet-derived moment-arm branches stated in stable QA variables.",
        "source_hierarchy": {
            "tier_1_formula_fragment": {
                "path": "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md",
                "refs": [
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:60",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:61",
                    "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:86",
                ],
                "role": "Arto/QA mapping note with the twin-circle torus/ellipse description and the `T_k ∝ v_k*r(Θ_k)` fragment.",
            },
            "tier_2_corpus_packet": {
                "path": "qa_corpus_text/qa-1__qa_1_all_pages__docx.md",
                "refs": [
                    "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:215",
                    "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:219",
                    "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:229",
                    "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:230",
                    "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:232",
                    "qa_corpus_text/qa-1__qa_1_all_pages__docx.md:240",
                ],
                "role": "Stable D/X/J/K/W/P packet used to rewrite radius language.",
            },
            "tier_3_local_rewrite_state": {
                "paths": [
                    "pythagoras_quantum_world_rt/archimedean_twin_circle_qa_candidate.json",
                    "pythagoras_quantum_world_rt/arto_contra_cyclic_formula_rewrites.json",
                ],
                "role": "Current project replacement rules for Archimedean radius statements.",
            },
        },
        "stable_replacement": {
            "direct_rule": "Do not use a free radius symbol for the contra-cyclic moment arm. Route it through the packet branches below.",
            "branches": [
                {
                    "id": "moment_arm_circle_branch",
                    "claim": "If the moment arm is the Archimedean circle branch, use `r_circle = W` and `Q_circle = W*W`, with `P = 2*W`.",
                    "status": "stable",
                },
                {
                    "id": "moment_arm_ellipse_branch",
                    "claim": "If the moment arm is the scaled ellipse branch, derive it from the d-scaled ellipse sample `(x(θ),y(θ)) = (K*cos(θ), D*sin(θ))`.",
                    "status": "stable",
                },
                {
                    "id": "moment_arm_rt_branch",
                    "claim": "For rational-trig use, replace the Euclidean arm by its quadrance carrier `Q_m(t)` under the rational parameter `t = tan(θ/2)`.",
                    "status": "stable",
                },
            ],
        },
        "packet_chain": {
            "claim": "The moment-arm packet is controlled by the exact X-step chain.",
            "relations": [
                "D = J + X",
                "K = D + X",
                "K = J + C",
                "W = K + X",
                "W = D + C",
                "P = 2*W",
            ],
            "status": "stable",
        },
        "moment_arm_forms": {
            "circle_branch": {
                "radius": "W",
                "quadrance": "W*W",
                "diameter": "P = 2*W",
            },
            "ellipse_branch_euclidean": {
                "coordinate_form": "(x(θ),y(θ)) = (K*cos(θ), D*sin(θ))",
                "moment_arm": "m(θ) = sqrt(K*K*cos(θ)*cos(θ) + D*D*sin(θ)*sin(θ))",
            },
            "ellipse_branch_rationalized": {
                "parameter": "t = tan(θ/2)",
                "x_t": "x(t) = K*(1 - t*t)/(1 + t*t)",
                "y_t": "y(t) = 2*D*t/(1 + t*t)",
                "quadrance_t": "Q_m(t) = (K*K*(1 - t*t)*(1 - t*t) + 4*D*D*t*t)/((1 + t*t)*(1 + t*t))",
            },
        },
        "interpretation_rules": [
            {
                "id": "mom_rule_01",
                "rule": "Use `W` when the source is talking about the Archimedean circle itself.",
                "status": "stable",
            },
            {
                "id": "mom_rule_02",
                "rule": "Use the `(K,D)` scaled ellipse sample when the source is talking about a varying arm over the cycle.",
                "status": "stable",
            },
            {
                "id": "mom_rule_03",
                "rule": "Keep `J` and `X` as packet offsets/bounds; do not invent a new free radius symbol from them.",
                "status": "stable",
            },
            {
                "id": "mom_rule_04",
                "rule": "Keep `a/d` and `F/C` as the gearing and drive laws feeding the moment-arm packet, not as replacements for it.",
                "status": "stable",
            },
        ],
        "sample_rows": rows,
        "verdict": {
            "moment_arm_packet_ready": True,
            "honest_summary": "The contra-cyclic `r(Θ)` fragment no longer needs a floating radius symbol. The stable replacement is a branch packet: constant circle arm `W`, or a varying scaled-ellipse arm carried by `(K,D)` and rationalized as `Q_m(t)`.",
        },
    }


def self_test() -> int:
    row = sample_row(make_tuple(1, 2))
    ok = True
    ok = ok and row["packet"]["D"] == 9
    ok = ok and row["packet"]["X"] == 6
    ok = ok and row["packet"]["J"] == 3
    ok = ok and row["packet"]["K"] == 15
    ok = ok and row["packet"]["W"] == 21
    ok = ok and row["packet_identities"]["D_equals_J_plus_X"]
    ok = ok and row["packet_identities"]["K_equals_D_plus_X"]
    ok = ok and row["packet_identities"]["W_equals_D_plus_C"]
    ok = ok and row["circle_branch"]["quadrance"] == 441
    ok = ok and row["ellipse_branch_samples"][1]["quadrance"] == "81/1"
    ok = ok and row["ellipse_branch_samples"][1]["matches_closed_form"]
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
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/arto_contra_cyclic_moment_arm_packet.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
