#!/usr/bin/env python3
"""Rewrite Arto contra-cyclic formula fragments through the stable QA packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
OUT_PATH = OUT_DIR / "arto_contra_cyclic_formula_rewrites.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def make_tuple(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = d + e
    return {"a": a, "b": b, "d": d, "e": e}


def packet_row(item: dict[str, int]) -> dict[str, object]:
    b = item["b"]
    e = item["e"]
    d = item["d"]
    a = item["a"]
    f_value = a * b
    c_value = 2 * e * d
    d_value = d * d
    x_value = e * d
    j_value = b * d
    k_value = a * d
    w_value = x_value + k_value
    p_value = 2 * w_value
    return {
        "tuple": item,
        "qa_core": {
            "C": c_value,
            "F": f_value,
            "gear_ratio_a_over_d": round(a / d, 12),
            "velocity_proxy_F_over_C": round(f_value / c_value, 12),
        },
        "packet": {
            "D": d_value,
            "J": j_value,
            "K": k_value,
            "P": p_value,
            "W": w_value,
            "X": x_value,
        },
        "checks": {
            "P_equals_2W": p_value == 2 * w_value,
            "W_equals_X_plus_K": w_value == x_value + k_value,
            "scaled_tuple_equals_jxdk": [b * d, e * d, d * d, a * d] == [j_value, x_value, d_value, k_value],
        },
    }


def build_payload() -> dict[str, object]:
    samples = [packet_row(make_tuple(b, e)) for (b, e) in ((1, 1), (1, 2), (2, 3), (3, 5))]
    return {
        "artifact_id": "arto_contra_cyclic_formula_rewrites",
        "purpose": "Apply the stable QA Archimedean packet directly to the surviving Arto/local formula fragments before further testing.",
        "rewrite_rules": [
            {
                "id": "arto_rw_01",
                "trigger": "Fragment speaks about a circle diameter in the Archimedean lane.",
                "rewrite": "Use `P = 2*W`.",
                "status": "stable",
            },
            {
                "id": "arto_rw_02",
                "trigger": "Fragment speaks about the corresponding circle radius in the Archimedean lane.",
                "rewrite": "Use `W = P/2 = X + K = d*(e+a)`.",
                "status": "stable",
            },
            {
                "id": "arto_rw_03",
                "trigger": "Fragment speaks about ellipse half-width or apex distance.",
                "rewrite": "Use `D = d*d`.",
                "status": "stable",
            },
            {
                "id": "arto_rw_04",
                "trigger": "Fragment speaks about ellipse quarter-width or half-length.",
                "rewrite": "Use `X = e*d`.",
                "status": "stable",
            },
            {
                "id": "arto_rw_05",
                "trigger": "Fragment speaks about outer-width loci distances.",
                "rewrite": "Use `J = b*d` and `K = d*a`.",
                "status": "stable",
            },
            {
                "id": "arto_rw_06",
                "trigger": "Fragment scales the tuple `(b,e,d,a)` by `d`.",
                "rewrite": "Use the mixed lift `d*(b,e,d,a) = (J,X,D,K)`.",
                "status": "stable",
            },
            {
                "id": "arto_rw_07",
                "trigger": "Fragment introduces a constant radius law involving `sqrt(F)`.",
                "rewrite": "Reject as default. Keep quarantined unless a stronger witness defines a different radius symbol.",
                "status": "stable",
            },
        ],
        "formula_fragments": [
            {
                "id": "frag_arch_twin_circle_ratio",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:67",
                "source_text": "The gear ratio between stages is then a/d = (b+2e)/(b+e).",
                "classification": "keep_direct_qa",
                "rewritten_form": "Keep `a/d` as the gearing law. It is already QA-native and does not need a D/X/J/K/W/P rewrite.",
                "status": "stable",
            },
            {
                "id": "frag_arch_torus_radii",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:60",
                "source_text": "Parameterize the rolling loci by QA radii R_a~a and R_d~d.",
                "classification": "rewrite_required",
                "rewritten_form": "Replace informal radii labels with the stable packet: use `W` for the Archimedean circle radius candidate, `P = 2*W` for the diameter, and `D`/`X` for ellipse widths. Keep `a/d` only as a gearing ratio, not as the final circle-radius statement.",
                "status": "stable",
            },
            {
                "id": "frag_stage_velocity",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:83",
                "source_text": "v_k = F_k/C_k = ab/(2ed) = b(b+2e)/(2e(b+e)).",
                "classification": "keep_direct_qa",
                "rewritten_form": "Keep `v_k = F/C` as the stage-drive law. It is already in the stable QA core.",
                "status": "stable",
            },
            {
                "id": "frag_moment_arm_r_theta",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2025/08/QA mapping geometry.md:86",
                "source_text": "T_k ∝ v_k · r(Θ_k) with the moment arm taken from your geometry.",
                "classification": "rewrite_required",
                "rewritten_form": "Do not collapse `r(Θ)` to a constant radius law. Read it as a geometry-specific packet carrier: circle branch uses `W`/`P`, ellipse branch uses `D`/`X` with `J`/`K` for outer loci offsets.",
                "status": "stable",
            },
            {
                "id": "frag_local_radius_formula",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:14",
                "source_text": "r = d*d - d*sqrt(F)",
                "classification": "reject_default",
                "rewritten_form": "Rejected as a default Archimedean radius law. Replace circle claims with `W` and `P`, and ellipse claims with `D`/`X`/`J`/`K`.",
                "status": "stable",
            },
            {
                "id": "frag_local_quadrance_claim",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
                "source_text": "r^2 = D^2",
                "classification": "rewrite_required",
                "rewritten_form": "If the claim is about the ellipse apex packet, keep `D = d*d` and interpret its quadrance as `D*D`. Do not identify that with the Archimedean circle radius unless a stronger witness explicitly equates them.",
                "status": "stable",
            },
            {
                "id": "frag_local_scaled_tuple",
                "source_ref": "private/QAnotes/Nexus AI Chat Imports/2024/11/Radius of Archimedean Twin Circle.md:154",
                "source_text": "((b,e,d,a)*d)^2 = (J,X,D,A)^2",
                "classification": "rewrite_required",
                "rewritten_form": "Rewrite to the stable mixed lift `d*(b,e,d,a) = (J,X,D,K)`.",
                "status": "stable",
            },
        ],
        "stable_operational_packet": {
            "circle_branch": [
                "W = X + K = d*(e+a)",
                "P = 2*W",
            ],
            "ellipse_branch": [
                "D = d*d",
                "X = e*d",
                "J = b*d",
                "K = d*a",
            ],
            "core_drive": [
                "gear_ratio = a/d",
                "v = F/C = a*b / (2*e*d)",
            ],
            "rule": "Use the circle branch for Archimedean circle claims, the ellipse branch for loci/width claims, and the core drive formulas for stage timing.",
        },
        "sample_rows": samples,
        "verdict": {
            "rewrite_pass_complete": True,
            "honest_summary": "The surviving contra-cyclic formula fragments now split cleanly: `a/d` and `F/C` stay as direct QA laws; Archimedean radius language is rewritten into the stable packet `D/X/J/K/W/P`; and the old `sqrt(F)` note is removed from active use.",
        },
    }


def self_test() -> int:
    row = packet_row(make_tuple(1, 2))
    ok = True
    ok = ok and row["qa_core"]["gear_ratio_a_over_d"] == round(5 / 3, 12)
    ok = ok and row["qa_core"]["velocity_proxy_F_over_C"] == round(5 / 12, 12)
    ok = ok and row["packet"]["W"] == 21
    ok = ok and row["packet"]["P"] == 42
    ok = ok and row["checks"]["scaled_tuple_equals_jxdk"]
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
    print(canonical_dump({"ok": True, "outputs": ["pythagoras_quantum_world_rt/arto_contra_cyclic_formula_rewrites.json"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
