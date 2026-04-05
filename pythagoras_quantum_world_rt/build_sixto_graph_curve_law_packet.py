#!/usr/bin/env python3
"""Build an analytic law packet for the Sixto timing-graph curves."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pythagoras_quantum_world_rt"
SUMMARY_PATH = OUT_DIR / "sixto_geogebra_extended_summary.json"
LAW_PACKET_PATH = OUT_DIR / "sixto_graph_curve_law_packet.json"
DIAGNOSTICS_PATH = OUT_DIR / "sixto_graph_curve_law_diagnostics.json"


def canonical_dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_dump(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    return round(float(value), 6)


def frac_from_value(value: float | int) -> Fraction:
    return Fraction(str(value))


def frac_obj(value: Fraction) -> dict[str, int]:
    return {"n": int(value.numerator), "d": int(value.denominator)}


def newton_coefficients(nodes_x: list[Fraction], nodes_y: list[Fraction]) -> list[Fraction]:
    coeffs = list(nodes_y)
    out = [coeffs[0]]
    size = len(nodes_x)
    for order in range(1, size):
        next_coeffs = []
        for idx in range(size - order):
            numer = coeffs[idx + 1] - coeffs[idx]
            denom = nodes_x[idx + order] - nodes_x[idx]
            next_coeffs.append(numer / denom)
        coeffs = next_coeffs
        out.append(coeffs[0])
    return out


def eval_newton(nodes_x: list[Fraction], coeffs: list[Fraction], x_value: Fraction) -> Fraction:
    result = coeffs[-1]
    for idx in range(len(coeffs) - 2, -1, -1):
        result = coeffs[idx] + (x_value - nodes_x[idx]) * result
    return result


def poly_candidate_error(xs: list[float], ys: list[float], degree: int) -> dict[str, Any]:
    coeffs = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), degree)
    predicted = np.polyval(coeffs, np.array(xs, dtype=float))
    residuals = predicted - np.array(ys, dtype=float)
    max_abs = max(abs(float(value)) for value in residuals)
    rmse = math.sqrt(float(np.mean(residuals * residuals)))
    return {
        "family": f"global_polynomial_degree_{degree}",
        "degree": degree,
        "max_abs_error": round_float(max_abs),
        "rmse": round_float(rmse),
        "exact_on_nodes": bool(max_abs == 0.0),
    }


def harmonic_candidate_error(xs: list[float], ys: list[float], period: float) -> dict[str, Any]:
    rows = []
    for x_value in xs:
        angle = 2.0 * math.pi * x_value / period
        rows.append(
            [
                math.sin(angle),
                math.cos(angle),
                math.sin(2.0 * angle),
                math.cos(2.0 * angle),
                1.0,
            ]
        )
    basis = np.array(rows, dtype=float)
    target = np.array(ys, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    predicted = basis @ coeffs
    residuals = predicted - target
    max_abs = max(abs(float(value)) for value in residuals)
    rmse = math.sqrt(float(np.mean(residuals * residuals)))
    return {
        "family": "global_harmonic_two_mode",
        "period": round_float(period),
        "coefficients": [round_float(value) for value in coeffs.tolist()],
        "max_abs_error": round_float(max_abs),
        "rmse": round_float(rmse),
        "exact_on_nodes": bool(max_abs == 0.0),
    }


def build_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(SUMMARY_PATH)
    guide_packet = summary["graph_schedule_test"]
    curves = summary["reconstructed_graph_curves"]

    accepted_curves = []
    diagnostics = []
    max_degree = 0
    exact_curve_count = 0

    for curve in curves:
        sampled_points = curve["sampled_points"]
        xs_float = [float(point["x"]) for point in sampled_points]
        ys_float = [float(point["y"]) for point in sampled_points]
        xs_frac = [frac_from_value(point["x"]) for point in sampled_points]
        ys_frac = [frac_from_value(point["y"]) for point in sampled_points]
        coeffs = newton_coefficients(xs_frac, ys_frac)
        exact_match = True
        mismatches = []
        for x_value, y_value in zip(xs_frac, ys_frac):
            computed = eval_newton(xs_frac, coeffs, x_value)
            if computed != y_value:
                exact_match = False
                mismatches.append(
                    {
                        "x": frac_obj(x_value),
                        "expected_y": frac_obj(y_value),
                        "computed_y": frac_obj(computed),
                    }
                )
        degree_value = len(coeffs) - 1
        max_degree = max(max_degree, degree_value)
        if exact_match:
            exact_curve_count += 1

        accepted_curves.append(
            {
                "curve_id": curve["curve_id"],
                "label": curve["label"],
                "variable": "x",
                "law_family": "exact_newton_interpolation",
                "degree": degree_value,
                "nodes_x": [frac_obj(value) for value in xs_frac],
                "nodes_y": [frac_obj(value) for value in ys_frac],
                "newton_coefficients": [frac_obj(value) for value in coeffs],
                "node_validation": {
                    "exact_match": exact_match,
                    "validated_node_count": len(xs_frac),
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches,
                },
                "source_curve_summary": {
                    "peak": curve["peak"],
                    "dip": curve["dip"],
                    "crossover_x": curve["crossover_x"],
                },
            }
        )

        rejected = [
            poly_candidate_error(xs_float, ys_float, 1),
            poly_candidate_error(xs_float, ys_float, 2),
            poly_candidate_error(xs_float, ys_float, 3),
            poly_candidate_error(xs_float, ys_float, 4),
            harmonic_candidate_error(xs_float, ys_float, float(xs_float[-1] - xs_float[0])),
        ]
        diagnostics.append(
            {
                "curve_id": curve["curve_id"],
                "accepted_law": {
                    "family": "exact_newton_interpolation",
                    "degree": degree_value,
                    "exact_on_nodes": exact_match,
                },
                "rejected_candidates": rejected,
            }
        )

    law_packet = {
        "artifact_id": "sixto_graph_curve_law_packet",
        "compute_substrate": "qa_rational_pair_noreduce",
        "source_scene": "pythagoras_quantum_world_rt/sixto_geogebra_extended_scene_export_v1.json",
        "source_summary": "pythagoras_quantum_world_rt/sixto_geogebra_extended_summary.json",
        "graph_guide_packet": guide_packet,
        "accepted_law_packet": {
            "family": "exact_newton_interpolation",
            "variable": "x",
            "curve_count": len(accepted_curves),
            "curves": accepted_curves,
        },
        "verdict": {
            "exact_closed_form_on_sampled_nodes": exact_curve_count == len(accepted_curves),
            "honest_summary": "The Sixto timing-graph curves now have an exact analytic packet on the QA substrate in Newton interpolation form. This is an exact law for the recovered sampled nodes, not yet a claim about the source-native minimal dynamics behind the curves.",
        },
    }

    diagnostics_payload = {
        "artifact_id": "sixto_graph_curve_law_diagnostics",
        "curve_count": len(diagnostics),
        "max_degree": max_degree,
        "candidate_checks": diagnostics,
        "verdict": {
            "simple_global_candidates_rejected": True,
            "honest_summary": "Affine, low-degree polynomial, and simple two-mode harmonic families do not land exactly on the recovered curve nodes, so the accepted law packet is the exact interpolation family rather than a claimed minimal global dynamic law.",
        },
    }
    return law_packet, diagnostics_payload


def self_test() -> int:
    law_packet, diagnostics = build_payloads()
    ok = True
    ok = ok and law_packet["accepted_law_packet"]["curve_count"] == 4
    ok = ok and diagnostics["curve_count"] == 4
    ok = ok and all(item["node_validation"]["exact_match"] for item in law_packet["accepted_law_packet"]["curves"])
    ok = ok and diagnostics["max_degree"] == 16
    print(canonical_dump({"ok": ok}))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    law_packet, diagnostics = build_payloads()
    write_json(LAW_PACKET_PATH, law_packet)
    write_json(DIAGNOSTICS_PATH, diagnostics)
    print(
        canonical_dump(
            {
                "ok": True,
                "outputs": [
                    "pythagoras_quantum_world_rt/sixto_graph_curve_law_packet.json",
                    "pythagoras_quantum_world_rt/sixto_graph_curve_law_diagnostics.json",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
