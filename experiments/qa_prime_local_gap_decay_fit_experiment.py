#!/usr/bin/env python3
"""
Fit decay models for the radius-2 local certificate gap.

The radius-2 gap is:
  exact_composite_coverage_at_radius_2 - exact_prime_coverage_at_radius_2

This experiment compares:
  1. the observed gap from local certificate decisions
  2. an exact structural formula derived from counting
  3. simple generic decay fits
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_lab.qa_core import canonical_json, domain_sha256, is_composite, is_prime, local_certificate_neighborhood_decision


def _parse_endpoints(raw: str) -> list[int]:
    endpoints = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 7:
            raise ValueError("Endpoints must be >= 7.")
        if value not in endpoints:
            endpoints.append(value)
    if not endpoints:
        raise ValueError("At least one endpoint is required.")
    return endpoints


def _pi(n: int) -> int:
    return sum(1 for value in range(2, n + 1) if is_prime(value))


def _observed_radius2_gap(end: int) -> dict[str, float | int]:
    integers = list(range(2, end + 1))
    prime_count = sum(1 for n in integers if is_prime(n))
    composite_count = sum(1 for n in integers if is_composite(n))

    exact_prime = sum(1 for n in integers if is_prime(n) and local_certificate_neighborhood_decision(n, 2) == "PRIME")
    exact_composite = sum(1 for n in integers if is_composite(n) and local_certificate_neighborhood_decision(n, 2) == "COMPOSITE")

    exact_prime_fraction = exact_prime / prime_count
    exact_composite_fraction = exact_composite / composite_count
    return {
        "prime_count": prime_count,
        "composite_count": composite_count,
        "exact_prime_fraction": exact_prime_fraction,
        "exact_composite_fraction": exact_composite_fraction,
        "gap": exact_composite_fraction - exact_prime_fraction,
    }


def _structural_radius2_gap(end: int) -> dict[str, float | int]:
    pi_end = _pi(end)
    pi_7 = _pi(min(end, 7))
    composite_count = (end - 1) - pi_end
    even_composite_count = (end // 2) - 1
    prime_fraction = pi_7 / pi_end
    composite_fraction = even_composite_count / composite_count
    return {
        "pi_end": pi_end,
        "pi_7": pi_7,
        "composite_count": composite_count,
        "even_composite_count": even_composite_count,
        "exact_prime_fraction": prime_fraction,
        "exact_composite_fraction": composite_fraction,
        "gap": composite_fraction - prime_fraction,
    }


def _fit_no_intercept(xs: list[float], ys: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(xs, ys))
    denominator = sum(x * x for x in xs)
    return 0.0 if denominator == 0.0 else numerator / denominator


def _fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    ss_xx = sum((x - mean_x) * (x - mean_x) for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = 0.0 if ss_xx == 0.0 else ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _rmse(observed: list[float], predicted: list[float]) -> float:
    mse = sum((obs - pred) * (obs - pred) for obs, pred in zip(observed, predicted)) / len(observed)
    return math.sqrt(mse)


def run_experiment(endpoints: list[int]) -> dict[str, object]:
    rows = []
    observed_gaps = []
    inv_logs = []
    inv_sqroots = []

    for end in endpoints:
        observed = _observed_radius2_gap(end)
        structural = _structural_radius2_gap(end)
        rows.append(
            {
                "end": end,
                "observed_gap": round(float(observed["gap"]), 12),
                "structural_gap": round(float(structural["gap"]), 12),
                "structural_abs_error": abs(float(observed["gap"]) - float(structural["gap"])),
                "observed_exact_prime_fraction": round(float(observed["exact_prime_fraction"]), 12),
                "observed_exact_composite_fraction": round(float(observed["exact_composite_fraction"]), 12),
                "structural_terms": {
                    "pi_end": int(structural["pi_end"]),
                    "pi_7": int(structural["pi_7"]),
                    "even_composite_count": int(structural["even_composite_count"]),
                    "composite_count": int(structural["composite_count"]),
                },
            }
        )
        observed_gaps.append(float(observed["gap"]))
        inv_logs.append(1.0 / math.log(end))
        inv_sqroots.append(1.0 / math.sqrt(end))

    excess = [gap - 0.5 for gap in observed_gaps]
    c_log = _fit_no_intercept(inv_logs, excess)
    c_sqrt = _fit_no_intercept(inv_sqroots, excess)
    a_log, b_log = _fit_affine(inv_logs, observed_gaps)

    fit_half_plus_c_over_log = [0.5 + c_log * x for x in inv_logs]
    fit_half_plus_c_over_sqrt = [0.5 + c_sqrt * x for x in inv_sqroots]
    fit_affine_inv_log = [a_log + b_log * x for x in inv_logs]
    structural_pred = [row["structural_gap"] for row in rows]

    fits = {
        "structural_exact_formula": {
            "rmse": _rmse(observed_gaps, structural_pred),
            "max_abs_error": max(abs(obs - pred) for obs, pred in zip(observed_gaps, structural_pred)),
        },
        "half_plus_c_over_log_n": {
            "c": c_log,
            "rmse": _rmse(observed_gaps, fit_half_plus_c_over_log),
            "max_abs_error": max(abs(obs - pred) for obs, pred in zip(observed_gaps, fit_half_plus_c_over_log)),
        },
        "half_plus_c_over_sqrt_n": {
            "c": c_sqrt,
            "rmse": _rmse(observed_gaps, fit_half_plus_c_over_sqrt),
            "max_abs_error": max(abs(obs - pred) for obs, pred in zip(observed_gaps, fit_half_plus_c_over_sqrt)),
        },
        "affine_in_inv_log_n": {
            "intercept": a_log,
            "slope": b_log,
            "rmse": _rmse(observed_gaps, fit_affine_inv_log),
            "max_abs_error": max(abs(obs - pred) for obs, pred in zip(observed_gaps, fit_affine_inv_log)),
        },
    }

    best_generic = min(
        ("half_plus_c_over_log_n", "half_plus_c_over_sqrt_n", "affine_in_inv_log_n"),
        key=lambda label: fits[label]["rmse"],
    )
    structural_exact = all(row["structural_abs_error"] < 1e-12 for row in rows)
    result = "PASS" if structural_exact and fits["half_plus_c_over_log_n"]["rmse"] < fits["half_plus_c_over_sqrt_n"]["rmse"] else "FAIL"

    payload = {
        "experiment_id": f"qa_prime_local_gap_decay_fit_{endpoints[0]}_{endpoints[-1]}",
        "hypothesis": (
            "The radius-2 local certificate gap should be better explained by a structural counting law and, among "
            "simple generic decay families, by inverse-log decay rather than inverse-sqrt decay."
        ),
        "success_criteria": (
            "PASS if the structural radius-2 formula matches the observed gap on every tested endpoint and "
            "half_plus_c_over_log_n has lower RMSE than half_plus_c_over_sqrt_n."
        ),
        "tested_endpoints": endpoints,
        "result": result,
        "rows": rows,
        "fits": fits,
        "best_generic_fit": best_generic,
        "honest_interpretation": (
            "The strongest result here is structural, not merely statistical: the radius-2 gap is exactly determined "
            "by counting even composites and the small set of primes certifiable with horizon <= 2. The inverse-log fit "
            "is useful as an asymptotic summary, but it is weaker than the exact finite-N identity."
        ),
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_PRIME_LOCAL_GAP_DECAY_FIT_EXPERIMENT.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit decay models for the radius-2 local certificate gap.")
    parser.add_argument("--endpoints", default="100,250,500,1000,2000,5000,10000,20000,50000")
    parser.add_argument(
        "--out",
        default="results/qa_prime_local_gap_decay_fit_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    payload = run_experiment(_parse_endpoints(args.endpoints))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"[qa_prime_local_gap_decay_fit_experiment] Wrote {out_path}")
    print(f"[qa_prime_local_gap_decay_fit_experiment] Overall result: {payload['result']}")
    print(f"[qa_prime_local_gap_decay_fit_experiment] Best generic fit: {payload['best_generic_fit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
