#!/usr/bin/env python3
"""
Scale sweep for local factor-certificate neighborhood asymmetry.

Tests whether, across multiple interval endpoints [2, N], the first strict
local radius where composite exact coverage exceeds prime exact coverage stays
at radius 2 while the full horizon still yields exact classification.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_lab.qa_core import (
    canonical_json,
    domain_sha256,
    is_composite,
    is_prime,
    is_semiprime,
    largest_prime_leq,
    local_certificate_exact_radius,
    local_certificate_neighborhood_decision,
)


def _parse_endpoints(raw: str) -> list[int]:
    endpoints = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 4:
            raise ValueError("Endpoints must be >= 4.")
        if value not in endpoints:
            endpoints.append(value)
    if not endpoints:
        raise ValueError("At least one endpoint is required.")
    return endpoints


def _primes_up_to(limit: int) -> list[int]:
    return [value for value in range(2, limit + 1) if is_prime(value)]


def _fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _decision_is_exact(n: int, decision: str | None) -> bool:
    if decision == "PRIME":
        return is_prime(n)
    if decision == "COMPOSITE":
        return is_composite(n)
    return False


def _median(values: list[int]) -> float | int | None:
    return None if not values else statistics.median(values)


def _coverage_rows(end: int) -> tuple[int, list[dict[str, float | int]]]:
    integers = list(range(2, end + 1))
    full_horizon = 0 if end < 4 else int(largest_prime_leq(math.isqrt(end)) or 0)
    candidate_radii = [0] + _primes_up_to(max(2, full_horizon))

    prime_count = sum(1 for n in integers if is_prime(n))
    composite_count = sum(1 for n in integers if is_composite(n))
    semiprime_count = sum(1 for n in integers if is_semiprime(n))

    rows = []
    for radius in candidate_radii:
        decisions = {n: local_certificate_neighborhood_decision(n, radius) for n in integers}
        exact_overall = sum(1 for n in integers if _decision_is_exact(n, decisions[n]))
        exact_prime = sum(1 for n in integers if is_prime(n) and decisions[n] == "PRIME")
        exact_composite = sum(1 for n in integers if is_composite(n) and decisions[n] == "COMPOSITE")
        exact_semiprime = sum(1 for n in integers if is_semiprime(n) and decisions[n] == "COMPOSITE")
        rows.append(
            {
                "prime_max": radius,
                "exact_fraction_overall": _fraction(exact_overall, len(integers)),
                "exact_fraction_prime": _fraction(exact_prime, prime_count),
                "exact_fraction_composite": _fraction(exact_composite, composite_count),
                "exact_fraction_semiprime": _fraction(exact_semiprime, semiprime_count),
            }
        )
    return full_horizon, rows


def _row_for_endpoint(end: int) -> dict[str, object]:
    full_horizon, coverage = _coverage_rows(end)
    integers = list(range(2, end + 1))
    prime_radii = [int(local_certificate_exact_radius(n)) for n in integers if is_prime(n)]
    semiprime_radii = [int(local_certificate_exact_radius(n)) for n in integers if is_semiprime(n)]
    composite_radii = [int(local_certificate_exact_radius(n)) for n in integers if is_composite(n)]

    first_advantage = next(
        (
            row for row in coverage[:-1]
            if row["exact_fraction_composite"] > row["exact_fraction_prime"]
        ),
        None,
    )
    full_row = coverage[-1]

    return {
        "end": end,
        "full_horizon_prime_max": full_horizon,
        "full_horizon_exact_fraction_overall": full_row["exact_fraction_overall"],
        "full_horizon_exact_fraction_prime": full_row["exact_fraction_prime"],
        "full_horizon_exact_fraction_composite": full_row["exact_fraction_composite"],
        "first_radius_with_composite_advantage": None if first_advantage is None else first_advantage["prime_max"],
        "composite_advantage_gap_at_first_radius": None if first_advantage is None else round(
            first_advantage["exact_fraction_composite"] - first_advantage["exact_fraction_prime"], 6
        ),
        "coverage_at_first_radius": None if first_advantage is None else first_advantage,
        "median_local_exact_radius": {
            "prime": _median(prime_radii),
            "semiprime": _median(semiprime_radii),
            "composite": _median(composite_radii),
        },
    }


def run_experiment(endpoints: list[int]) -> dict[str, object]:
    rows = [_row_for_endpoint(end) for end in endpoints]
    all_pass = all(
        row["full_horizon_exact_fraction_overall"] == 1.0
        and row["first_radius_with_composite_advantage"] == 2
        and row["composite_advantage_gap_at_first_radius"] is not None
        and row["composite_advantage_gap_at_first_radius"] > 0
        for row in rows
    )

    payload = {
        "experiment_id": f"qa_prime_local_certificate_neighborhood_scaling_{endpoints[0]}_{endpoints[-1]}",
        "hypothesis": (
            "Across tested intervals [2,N], the locality asymmetry in the factor-certificate graph remains stable: "
            "the full horizon yields exact classification, and the first strict local radius where exact composite "
            "coverage exceeds exact prime coverage remains 2."
        ),
        "success_criteria": (
            "PASS if every tested endpoint has full-horizon exact overall coverage 1.0, "
            "first_radius_with_composite_advantage = 2, and composite_advantage_gap_at_first_radius > 0."
        ),
        "tested_endpoints": endpoints,
        "result": "PASS" if all_pass else "FAIL",
        "rows": rows,
        "honest_interpretation": (
            "This is an empirical scaling check for locality asymmetry, not a new prime theorem. "
            "A PASS means the composite-local advantage appeared immediately at radius 2 on every tested interval, "
            "while exact prime certification still required the full no-witness horizon."
        ),
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_PRIME_LOCAL_CERTIFICATE_NEIGHBORHOOD_SCALING_EXPERIMENT.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Scale sweep for local certificate-neighborhood asymmetry.")
    parser.add_argument("--endpoints", default="100,250,500,1000")
    parser.add_argument(
        "--out",
        default="results/qa_prime_local_certificate_neighborhood_scaling_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    payload = run_experiment(_parse_endpoints(args.endpoints))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"[qa_prime_local_certificate_neighborhood_scaling_experiment] Wrote {out_path}")
    print(f"[qa_prime_local_certificate_neighborhood_scaling_experiment] Overall result: {payload['result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
