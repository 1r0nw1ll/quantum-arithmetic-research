#!/usr/bin/env python3
"""
Measure how much prime/composite classification is visible in local
factor-certificate neighborhoods.

A bounded neighborhood of radius prime_max can certify compositeness as soon as
a witness prime is visible. Primehood is harder: it becomes exact only once the
no-witness horizon reaches the largest prime <= sqrt(n).
"""

from __future__ import annotations

import argparse
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


def _primes_up_to(limit: int) -> list[int]:
    return [value for value in range(2, limit + 1) if is_prime(value)]


def _class_label(n: int) -> str:
    if is_prime(n):
        return "prime"
    if is_semiprime(n):
        return "semiprime"
    if is_composite(n):
        return "composite"
    return "other"


def _decision_is_exact(n: int, decision: str | None) -> bool:
    if decision == "PRIME":
        return is_prime(n)
    if decision == "COMPOSITE":
        return is_composite(n)
    return False


def _fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _median(values: list[int]) -> float | int | None:
    return None if not values else statistics.median(values)


def run_experiment(start: int, end: int) -> dict[str, object]:
    if start < 2:
        raise ValueError("start must be >= 2")
    if end < start:
        raise ValueError("end must be >= start")

    integers = list(range(start, end + 1))
    rows = [
        {
            "n": n,
            "class_label": _class_label(n),
            "local_exact_radius": local_certificate_exact_radius(n),
        }
        for n in integers
    ]
    full_horizon = 0 if end < 4 else int(largest_prime_leq(int(end ** 0.5)) or 0)
    candidate_radii = [0] + _primes_up_to(max(2, full_horizon))

    coverage_by_radius = []
    for radius in candidate_radii:
        decisions = {n: local_certificate_neighborhood_decision(n, radius) for n in integers}
        exact_overall = sum(1 for n in integers if _decision_is_exact(n, decisions[n]))
        exact_prime = sum(1 for n in integers if is_prime(n) and decisions[n] == "PRIME")
        exact_composite = sum(1 for n in integers if is_composite(n) and decisions[n] == "COMPOSITE")
        exact_semiprime = sum(1 for n in integers if is_semiprime(n) and decisions[n] == "COMPOSITE")
        undecided = sum(1 for n in integers if decisions[n] == "UNDECIDED")
        prime_count = sum(1 for n in integers if is_prime(n))
        composite_count = sum(1 for n in integers if is_composite(n))
        semiprime_count = sum(1 for n in integers if is_semiprime(n))
        coverage_by_radius.append(
            {
                "prime_max": radius,
                "exact_fraction_overall": _fraction(exact_overall, len(integers)),
                "exact_fraction_prime": _fraction(exact_prime, prime_count),
                "exact_fraction_composite": _fraction(exact_composite, composite_count),
                "exact_fraction_semiprime": _fraction(exact_semiprime, semiprime_count),
                "undecided_fraction_overall": _fraction(undecided, len(integers)),
            }
        )

    prime_radii = [int(row["local_exact_radius"]) for row in rows if row["class_label"] == "prime"]
    semiprime_radii = [int(row["local_exact_radius"]) for row in rows if row["class_label"] == "semiprime"]
    composite_radii = [int(row["local_exact_radius"]) for row in rows if row["class_label"] in {"semiprime", "composite"}]

    full_row = coverage_by_radius[-1]
    local_advantage_row = next(
        (
            row for row in coverage_by_radius[:-1]
            if row["exact_fraction_composite"] > row["exact_fraction_prime"]
        ),
        None,
    )
    result = (
        "PASS"
        if full_row["exact_fraction_overall"] == 1.0 and local_advantage_row is not None
        else "FAIL"
    )

    payload = {
        "experiment_id": f"qa_prime_local_certificate_neighborhood_{start}_{end}",
        "hypothesis": (
            "Compositehood should often be visible in small local certificate neighborhoods, while primehood should "
            "require the full no-witness horizon up to the largest prime <= sqrt(n)."
        ),
        "success_criteria": (
            "PASS if the full tested horizon yields exact classification on the whole interval and at least one "
            "strictly smaller neighborhood radius yields higher exact coverage for composites than for primes."
        ),
        "interval": {"start": start, "end": end},
        "candidate_radii": candidate_radii,
        "result": result,
        "summary": {
            "full_horizon_prime_max": full_horizon,
            "full_horizon_exact_fraction_overall": full_row["exact_fraction_overall"],
            "full_horizon_exact_fraction_prime": full_row["exact_fraction_prime"],
            "full_horizon_exact_fraction_composite": full_row["exact_fraction_composite"],
            "median_local_exact_radius": {
                "prime": _median(prime_radii),
                "semiprime": _median(semiprime_radii),
                "composite": _median(composite_radii),
            },
            "first_radius_with_composite_advantage": None if local_advantage_row is None else local_advantage_row["prime_max"],
            "composite_advantage_gap_at_first_radius": None if local_advantage_row is None else round(
                local_advantage_row["exact_fraction_composite"] - local_advantage_row["exact_fraction_prime"], 6
            ),
        },
        "coverage_by_radius": coverage_by_radius,
        "examples": {
            "small_local_composite_witnesses": [
                row["n"] for row in rows
                if row["class_label"] in {"semiprime", "composite"} and int(row["local_exact_radius"]) <= 5
            ][:15],
            "primes_requiring_full_horizon_examples": [
                row["n"] for row in rows
                if row["class_label"] == "prime" and int(row["local_exact_radius"]) == full_horizon
            ][:15],
        },
        "honest_interpretation": (
            "This is not a new prime theorem. It quantifies the asymmetry of the certificate graph: composites are "
            "often decided locally by a nearby witness, while primes require the absence of all witness primes up to "
            "their no-witness horizon."
        ),
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_PRIME_LOCAL_CERTIFICATE_NEIGHBORHOOD_EXPERIMENT.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure local certificate-neighborhood sufficiency for primes/composites.")
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument(
        "--out",
        default="results/qa_prime_local_certificate_neighborhood_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    payload = run_experiment(int(args.start), int(args.end))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"[qa_prime_local_certificate_neighborhood_experiment] Wrote {out_path}")
    print(f"[qa_prime_local_certificate_neighborhood_experiment] Overall result: {payload['result']}")
    print(
        "[qa_prime_local_certificate_neighborhood_experiment] "
        f"first composite advantage radius={payload['summary']['first_radius_with_composite_advantage']} "
        f"full_horizon={payload['summary']['full_horizon_prime_max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
