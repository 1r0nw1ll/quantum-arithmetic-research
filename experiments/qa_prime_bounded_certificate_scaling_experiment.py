#!/usr/bin/env python3
"""
Empirical scaling law for bounded factor-certificate witnesses.

Tests whether the minimal passing prime_max for exact prime/composite separation
on [2, N] matches the largest prime <= sqrt(N).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from qa_lab.qa_core import canonical_json, domain_sha256, is_prime
from qa_prime_bounded_certificate_sweep import run_sweep


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
    return [n for n in range(2, limit + 1) if is_prime(n)]


def _largest_prime_leq(limit: int) -> int | None:
    primes = _primes_up_to(limit)
    return None if not primes else primes[-1]


def run_experiment(endpoints: list[int]) -> dict[str, object]:
    rows = []
    all_match = True

    for end in endpoints:
        sqrt_floor = math.isqrt(end)
        candidate_caps = _primes_up_to(max(2, sqrt_floor))
        sweep = run_sweep(2, end, candidate_caps)
        observed = sweep["minimal_pass_prime_max"]
        predicted = _largest_prime_leq(sqrt_floor)
        matches = observed == predicted
        all_match = all_match and matches
        rows.append(
            {
                "end": end,
                "sqrt_floor": sqrt_floor,
                "candidate_caps": candidate_caps,
                "observed_minimal_pass_prime_max": observed,
                "predicted_largest_prime_leq_sqrt_end": predicted,
                "matches_prediction": matches,
            }
        )

    payload = {
        "experiment_id": f"qa_prime_bounded_certificate_scaling_{endpoints[0]}_{endpoints[-1]}",
        "hypothesis": (
            "For interval [2, N], the minimal bounded witness cap needed for exact prime/composite separation "
            "in the factor-certificate graph is the largest prime <= sqrt(N)."
        ),
        "success_criteria": (
            "PASS if the observed minimal passing prime_max matches the largest prime <= sqrt(N) for every tested endpoint."
        ),
        "tested_endpoints": endpoints,
        "result": "PASS" if all_match else "FAIL",
        "rows": rows,
        "honest_interpretation": (
            "This is an empirical scaling check over tested endpoints, not a general proof. "
            "A PASS means the observed bounded-witness threshold follows the expected smallest-factor barrier "
            "on all tested intervals."
        ),
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_PRIME_BOUNDED_CERTIFICATE_SCALING.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the scaling law for bounded factor-certificate witnesses.")
    parser.add_argument("--endpoints", default="100,250,500,1000")
    parser.add_argument(
        "--out",
        default="results/qa_prime_bounded_certificate_scaling_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    payload = run_experiment(_parse_endpoints(args.endpoints))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"[qa_prime_bounded_certificate_scaling_experiment] Wrote {out_path}")
    print(f"[qa_prime_bounded_certificate_scaling_experiment] Overall result: {payload['result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
