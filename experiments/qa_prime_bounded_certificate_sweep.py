#!/usr/bin/env python3
"""
Sweep bounded witness sets for the factor-certificate graph.

The goal is to find the smallest prime_max that still preserves exact
prime/composite/semiprime separation on the tested interval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from qa_lab.qa_core import canonical_json, domain_sha256
from qa_prime_certificate_graph_experiment import run_experiment


def _parse_prime_caps(raw: str) -> list[int]:
    caps = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 2:
            raise ValueError("prime caps must be >= 2")
        if value not in caps:
            caps.append(value)
    if not caps:
        raise ValueError("At least one prime cap is required.")
    return caps


def run_sweep(start: int, end: int, prime_caps: list[int]) -> dict[str, object]:
    runs = []
    for prime_cap in prime_caps:
        payload = run_experiment(start, end, prime_max=prime_cap)
        runs.append(
            {
                "prime_max": prime_cap,
                "result": payload["result"],
                "summary": payload["summary"],
            }
        )

    passing = [run for run in runs if run["result"] == "PASS"]
    minimal_pass = None if not passing else min(passing, key=lambda run: run["prime_max"])

    payload = {
        "experiment_id": f"qa_prime_bounded_certificate_sweep_{start}_{end}",
        "hypothesis": (
            "Exact prime/composite separation in the factor-certificate graph should survive under a bounded "
            "witness set if the bound still includes enough small prime divisibility witnesses to resolve every "
            "composite in the tested interval."
        ),
        "success_criteria": (
            "Report the smallest prime_max that still yields PASS under the exact certificate-graph checks. "
            "If no tested bound yields PASS, the sweep fails."
        ),
        "interval": {"start": start, "end": end},
        "prime_caps_tested": prime_caps,
        "result": "PASS" if minimal_pass is not None else "FAIL",
        "minimal_pass_prime_max": None if minimal_pass is None else minimal_pass["prime_max"],
        "runs": runs,
        "honest_interpretation": (
            "This sweep measures witness-budget sufficiency, not a free-standing prime theorem. "
            "When a bounded prime_max passes, it means small-prime certificate generators are sufficient "
            "to resolve all composites in the interval."
        ),
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_PRIME_BOUNDED_CERTIFICATE_SWEEP.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep bounded witness sets for the factor-certificate graph.")
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--prime-caps", default="2,3,5,7,11,13,17,19")
    parser.add_argument(
        "--out",
        default="results/qa_prime_bounded_certificate_sweep.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    payload = run_sweep(int(args.start), int(args.end), _parse_prime_caps(args.prime_caps))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"[qa_prime_bounded_certificate_sweep] Wrote {out_path}")
    print(f"[qa_prime_bounded_certificate_sweep] Overall result: {payload['result']}")
    print(f"[qa_prime_bounded_certificate_sweep] Minimal PASS prime_max: {payload['minimal_pass_prime_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
