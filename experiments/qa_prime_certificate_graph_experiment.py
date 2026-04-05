#!/usr/bin/env python3
"""
Test whether an exact factor-certificate graph separates primes from composites.

This is intentionally stronger than the earlier residue/path-profile experiment:
the graph now carries explicit divisibility witnesses, so any exact separation
must come from certificate structure rather than QA residue aliasing alone.
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
    decomposition_depth_to_prime_terminal,
    domain_sha256,
    factor_certificate_edges,
    factor_chain_to_prime,
    is_composite,
    is_prime,
    is_prime_power,
    is_semiprime,
)


def _row(n: int, prime_max: int | None) -> dict[str, object]:
    return {
        "n": n,
        "is_prime": is_prime(n),
        "is_composite": is_composite(n),
        "is_semiprime": is_semiprime(n),
        "is_prime_power": is_prime_power(n),
        "outgoing_certificate_edge_count": len(factor_certificate_edges(n, prime_max=prime_max)),
        "decomposition_depth_to_prime_terminal": decomposition_depth_to_prime_terminal(n, prime_max=prime_max),
        "smallest_prime_chain": factor_chain_to_prime(n, strategy="smallest", prime_max=prime_max),
        "largest_prime_chain": factor_chain_to_prime(n, strategy="largest", prime_max=prime_max),
    }


def run_experiment(start: int, end: int, prime_max: int | None = None) -> dict[str, object]:
    if start < 2:
        raise ValueError("start must be >= 2")
    if end < start:
        raise ValueError("end must be >= start")

    rows = [_row(n, prime_max=prime_max) for n in range(start, end + 1)]
    prime_rows = [row for row in rows if row["is_prime"]]
    semiprime_rows = [row for row in rows if row["is_semiprime"]]
    composite_rows = [row for row in rows if row["is_composite"]]

    prime_terminal_exact = all(
        row["decomposition_depth_to_prime_terminal"] == 0 and row["outgoing_certificate_edge_count"] == 0
        for row in prime_rows
    )
    composite_positive_depth_exact = all(
        row["decomposition_depth_to_prime_terminal"] is not None
        and row["decomposition_depth_to_prime_terminal"] >= 1
        and row["outgoing_certificate_edge_count"] >= 1
        for row in composite_rows
    )
    semiprime_depth_exact = all(
        row["decomposition_depth_to_prime_terminal"] == 1
        for row in semiprime_rows
    )

    def _median_depth(label_rows: list[dict[str, object]]) -> float | int | None:
        values = [
            row["decomposition_depth_to_prime_terminal"]
            for row in label_rows
            if row["decomposition_depth_to_prime_terminal"] is not None
        ]
        return None if not values else statistics.median(values)

    if prime_terminal_exact and composite_positive_depth_exact and semiprime_depth_exact:
        verdict = "PASS"
    elif prime_terminal_exact and composite_positive_depth_exact:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    payload = {
        "experiment_id": f"qa_prime_certificate_graph_experiment_{start}_{end}",
        "hypothesis": (
            "If explicit divisibility witnesses are added as lawful graph generators, then primes should become "
            "exactly the zero-depth terminals of the factor-certificate graph, while composites admit positive-depth "
            "certificate paths and semiprimes occupy depth 1."
        ),
        "success_criteria": (
            "PASS if every prime is a zero-depth terminal with no outgoing certificate edges, every composite has "
            "positive certificate depth with at least one outgoing witness edge, and every semiprime has depth 1."
        ),
        "interval": {"start": start, "end": end},
        "prime_max": prime_max,
        "result": verdict,
        "summary": {
            "prime_count": len(prime_rows),
            "composite_count": len(composite_rows),
            "semiprime_count": len(semiprime_rows),
            "prime_terminal_exact": prime_terminal_exact,
            "composite_positive_depth_exact": composite_positive_depth_exact,
            "semiprime_depth_exact": semiprime_depth_exact,
            "median_decomposition_depth": {
                "prime": _median_depth(prime_rows),
                "semiprime": _median_depth(semiprime_rows),
                "composite": _median_depth(composite_rows),
            },
        },
        "examples": {
            "prime_terminals": [row["n"] for row in prime_rows[:10]],
            "semiprime_depth_one": [row["n"] for row in semiprime_rows[:10]],
            "prime_power_examples": [row["n"] for row in rows if row["is_prime_power"]][:10],
            "composite_examples": [row["n"] for row in composite_rows[:10]],
        },
        "honest_interpretation": (
            "This is an exact certificate-graph result, not a new QA prime theorem. The separation is obtained by "
            "adding divisibility witnesses to the graph, so the graph becomes a primality-certificate structure rather "
            "than a residue-only QA reachability profile."
        ),
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_PRIME_CERTIFICATE_GRAPH_EXPERIMENT.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the exact factor-certificate graph experiment for primes.")
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--prime-max", type=int)
    parser.add_argument(
        "--out",
        default="results/qa_prime_certificate_graph_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    payload = run_experiment(int(args.start), int(args.end), prime_max=args.prime_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"[qa_prime_certificate_graph_experiment] Wrote {out_path}")
    print(f"[qa_prime_certificate_graph_experiment] Overall result: {payload['result']}")
    print(
        "[qa_prime_certificate_graph_experiment] Exact checks: "
        f"prime_terminal={payload['summary']['prime_terminal_exact']} "
        f"composite_positive_depth={payload['summary']['composite_positive_depth_exact']} "
        f"semiprime_depth_one={payload['summary']['semiprime_depth_exact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
