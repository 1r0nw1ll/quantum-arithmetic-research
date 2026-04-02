#!/usr/bin/env python3
"""
Build an exact divisibility-certificate graph over an integer interval.

Examples
--------
python tools/qa_factor_certificate_graph.py --start 2 --end 100
python tools/qa_factor_certificate_graph.py --start 2 --end 500 --focus 221 --pretty
"""

from __future__ import annotations

import argparse
import json
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
    factor_integer,
    is_composite,
    is_prime,
    is_prime_power,
    is_semiprime,
    omega,
    big_omega,
)


SCHEMA_PATH = "schemas/qa_number_graph.schema.json"


def _node_record(n: int, prime_max: int | None) -> dict[str, object]:
    return {
        "n": n,
        "is_prime": is_prime(n),
        "is_composite": is_composite(n),
        "is_semiprime": is_semiprime(n),
        "is_prime_power": is_prime_power(n),
        "omega": omega(n),
        "Omega": big_omega(n),
        "decomposition_depth_to_prime_terminal": decomposition_depth_to_prime_terminal(n, prime_max=prime_max),
        "factorization": [{"p": prime, "e": exponent} for prime, exponent in factor_integer(n)],
    }


def build_graph(start: int, end: int, focus: int | None, prime_max: int | None) -> dict[str, object]:
    if start < 2:
        raise ValueError("start must be >= 2")
    if end < start:
        raise ValueError("end must be >= start")

    nodes = [_node_record(n, prime_max=prime_max) for n in range(start, end + 1)]
    node_set = {node["n"] for node in nodes}
    edges = []
    for n in range(start, end + 1):
        for edge in factor_certificate_edges(n, prime_max=prime_max):
            if edge["target"] in node_set:
                edges.append(edge)

    payload = {
        "schema": SCHEMA_PATH,
        "graph_kind": "factor_certificate",
        "range": {"start": start, "end": end},
        "prime_max": prime_max,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "prime_count": sum(1 for node in nodes if node["is_prime"]),
            "composite_count": sum(1 for node in nodes if node["is_composite"]),
            "semiprime_count": sum(1 for node in nodes if node["is_semiprime"]),
            "prime_power_count": sum(1 for node in nodes if node["is_prime_power"]),
        },
        "nodes": nodes,
        "edges": edges,
    }

    if focus is not None:
        payload["focus"] = {
            "n": focus,
            "smallest_prime_chain": factor_chain_to_prime(focus, strategy="smallest", prime_max=prime_max),
            "largest_prime_chain": factor_chain_to_prime(focus, strategy="largest", prime_max=prime_max),
            "outgoing_certificate_edges": factor_certificate_edges(focus, prime_max=prime_max),
            "decomposition_depth_to_prime_terminal": decomposition_depth_to_prime_terminal(focus, prime_max=prime_max),
        }

    payload["canonical_hash"] = domain_sha256(
        "QA_FACTOR_CERTIFICATE_GRAPH.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an exact divisibility-certificate graph over an integer interval.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--focus", type=int)
    parser.add_argument("--prime-max", type=int)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = build_graph(int(args.start), int(args.end), args.focus, args.prime_max)
    if args.pretty:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
