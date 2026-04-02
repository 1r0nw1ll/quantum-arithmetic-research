#!/usr/bin/env python3
"""
Breadth-first QA reachability verifier with shortest witnesses and prime targets.

Examples
--------
python bfs_verify.py --source 1 1 --target 1 2 --modulus 24
python bfs_verify.py --source 1 1 --prime-target norm --limit 5 --modulus 24
python bfs_verify.py --source 1 1 --prime-target norm --emit-graph /tmp/qa_graph.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

from qa_lab.qa_core import (
    canonical_json,
    nearest_semiprime_norm_targets,
    obstructed_semiprime_residues,
    nearest_prime_norm_targets,
    semiprime_residues,
    obstructed_prime_residues,
    prime_residues,
    is_semiprime,
    qa_norm_mod,
    reachable_subgraph,
    shortest_witness,
    structural_obstruction,
)

SCHEMA_PATH = "schemas/qa_number_graph.schema.json"


def _parse_generators(raw: str) -> Tuple[str, ...]:
    parts = tuple(piece.strip().upper() for piece in raw.split(",") if piece.strip())
    if not parts:
        raise argparse.ArgumentTypeError("Generator list must not be empty.")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify QA graph reachability and shortest paths."
    )
    parser.add_argument("--source", nargs=2, type=int, metavar=("B", "E"), required=True)
    parser.add_argument("--target", nargs=2, type=int, metavar=("B", "E"))
    parser.add_argument(
        "--prime-target",
        choices=["norm"],
        help="Find shortest witnesses to reachable prime QA norm residues.",
    )
    parser.add_argument(
        "--semiprime-target",
        choices=["norm"],
        help="Find shortest witnesses to reachable semiprime QA norm residues.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Max prime targets to return.")
    parser.add_argument("--modulus", type=int, default=24)
    parser.add_argument(
        "--generators",
        type=_parse_generators,
        default=("Q",),
        help="Comma-separated generator set. Allowed: Q,T",
    )
    parser.add_argument(
        "--emit-graph",
        help="Write the reachable subgraph rooted at --source as JSON.",
    )
    return parser


def _target_query(
    source: Sequence[int],
    target: Sequence[int],
    modulus: int,
    generators: Sequence[str],
) -> dict:
    witness = shortest_witness(
        source[0],
        source[1],
        target[0],
        target[1],
        modulus,
        generators=generators,
    )
    target_norm = qa_norm_mod(target[0], target[1], modulus)
    return {
        "source": list(source),
        "target": list(target),
        "reachable": witness is not None,
        "shortest_steps": None if witness is None else witness["steps"],
        "generator_trace": [] if witness is None else witness["generator_trace"],
        "witness_states": [] if witness is None else [list(state) for state in witness["witness_states"]],
        "target_norm_mod": target_norm,
        "target_norm_is_prime": target_norm in prime_residues(modulus),
        "target_norm_is_semiprime": is_semiprime(target_norm),
        "target_norm_obstructed": structural_obstruction(target_norm, modulus),
    }


def _prime_query(
    source: Sequence[int],
    modulus: int,
    generators: Sequence[str],
    limit: int,
) -> dict:
    witnesses = nearest_prime_norm_targets(
        source[0],
        source[1],
        modulus,
        generators=generators,
        limit=limit,
    )
    return {
        "mode": "norm",
        "source": list(source),
        "candidate_prime_residues": prime_residues(modulus),
        "obstructed_prime_residues": obstructed_prime_residues(modulus),
        "reachable_prime_targets": [
            {
                "prime": item["prime"],
                "target": list(item["target"]),
                "steps": item["steps"],
                "generator_trace": item["generator_trace"],
                "witness_states": [list(state) for state in item["witness_states"]],
            }
            for item in witnesses
        ],
    }


def _semiprime_query(
    source: Sequence[int],
    modulus: int,
    generators: Sequence[str],
    limit: int,
) -> dict:
    witnesses = nearest_semiprime_norm_targets(
        source[0],
        source[1],
        modulus,
        generators=generators,
        limit=limit,
    )
    return {
        "mode": "norm",
        "source": list(source),
        "candidate_semiprime_residues": semiprime_residues(modulus),
        "obstructed_semiprime_residues": obstructed_semiprime_residues(modulus),
        "reachable_semiprime_targets": [
            {
                "semiprime": item["semiprime"],
                "target": list(item["target"]),
                "steps": item["steps"],
                "generator_trace": item["generator_trace"],
                "witness_states": [list(state) for state in item["witness_states"]],
            }
            for item in witnesses
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    source = tuple(args.source)
    modulus = int(args.modulus)
    generators = tuple(args.generators)

    payload = {
        "schema": SCHEMA_PATH,
        "control_basis": {
            "generator_relative_reachability": list(generators),
            "qa_time": "path_length_in_generator_steps",
            "obstruction_rule": "v_p(r)=1 for an inert prime makes arithmetic class r unreachable",
        },
        "modulus": modulus,
        "source": list(source),
    }

    if args.target:
        payload["target_query"] = _target_query(
            source=source,
            target=tuple(args.target),
            modulus=modulus,
            generators=generators,
        )

    if args.prime_target == "norm":
        payload["prime_query"] = _prime_query(
            source=source,
            modulus=modulus,
            generators=generators,
            limit=int(args.limit),
        )

    if args.semiprime_target == "norm":
        payload["semiprime_query"] = _semiprime_query(
            source=source,
            modulus=modulus,
            generators=generators,
            limit=int(args.limit),
        )

    if args.emit_graph:
        graph_payload = reachable_subgraph(
            source[0],
            source[1],
            modulus,
            generators=generators,
        )
        graph_path = Path(args.emit_graph)
        graph_path.write_text(canonical_json(graph_payload) + "\n", encoding="utf-8")
        payload["graph_path"] = str(graph_path)

    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
