#!/usr/bin/env python3
"""
qa_graph_builder_rust.py

Torch-free QA graph builder using canonical generator edges.

Entry points:
- qa_theorem_discovery_orchestrator_rust.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GraphBuilderRust")


Edge = Tuple[int, int, str]
SCHEMA_VERSION = "qa_graph_edges@1"


def _build_tuple_index(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    return {(int(row.b), int(row.e)): int(idx) for idx, row in df.iterrows()}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _qa_lab_rs_meta() -> dict | None:
    rs_path = Path("qa_lab") / "qa_lab_rs.so"
    if not rs_path.exists():
        return None
    return {
        "path": str(rs_path),
        "sha256": _sha256_file(rs_path),
        "mtime": rs_path.stat().st_mtime,
        "size": rs_path.stat().st_size,
    }


def build_canonical_edges(df: pd.DataFrame) -> Tuple[List[Edge], Dict[str, int]]:
    tuple_to_idx = _build_tuple_index(df)
    edges: List[Edge] = []
    stats = {"sigma": 0, "mu": 0, "lambda2": 0, "nu": 0}

    for idx, row in df.iterrows():
        b = int(row["b"])
        e = int(row["e"])

        sigma_key = (b, e + 1)
        if sigma_key in tuple_to_idx:
            edges.append((int(idx), tuple_to_idx[sigma_key], "sigma"))
            stats["sigma"] += 1

        mu_key = (e, b)
        if mu_key in tuple_to_idx:
            edges.append((int(idx), tuple_to_idx[mu_key], "mu"))
            stats["mu"] += 1

        lambda2_key = (2 * b, 2 * e)
        if lambda2_key in tuple_to_idx:
            edges.append((int(idx), tuple_to_idx[lambda2_key], "lambda2"))
            stats["lambda2"] += 1

        if b % 2 == 0 and e % 2 == 0:
            nu_key = (b // 2, e // 2)
            if nu_key in tuple_to_idx:
                edges.append((int(idx), tuple_to_idx[nu_key], "nu"))
                stats["nu"] += 1

    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    return edges, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical QA edges without torch")
    parser.add_argument("--input", default="qa_10000_balanced_tuples.csv", help="Input CSV")
    parser.add_argument("--output", default="qa_graph_edges.csv", help="Output edge CSV")
    parser.add_argument("--stats", default="qa_graph_edge_stats.json", help="Stats JSON")
    parser.add_argument("--schema-version", default=SCHEMA_VERSION, help="Schema version tag")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)

    logger.info("Loading dataset from %s", input_path)
    df = pd.read_csv(input_path)

    required = {"b", "e", "d", "a"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Dataset contains NaN values in required columns")

    b = df["b"].to_numpy()
    e = df["e"].to_numpy()
    d = df["d"].to_numpy()
    a = df["a"].to_numpy()
    if not ((d == b + e).all() and (a == b + 2 * e).all()):
        raise ValueError("Dataset violates canonical closure (d=b+e, a=b+2e)")

    logger.info("Building canonical edges (sigma, mu, lambda2, nu)")
    edges, stats = build_canonical_edges(df)

    logger.info("Writing edges to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("src_id,dst_id,move,move_param,fail_type,invariant_diff,is_legal\n")
        for src, dst, etype in edges:
            move_param = "2" if etype == "lambda2" else ""
            f.write(f"{src},{dst},{etype},{move_param},,,1\n")

    dataset_sha = _sha256_file(input_path)
    git_commit = _git_commit()
    stats_payload = {
        "schema_version": args.schema_version,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_path": str(input_path),
        "dataset_sha256": dataset_sha,
        "dataset_rows": int(len(df)),
        "git_commit": git_commit,
        "qa_lab_rs": _qa_lab_rs_meta(),
        "num_nodes": int(len(df)),
        "num_edges": int(len(edges)),
        "edge_counts": stats,
        "edge_columns": [
            "src_id",
            "dst_id",
            "move",
            "move_param",
            "fail_type",
            "invariant_diff",
            "is_legal",
        ],
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    logger.info("Edges: %d", stats_payload["num_edges"])
    logger.info("  sigma: %d", stats["sigma"])
    logger.info("  mu: %d", stats["mu"])
    logger.info("  lambda2: %d", stats["lambda2"])
    logger.info("  nu: %d", stats["nu"])
    logger.info("Stats written to %s", stats_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
