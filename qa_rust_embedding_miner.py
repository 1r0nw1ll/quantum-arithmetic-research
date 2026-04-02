#!/usr/bin/env python3
"""
qa_rust_embedding_miner.py

Torch-free QA embedding + conjecture mining using the Rust backend (qa_lab_rs).

Entry points:
- qa_theorem_discovery_orchestrator_rust.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RustEmbeddingMiner")

SCHEMA_VERSION = "qa_conjectures@1"
CANONICAL_VALIDATOR = Path(
    "Formalizing tuple drift in quantum-native learning/files/files(1)/validate_canonical_v2.py"
)


def _ensure_rust_backend(qa_lab_root: Path) -> object:
    repo_root = Path(__file__).resolve().parent
    qa_lab_path = qa_lab_root
    if not qa_lab_path.is_absolute():
        qa_lab_path = repo_root / qa_lab_path
    if qa_lab_path.exists():
        sys.path.insert(0, str(qa_lab_path))
    try:
        import qa_lab_rs
    except Exception as exc:
        raise RuntimeError(
            "qa_lab_rs not importable. Build it with `make rust-py-build` in qa_lab."  # noqa: E501
        ) from exc
    if getattr(qa_lab_rs, "ping", lambda: "")() != "qa_lab_rs:ok":
        raise RuntimeError("qa_lab_rs ping failed")
    return qa_lab_rs


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _load_canonical_validator():
    if not CANONICAL_VALIDATOR.exists():
        raise RuntimeError(f"Canonical validator missing: {CANONICAL_VALIDATOR}")
    spec = importlib.util.spec_from_file_location("qa_canonical_validator", CANONICAL_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load canonical validator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spot_check_invariants(
    qa_lab_rs: object,
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(df))
    if sample_size <= 0:
        return
    validator = _load_canonical_validator()
    indices = rng.choice(len(df), size=sample_size, replace=False)
    for idx in indices:
        row = df.iloc[int(idx)]
        b = int(row["b"])
        e = int(row["e"])
        d = int(row["d"])
        a = int(row["a"])
        rust = qa_lab_rs.compute_bundle_py(float(b), float(e), float(d), float(a))
        canonical = validator.construct_qa_state(b, e)
        checks = {
            "J": canonical.J,
            "K": canonical.K,
            "X": canonical.X,
            "W": canonical.W,
            "Y": canonical.Y,
            "Z": canonical.Z,
            "C": canonical.C,
            "F": canonical.F,
            "G": canonical.G,
        }
        for key, value in checks.items():
            if float(rust[key]) != float(value):
                raise RuntimeError(
                    f"Rust invariant mismatch at row {idx}: {key} rust={rust[key]} canonical={value}"
                )


def _kmeans_numpy(x: np.ndarray, k: int, max_iters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, n))
    centers = x[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = x[mask].mean(axis=0)
            else:
                centers[i] = x[rng.integers(0, n)]

    return labels


def _extract_patterns(df: pd.DataFrame) -> List[Dict[str, object]]:
    patterns: List[Dict[str, object]] = []

    # Modular constants
    for mod_col in ["b_mod24", "e_mod24", "d_mod24", "a_mod24"]:
        if mod_col in df.columns:
            unique_vals = df[mod_col].unique()
            if len(unique_vals) == 1:
                patterns.append({
                    "type": "modular_constant",
                    "column": mod_col,
                    "value": int(unique_vals[0]),
                    "count": int(len(df)),
                })

    # Modular identity
    if {"a_mod24", "b_mod24", "e_mod24"}.issubset(df.columns):
        expected = (df["b_mod24"] + 2 * df["e_mod24"]) % 24
        if (df["a_mod24"] == expected).all():
            patterns.append({
                "type": "modular_identity",
                "formula": "a ≡ b + 2e (mod 24)",
                "count": int(len(df)),
            })

    # Algebraic identities
    if {"b", "e", "d"}.issubset(df.columns):
        if (df["d"] == df["b"] + df["e"]).all():
            patterns.append({
                "type": "algebraic_identity",
                "formula": "d = b + e",
                "count": int(len(df)),
            })
    if {"a", "b", "e"}.issubset(df.columns):
        if (df["a"] == df["b"] + 2 * df["e"]).all():
            patterns.append({
                "type": "algebraic_identity",
                "formula": "a = b + 2e",
                "count": int(len(df)),
            })

    # Geometry dominance
    if "geometry" in df.columns:
        geom_counts = df["geometry"].value_counts()
        for geom_type, count in geom_counts.items():
            if count / len(df) > 0.8:
                patterns.append({
                    "type": "geometric_class",
                    "geometry": str(geom_type),
                    "proportion": float(count / len(df)),
                    "count": int(count),
                })

    return patterns


def _rank_score(patterns: List[Dict[str, object]]) -> float:
    score = 0.0
    for pattern in patterns:
        ptype = pattern.get("type")
        if ptype == "modular_identity":
            score += 10
        elif ptype == "algebraic_identity":
            score += 8
        elif ptype == "geometric_class":
            score += float(pattern.get("proportion", 0.0)) * 5
        elif ptype == "modular_constant":
            score += 3
        score += np.log(float(pattern.get("count", 1)))
    return float(score)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rust-backed embedding + conjecture mining")
    parser.add_argument("--dataset", default="qa_10000_balanced_tuples.csv", help="Input CSV")
    parser.add_argument("--embeddings-out", default="qa_embeddings.npy", help="Embeddings output")
    parser.add_argument("--clusters-out", default="qa_clusters.npy", help="Cluster labels output")
    parser.add_argument("--conjectures-out", default="conjectures.json", help="Conjectures output")
    parser.add_argument(
        "--conjectures-list-out",
        default="conjectures_list.json",
        help="List-only conjectures for Lean verifier",
    )
    parser.add_argument("--k", type=int, default=32, help="K-means clusters")
    parser.add_argument("--max-iters", type=int, default=50, help="K-means max iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--min-cluster", type=int, default=3, help="Minimum cluster size")
    parser.add_argument("--qa-lab-root", default="qa_lab", help="Path to qa_lab root")
    parser.add_argument("--skip-closure-check", action="store_true", help="Skip closure validation")
    parser.add_argument("--skip-spot-check", action="store_true", help="Skip canonical invariant check")
    parser.add_argument("--spot-check-n", type=int, default=50, help="Spot-check sample size")

    args = parser.parse_args()

    qa_lab_rs = _ensure_rust_backend(Path(args.qa_lab_root))

    dataset_path = Path(args.dataset)
    logger.info("Loading dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path)

    required = {"b", "e", "d", "a"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Dataset contains NaN values in required columns")

    b = df["b"].to_numpy(dtype=np.float64)
    e = df["e"].to_numpy(dtype=np.float64)
    d = df["d"].to_numpy(dtype=np.float64)
    a = df["a"].to_numpy(dtype=np.float64)

    if not args.skip_closure_check:
        if not (np.allclose(d, b + e, rtol=0.0, atol=1e-9) and np.allclose(a, b + 2 * e, rtol=0.0, atol=1e-9)):
            raise ValueError("Dataset violates canonical closure (d=b+e, a=b+2e)")

    if not args.skip_spot_check:
        logger.info("Running canonical invariant spot-check")
        _spot_check_invariants(qa_lab_rs, df, args.spot_check_n, args.seed)

    logger.info("Computing invariants via qa_lab_rs")
    inv = qa_lab_rs.compute_bundle_batch_numpy_closure_py(b, e)

    embeddings = np.stack(
        [
            b,
            e,
            d,
            a,
            np.asarray(inv["J"], dtype=np.float64),
            np.asarray(inv["K"], dtype=np.float64),
            np.asarray(inv["X"], dtype=np.float64),
            np.asarray(inv["W"], dtype=np.float64),
        ],
        axis=1,
    )

    embeddings_path = Path(args.embeddings_out)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    logger.info("Embeddings saved to %s", embeddings_path)

    logger.info("Clustering embeddings with k=%d", args.k)
    labels = _kmeans_numpy(embeddings, args.k, args.max_iters, args.seed)
    clusters_path = Path(args.clusters_out)
    clusters_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(clusters_path, labels)

    df["cluster"] = labels
    conjectures: List[Dict[str, object]] = []

    for cluster_id in sorted(set(labels)):
        cluster_df = df[df["cluster"] == cluster_id]
        if len(cluster_df) < args.min_cluster:
            continue
        patterns = _extract_patterns(cluster_df)
        if not patterns:
            continue
        rep_index = int(cluster_df.index[0])
        rep_tuple = {
            "b": int(b[rep_index]),
            "e": int(e[rep_index]),
            "d": int(d[rep_index]),
            "a": int(a[rep_index]),
        }
        rep_packet = {
            "J": int(inv["J"][rep_index]),
            "K": int(inv["K"][rep_index]),
            "X": int(inv["X"][rep_index]),
            "W": int(inv["W"][rep_index]),
            "Y": int(inv["Y"][rep_index]),
            "Z": int(inv["Z"][rep_index]),
            "C": int(inv["C"][rep_index]),
            "F": int(inv["F"][rep_index]),
            "G": int(inv["G"][rep_index]),
        }
        packet_hash = _sha256_text(json.dumps(rep_packet, sort_keys=True))
        tuple_hash = _sha256_text(json.dumps(rep_tuple, sort_keys=True))
        conjectures.append({
            "cluster_id": int(cluster_id),
            "patterns": patterns,
            "tuple_count": int(len(cluster_df)),
            "rank_score": _rank_score(patterns),
            "representative": {
                "row_index": rep_index,
                "tuple": rep_tuple,
                "tuple_hash": tuple_hash,
                "packet": rep_packet,
                "packet_hash": packet_hash,
            },
        })

    conjectures.sort(key=lambda x: (-x["rank_score"], x["cluster_id"]))

    dataset_path = Path(args.dataset)
    dataset_sha = _sha256_file(dataset_path)
    git_commit = _git_commit()
    rs_path = Path(getattr(qa_lab_rs, "__file__", ""))
    rs_meta = None
    if rs_path and rs_path.exists():
        rs_meta = {
            "path": str(rs_path),
            "sha256": _sha256_file(rs_path),
            "mtime": rs_path.stat().st_mtime,
            "size": rs_path.stat().st_size,
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "dataset_rows": int(len(df)),
        "git_commit": git_commit,
        "qa_lab_rs": rs_meta,
        "cluster_config": {
            "k": args.k,
            "max_iters": args.max_iters,
            "seed": args.seed,
            "min_cluster": args.min_cluster,
        },
        "embeddings_path": str(embeddings_path),
        "clusters_path": str(clusters_path),
    }

    conjectures_path = Path(args.conjectures_out)
    conjectures_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "conjectures": conjectures}
    conjectures_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Conjectures saved to %s", conjectures_path)

    list_path = Path(args.conjectures_list_out)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(json.dumps(conjectures, indent=2), encoding="utf-8")
    logger.info("Conjectures list saved to %s", list_path)

    logger.info("Total conjectures: %d", len(conjectures))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
