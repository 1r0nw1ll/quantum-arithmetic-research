"""QA reachability graph: nodes = (b, e) in {1,...,m}^2, edges = legal generators.

Each (b, e) is assigned a stable index by enumerating the grid in row-major
order (matching tools.qa_ml.qa_dataset.all_pairs). Edges are produced by
applying every generator in qa_ml.qa_generators.GENERATORS to every node and
recording the (src_idx, dst_idx, gen_name) triples for which the generator is
defined.

QA_COMPLIANCE = "qa_ml_graph — deterministic enumeration; no random graph models"
"""

from __future__ import annotations

import numpy as np

from .qa_dataset import all_pairs
from .qa_generators import GENERATORS


def build_edges(m: int) -> list[tuple[int, int, str]]:
    """Return list of (src_idx, dst_idx, gen_name) edges over the full mod-m grid."""
    pairs = all_pairs(m)
    idx = {p: i for i, p in enumerate(pairs)}
    edges: list[tuple[int, int, str]] = []
    for src_idx, (b, e) in enumerate(pairs):
        for gen_name, gen_fn in GENERATORS.items():
            result = gen_fn(b, e, m)
            if result is None:
                continue
            edges.append((src_idx, idx[result], gen_name))
    return edges


def dense_adjacency(m: int, *, symmetric: bool = True) -> np.ndarray:
    """Dense float64 adjacency matrix of the QA reachability graph for modulus m.

    symmetric=True (default for spectral GCN) adds reverse edges so that
    message passing flows both ways. The float64 cast is observer-side; node
    states stay int.
    """
    n = m * m
    a = np.zeros((n, n), dtype=np.float64)
    for src, dst, _ in build_edges(m):
        a[src, dst] = 1.0
        if symmetric:
            a[dst, src] = 1.0
    return a


def gcn_normalize(adj: np.ndarray) -> np.ndarray:
    """Symmetric GCN normalization: D^(-1/2) (A + I) D^(-1/2)."""
    a_hat = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    deg = a_hat.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    return (a_hat * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


def edge_count_by_generator(m: int) -> dict[str, int]:
    """Diagnostic: how many edges does each generator contribute?"""
    counts: dict[str, int] = {name: 0 for name in GENERATORS}
    for _, _, gen_name in build_edges(m):
        counts[gen_name] += 1
    return counts
