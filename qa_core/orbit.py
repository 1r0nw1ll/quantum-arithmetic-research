"""Orbit and coupling helpers shared by QA simulations."""

from __future__ import annotations

import numpy as np


def complete_graph_adjacency(num_nodes: int) -> np.ndarray:
    """Return the dense zero-diagonal adjacency used by legacy signal scripts."""

    return np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)


def resonance_matrix(tuples: np.ndarray, modulus: int) -> np.ndarray:
    """Compute the modular tuple resonance matrix."""

    return np.einsum("ik,jk->ij", tuples, tuples) % modulus


def weighted_adjacency(adjacency_matrix: np.ndarray, resonance: np.ndarray, modulus: int) -> np.ndarray:
    """Apply the legacy resonance weighting to an adjacency matrix."""

    scale = resonance / modulus
    return adjacency_matrix * scale * scale


def neighbor_pull(weighted_adj: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Compute weighted neighbor averages with zero-safe normalization."""

    sum_w = np.sum(weighted_adj, axis=1)
    sum_w[sum_w == 0] = 1
    return (weighted_adj @ values) / sum_w
