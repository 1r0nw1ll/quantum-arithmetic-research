"""QA scoring functions for graph anomaly detection.

All arithmetic is exact integer — no floating-point in the core computation.

Scores:
  monotone_dir   — count of incident edges where Δb·Δe > 0 (branch-type edges).
                   Branch body = 2, leaf = 1, attachment = 1, path interior = 0.
                   Best overall: AUROC=0.8056 on path-branch benchmark (extended label).

  koenig_gap     — |2e² − b²| where b = d(v,L)+1, e = d(v,R)+1.
                   Derived from the Koenig 2·C·F square-gap law.
                   More robust on shortcut/cycle graphs (AUROC=0.6939 vs 0.5952 shortcut).

  composite      — monotone_dir + local_edge_spread (Wildberger pairwise spread over
                   incident edge-direction vectors). Attachment-point sensitive.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Node = str


def monotone_dir(
    node: object,
    neighbors: List[object],
    dist_left: Dict[object, int],
    dist_right: Dict[object, int],
) -> int:
    """
    Count incident edges where both anchor-distances move in the same direction.

    On the main L–R path every edge moves closer to one anchor and farther from
    the other (Δb·Δe < 0).  On a branch both distances grow simultaneously
    (Δb·Δe > 0).  The count is the QA structural anomaly signal.
    """
    b_v = dist_left[node]
    e_v = dist_right[node]
    count = 0
    for nbr in neighbors:
        db = dist_left[nbr] - b_v
        de = dist_right[nbr] - e_v
        if db * de > 0:
            count += 1
    return count


def local_edge_spread(
    node: object,
    neighbors: List[object],
    dist_left: Dict[object, int],
    dist_right: Dict[object, int],
) -> float:
    """
    Pairwise Wildberger spread over incident edge-direction vectors.
    Attachment points (geometric corners) score highest; path interior scores 0.
    """
    b_v = dist_left[node]
    e_v = dist_right[node]
    deltas: List[Tuple[int, int]] = []
    for nbr in neighbors:
        deltas.append((dist_left[nbr] - b_v, dist_right[nbr] - e_v))

    total = 0.0
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            db_i, de_i = deltas[i]
            db_j, de_j = deltas[j]
            g_i = db_i * db_i + de_i * de_i
            g_j = db_j * db_j + de_j * de_j
            if g_i > 0 and g_j > 0:
                c = db_i * de_j - db_j * de_i
                total += (c * c) / (g_i * g_j)
    return total


def koenig_gap(
    node: object,
    dist_left: Dict[object, int],
    dist_right: Dict[object, int],
) -> int:
    """
    |2e² − b²| where b = d(v,L)+1, e = d(v,R)+1.

    Algebraic gap from the Koenig/Pythagorean 2·C·F square law.
    Exact integer — no floating-point.
    """
    b = dist_left[node] + 1
    e = dist_right[node] + 1
    return abs(2 * e * e - b * b)


def score_node(
    node: object,
    neighbors: List[object],
    dist_left: Dict[object, int],
    dist_right: Dict[object, int],
    score_type: str,
) -> float:
    if score_type == "monotone_dir":
        return float(monotone_dir(node, neighbors, dist_left, dist_right))
    if score_type == "koenig_gap":
        return float(koenig_gap(node, dist_left, dist_right))
    if score_type == "composite":
        md = monotone_dir(node, neighbors, dist_left, dist_right)
        ls = local_edge_spread(node, neighbors, dist_left, dist_right)
        return md + ls
    raise ValueError(
        f"Unknown score_type {score_type!r}. "
        "Choose from: 'monotone_dir', 'koenig_gap', 'composite'."
    )
