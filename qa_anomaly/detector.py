"""QAGraphAnomalyDetector — main public class."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ._graph import (
    Adjacency,
    bfs_distances,
    connected_components,
    normalize,
    pseudo_diameter_anchors,
)
from ._scores import score_node


class QAGraphAnomalyDetector:
    """
    Zero-parameter graph anomaly detector based on QA discrete versor algebra.

    Assigns each node an anomaly score using the QA monotone-direction criterion:
    nodes on structural branches score high; nodes on the main path score zero.
    No training required. No hyperparameters. Pure integer arithmetic.

    Parameters
    ----------
    score_type : str
        'monotone_dir'  (default) — AUROC 0.81 on path-branch benchmark.
                                    Best for tree-like graphs.
        'koenig_gap'    — more robust on graphs with cycles/shortcuts.
        'composite'     — monotone_dir + Wildberger spread; attachment-sensitive.

    anchor : 'auto' | tuple
        'auto' (default) — double-BFS pseudo-diameter; no domain knowledge.
        (L, R)           — supply specific anchor nodes.

    handle_components : 'largest' | 'all'
        'largest'  — score only the largest connected component (default).
        'all'      — score each component independently with its own anchors.

    Examples
    --------
    >>> from qa_anomaly import QAGraphAnomalyDetector
    >>> det = QAGraphAnomalyDetector()

    # networkx
    >>> import networkx as nx
    >>> G = nx.path_graph(20)
    >>> nx.add_path(G, [8, 100, 101, 102])   # branch at node 8
    >>> scores = det.score(G)
    >>> sorted(scores, key=scores.get, reverse=True)[:5]

    # edge list
    >>> edges = [(0,1),(1,2),(2,3),(3,4),(2,10),(10,11)]
    >>> scores = det.score(edges)

    # dict-of-lists
    >>> adj = {0:[1,2], 1:[0,3], 2:[0,3], 3:[1,2]}
    >>> scores = det.score(adj)
    """

    def __init__(
        self,
        score_type: str = "monotone_dir",
        anchor: Union[str, Tuple[Any, Any]] = "auto",
        handle_components: str = "largest",
    ) -> None:
        if score_type not in ("monotone_dir", "koenig_gap", "composite"):
            raise ValueError(f"Unknown score_type: {score_type!r}")
        if handle_components not in ("largest", "all"):
            raise ValueError(f"Unknown handle_components: {handle_components!r}")
        self.score_type = score_type
        self.anchor = anchor
        self.handle_components = handle_components

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, graph: Any) -> Dict[Any, float]:
        """
        Compute anomaly scores for every node.

        Parameters
        ----------
        graph : networkx.Graph | list of (u,v) edges | dict-of-lists adjacency

        Returns
        -------
        dict mapping node → float anomaly score.
        Higher = more anomalous.
        """
        adj = normalize(graph)
        return self._score_adjacency(adj)

    def top_k(self, graph: Any, k: int = 10) -> List[Tuple[Any, float]]:
        """Return the k highest-scoring (most anomalous) nodes."""
        scores = self.score(graph)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def fit(self, graph: Any) -> "QAGraphAnomalyDetector":
        """No-op — kept for sklearn pipeline compatibility."""
        return self

    def fit_score(self, graph: Any) -> Dict[Any, float]:
        return self.score(graph)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_adjacency(self, adj: Adjacency) -> Dict[Any, float]:
        scores: Dict[Any, float] = {}

        if self.handle_components == "largest":
            components = connected_components(adj)
            components.sort(key=len, reverse=True)
            work = [components[0]]
        else:
            work = connected_components(adj)

        for component in work:
            if len(component) < 3:
                for node in component:
                    scores[node] = 0.0
                continue

            sub_adj = {n: [nb for nb in adj[n] if nb in set(component)]
                       for n in component}

            if self.anchor == "auto":
                left, right = pseudo_diameter_anchors(sub_adj)
            else:
                left, right = self.anchor

            dist_left  = bfs_distances(sub_adj, left)
            dist_right = bfs_distances(sub_adj, right)

            for node in component:
                neighbors = sub_adj.get(node, [])
                scores[node] = score_node(
                    node, neighbors, dist_left, dist_right, self.score_type
                )

        # Nodes not scored (tiny components) get 0
        for node in adj:
            scores.setdefault(node, 0.0)

        return scores
