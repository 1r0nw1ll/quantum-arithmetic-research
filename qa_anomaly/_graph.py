"""Graph primitives — adjacency normalization and BFS. No external dependencies."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

Node = Hashable
Adjacency = Dict[Node, List[Node]]


def from_networkx(G: Any) -> Adjacency:
    adj: Adjacency = {}
    for node in G.nodes():
        adj[node] = list(G.neighbors(node))
    return adj


def from_edge_list(
    edges: Iterable[Tuple[Any, Any]],
    *,
    directed: bool = False,
) -> Adjacency:
    adj: Adjacency = {}
    for u, v in edges:
        adj.setdefault(u, [])
        adj.setdefault(v, [])
        if v not in adj[u]:
            adj[u].append(v)
        if not directed and u not in adj[v]:
            adj[v].append(u)
    return adj


def from_adjacency_dict(d: Mapping[Any, Iterable[Any]]) -> Adjacency:
    return {node: list(neighbors) for node, neighbors in d.items()}


def normalize(graph: Any) -> Adjacency:
    """Accept networkx Graph, edge list, or dict-of-lists. Return Adjacency."""
    if isinstance(graph, dict):
        return from_adjacency_dict(graph)
    if isinstance(graph, (list, tuple)):
        return from_edge_list(graph)
    # networkx duck-typing: has .nodes() and .neighbors()
    if hasattr(graph, "nodes") and hasattr(graph, "neighbors"):
        return from_networkx(graph)
    raise TypeError(
        f"Unsupported graph type {type(graph).__name__}. "
        "Pass a networkx Graph, edge list, or dict-of-lists adjacency."
    )


def bfs_distances(adj: Adjacency, source: Node) -> Dict[Node, int]:
    dist: Dict[Node, int] = {source: 0}
    queue: deque = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


def pseudo_diameter_anchors(adj: Adjacency) -> Tuple[Node, Node]:
    """Double-BFS pseudo-diameter: no domain knowledge required."""
    start = next(iter(adj))
    d1 = bfs_distances(adj, start)
    u = max(d1, key=d1.__getitem__)
    d2 = bfs_distances(adj, u)
    v = max(d2, key=d2.__getitem__)
    return u, v


def connected_components(adj: Adjacency) -> List[List[Node]]:
    visited: set = set()
    components: List[List[Node]] = []
    for start in adj:
        if start in visited:
            continue
        component: List[Node] = []
        queue: deque = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components
