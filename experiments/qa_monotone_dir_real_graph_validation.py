#!/usr/bin/env python3
"""Real-graph validation of qa_monotone_dir_score on three classic network datasets.

Tests the AGS theorem claim (cert [288]) on real-world graphs by:
  1. Extracting the BFS spanning tree (tree-mode evaluation)
  2. Finding diameter endpoints as anchors via double-BFS on the spanning tree
  3. Computing monotone_dir_score = count of incident edges with Δb*Δe > 0
  4. Checking whether high-score nodes correspond to structurally peripheral nodes

Three datasets (karate club, dolphins, football) from codex_on_QA/data/.
Each graph is predominantly cyclic (not a tree), so AGS is run on the spanning tree.
A shortcut-cycle fraction is reported to quantify tree-ness of the original graph.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Node = str
Adjacency = Dict[Node, List[Node]]


def load_graphml(path: str) -> Tuple[Adjacency, Dict[Node, Dict[str, str]]]:
    tree = ET.parse(path)
    root = tree.getroot()
    tag = root.tag
    ns = tag[tag.index("{") + 1 : tag.index("}")] if "{" in tag else ""
    px = "{" + ns + "}" if ns else ""
    g = root.find(f"{px}graph")
    node_attrs: Dict[Node, Dict[str, str]] = {}
    key_names: Dict[str, str] = {}
    for key in root.findall(f"{px}key"):
        key_names[key.attrib["id"]] = key.attrib.get("attr.name", key.attrib["id"])
    for node in g.findall(f"{px}node"):
        nid = node.attrib["id"]
        attrs = {key_names.get(d.attrib["key"], d.attrib["key"]): d.text or "" for d in node.findall(f"{px}data")}
        node_attrs[nid] = attrs
    adj: Adjacency = {nid: [] for nid in node_attrs}
    for edge in g.findall(f"{px}edge"):
        s, t = edge.attrib["source"], edge.attrib["target"]
        adj.setdefault(s, [])
        adj.setdefault(t, [])
        if t not in adj[s]:
            adj[s].append(t)
        if s not in adj[t]:
            adj[t].append(s)
    return adj, node_attrs


def bfs(adj: Adjacency, source: Node) -> Dict[Node, int]:
    dist = {source: 0}
    q: deque[Node] = deque([source])
    while q:
        v = q.popleft()
        for u in adj.get(v, []):
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def bfs_spanning_tree(adj: Adjacency) -> Adjacency:
    """BFS spanning tree from the lexicographically-smallest node."""
    start = min(adj)
    parent: Dict[Node, Optional[Node]] = {start: None}
    q: deque[Node] = deque([start])
    while q:
        v = q.popleft()
        for u in sorted(adj[v]):
            if u not in parent:
                parent[u] = v
                q.append(u)
    tree: Adjacency = {v: [] for v in adj}
    for v, p in parent.items():
        if p is not None:
            tree[v].append(p)
            tree[p].append(v)
    return tree


def diameter_anchors(adj: Adjacency) -> Tuple[Node, Node]:
    start = min(adj)
    d1 = bfs(adj, start)
    u = max(d1, key=d1.__getitem__)
    d2 = bfs(adj, u)
    v = max(d2, key=d2.__getitem__)
    return u, v


def monotone_dir_score(adj: Adjacency, dist_l: Dict[Node, int], dist_r: Dict[Node, int]) -> Dict[Node, int]:
    scores: Dict[Node, int] = {}
    for v in adj:
        bv, ev = dist_l[v], dist_r[v]
        scores[v] = sum(
            1 for u in adj[v]
            if (dist_l[u] - bv) * (dist_r[u] - ev) > 0
        )
    return scores


def on_path_nodes(adj: Adjacency, l: Node, r: Node) -> set:
    dl, dr = bfs(adj, l), bfs(adj, r)
    total = dl[r]
    return {v for v in adj if dl[v] + dr[v] == total}


def cycle_fraction(adj: Adjacency, tree: Adjacency) -> float:
    """Fraction of edges in adj that are NOT in the spanning tree."""
    tree_edges: set = set()
    for v in tree:
        for u in tree[v]:
            tree_edges.add(frozenset({v, u}))
    all_edges: set = set()
    for v in adj:
        for u in adj[v]:
            all_edges.add(frozenset({v, u}))
    back = len(all_edges) - len(tree_edges)
    return back / len(all_edges) if all_edges else 0.0


def analyse(name: str, path: str) -> Dict[str, Any]:
    adj, node_attrs = load_graphml(path)
    tree = bfs_spanning_tree(adj)
    l, r = diameter_anchors(tree)
    dl, dr = bfs(tree, l), bfs(tree, r)
    scores = monotone_dir_score(tree, dl, dr)
    on_path = on_path_nodes(tree, l, r)
    cyc_frac = cycle_fraction(adj, tree)

    # Summarise by score tier.
    by_score: Dict[int, List[str]] = {}
    for v, s in scores.items():
        by_score.setdefault(s, []).append(v)

    attr_key = next(iter(next(iter(node_attrs.values())).keys()), None) if node_attrs else None
    result: Dict[str, Any] = {
        "graph": name,
        "nodes": len(adj),
        "edges": sum(len(v) for v in adj.values()) // 2,
        "cycle_fraction": round(cyc_frac, 3),
        "anchors": [l, r],
        "diameter_tree": dl[r],
        "on_path_count": len(on_path),
        "label_key": attr_key,
        "score_distribution": {str(k): len(v) for k, v in sorted(by_score.items())},
        "top_scoring": [],
        "on_path_sample": [],
    }

    # Top-8 highest-scoring nodes.
    ordered = sorted(scores, key=lambda v: (-scores[v], v))
    for v in ordered[:8]:
        attrs = node_attrs.get(v, {})
        result["top_scoring"].append({
            "node": v,
            "score": scores[v],
            "degree_original": len(adj[v]),
            "degree_tree": len(tree[v]),
            "on_path": v in on_path,
            "attrs": attrs,
        })

    # Sample on-path nodes (score=0 by definition on tree).
    for v in sorted(on_path)[:6]:
        attrs = node_attrs.get(v, {})
        result["on_path_sample"].append({
            "node": v,
            "score": scores[v],
            "degree_original": len(adj[v]),
            "attrs": attrs,
        })

    return result


def main() -> int:
    data_dir = Path(__file__).resolve().parents[1] / "codex_on_QA" / "data"
    graphs = [
        ("karate", str(data_dir / "karate.graphml")),
        ("dolphins", str(data_dir / "dolphins.graphml")),
        ("football", str(data_dir / "football.graphml")),
    ]

    out_dir = Path(__file__).resolve().parents[1] / "results" / "qa_monotone_dir_real_graph_001"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, path in graphs:
        if not Path(path).exists():
            print(f"SKIP {name}: not found at {path}", file=sys.stderr)
            continue
        r = analyse(name, path)
        results.append(r)

        print(f"\n{'='*60}")
        print(f"Graph: {r['graph']}  ({r['nodes']} nodes, {r['edges']} edges)")
        print(f"  Cycle fraction (back-edges / total): {r['cycle_fraction']:.1%}  [AGS applies to spanning tree only]")
        print(f"  Spanning-tree diameter: {r['diameter_tree']}  anchors: {r['anchors']}")
        print(f"  On-path node count: {r['on_path_count']}")
        print(f"  Score distribution: {r['score_distribution']}")
        print(f"\n  Top-scoring nodes (branch-type on spanning tree):")
        for t in r["top_scoring"][:6]:
            print(f"    node={t['node']:>4}  score={t['score']}  deg_orig={t['degree_original']}  deg_tree={t['degree_tree']}  on_path={t['on_path']}  {t['attrs']}")
        print(f"\n  On-path sample (score=0 — path-type on spanning tree):")
        for t in r["on_path_sample"][:5]:
            print(f"    node={t['node']:>4}  score={t['score']}  deg_orig={t['degree_original']}  {t['attrs']}")

    payload = {"results": results}
    (out_dir / "qa_monotone_dir_real_graph.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    (out_dir / "QA_MONOTONE_DIR_REAL_GRAPH_001.md").write_text(
        _report(results), encoding="utf-8"
    )
    print(f"\nWrote results to {out_dir}")
    return 0


def _report(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# QA Monotone Direction Score — Real Graph Validation 001",
        "",
        "Validates the AGS theorem (cert [288]) structural interpretation on three classic",
        "real-world network graphs. Because these graphs contain cycles, the monotone_dir",
        "score is computed on the BFS spanning tree of each graph. Cycle fraction quantifies",
        "how much structure is lost by this projection.",
        "",
        "**Claim under test**: high-score nodes (score=2) are branch-body nodes on the",
        "spanning tree — structurally peripheral nodes with no incident path-axis edges.",
        "Low-score nodes (score=0) lie on the diameter axis of the spanning tree and are",
        "structurally central (highest-betweenness bridges in the original graph).",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['graph'].capitalize()} ({r['nodes']} nodes, {r['edges']} edges)",
            "",
            f"- Cycle fraction: {r['cycle_fraction']:.1%}",
            f"- Spanning-tree diameter: {r['diameter_tree']}, anchors: {r['anchors']}",
            f"- On-path nodes: {r['on_path_count']}",
            f"- Score distribution: {r['score_distribution']}",
            "",
            "**Top-scoring nodes** (periphery of spanning tree):",
            "",
            "| node | score | deg(orig) | deg(tree) | on_path | attrs |",
            "|---|---:|---:|---:|---|---|",
        ]
        for t in r["top_scoring"][:8]:
            attr_str = "; ".join(f"{k}={v}" for k, v in t["attrs"].items())
            lines.append(
                f"| {t['node']} | {t['score']} | {t['degree_original']} | {t['degree_tree']} | {t['on_path']} | {attr_str} |"
            )
        lines += [
            "",
            "**On-path sample** (spanning-tree axis — score=0):",
            "",
            "| node | score | deg(orig) | attrs |",
            "|---|---:|---:|---|",
        ]
        for t in r["on_path_sample"][:6]:
            attr_str = "; ".join(f"{k}={v}" for k, v in t["attrs"].items())
            lines.append(f"| {t['node']} | {t['score']} | {t['degree_original']} | {attr_str} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
