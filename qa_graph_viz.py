#!/usr/bin/env python3
"""
qa_graph_viz.py — Visualize the QA knowledge graph

Entry point:
  python qa_graph_viz.py --graph artifacts/knowledge/qa_knowledge_graph.graphml \
                         --out artifacts/plots/qa_knowledge_graph.png

Notes
- Colors nodes by HI (or E8 alignment as fallback); sizes by degree.
- Saves a PNG to artifacts/plots/ suitable for reports.

Usage
  - Minimal: python qa_graph_viz.py
  - Custom:  python qa_graph_viz.py --graph artifacts/knowledge/qa_knowledge_graph.graphml --out artifacts/plots/qa_knowledge_graph.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import networkx as nx  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
except Exception as e:  # pragma: no cover
    plt = None
    mpl = None


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_graph(graph_path: str) -> nx.DiGraph:
    G = nx.read_graphml(graph_path)
    for n, attrs in G.nodes(data=True):
        for key in ("e8_alignment", "hi"):
            if key in attrs:
                try:
                    attrs[key] = float(attrs[key])
                except Exception:
                    pass
    return G


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize the QA knowledge graph")
    parser.add_argument("--graph", dest="graph_path", default="artifacts/knowledge/qa_knowledge_graph.graphml",
                        help="GraphML path")
    parser.add_argument("--out", dest="out_path", default="artifacts/plots/qa_knowledge_graph.png",
                        help="Output PNG path")
    args = parser.parse_args(argv)

    if plt is None:
        print("[ERROR] matplotlib not available. Please install it in the venv.", file=sys.stderr)
        return 2
    if not os.path.exists(args.graph_path):
        print(f"[ERROR] GraphML not found: {args.graph_path}", file=sys.stderr)
        return 2

    G = load_graph(args.graph_path)
    pos = nx.spring_layout(G, seed=42)

    # Node color by HI (fallback to e8)
    his = [float(G.nodes[n].get("hi", G.nodes[n].get("e8_alignment", 0.0))) for n in G.nodes()]
    degrees = dict(G.degree())
    sizes = [300 + 50 * degrees.get(n, 0) for n in G.nodes()]

    cmap = mpl.cm.viridis
    vmin = min(his) if his else 0.0
    vmax = max(his) if his else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(10, 8), dpi=160)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.0)
    nodes = nx.draw_networkx_nodes(G, pos,
                                   node_size=sizes,
                                   node_color=[norm(v) for v in his],
                                   cmap=cmap)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.colorbar(nodes, shrink=0.7, label="HI (or E8 alignment)")
    plt.axis('off')

    ensure_dir(args.out_path)
    plt.tight_layout()
    plt.savefig(args.out_path)
    plt.close()

    print(f"[qa_graph_viz] Saved visualization → {args.out_path}")
    return 0


if __name__ == "__main__":  # --- Main ---
    raise SystemExit(main())
