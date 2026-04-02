#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_graph_query.py — Query the QA knowledge graph via QA tuple semantics

Entry point:
  python qa_graph_query.py "What is Harmonic Index?" \
      --graph artifacts/knowledge/qa_knowledge_graph.graphml \
      --top-k 5 --save

Notes
- Converts query text → QA tuple (hash-based) and scores nodes by HI of query→node transition.
- HI = E8_alignment × exp(-k × loss), with loss a simple normalized triangle residual.
- Saves optional JSON snapshot under artifacts/evals/ when --save is passed.

Usage
  - Minimal: python qa_graph_query.py "Find all Bell test experiments"
  - With explicit graph: python qa_graph_query.py "..." --graph artifacts/knowledge/qa_knowledge_graph.graphml
  - Save results: add --save to write artifacts/evals/qa_graph_query_*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import networkx as nx  # type: ignore


def _import_fastpath():
    try:
        import qa_lab.qa_fastpath as fp  # type: ignore
        return fp
    except Exception:
        pass
    try:
        sys.path.append(os.path.abspath("qa_lab"))
        import qa_fastpath as fp  # type: ignore
        return fp
    except Exception:
        return None


def _import_e8_simple():
    try:
        import qa_lab.qa_e8_alignment as e8  # type: ignore
        return e8
    except Exception:
        try:
            sys.path.append(os.path.abspath("qa_lab"))
            import qa_e8_alignment as e8  # type: ignore
            return e8
        except Exception:
            return None


def hash_to_qa_tuple(text: str, modulus: int = 24) -> Tuple[int, int, int, int]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    b = int.from_bytes(h[0:4], "big") % modulus
    e = int.from_bytes(h[4:8], "big") % modulus
    d = (b + e) % modulus
    a = (b + 2 * e) % modulus
    return int(b), int(e), int(d), int(a)


def triangle_residual(b: int, e: int, d: int, a: int) -> float:
    c = 2.0 * e * d
    f = b * a
    g = e * e + d * d
    return float(abs(c * c + f * f - g * g))


def e8_alignment(fp, e8_module, b: int, e: int, d: int, a: int) -> float:
    try:
        if fp is not None:
            roots_info = fp.get_e8_roots()
            if roots_info is not None:
                roots, _unit = roots_info
                import numpy as np
                vec = fp.build_e8_vectors(np.array([b], dtype=float),
                                          np.array([e], dtype=float),
                                          np.array([d], dtype=float),
                                          np.array([a], dtype=float))
                scores = fp.e8_scores_auto(vec, roots)
                return float(scores[0])
    except Exception:
        pass
    if e8_module is not None:
        try:
            return float(e8_module.e8_alignment_single(float(b), float(e), float(d), float(a)))
        except Exception:
            pass
    return 0.0


def score_query_to_node(fp, e8_module, q: Tuple[int, int, int, int], n: Tuple[int, int, int, int]) -> float:
    qb, qe, qd, qa = q
    nb, ne, nd, na = n
    # Transition tuple modulo 24
    tb = (nb - qb) % 24
    te = (ne - qe) % 24
    td = (nd - qd) % 24
    ta = (na - qa) % 24
    e8 = e8_alignment(fp, e8_module, tb, te, td, ta)
    tri = triangle_residual(tb, te, td, ta)
    denom = 1.0 + (tb*tb + te*te + td*td + ta*ta)
    loss = float(tri / denom)
    k = 0.1
    hi = float(e8 * math.exp(-k * loss))
    return hi


def load_graph(graph_path: str) -> nx.DiGraph:
    G = nx.read_graphml(graph_path)
    # GraphML loads attributes as strings; ensure numeric fields are numeric
    for n, attrs in G.nodes(data=True):
        for key in ("b", "e", "d", "a"):
            if key in attrs:
                try:
                    attrs[key] = int(attrs[key])
                except Exception:
                    pass
        for key in ("e8_alignment", "hi", "loss"):
            if key in attrs:
                try:
                    attrs[key] = float(attrs[key])
                except Exception:
                    pass
    return G


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query the QA knowledge graph")
    parser.add_argument("query", help="Free-text query")
    parser.add_argument("--graph", dest="graph_path", default="artifacts/knowledge/qa_knowledge_graph.graphml",
                        help="GraphML path to load")
    parser.add_argument("--top-k", dest="top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--save", dest="save", action="store_true", help="Save results JSON under artifacts/evals/")
    args = parser.parse_args(argv)

    if not os.path.exists(args.graph_path):
        print(f"[ERROR] GraphML not found: {args.graph_path}", file=sys.stderr)
        return 2

    G = load_graph(args.graph_path)
    fp = _import_fastpath()
    e8m = _import_e8_simple()

    q_tuple = hash_to_qa_tuple(args.query)
    scores: List[Tuple[str, float]] = []
    for n in G.nodes():
        nb, ne, nd, na = (G.nodes[n].get("b"), G.nodes[n].get("e"), G.nodes[n].get("d"), G.nodes[n].get("a"))
        if None in (nb, ne, nd, na):
            continue
        s = score_query_to_node(fp, e8m, q_tuple, (int(nb), int(ne), int(nd), int(na)))
        scores.append((n, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[: max(0, args.top_k)]

    print(f"[qa_graph_query] Query: {args.query}")
    for rank, (name, sc) in enumerate(top, start=1):
        node = G.nodes[name]
        print(f"{rank:2d}. {name}  score={sc:.4f}  e8={node.get('e8_alignment', 0.0):.4f}  hi={node.get('hi', 0.0):.4f}")

    if args.save:
        out_dir = os.path.abspath("artifacts/evals")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"qa_graph_query_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
        payload = {
            "query": args.query,
            "query_tuple": q_tuple,
            "graph": os.path.abspath(args.graph_path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "results": [
                {
                    "name": name,
                    "score": float(sc),
                    "b": int(G.nodes[name].get("b", 0)),
                    "e": int(G.nodes[name].get("e", 0)),
                    "d": int(G.nodes[name].get("d", 0)),
                    "a": int(G.nodes[name].get("a", 0)),
                    "definition": G.nodes[name].get("definition", ""),
                }
                for (name, sc) in top
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[qa_graph_query] Saved results → {out_path}")

    return 0


if __name__ == "__main__":  # --- Main ---
    raise SystemExit(main())

