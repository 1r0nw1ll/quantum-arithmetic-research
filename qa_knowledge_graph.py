#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_knowledge_graph.py — Build a QA knowledge graph with E8 alignment

Entry point:
  python qa_knowledge_graph.py --enc artifacts/knowledge/qa_entity_encodings.json \
                               --out artifacts/knowledge/qa_knowledge_graph.graphml

Notes
- Nodes: canonical entities with QA tuples and metrics (E8, HI).
- Edges: simple mention-based relationships derived from definitions.
- E8 calculations reuse QA Lab utilities if available; falls back to simplified metric.

Usage
  - Minimal: python qa_knowledge_graph.py
  - Custom:  python qa_knowledge_graph.py --enc artifacts/knowledge/qa_entity_encodings.json --out artifacts/knowledge/qa_knowledge_graph.graphml
  - Prints a short summary and writes GraphML to artifacts/knowledge/.
"""

from __future__ import annotations

import argparse
import json
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


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def compute_transition_strength(fp, e8_module, t: Tuple[int, int, int, int]) -> float:
    b, e, d, a = t
    # Prefer real E8 roots path
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
    # Fallback: simplified cosine against ideal root in 4D
    if e8_module is not None:
        try:
            return float(e8_module.e8_alignment_single(float(b), float(e), float(d), float(a)))
        except Exception:
            pass
    return 0.0


def tuple_diff_mod24(src: Tuple[int, int, int, int], tgt: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return tuple(int((tgt[i] - src[i]) % 24) for i in range(4))  # type: ignore


def build_graph(enc_path: str) -> nx.DiGraph:
    with open(enc_path, "r", encoding="utf-8") as f:
        enc_data = json.load(f)
    encs: List[Dict] = enc_data.get("encodings", [])

    fp = _import_fastpath()
    e8m = _import_e8_simple()

    G = nx.DiGraph()

    # Add nodes
    for rec in encs:
        name = rec["name"]
        qa_tuple = (int(rec["b"]), int(rec["e"]), int(rec["d"]), int(rec["a"]))
        G.add_node(name,
                   name=name,
                   slug=rec.get("slug", ""),
                   section=rec.get("section", ""),
                   definition=rec.get("definition", ""),
                   symbol=rec.get("symbol", ""),
                   qa_tuple=",".join(str(x) for x in qa_tuple),  # GraphML-safe
                   b=int(rec["b"]), e=int(rec["e"]), d=int(rec["d"]), a=int(rec["a"]),
                   loss=float(rec.get("loss", 0.0)),
                   e8_alignment=float(rec.get("e8_alignment", 0.0)),
                   hi=float(rec.get("hi", 0.0)))

    # Simple mention-based edges: if entity j's name appears in entity i's definition
    # Case-insensitive substring match, excluding self
    names = list(G.nodes())
    name_lc = {n: n.lower() for n in names}
    defs_lc = {n: (G.nodes[n].get("definition", "").lower()) for n in names}

    edge_count = 0
    for src in names:
        dtext = defs_lc.get(src, "")
        if not dtext:
            continue
        for tgt in names:
            if tgt == src:
                continue
            if name_lc[tgt] and name_lc[tgt] in dtext:
                sb, se, sd, sa = (G.nodes[src]["b"], G.nodes[src]["e"], G.nodes[src]["d"], G.nodes[src]["a"])
                tb, te, td, ta = (G.nodes[tgt]["b"], G.nodes[tgt]["e"], G.nodes[tgt]["d"], G.nodes[tgt]["a"])
                trans = tuple_diff_mod24((sb, se, sd, sa), (tb, te, td, ta))
                strength = compute_transition_strength(fp, e8m, trans)
                G.add_edge(src, tgt,
                           relationship="mentions",
                           transition=",".join(str(x) for x in trans),
                           strength=float(strength))
                edge_count += 1

    # Secondary rule: connect nodes that share important keywords in definitions/names
    keywords = {
        "e₈", "e8", "harmonic", "index", "fingerprint", "markovian", "equivariant",
        "operator", "coherence", "deviation", "baseline", "seismic", "finance", "detector",
        "rotation", "dashboard", "probabilistic", "kernel", "alignment"
    }
    def tokens(text: str) -> set:
        toks = set()
        for w in re.split(r"[^\w₈]+", text.lower()):
            if not w:
                continue
            if w in {"the", "and", "of", "a", "to", "in", "on", "for", "with", "is", "are"}:
                continue
            toks.add(w)
        return toks

    import re  # local import to avoid top-level dep for simple tokenization
    name_tokens = {n: tokens(n) for n in names}
    def_tokens = {n: tokens(defs_lc.get(n, "")) for n in names}
    for i, src in enumerate(names):
        for tgt in names[i+1:]:
            if src == tgt:
                continue
            ts = (name_tokens[src] | def_tokens[src]) & (name_tokens[tgt] | def_tokens[tgt])
            ts = {t for t in ts if t in keywords}
            if ts:
                sb, se, sd, sa = (G.nodes[src]["b"], G.nodes[src]["e"], G.nodes[src]["d"], G.nodes[src]["a"])
                tb, te, td, ta = (G.nodes[tgt]["b"], G.nodes[tgt]["e"], G.nodes[tgt]["d"], G.nodes[tgt]["a"])
                trans = tuple_diff_mod24((sb, se, sd, sa), (tb, te, td, ta))
                strength = compute_transition_strength(fp, e8m, trans)
                G.add_edge(src, tgt,
                           relationship="related",
                           shared_terms=",".join(sorted(ts)),
                           transition=",".join(str(x) for x in trans),
                           strength=float(strength))
                edge_count += 1

    return G


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build QA knowledge graph from encodings")
    parser.add_argument("--enc", dest="enc_path", default="artifacts/knowledge/qa_entity_encodings.json",
                        help="Input encodings JSON path")
    parser.add_argument("--out", dest="out_path", default="artifacts/knowledge/qa_knowledge_graph.graphml",
                        help="Output GraphML path")
    args = parser.parse_args(argv)

    if not os.path.exists(args.enc_path):
        print(f"[ERROR] Encodings JSON not found: {args.enc_path}", file=sys.stderr)
        return 2

    G = build_graph(args.enc_path)

    ensure_dir(args.out_path)
    nx.write_graphml(G, args.out_path)
    print(f"[qa_knowledge_graph] Nodes={G.number_of_nodes()} Edges={G.number_of_edges()} → {args.out_path}")
    return 0


if __name__ == "__main__":  # --- Main ---
    raise SystemExit(main())
