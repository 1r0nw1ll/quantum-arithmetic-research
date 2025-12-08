#!/usr/bin/env python3
"""
Add classic benchmark graphs (Karate, Dolphins) into codex_on_QA/data as GraphML.

- Karate: generated via networkx.karate_club_graph; node attribute 'value' is club id {0,1}.
- Dolphins: if a GML/GraphML is present at codex_on_QA/data/dolphins.(gml|graphml),
  standardize to GraphML and ensure a 'value' attribute if possible (left null otherwise).

This script does not download from network.
"""
from __future__ import annotations

import os
from typing import Optional


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def write_karate_graphml(out_path: str) -> None:
    import networkx as nx
    G = nx.karate_club_graph()
    # Map club string to int {0,1}
    clubs = sorted({G.nodes[n].get('club') for n in G.nodes})
    club2id = {c: i for i, c in enumerate(clubs)}
    for n in G.nodes:
        c = G.nodes[n].get('club')
        G.nodes[n]['value'] = club2id.get(c, 0)
    nx.write_graphml(G, out_path)
    print('Wrote', out_path)


def convert_dolphins(src_path: str, out_path: str) -> None:
    import networkx as nx
    if src_path.lower().endswith('.gml'):
        G = nx.read_gml(src_path)
    else:
        G = nx.read_graphml(src_path)
    # Ensure undirected
    G = G.to_undirected()
    # If no 'value' labels present, leave unlabeled (None)
    nx.write_graphml(G, out_path)
    print('Wrote', out_path)


def main() -> int:
    data_dir = os.path.join('codex_on_QA', 'data')
    ensure_dir(data_dir)
    # Karate
    kar_path = os.path.join(data_dir, 'karate.graphml')
    try:
        write_karate_graphml(kar_path)
    except Exception as e:
        print('Karate generation failed:', e)
    # Dolphins (optional, if source exists)
    dol_src_gml = os.path.join(data_dir, 'dolphins.gml')
    dol_src_xml = os.path.join(data_dir, 'dolphins.graphml')
    dol_out = os.path.join(data_dir, 'dolphins.graphml')
    if os.path.exists(dol_src_gml):
        convert_dolphins(dol_src_gml, dol_out)
    elif os.path.exists(dol_src_xml):
        convert_dolphins(dol_src_xml, dol_out)
    else:
        print('Dolphins not found. Place dolphins.gml or dolphins.graphml under', data_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

