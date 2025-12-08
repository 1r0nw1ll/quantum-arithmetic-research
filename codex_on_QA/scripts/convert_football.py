#!/usr/bin/env python3
"""
Convert an existing football.gml into football.graphml using networkx.

Usage:
  python codex_on_QA/scripts/convert_football.py \
    --in-gml codex_on_QA/data/football.gml \
    --out-graphml codex_on_QA/data/football.graphml
"""
import argparse, os

def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-gml', default='codex_on_QA/data/football.gml')
    ap.add_argument('--out-graphml', default='codex_on_QA/data/football.graphml')
    args = ap.parse_args()

    try:
        import networkx as nx  # type: ignore
    except Exception:
        raise SystemExit('networkx not installed. Activate venv then: pip install networkx')

    if not os.path.exists(args.in_gml):
        raise SystemExit(f'Input not found: {args.in_gml}')

    G = nx.read_gml(args.in_gml)
    ensure_dir(args.out_graphml)
    nx.write_graphml(G, args.out_graphml)
    print(f'Wrote GraphML: {args.out_graphml} (nodes={G.number_of_nodes()} edges={G.number_of_edges()})')

if __name__ == '__main__':
    main()

