#!/usr/bin/env python3
"""
Fetch the classic college football network (football.gml) and convert to GraphML.

Tries multiple known URLs. Requires networkx for conversion.

Usage:
  python codex_on_QA/scripts/fetch_football.py \
    --out-gml codex_on_QA/data/football.gml \
    --out-graphml codex_on_QA/data/football.graphml
"""
import argparse, os, sys, urllib.request, ssl

URLS = [
    # Gephi test resources (commonly used football.gml)
    "https://raw.githubusercontent.com/gephi/gephi/master/modules/ImportPluginExample/src/test/resources/org/gephi/io/importer/plugin/file/gml/resources/football.gml",
    # NetworkX legacy mirror (may change)
    "https://raw.githubusercontent.com/networkx/networkx/main/networkx/generators/tests/football.gml",
]

def download(url, dest):
    print(f"[fetch] Trying {url}")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
        data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
    print(f"[fetch] Saved {dest} ({len(data)} bytes)")

def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def convert_gml_to_graphml(gml_path, graphml_path):
    try:
        import networkx as nx
    except Exception as e:
        print("[convert] networkx not installed. Install via: pip install networkx", file=sys.stderr)
        raise
    print(f"[convert] Reading {gml_path}")
    G = nx.read_gml(gml_path)
    print(f"[convert] Nodes={G.number_of_nodes()} Edges={G.number_of_edges()}")
    ensure_dir(graphml_path)
    nx.write_graphml(G, graphml_path)
    print(f"[convert] Wrote GraphML to {graphml_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-gml", default="codex_on_QA/data/football.gml")
    ap.add_argument("--out-graphml", default="codex_on_QA/data/football.graphml")
    args = ap.parse_args()

    ensure_dir(args.out_gml)
    last_err = None
    for url in URLS:
        try:
            download(url, args.out_gml)
            break
        except Exception as e:
            last_err = e
            print(f"[fetch] Failed from {url}: {e}")
    else:
        print("[fetch] All URLs failed", file=sys.stderr)
        if last_err:
            raise last_err
        sys.exit(2)

    try:
        convert_gml_to_graphml(args.out_gml, args.out_graphml)
    except Exception as e:
        print(f"[convert] Conversion failed: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()

