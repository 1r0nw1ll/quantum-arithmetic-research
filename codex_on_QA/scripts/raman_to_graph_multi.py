#!/usr/bin/env python3
"""
Build a multi‑tuple Raman QA graph (GraphML) from a CSV containing
multiple (b,e) tuples per sample.

Input CSV schema (header required):
  id, b1, e1, b2, e2, ..., label

For each tuple index n detected from the header (b{n}/e{n} pairs), the
script computes canonical QA invariants and attaches them to each node
using the prefix t{n}_*. A compact concatenated feature space is built
from [X, J, K] per tuple to construct a k‑NN graph.

Outputs a GraphML compatible with the existing Rust spectral pipeline
(`qa_graph_experiments`). Edge weights are set to 1.0; a `dist` edge
attribute (pre‑zscored Euclidean distance in the concatenated [X,J,K]
space) is included for reference.

Usage (from repo root):
  PYTHONPATH=. python codex_on_QA/scripts/raman_to_graph_multi.py \
    --csv codex_on_QA/out/raman_multi_fundovt_fingerprint_multiseg.csv \
    --qa-mode qa21 \
    --k 8 \
    --out codex_on_QA/data/raman_multi_qa21.graphml

Optional Markovian HI ingestion (CSV with columns: id, t1_hi, t2_hi, ... or id, hi1, hi2, ...):
  PYTHONPATH=. python codex_on_QA/scripts/raman_to_graph_multi.py \
    --csv codex_on_QA/out/raman_multi_fundovt_fingerprint_multiseg.csv \
    --hi-csv codex_on_QA/out/raman_markovian_hi.csv \
    --qa-mode qa21 --k 8 \
    --out codex_on_QA/data/raman_multi_qa21_markovhi.graphml

Dependencies: numpy
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from codex_on_QA.feature_maps.qa_feature_map_v3 import compute_qa_invariants, CANONICAL_21


# --- helpers -----------------------------------------------------------------

def detect_tuple_indices(headers: List[str]) -> List[int]:
    """Detect n for which both b{n} and e{n} columns exist."""
    b_idx = set()
    e_idx = set()
    for h in headers:
        if h.startswith("b") and h[1:].isdigit():
            b_idx.add(int(h[1:]))
        if h.startswith("e") and h[1:].isdigit():
            e_idx.add(int(h[1:]))
    return sorted(b_idx & e_idx)


def zscore_columns(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    mu = mat.mean(axis=0)
    sig = mat.std(axis=0)
    sig[sig == 0.0] = 1.0
    return (mat - mu) / sig


def knn_edges(features: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    """Return undirected k‑NN edges as (i, j, dist) with i<j, using L2 on rows."""
    n = features.shape[0]
    # pairwise dists via simple loops to bound memory
    edges_set = set()
    out: List[Tuple[int, int, float]] = []
    for i in range(n):
        # compute dists to all j
        diffs = features - features[i]
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)
        nn = [int(j) for j in order[1 : k + 1] if int(j) != i]
        for j in nn:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edges_set:
                continue
            edges_set.add((a, b))
            out.append((a, b, float(dists[j])))
    return out


def build_node_attrs(
    row: Dict[str, str], tuple_indices: List[int], qa_mode: str
) -> Dict[str, float]:
    attrs: Dict[str, float] = {}
    # compute invariants per tuple and prefix with t{n}_
    for n in tuple_indices:
        try:
            b = float(row[f"b{n}"])
            e = float(row[f"e{n}"])
        except Exception:
            # skip malformed tuples
            continue
        inv = compute_qa_invariants(b, e)
        # Store canonical 21 keys with prefix
        for key in CANONICAL_21:
            if key in inv:
                attrs[f"t{n}_" + key] = float(inv[key])
        # Always keep raw b/e as well (redundant with canonical, but explicit)
        attrs[f"t{n}_b"] = b
        attrs[f"t{n}_e"] = e
    return attrs


def features_from_attrs(node_attrs: List[Dict[str, float]], tuple_indices: List[int]) -> np.ndarray:
    """Concatenate [X, J, K] per tuple to form a compact space for kNN."""
    keys: List[str] = []
    for n in tuple_indices:
        keys.extend([f"t{n}_X", f"t{n}_J", f"t{n}_K"])
    # drop keys not present anywhere
    present = [k for k in keys if any(k in d for d in node_attrs)]
    if not present:
        return np.zeros((len(node_attrs), 0), dtype=float)
    mat = np.zeros((len(node_attrs), len(present)), dtype=float)
    for i, attrs in enumerate(node_attrs):
        for j, k in enumerate(present):
            mat[i, j] = float(attrs.get(k, 0.0))
    return mat


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Build multi‑tuple Raman QA GraphML from CSV")
    ap.add_argument("--csv", required=True, help="Input CSV with id,b1,e1,b2,e2,...,label")
    ap.add_argument("--qa-mode", default="qa21", choices=["qa21", "qa27", "qa83"], help="QA mode (affects invariants computed; edge features use X/J/K per tuple)")
    ap.add_argument("--k", type=int, default=8, help="k for kNN construction")
    ap.add_argument("--out", required=True, help="Output GraphML path")
    ap.add_argument("--hi-csv", default=None, help="Optional CSV with per-tuple Markovian HI: id,(t1_hi|hi1),(t2_hi|hi2),...")
    ap.add_argument("--id-col", default="id", help="ID column name")
    ap.add_argument("--label-col", default="label", help="Label column name")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional Markovian HI map
    markov_hi: Dict[str, Dict[int, float]] = {}
    if args.hi_csv:
        hi_path = Path(args.hi_csv)
        if not hi_path.exists():
            raise SystemExit(f"--hi-csv not found: {hi_path}")
        with hi_path.open("r", newline="") as hf:
            r = csv.DictReader(hf)
            hcols = r.fieldnames or []
            # detect tuple hi columns: t{n}_hi or hi{n}
            def tuple_hi_cols(cols: List[str]) -> List[Tuple[int, str]]:
                out: List[Tuple[int, str]] = []
                for c in cols:
                    if c.startswith("t") and c.endswith("_hi") and c[1:-3].isdigit():
                        out.append((int(c[1:-3]), c))
                    elif c.startswith("hi") and c[2:].isdigit():
                        out.append((int(c[2:]), c))
                return sorted(out)
            tcols = tuple_hi_cols(hcols)
            if not tcols:
                raise SystemExit("--hi-csv has no tuple HI columns (expected t1_hi, t2_hi,... or hi1,hi2,...)")
            for row in r:
                rid = str(row.get("id"))
                if rid is None:
                    continue
                d: Dict[int, float] = {}
                for n, cname in tcols:
                    v = row.get(cname)
                    if v is None or v == "":
                        continue
                    try:
                        d[n] = float(v)
                    except Exception:
                        pass
                markov_hi[rid] = d

    with in_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if args.id_col not in headers:
            raise SystemExit(f"Missing id column '{args.id_col}' in {in_path}")
        if args.label_col not in headers:
            raise SystemExit(f"Missing label column '{args.label_col}' in {in_path}")
        tuple_indices = detect_tuple_indices(headers)
        if not tuple_indices:
            raise SystemExit("No tuple columns detected (expect b1/e1, b2/e2, ...) in header")

        node_ids: List[str] = []
        node_labels: List[str] = []
        node_attrs: List[Dict[str, float]] = []
        for row in reader:
            node_ids.append(str(row[args.id_col]))
            node_labels.append(str(row[args.label_col]))
            attrs = build_node_attrs(row, tuple_indices, args.qa_mode)
            node_attrs.append(attrs)

    # Build compact features and z‑score per column
    feats = features_from_attrs(node_attrs, tuple_indices)
    if feats.shape[1] == 0:
        raise SystemExit("Could not assemble feature matrix from X/J/K per tuple")
    feats = zscore_columns(feats)

    # Manual GraphML write with explicit keys compatible with Rust parser
    # k_value -> node ground truth (attr.name="value")
    # k_d1..k_d4 -> (b,e,d,a) from tuple1 (FO)
    # k_w -> edge weight
    edges = knn_edges(feats, args.k)

    with out_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        f.write('  <key id="k_value" for="node" attr.name="value" attr.type="int"/>\n')
        f.write('  <key id="k_d1" for="node" attr.name="d1" attr.type="double"/>\n')
        f.write('  <key id="k_d2" for="node" attr.name="d2" attr.type="double"/>\n')
        f.write('  <key id="k_d3" for="node" attr.name="d3" attr.type="double"/>\n')
        f.write('  <key id="k_d4" for="node" attr.name="d4" attr.type="double"/>\n')
        # declare keys for per‑tuple raw (b,e) so Rust can build multi‑tuple kernels
        for n in tuple_indices:
            f.write(f'  <key id="t{n}_b" for="node" attr.name="t{n}_b" attr.type="double"/>\n')
            f.write(f'  <key id="t{n}_e" for="node" attr.name="t{n}_e" attr.type="double"/>\n')
            f.write(f'  <key id="t{n}_hi" for="node" attr.name="t{n}_hi" attr.type="double"/>\n')
        f.write('  <key id="k_w" for="edge" attr.name="weight" attr.type="double"/>\n')
        f.write('  <graph id="raman_multi" edgedefault="undirected">\n')

        # nodes
        for i, (nid, lab, attrs) in enumerate(zip(node_ids, node_labels, node_attrs)):
            # tuple1 (FO) for (b,e)
            b1 = float(attrs.get("t1_b", 0.0))
            e1 = float(attrs.get("t1_e", 0.0))
            d1 = b1 + e1
            a1 = b1 + 2.0 * e1
            f.write(f'    <node id="{nid}">\n')
            f.write(f'      <data key="k_value">{int(lab)}</data>\n')
            f.write(f'      <data key="k_d1">{b1:.9f}</data>\n')
            f.write(f'      <data key="k_d2">{e1:.9f}</data>\n')
            f.write(f'      <data key="k_d3">{d1:.9f}</data>\n')
            f.write(f'      <data key="k_d4">{a1:.9f}</data>\n')
            # emit t{n}_b/e and a canonical harmonicity index t{n}_hi for all tuples present
            for n in tuple_indices:
                vb = attrs.get(f"t{n}_b")
                ve = attrs.get(f"t{n}_e")
                if vb is not None:
                    f.write(f'      <data key="t{n}_b">{float(vb):.9f}</data>\n')
                if ve is not None:
                    f.write(f'      <data key="t{n}_e">{float(ve):.9f}</data>\n')
                # prefer Markovian HI if provided, else canonical closure dev
                if vb is not None and ve is not None:
                    hi_val = None
                    if nid in markov_hi:
                        hi_val = markov_hi[nid].get(n)
                    if hi_val is None:
                        b = float(vb); e = float(ve)
                        d = b + e; a = b + 2.0*e
                        c = 2.0 * d * e
                        ff = b * a
                        g = e*e + d*d
                        hi_val = abs((g*g) - (c*c + ff*ff)) / (1.0 + g*g + c*c + ff*ff)
                    f.write(f'      <data key="t{n}_hi">{float(hi_val):.9f}</data>\n')
            f.write('    </node>\n')

        # edges (use node_ids for source/target)
        for i, j, _dist in edges:
            src = node_ids[i]
            dst = node_ids[j]
            f.write(f'    <edge source="{src}" target="{dst}">\n')
            f.write(f'      <data key="k_w">1.0</data>\n')
            f.write('    </edge>\n')

        f.write('  </graph>\n')
        f.write('</graphml>\n')

    print(f"[raman_to_graph_multi] Wrote graph with {len(node_ids)} nodes, {len(edges)} edges to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
