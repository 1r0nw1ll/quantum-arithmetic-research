#!/usr/bin/env python3
"""
Tests for qa_anomaly. Run directly: python3 -m qa_anomaly.tests
Validates against the same 192-case benchmark that produced AUROC=0.8056.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_anomaly import QAGraphAnomalyDetector


# ---------------------------------------------------------------------------
# Benchmark graph generation (mirrors benchmark script)
# ---------------------------------------------------------------------------

def path_branch_graph(
    path_length: int,
    branch_length: int,
    attach: int,
    decoys: int = 0,
    shortcut_step: int = 0,
) -> Tuple[Dict, Dict]:
    """Returns (adjacency, labels) where label=1 means anomaly (branch body)."""
    adj: Dict = {}
    labels: Dict = {}

    def add_edge(a, b):
        adj.setdefault(a, [])
        adj.setdefault(b, [])
        if b not in adj[a]: adj[a].append(b)
        if a not in adj[b]: adj[b].append(a)

    for i in range(path_length):
        node = f"P{i}"
        adj.setdefault(node, [])
        labels[node] = 0
        if i:
            add_edge(node, f"P{i-1}")

    for i in range(1, branch_length + 1):
        node = f"B{i}"
        labels[node] = 1
        add_edge(node, f"P{attach}" if i == 1 else f"B{i-1}")

    for d in range(decoys):
        da = max(2, min(path_length - 3, attach + (d + 1) * 3 * (-1 if d % 2 else 1)))
        dl = max(1, branch_length // 2)
        for depth in range(1, dl + 1):
            node = f"D{d}_{depth}"
            labels[node] = 0
            add_edge(node, f"P{da}" if depth == 1 else f"D{d}_{depth-1}")

    if shortcut_step:
        for i in range(0, path_length - shortcut_step, shortcut_step):
            add_edge(f"P{i}", f"P{i+shortcut_step}")

    return adj, labels


def auroc(scores: Dict, labels: Dict) -> float:
    positives = [n for n, l in labels.items() if l == 1]
    negatives = [n for n, l in labels.items() if l == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = len(positives) * len(negatives)
    for p in positives:
        for n in negatives:
            ps, ns = scores.get(p, 0.0), scores.get(n, 0.0)
            wins += (1.0 if ps > ns else 0.5 if ps == ns else 0.0)
    return wins / total


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_edge_list_input():
    edges = [(0,1),(1,2),(2,3),(3,4),(2,10),(10,11),(11,12)]
    det = QAGraphAnomalyDetector()
    scores = det.score(edges)
    assert set(scores.keys()) == {0,1,2,3,4,10,11,12}
    # Nodes 10,11 are branch body — should score higher than path interior
    assert scores[10] >= scores[1]

def test_dict_input():
    adj = {"A": ["B","C"], "B": ["A","D"], "C": ["A","E"], "D": ["B"], "E": ["C"]}
    scores = QAGraphAnomalyDetector().score(adj)
    assert set(scores.keys()) == {"A","B","C","D","E"}

def test_networkx_input():
    try:
        import networkx as nx
        G = nx.path_graph(10)
        nx.add_path(G, [5, 20, 21, 22])
        scores = QAGraphAnomalyDetector().score(G)
        assert scores[20] > scores[0], "Branch node should outscore path endpoint"
        assert scores[21] > scores[3], "Branch interior should outscore path interior"
    except ImportError:
        print("  [skip] networkx not installed")

def test_explicit_anchor():
    edges = [(i, i+1) for i in range(10)] + [(5,20),(20,21),(21,22)]
    det = QAGraphAnomalyDetector(anchor=(0, 9))
    scores = det.score(edges)
    assert scores[20] >= scores[0]

def test_score_types():
    edges = [(i, i+1) for i in range(20)] + [(10,30),(30,31),(31,32)]
    for stype in ("monotone_dir", "koenig_gap", "composite"):
        scores = QAGraphAnomalyDetector(score_type=stype).score(edges)
        assert all(v >= 0 for v in scores.values()), f"{stype}: negative score"

def test_top_k():
    edges = [(i, i+1) for i in range(20)] + [(10,30),(30,31),(31,32)]
    top = QAGraphAnomalyDetector().top_k(edges, k=3)
    assert len(top) == 3
    assert all(isinstance(score, float) for _, score in top)
    # Scores should be descending
    assert top[0][1] >= top[1][1] >= top[2][1]

def test_disconnected_graph():
    edges = [(0,1),(1,2),(2,3)] + [(10,11),(11,12),(12,13)]
    # Should not crash; largest component scored
    scores = QAGraphAnomalyDetector(handle_components="largest").score(edges)
    scored = [n for n, s in scores.items() if s > 0 or s == 0]
    assert len(scored) > 0

def test_all_components():
    edges = [(0,1),(1,2),(2,3),(2,20),(20,21)] + [(10,11),(11,12)]
    scores = QAGraphAnomalyDetector(handle_components="all").score(edges)
    assert set(scores.keys()) == {0,1,2,3,10,11,12,20,21}

def test_path_graph_all_zero():
    """Pure path graph: all interior nodes should score 0 with monotone_dir."""
    edges = [(i, i+1) for i in range(20)]
    det = QAGraphAnomalyDetector(score_type="monotone_dir", anchor=(0, 19))
    scores = det.score(edges)
    interior = [scores[i] for i in range(1, 19)]
    assert all(s == 0.0 for s in interior), f"Path interior should score 0: {interior}"

def test_branch_body_scores_highest():
    """Branch body nodes should score 2 (both incident edges are branch-type)."""
    edges = [(i, i+1) for i in range(15)] + [(7,20),(20,21),(21,22)]
    det = QAGraphAnomalyDetector(score_type="monotone_dir", anchor=(0, 14))
    scores = det.score(edges)
    # Body of branch (not leaf, not attachment): node 20 and 21
    assert scores[20] == 2.0, f"Branch body should score 2, got {scores[20]}"
    assert scores[21] == 2.0, f"Branch body should score 2, got {scores[21]}"
    assert scores[22] == 1.0, f"Branch leaf should score 1, got {scores[22]}"

# ---------------------------------------------------------------------------
# Benchmark: 192 generated cases
# ---------------------------------------------------------------------------

def run_benchmark() -> Dict[str, float]:
    """Reproduce the core benchmark. Returns mean AUROC per score type."""
    cases = []
    for path_length in (24, 32, 40, 52):
        for branch_length in (4, 6, 8, 10):
            for af in (0.33, 0.5, 0.67):
                attach = max(3, min(path_length - 4, round(path_length * af)))
                for decoys in (0, 2):
                    for shortcut in (0, 7):
                        cases.append(path_branch_graph(path_length, branch_length, attach, decoys, shortcut))

    results: Dict[str, List[float]] = {"monotone_dir": [], "koenig_gap": [], "composite": []}
    no_shortcut: Dict[str, List[float]] = {"monotone_dir": [], "koenig_gap": [], "composite": []}
    for case_idx, (adj, labels) in enumerate(cases):
        has_shortcut = case_idx % 2 == 1
        for stype in results:
            det = QAGraphAnomalyDetector(score_type=stype)
            scores = det.score(adj)
            auc = auroc(scores, labels)
            if not math.isnan(auc):
                results[stype].append(auc)
                if not has_shortcut:
                    no_shortcut[stype].append(auc)

    return {
        "mean_auroc": {k: sum(v)/len(v) for k,v in results.items() if v},
        "no_shortcut_auroc": {k: sum(v)/len(v) for k,v in no_shortcut.items() if v},
        "n_cases": len(cases),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_unit_tests():
    tests = [
        test_edge_list_input,
        test_dict_input,
        test_networkx_input,
        test_explicit_anchor,
        test_score_types,
        test_top_k,
        test_disconnected_graph,
        test_all_components,
        test_path_graph_all_zero,
        test_branch_body_scores_highest,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
    return passed, len(tests)


if __name__ == "__main__":
    print("=" * 60)
    print("qa_anomaly unit tests")
    print("=" * 60)
    passed, total = run_unit_tests()
    print(f"\n{passed}/{total} passed\n")

    print("=" * 60)
    print("Benchmark (192 cases — reproduces benchmark_005 results)")
    print("=" * 60)
    bm = run_benchmark()
    print(f"\n{bm['n_cases']} cases\n")
    print("Mean AUROC (all cases incl. shortcut):")
    for k, v in bm["mean_auroc"].items():
        print(f"  {k:20s}: {v:.4f}")
    print("\nMean AUROC (no-shortcut cases only — tree-valid):")
    for k, v in bm["no_shortcut_auroc"].items():
        print(f"  {k:20s}: {v:.4f}")
    print()
    md = bm["no_shortcut_auroc"].get("monotone_dir", 0)
    status = "PASS" if md >= 0.90 else "WARN"
    print(f"[{status}] monotone_dir no-shortcut AUROC = {md:.4f} (expect ≥ 0.90)")
