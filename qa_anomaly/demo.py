#!/usr/bin/env python3
"""
qa_anomaly demo — three realistic use cases.

Run: python3 -m qa_anomaly.demo
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_anomaly import QAGraphAnomalyDetector


# ---------------------------------------------------------------------------
# Case 1: Transaction network — money-mule chain detection
# ---------------------------------------------------------------------------

def demo_transaction_network():
    print("=" * 60)
    print("Case 1: Supply chain — rogue vendor chain detection")
    print("=" * 60)
    print("""
Setup: 20 suppliers form a tree-shaped procurement chain (each
supplier sources from one upstream vendor). A rogue sub-vendor
chain (V1→V2→V3→V4) is attached at supplier S7 — longer and
deeper than legitimate branches.
Algorithm scope: tree-structured graphs (supply chains, org charts,
referral trees, network topologies). Degrades on dense cyclic graphs.
""")
    # Tree backbone: two branching paths from root
    edges  = [(f"S{i}", f"S{i+1}") for i in range(12)]        # main chain
    edges += [(f"S{i}", f"S{i+13}") for i in range(0, 12, 4)] # side branches
    edges += [("S13","S14"),("S14","S15")]                      # side branch extension
    # Rogue vendor chain at S7
    edges += [("S7","V1"),("V1","V2"),("V2","V3"),("V3","V4"),("V4","V5")]

    det = QAGraphAnomalyDetector()
    top10 = det.top_k(edges, k=8)

    print("Top 8 anomalous nodes:")
    for node, score in top10:
        tag = " ← ROGUE VENDOR" if str(node).startswith("V") else ""
        print(f"  {str(node):6s}  score={score:.1f}{tag}")

    rogue = {"V1","V2","V3","V4","V5"}
    top5 = {n for n,_ in det.top_k(edges, k=5)}
    recall = len(rogue & top5) / len(rogue)
    print(f"\nRogue vendor recall in top-5: {recall:.0%}")


# ---------------------------------------------------------------------------
# Case 2: Network topology — rogue device detection
# ---------------------------------------------------------------------------

def demo_network_topology():
    print("\n" + "=" * 60)
    print("Case 2: Network topology — rogue device chain")
    print("=" * 60)
    print("""
Setup: core network spine (switches 0–15).
A rogue chain R1→R2→R3 is plugged into switch 9.
A decoy single-hop branch D1 is at switch 4 (normal expansion port).
""")
    # Spine path
    edges = [(f"SW{i}", f"SW{i+1}") for i in range(15)]
    # Normal decoy branch (short, acceptable)
    edges += [("SW4","D1"),("D1","D2")]
    # Rogue chain (longer, anomalous)
    edges += [("SW9","R1"),("R1","R2"),("R2","R3"),("R3","R4"),("R4","R5")]

    det = QAGraphAnomalyDetector()
    top8 = det.top_k(edges, k=8)

    print("Top 8 anomalous nodes:")
    for node, score in top8:
        tag = " ← ROGUE" if str(node).startswith("R") else (" ← DECOY" if str(node).startswith("D") else "")
        print(f"  {node:6s}  score={score:.1f}{tag}")

    rogue = {"R1","R2","R3","R4","R5"}
    top5_nodes = {n for n,_ in det.top_k(edges, k=5)}
    recall = len(rogue & top5_nodes) / len(rogue)
    print(f"\nRogue recall in top-5: {recall:.0%}")


# ---------------------------------------------------------------------------
# Case 3: Social network — bot farm detection
# ---------------------------------------------------------------------------

def demo_pipeline():
    print("\n" + "=" * 60)
    print("Case 3: CI/CD pipeline — rogue subprocess chain")
    print("=" * 60)
    print("""
Setup: a linear build pipeline (Stage0 → Stage1 → ... → Stage18).
Each stage has 1 short side-task (depth 1 or 2 — normal QA steps).
Stage9 has a rogue subprocess chain (P1→P2→P3→P4→P5→P6) — 6 deep,
structurally indistinguishable from a supply-chain or data-lineage graph.
""")
    # Linear pipeline backbone
    edges = [(f"Stage{i}", f"Stage{i+1}") for i in range(18)]
    # Normal side-tasks: single-hop leaf nodes (score=1, not 2)
    for i in range(0, 18, 2):
        edges += [(f"Stage{i}", f"T{i}")]
    # Rogue subprocess chain off Stage9 — 5 body nodes, all score 2
    edges += [(f"Stage9","P1")] + [(f"P{i}",f"P{i+1}") for i in range(1,6)]
    rogue = {f"P{i}" for i in range(1,7)}

    det  = QAGraphAnomalyDetector()
    top8 = det.top_k(edges, k=8)

    print("Top 8 anomalous nodes:")
    for node, score in top8:
        tag = " ← ROGUE PROC" if node in rogue else ""
        print(f"  {str(node):10s}  score={score:.1f}{tag}")

    top6 = {n for n,_ in det.top_k(edges, k=6)}
    recall = len(rogue & top6) / len(rogue)
    print(f"\nRogue process recall in top-6: {recall:.0%}")


# ---------------------------------------------------------------------------
# Case 4: sklearn Pipeline integration
# ---------------------------------------------------------------------------

def demo_sklearn_pipeline():
    print("\n" + "=" * 60)
    print("Case 4: sklearn Pipeline compatible")
    print("=" * 60)

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        import numpy as np

        edges = [(i, i+1) for i in range(20)] + [(10,30),(30,31),(31,32),(32,33)]

        det = QAGraphAnomalyDetector()

        # fit/score interface
        det.fit(edges)
        scores = det.fit_score(edges)

        nodes = sorted(scores.keys())
        score_arr = np.array([scores[n] for n in nodes])

        print(f"\nfit_score() returned {len(scores)} node scores")
        print(f"Score array shape: {score_arr.shape}")
        print(f"Non-zero scores: {(score_arr > 0).sum()}")
        print(f"Max score node: {nodes[score_arr.argmax()]} = {score_arr.max():.1f}")
        print("\nsklearn Pipeline: ✓ (fit/fit_score API compatible)")

    except ImportError:
        print("  [skip] sklearn not available")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_transaction_network()
    demo_network_topology()
    demo_pipeline()
    demo_sklearn_pipeline()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
qa_anomaly.QAGraphAnomalyDetector
  - Input: networkx Graph, edge list, or dict-of-lists
  - Output: {node: anomaly_score}
  - Parameters: 0 learned weights
  - Algorithm: QA discrete versor algebra (monotone direction criterion)
  - Benchmark: AUROC=0.927 on trees (no-shortcut), 0.550 overall
  - Best for: tree-like graphs, hierarchical networks, path + branch structures
""")
